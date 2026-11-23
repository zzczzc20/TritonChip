"""ArchBench's own checks: the definitions are what they claim, and the
scoring refuses what it must.

    python -m archbench.selftest [--tier tiny small]

  1. MACs never overcount: each point's declared MACs are at most what
     torch.utils.flop_counter counts for the same call (FLOPs / 2), and equal
     to it where nothing is masked away (GEMM-like families). Overcounting
     would let a platform beat its own bound.
  2. Bytes: declared writes equal the bytes of the outputs; declared reads
     are at most the bytes of every input, parameter and buffer.
  3. Every point has a positive strict bound on TNCA (cycles) and on the GPU
     spec (seconds).
  4. The frozen suites match the registry (archbench/suites/*.json).
  5. Scoring refuses: eta > 1 raises BoundViolation; edited weights raise
     WeightMismatch; a missing point makes E zero and uncertified.
"""
import argparse
import contextlib
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

from . import registry, suite
from .bounds import gpu_bound, r38_score as S, tnca_bound
from .platforms.common import flatten

# Families whose MACs are exactly the dense count (no causal mask, no
# padding taps): the declared demand must EQUAL the counter's.
EXACT_MACS = {'matmul', 'linear', 'gemv', 'bmm', 'mlp', 'attention_decode', 'fp8_linear'}


def _bytes(ts):
    seen, total = set(), 0
    for t in ts:
        st = t.untyped_storage()
        # a meta tensor has no address: its storage object identifies it (xl points are checked on meta)
        key = st._cdata if t.device.type == 'meta' else st.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total


def _math_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Attention written as matmul-softmax-matmul, for counting only: the
    FLOP counter has no formula for the CPU flash-attention op and counts
    it as zero. This counts the DENSE work (every score, masked or not), so
    a causal point's declared MACs must come out at most this."""
    s = (q @ k.transpose(-2, -1)) * (scale or q.shape[-1] ** -0.5)
    if is_causal:
        n, m = s.shape[-2], s.shape[-1]
        s = s.masked_fill(torch.ones(n, m, dtype=torch.bool, device=s.device).triu(m - n + 1), float('-inf'))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v


def check_point(p):
    problems = []
    # xl points are checked on the meta device: their shapes are real LLM sizes (hundreds of MiB of weights), and
    # everything checked here -- MACs, bytes, memory -- is a property of shapes. Their FP8 modules are run in `emu`
    # mode (the CPU twin of the Hub kernels), which has the same shapes, dtypes and matrix work.
    xl = p.tier == 'xl'
    m, ins = p.make(0, device='meta') if xl else p.make(0)
    m = m.eval()
    real = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = _math_sdpa
    try:
        from . import xl as _xl
        with torch.no_grad(), FlopCounterMode(display=False) as fc, \
                (_xl.mode('emu') if xl else contextlib.nullcontext()):
            out = m(*ins)
    finally:
        torch.nn.functional.scaled_dot_product_attention = real
    counted = fc.get_total_flops() / 2
    d = p.demand
    if d.macs > counted + 1e-6:
        problems.append(f'declares {d.macs:.6g} MACs, the counter finds {counted:.6g}')
    if p.family in EXACT_MACS and abs(d.macs - counted) > 1e-6 * max(1, counted):
        problems.append(f'dense family: declares {d.macs:.6g} MACs, counter {counted:.6g}')
    outs = flatten(out)
    wb = sum(t.numel() * t.element_size() for t in outs)
    if abs(d.write_bytes - wb) > 0.5:
        problems.append(f'declares {d.write_bytes} bytes written, outputs are {wb}')
    rb = _bytes(list(ins) + list(m.parameters()) + list(m.buffers()))
    if d.read_bytes > rb + 0.5:
        problems.append(f'declares {d.read_bytes} bytes read, inputs+weights are only {rb}')
    if xl:
        problems += check_xl_memory(p, ins, m, outs)
    for name, fn in (('tnca', lambda: tnca_bound(p, 'core-sim')),
                     ('tnca/hbm', lambda: tnca_bound(p, 'core-vrf-sync-v5/hbm')),
                     ('tnca/dp', lambda: tnca_bound(p, 'core-vrf-sync-v5-x8-g2c4-dp')),
                     ('gpu', lambda: gpu_bound(p))):
        L, _binding, _terms = fn()
        if not L > 0:
            problems.append(f'{name} bound {L} is not positive')
    return problems


def check_xl_memory(p, ins, m, outs):
    """xl: the live device bytes the point declares (arguments + outputs + scratch) are what its tensors say,
    and within the cap (3 GiB of the 4 GiB a launch sees)."""
    from .xl import CAP_BYTES
    problems = []
    mem = p.extra['memory']
    args = _bytes(list(ins) + list(m.parameters()) + list(m.buffers()))
    out = sum(t.numel() * t.element_size() for t in outs)
    if mem['args_bytes'] != args:
        problems.append(f'declares {mem["args_bytes"]} argument bytes, the tensors are {args}')
    if mem['out_bytes'] != out:
        problems.append(f'declares {mem["out_bytes"]} output bytes, the outputs are {out}')
    live = args + out + mem['scratch_bytes']
    if live != mem['live_bytes']:
        problems.append(f'live_bytes {mem["live_bytes"]} is not args + outputs + scratch = {live}')
    if mem['cap_bytes'] != CAP_BYTES or live > CAP_BYTES:
        problems.append(f'live device bytes {live} exceed the {CAP_BYTES}-byte cap')
    if p.family == 'argmax':                       # exact point: the data of seed 0 must have no tie at the maximum
        x = p.make(0)[1][0]
        top = torch.topk(x.float(), 2, dim=-1).values
        if bool((top[:, 0] == top[:, 1]).any()):
            problems.append('a row of the argmax input has a tie at its maximum')
    return problems


def check_scoring():
    out = []
    try:
        S.eta(10.0, 5.0, id='x')
        out.append('eta > 1 was accepted')
    except S.BoundViolation:
        pass
    w = {'a': 0.5, 'b': 0.5}
    doc = dict(weights=dict(w), sha256=S.weights_hash(w))
    doc['weights']['a'] = 0.6
    try:
        S.frozen_weights(doc)
        out.append('edited weights were accepted')
    except S.WeightMismatch:
        pass
    E, _A, _ = S.efficiency([S.Point('a', 'f', 'pass', 10, 5, 'v')], w)
    if E.value != 0 or E.certified:
        out.append('a missing point did not zero E')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tier', nargs='+', default=['tiny', 'small', 'xl'])
    args = ap.parse_args(argv)
    fails = 0
    for p in registry.points(tier=args.tier):
        probs = check_point(p)
        fails += bool(probs)
        print(f'{"FAIL" if probs else "PASS"} {p.id}' + ''.join(f'\n     {q}' for q in probs))
    if suite.main(['--check']):
        fails += 1
        print('FAIL suites: the registry does not match archbench/suites (run `python -m archbench.suite --freeze` '
              'if the change is intended)')
    probs = check_scoring()
    fails += bool(probs)
    print(f'{"FAIL" if probs else "PASS"} scoring refusals' + ''.join(f'\n     {q}' for q in probs))
    print(f'archbench selftest: {"FAIL" if fails else "PASS"} ({fails} failing check(s))')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
