"""Greedy LLaMA generation on TNCA, the way a model is used.

    python -m archbench.generate --tier small --new 8 --profile core-vrf-sync-v1 \\
        --backend fpga-u280 --backend-args '{"xclbin": "<build>/tnca_u280.xclbin"}' \\
        --out results/u280_generate.json

The ArchBench points time one prefill, and one decode step against a random
cache. This chains them: the device prefills the prompt and returns its K/V,
every decode step appends to the cache the device produced the step before,
and the token fed back is the argmax of the logits the device computed. The
weights and prompt are the `model.llama_prefill/<tier>` point's (same seed,
same construction), so the first step is that point plus its K/V outputs.

Two checks, both against eager PyTorch in binary32 on the CPU:

  per step     the step's outputs against the reference given the same inputs
               (the device's own tokens and cache), at the model tolerance --
               the point check, applied to every step of a real sequence
  sequence     the generated token ids against a generation run wholly in the
               reference. fp16 against fp32 can legitimately pick a different
               token where two logits nearly tie; where the ids differ, the
               reference's margin between the two is recorded so that case
               can be told apart from a wrong result

Time is what the points report: the sum of launch cycles, converted at the
board's clock where there is one (archbench/platforms/tnca.py). tokens/s is
computed from device time only. Host time -- compiling each step (every
cache length is a new shape) and moving every argument across PCIe per
launch -- is reported beside it, separately, and is not device time.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tier', default='small', choices=('tiny', 'small'))
    ap.add_argument('--new', type=int, default=8, help='tokens to generate after the prompt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--profile', default='core-sim')
    ap.add_argument('--opt-level', type=int, default=2)
    ap.add_argument('--backend', default=os.environ.get('TNCA_BACKEND', 'sim-file'))
    ap.add_argument('--backend-args', default=os.environ.get('TNCA_BACKEND_ARGS'))
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)

    os.environ['TNCA_PROFILE'] = args.profile
    os.environ['TNCA_OPT_LEVEL'] = str(args.opt_level)
    os.environ['TNCA_BACKEND'] = args.backend
    if args.backend_args:
        os.environ['TNCA_BACKEND_ARGS'] = args.backend_args
    os.environ.setdefault('TNCA_HOST_TIMEOUT', '3600')

    import torch
    import torch.nn.functional as F
    import triton                                            # noqa: F401  (before the driver)
    import tnca.inductor
    from tnca.triton_backend.driver import backend_choice, launch_log
    from torch._inductor.utils import run_and_get_code

    from .models import LLAMA, MODEL_TOL, LlamaDecode, LlamaPrefill
    from .ops import gen
    from .platforms.common import _to_fp32, compare, environment, flatten, ieee_fp32, now
    from .platforms.tnca import BOARD_CLOCK_HZ, EVIDENCE

    class PrefillKV(LlamaPrefill):
        """The prefill point's model, returning the prompt's K/V as well."""

        def forward(self, tokens):
            x = F.embedding(tokens, self.tok)
            ks, vs = [], []
            for layer in self.layers:
                x, k, v = layer(x, self.cos, self.sin)
                ks.append(k)
                vs.append(v)
            return self._head(x[:, -1:, :]), torch.stack(ks), torch.stack(vs)

    c, bsz, s, _p = LLAMA[args.tier]
    tol = MODEL_TOL['fp16']
    # Exactly mk_prefill's construction (archbench/models.py): the model from
    # the generator first, then the prompt from the same generator.
    g = gen(args.seed)
    first = PrefillKV(c, g, 'fp16', s).eval()
    prompt = torch.randint(0, c.vocab, (bsz, s), generator=g)
    weights = first.state_dict()

    def decoder(pos):
        m = LlamaDecode(c, gen(args.seed), 'fp16', pos).eval()
        m.load_state_dict(weights)
        return m

    def reference(module, inputs):
        m, ins = _to_fp32(copy.deepcopy(module), inputs, 'cpu')
        with torch.no_grad(), ieee_fp32():
            return tuple(t.detach().cpu() for t in flatten(m(*ins)))

    config = tnca.inductor.configure()
    backend, backend_kwargs = backend_choice()
    clock = BOARD_CLOCK_HZ.get(backend)
    xclbin = backend_kwargs.get('xclbin')
    head = dict(schema='archbench-generate/1', tier=args.tier, seed=args.seed,
                config=dict(dim=c.dim, n_layers=c.n_layers, n_heads=c.n_heads,
                            n_kv_heads=c.n_kv_heads, ffn=c.ffn, vocab=c.vocab),
                batch=bsz, prompt_tokens=s, new_tokens=args.new,
                platform=dict(kind='tnca', name=f'tnca:{args.profile}@{backend}', profile=args.profile,
                              backend=backend, backend_args=backend_kwargs, opt_level=args.opt_level,
                              inductor=config),
                evidence_level=EVIDENCE.get(backend, 'measured-rtl'), clock_hz=clock,
                xclbin_sha256=_sha256(xclbin) if xclbin and os.path.isfile(xclbin) else None,
                command=' '.join([sys.executable, '-m', 'archbench.generate'] + (argv or sys.argv[1:])),
                env=environment(), timestamp=now())
    print(f"generate: llama {args.tier} (dim {c.dim}, {c.n_layers} layers, vocab {c.vocab}), "
          f"prompt {s}, {args.new} new token(s), on {head['platform']['name']}"
          + (f", xclbin sha256 {head['xclbin_sha256'][:16]}..." if head['xclbin_sha256'] else ''),
          flush=True)

    def run_step(kind, pos, module, inputs):
        ref = reference(module, inputs)
        torch._dynamo.reset()
        t0 = time.monotonic()
        with torch.no_grad(), launch_log() as log:
            out, code = run_and_get_code(torch.compile(module), *inputs)
        wall = time.monotonic() - t0
        cmp = compare(out, ref, tol)
        cycles = sum(e['device_cycles'] or 0 for e in log)
        in_launch = sum(sum(e['host_seconds'].values()) for e in log)
        step = dict(kind=kind, position=pos, cache_len=0 if kind == 'prefill' else pos,
                    launches=len(log), fallbacks=tnca.inductor.host_fallbacks(code),
                    device_cycles=cycles, device_seconds=(cycles / clock) if clock else None,
                    max_rel_err=cmp['max_rel_err'], ok=cmp['ok'] and bool(log), detail=cmp['detail'],
                    host=dict(wall_s=round(wall, 3), in_launches_s=round(in_launch, 3),
                              compile_and_other_s=round(wall - in_launch, 3),
                              upload_bytes=sum(e['upload_bytes'] for e in log),
                              download_bytes=sum(e['download_bytes'] for e in log)))
        if step['fallbacks']:
            step['ok'] = False
            step['detail'] = (step['detail'] + '; ' if step['detail'] else '') + \
                f'host fallback: {step["fallbacks"]}'
        return out, ref, step

    # The device's sequence.
    steps, tokens = [], []
    logits, k, v = None, None, None
    for i in range(args.new):
        if i == 0:
            out, ref, step = run_step('prefill', 0, first, (prompt,))
            logits, k, v = out
        else:
            pos = s + i - 1
            out, ref, step = run_step('decode', pos, decoder(pos), (tokens[-1], k, v))
            logits, nk, nv = out
            k, v = torch.cat((k, nk), dim=3), torch.cat((v, nv), dim=3)
        nxt = logits[:, -1, :].float().argmax(-1, keepdim=True)
        tokens.append(nxt)
        step['token'] = nxt.flatten().tolist()
        steps.append(step)
        sec = step['device_seconds']
        print(f"  {step['kind']:7} pos {step['position']:3}  {'ok ' if step['ok'] else 'BAD'} "
              f"token {step['token']}  {step['device_cycles']:>10} cycles"
              + (f" = {sec * 1e3:8.2f} ms device" if sec else '')
              + f"  err {step['max_rel_err']:.2e}  launches {step['launches']}"
              f"  host {step['host']['wall_s']:.1f}s ({step['host']['in_launches_s']:.1f}s in launches)",
              flush=True)

    # The reference's own sequence, run wholly in binary32.
    with torch.no_grad(), ieee_fp32():
        m, (p32,) = _to_fp32(copy.deepcopy(first), (prompt,), 'cpu')
        rl, rk, rv = m(p32)
        ref_tokens, margins = [], []
        for i in range(args.new):
            if i:
                pos = s + i - 1
                m, ins = _to_fp32(decoder(pos), (ref_tokens[-1], rk, rv), 'cpu')
                rl, nk, nv = m(*ins)
                rk, rv = torch.cat((rk, nk), dim=3), torch.cat((rv, nv), dim=3)
            row = rl[:, -1, :]
            ref_tokens.append(row.argmax(-1, keepdim=True))
            top2 = row.topk(2, dim=-1).values
            margins.append((top2[:, 0] - top2[:, 1]).tolist())
    got_ids = [t.flatten().tolist() for t in tokens]
    ref_ids = [t.flatten().tolist() for t in ref_tokens]
    first_diff = next((i for i, (a, b) in enumerate(zip(got_ids, ref_ids)) if a != b), None)

    pre = [x for x in steps if x['kind'] == 'prefill']
    dec = [x for x in steps if x['kind'] == 'decode']
    dec_s = sum(x['device_seconds'] or 0 for x in dec) if clock else None
    summary = dict(
        steps_ok=all(x['ok'] for x in steps),
        tokens=got_ids, reference_tokens=ref_ids, tokens_match=first_diff is None,
        first_difference=None if first_diff is None else dict(
            step=first_diff, got=got_ids[first_diff], reference=ref_ids[first_diff],
            reference_top2_margin=margins[first_diff]),
        prefill_device_seconds=pre[0]['device_seconds'] if pre else None,
        prefill_tokens_per_s=(s * bsz / pre[0]['device_seconds']) if pre and clock else None,
        decode_steps=len(dec), decode_device_seconds=dec_s,
        decode_tokens_per_s=(len(dec) * bsz / dec_s) if dec and dec_s else None,
        device_cycles=sum(x['device_cycles'] for x in steps),
        host_wall_s=round(sum(x['host']['wall_s'] for x in steps), 2),
        host_in_launches_s=round(sum(x['host']['in_launches_s'] for x in steps), 2))
    summary['status'] = 'pass' if summary['steps_ok'] else 'numeric-error'

    print(f"  every step within tolerance {tol}: {'yes' if summary['steps_ok'] else 'NO'}")
    if first_diff is None:
        print(f"  tokens match the binary32 reference: {got_ids}")
    else:
        fd = summary['first_difference']
        print(f"  tokens diverge from the binary32 reference at step {first_diff}: got {fd['got']}, "
              f"reference {fd['reference']} (reference top-2 margin {fd['reference_top2_margin']})")
    if clock:
        print(f"  prefill: {s} tokens in {summary['prefill_device_seconds'] * 1e3:.2f} ms device "
              f"= {summary['prefill_tokens_per_s']:.1f} tokens/s")
        if dec:
            print(f"  decode:  {len(dec)} tokens in {dec_s * 1e3:.2f} ms device "
                  f"= {summary['decode_tokens_per_s']:.2f} tokens/s (device time at {clock / 1e6:g} MHz)")
    print(f"  host: {summary['host_wall_s']} s wall, {summary['host_in_launches_s']} s of it in launches "
          f"(transfers and waiting); the rest is compiling each step")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(head, steps=steps, summary=summary), indent=1) + '\n')
        print(f"  -> {args.out}")
    return 0 if summary['steps_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
