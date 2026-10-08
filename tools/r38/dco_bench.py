#!/usr/bin/env python3
"""
DCO on the 2x2 machine: Triton kernels, the compiler's TMU registrations, and
every cache policy of doc/DCO.pdf, measured on the RTL.

Each workload is an ordinary @triton.jit kernel. It is compiled once, with
the DCO pass on (tnca_cc/dco.py): the kernel starts by telling the Tensor
Management Unit how many times each tile of each input will be read. The same
image then runs on the four-tile system under each replacement and bypass
policy, which the host selects through the TMU's POLICY register
(rtl/nuca/tnca_tmu.v) -- the program is identical, only the cache differs.

The system is built small so these kernels thrash it: 8 sets x 4 ways per
slice (8 KiB of last level across four slices) and a DRAM that accepts one
line every DRAM_INTERVAL cycles. Everything else is the machine as it is.
The four-tile RTL simulates at roughly 300 cycles a second, so the kernels
are sized for runs of minutes: the working sets are 1.5-2x the cache, which
is the ratio that matters, not the bytes.

Two things are checked, separately:

  correctness  every policy must produce the same bits. A policy moves where
               a line lives, never what a load returns, so one differing word
               is a bug. The first run is also checked against NumPy.
  benefit      cycles and DRAM lines read, per policy, against round-robin
               (the slice's baseline) and LRU.

    python tools/r38/dco_bench.py                       # every workload
    python tools/r38/dco_bench.py --only attn --policies rr,lru,all
    python tools/r38/dco_bench.py --json sim/build/r38/dco_bench.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'test'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'model'))

BUILD = os.path.join(ROOT, 'sim', 'build')
OUT = os.path.join(BUILD, 'dco')
# The bench, per DRAM interval (make dco-bench DCO_DRAM_INTERVAL=n builds
# tb_top_dco_i<n>.vvp): 8 is the compute-bound regime, 48 the memory-bound one.
DRAM_INTERVAL = 48
BENCH = None
ASM = os.path.join(ROOT, 'tools', 'asm', 'tnca_asm.py')
GRID = (4, 1, 1)                 # one program per tile

# POLICY register bits (rtl/nuca/tnca_tmu.v):
#   [1:0] 0 round-robin, 1 LRU, 2 anti-thrash   [2] dead-block prediction
#   [3] dynamic bypass   [4] gqa_bypass   [5] bypass nAcc = 1 tensors
POLICIES = {
    'rr':        0,              # the slice's baseline
    'lru':       1,
    'at':        2,
    'at+dbp':    2 | 4,
    'at+bypass': 2 | 8 | 32,
    'all':       2 | 4 | 8 | 32,
    'all+gqa':   2 | 4 | 8 | 16 | 32,
    'lru+dbp+tb': 1 | 4 | 32,    # what the compiler alone buys an LRU cache
    'at+fix2':   2 | 8 | 64,     # static bypass at B_GEAR = 2 (the paper's "fix" variants)
}
DEFAULT_POLICIES = ['rr', 'lru', 'at', 'at+dbp', 'at+bypass', 'at+fix2', 'all', 'all+gqa']

# B_BITS 3 as in the paper. The dynamic gear moves on valid-line evictions per
# 1024 cycles in each slice: above 4 it bypasses one tier more, below 1 one
# less (these kernels evict a few lines per window per slice; the first
# setting, 24 per 512, never moved it). `gear` is GEAR_INIT, which only the
# static variant reads -- a dynamic gear starts at 0.
KNOBS = {'bbits': 3, 'gear': 2, 'window': 1024, 'ub': 4, 'lb': 1}

import triton                     # noqa: E402
import triton.language as tl      # noqa: E402


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------
# The kernels move their tiles with tensor descriptors (TMA-style): one
# transfer-engine instruction per 16x16 tile instead of a few hundred LSU and
# repack instructions, which is what a Triton kernel on a TMA machine is
# written as -- and what keeps a four-tile RTL simulation to minutes. The
# accumulation is the dot's own (`acc = tl.dot(a, b, acc)`): the matrix unit
# adds into its accumulator, where `acc += tl.dot(a, b)` would bring every
# row back through the vector unit.
@triton.jit
def attn_kernel(q_ptr, kt_ptr, v_ptr, o_ptr,
                H: tl.constexpr, NQB: tl.constexpr, NKB: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    """Attention's memory behaviour without its softmax (the system has no
    SFU): every Q block of a head streams the whole of that head's K and V.
    K is stored pre-transposed, one contiguous D x BN tile per key block."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    q_d = tl.make_tensor_descriptor(q_ptr, shape=[H * NQB * BM, D], strides=[D, 1],
                                    block_shape=[BM, D])
    k_d = tl.make_tensor_descriptor(kt_ptr, shape=[H * NKB * D, BN], strides=[BN, 1],
                                    block_shape=[D, BN])
    v_d = tl.make_tensor_descriptor(v_ptr, shape=[H * NKB * BN, D], strides=[D, 1],
                                    block_shape=[BN, D])
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[H * NQB * BM, D], strides=[D, 1],
                                    block_shape=[BM, D])
    for t in range(pid, H * NQB, npid):
        h = t // NQB
        qb = t % NQB
        row = (h * NQB + qb) * BM
        q = q_d.load([row, 0])
        acc = tl.zeros((BM, D), dtype=tl.float32)
        for kb in range(NKB):
            kt = k_d.load([(h * NKB + kb) * D, 0])
            s = tl.dot(q, kt)
            v = v_d.load([(h * NKB + kb) * BN, 0])
            acc = tl.dot(s.to(tl.float16), v, acc)
        o_d.store([row, 0], acc.to(tl.float16))


@triton.jit
def xw_kernel(x_ptr, w_ptr, o_ptr,
              NMB: tl.constexpr, BM: tl.constexpr, K: tl.constexpr,
              BK: tl.constexpr, N: tl.constexpr):
    """A stream of activations through one small weight: X is read once, W
    by every row block. X's tiles are half a line wide, so the compiler
    marks it nAcc = 1 but not for bypass; dead-block prediction sees each of
    its lines die after its last word is read."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    x_d = tl.make_tensor_descriptor(x_ptr, shape=[NMB * BM, K], strides=[K, 1],
                                    block_shape=[BM, BK])
    w_d = tl.make_tensor_descriptor(w_ptr, shape=[K, N], strides=[N, 1],
                                    block_shape=[BK, N])
    o_d = tl.make_tensor_descriptor(o_ptr, shape=[NMB * BM, N], strides=[N, 1],
                                    block_shape=[BM, N])
    for mb in range(pid, NMB, npid):
        acc = tl.zeros((BM, N), dtype=tl.float32)
        for kb in range(K // BK):
            x = x_d.load([mb * BM, kb * BK])
            w = w_d.load([kb * BK, 0])
            acc = tl.dot(x, w, acc)
        o_d.store([mb * BM, 0], acc.to(tl.float16))


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr,
                NMB: tl.constexpr, NNB: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
                N: tl.constexpr):
    """C = A @ B over a row-major walk of C's blocks. A row panel is reused
    by the NNB blocks of its row and is then dead; all of B is reused by
    every row of blocks, and B is larger than the cache. B is stored
    block-packed: tile (nb, kb) is one contiguous BK x BN block."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[NMB * BM, K], strides=[K, 1],
                                    block_shape=[BM, BK])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[NNB * K, BN], strides=[BN, 1],
                                    block_shape=[BK, BN])
    c_d = tl.make_tensor_descriptor(c_ptr, shape=[NMB * BM, N], strides=[N, 1],
                                    block_shape=[BM, BN])
    for t in range(pid, NMB * NNB, npid):
        mb = t // NNB
        nb = t % NNB
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(K // BK):
            a = a_d.load([mb * BM, kb * BK])
            b = b_d.load([(nb * (K // BK) + kb) * BK, 0])
            acc = tl.dot(a, b, acc)
        c_d.store([mb * BM, nb * BN], acc.to(tl.float16))


def _f16(rng, shape, scale=0.25):
    return (rng.uniform(-scale, scale, size=shape)).astype(np.float16)


def _mm(a, b):
    return a.astype(np.float32) @ b.astype(np.float32)


def w_attn():
    H, NQB, NKB, BM, BN, D = 2, 8, 16, 16, 16, 16
    rng = np.random.default_rng(1)
    q = _f16(rng, (H, NQB * BM, D))
    k = _f16(rng, (H, NKB * BN, D))
    v = _f16(rng, (H, NKB * BN, D))
    # K pre-transposed per key block: tile (h, kb) is k[h, kb*BN:(kb+1)*BN, :].T
    kt = np.stack([np.stack([k[h, kb * BN:(kb + 1) * BN, :].T for kb in range(NKB)])
                   for h in range(H)])
    o = np.zeros((H, NQB * BM, D), np.float16)
    ref = np.zeros_like(o)
    for h in range(H):
        for qb in range(NQB):
            qq = q[h, qb * BM:(qb + 1) * BM]
            acc = np.zeros((BM, D), np.float32)
            for kb in range(NKB):
                s = _mm(qq, kt[h, kb]).astype(np.float16)
                acc += _mm(s, v[h, kb * BN:(kb + 1) * BN])
            ref[h, qb * BM:(qb + 1) * BM] = acc.astype(np.float16)
    return dict(
        name='attn', fn=attn_kernel,
        sig={'q_ptr': '*fp16', 'kt_ptr': '*fp16', 'v_ptr': '*fp16', 'o_ptr': '*fp16',
             'H': 'constexpr', 'NQB': 'constexpr', 'NKB': 'constexpr',
             'BM': 'constexpr', 'BN': 'constexpr', 'D': 'constexpr'},
        const=dict(H=H, NQB=NQB, NKB=NKB, BM=BM, BN=BN, D=D),
        inputs=[('q_ptr', q), ('kt_ptr', kt), ('v_ptr', v)],
        output=('o_ptr', o), ref=ref,
        what=f'{H} heads, {NQB * BM} queries x {NKB * BN} keys, d={D}: '
             f'per head K+V = {2 * k[0].nbytes // 1024} KiB')


def w_xw():
    NMB, BM, K, BK, N = 24, 16, 128, 16, 16
    rng = np.random.default_rng(2)
    x = _f16(rng, (NMB * BM, K))
    w = _f16(rng, (K, N))
    o = np.zeros((NMB * BM, N), np.float16)
    ref = _mm(x, w).astype(np.float16)
    return dict(
        name='xw', fn=xw_kernel,
        sig={'x_ptr': '*fp16', 'w_ptr': '*fp16', 'o_ptr': '*fp16',
             'NMB': 'constexpr', 'BM': 'constexpr', 'K': 'constexpr',
             'BK': 'constexpr', 'N': 'constexpr'},
        const=dict(NMB=NMB, BM=BM, K=K, BK=BK, N=N),
        inputs=[('x_ptr', x), ('w_ptr', w)],
        output=('o_ptr', o), ref=ref,
        what=f'X {NMB * BM}x{K} streamed through W {K}x{N} '
             f'({w.nbytes // 1024} KiB, reused {NMB}x)')


def w_gemm():
    NMB, NNB, BM, BN, K, BK = 6, 8, 16, 16, 64, 16
    N = NNB * BN
    rng = np.random.default_rng(3)
    a = _f16(rng, (NMB * BM, K))
    b = _f16(rng, (K, N))
    # B stored block-column-major: block (nb, kb) is one contiguous BK x BN tile
    bt = np.stack([b[kb * BK:(kb + 1) * BK, nb * BN:(nb + 1) * BN]
                   for nb in range(NNB) for kb in range(K // BK)])
    c = np.zeros((NMB * BM, N), np.float16)
    ref = np.zeros_like(c)
    for mb in range(NMB):
        for nb in range(NNB):
            acc = np.zeros((BM, BN), np.float32)
            for kb in range(K // BK):
                acc += _mm(a[mb * BM:(mb + 1) * BM, kb * BK:(kb + 1) * BK],
                           bt[nb * (K // BK) + kb])
            ref[mb * BM:(mb + 1) * BM, nb * BN:(nb + 1) * BN] = acc.astype(np.float16)
    return dict(
        name='gemm', fn=gemm_kernel,
        sig={'a_ptr': '*fp16', 'b_ptr': '*fp16', 'c_ptr': '*fp16',
             'NMB': 'constexpr', 'NNB': 'constexpr', 'BM': 'constexpr',
             'BN': 'constexpr', 'K': 'constexpr', 'BK': 'constexpr',
             'N': 'constexpr'},
        const=dict(NMB=NMB, NNB=NNB, BM=BM, BN=BN, K=K, BK=BK, N=N),
        inputs=[('a_ptr', a), ('b_ptr', bt)],
        output=('c_ptr', c), ref=ref,
        what=f'C {NMB * BM}x{N} = A {NMB * BM}x{K} @ B {K}x{N} '
             f'(B = {b.nbytes // 1024} KiB)')


WORKLOADS = {'attn': w_attn, 'xw': w_xw, 'gemm': w_gemm}


# ---------------------------------------------------------------------------
# Build: compile, place, assemble
# ---------------------------------------------------------------------------
DATA_BASE = 0x10000
SCRATCH_STRIDE = 0x4000


def place(w):
    """Give every tensor an address: line aligned, 4 KiB apart, in order."""
    at = DATA_BASE
    addrs = {}
    for name, arr in w['inputs'] + [w['output']]:
        addrs[name] = at
        at += (arr.nbytes + 4095) // 4096 * 4096 + 4096
    return addrs, at


def data_file(w, addrs, path):
    """$readmemh image: @word-address, then 64-bit words."""
    with open(path, 'w') as fh:
        for name, arr in w['inputs'] + [w['output']]:
            raw = np.ascontiguousarray(arr).tobytes()
            raw += b'\0' * (-len(raw) % 8)
            fh.write(f"@{addrs[name] // 8:x}\n")
            words = np.frombuffer(raw, dtype='<u8')
            for k in range(0, len(words), 4):
                fh.write(' '.join(f"{x:016x}" for x in words[k:k + 4]) + '\n')


def compile_workload(w, dco=True):
    import rtl_harness as H
    import triton_bridge
    from tnca_cc import TargetInfo
    from tnca_cc.codegen import CodeGen
    from tnca_cc.frontend import TTIRImporter

    target = TargetInfo.load('tnca-sim-2x2')
    addrs, top = place(w)
    mod, _ctx = triton_bridge.make_ttir(w['fn'], w['sig'], w['const'])
    # Scratch for TensorMaps and repacks, one area per program: the four run
    # at once. Above the tensors, so it is never mistaken for one of them.
    imp = TTIRImporter(target, w['fn'].__name__, scratch_addr=top,
                       scratch_stride=SCRATCH_STRIDE)
    kernel, _ = imp.run(mod, w['fn'].__name__, dict(addrs),
                        param_names=triton_bridge.param_names(w['fn']),
                        dco={'grid': GRID} if dco else None)
    cg = CodeGen(kernel, target, check_protocol=False)
    cg.assign_shared()
    cg.run()
    tag = w['name'] + ('' if dco else '_nodco')
    p = H.Program(f"dco_{tag}")
    p.kernel(cg)
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(OUT, f"{tag}.s")
    hexf = os.path.join(OUT, f"{tag}.hex")
    with open(src, 'w') as fh:
        fh.write(p.text())
    asm = subprocess.run([sys.executable, ASM, src, '-o', hexf, '--size', '32768'],
                         capture_output=True, text=True)
    if asm.returncode != 0:
        raise RuntimeError(f"assembling {src} failed:\n{asm.stderr}{asm.stdout}")
    dataf = os.path.join(OUT, f"{w['name']}.data.hex")
    data_file(w, addrs, dataf)
    if imp.scratch_used > SCRATCH_STRIDE:
        raise RuntimeError(f"{tag} needs {imp.scratch_used} bytes of scratch per program")
    plan = imp.dco_plan
    return dict(hex=hexf, data=dataf, addrs=addrs, top=top, note=imp.dco_note,
                plan=None if plan is None else dict(
                    d_lsb=plan.d_lsb,
                    tensors=[dict(name=t.name, nacc=t.nacc, bypass=t.bypass,
                                  tiles=t.tiles, loads=t.loads, size=t.size,
                                  run=t.run, lsb=t.lsb) for t in plan.tensors]))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
STAT = {
    'cycles': r'cycles=(\d+)',
    'hit': r'totals: hit=(\d+)',
    'miss': r'totals: hit=\d+ miss=(\d+)',
    'wb': r'totals: hit=\d+ miss=\d+ wb=(\d+)',
    'bypass': r'dco: policy=\d+ bypass=(\d+)',
    'evict': r'dco: .* evict=(\d+)',
    'dead_evict': r'dco: .* dead_evict=(\d+)',
    'registered': r'registered=(\d+)',
    'tlw': r'tlw=(\d+)',
    'dead': r' dead=(\d+)',
    'dropped': r'dropped=(\d+)',
    'dram_reads': r'dram: reads=(\d+)',
    'dram_writes': r'dram: reads=\d+ writes=(\d+)',
}


def run_one(w, build, policy, knobs, tag):
    out_name, out_arr = w['output']
    base = build['addrs'][out_name]
    nwords = (out_arr.nbytes + 7) // 8
    dumpf = os.path.join(OUT, f"{w['name']}.{tag}.dump")
    if os.path.exists(dumpf):
        os.remove(dumpf)
    args = ['vvp', BENCH, f"+hex={build['hex']}", f"+data={build['data']}",
            '+maxcycles=20000000', '+progress=20000',
            f"+dump={dumpf}", f"+dumpbase={base:x}",
            f"+dumpwords={nwords}", f"+dco_policy={policy}"]
    args += [f"+dco_{k}={v}" for k, v in knobs.items()]
    t0 = time.time()
    # The log is kept: a run takes minutes, and its progress lines are how
    # to tell a slow run from a stuck one.
    logf = os.path.join(OUT, f"{w['name']}.{tag}.log")
    with open(logf, 'w') as fh:
        subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, text=True)
    proc = subprocess.CompletedProcess(args, 0, open(logf).read(), '')
    res = dict(policy=tag, code=policy, seconds=round(time.time() - t0, 1),
               ok='PASS' in proc.stdout and 'FAIL' not in proc.stdout)
    for k, rx in STAT.items():
        m = re.search(rx, proc.stdout)
        res[k] = int(m.group(1)) if m else None
    if not res['ok']:
        res['log'] = proc.stdout[-2000:] + proc.stderr[-1000:]
    words = []
    if os.path.exists(dumpf):
        with open(dumpf) as fh:
            words = [int(x, 16) for x in fh if x.strip()]
    raw = np.array(words, dtype='<u8').tobytes()[:out_arr.nbytes]
    res['_out'] = raw
    return res


def check(w, raw):
    out = np.frombuffer(raw, dtype=w['output'][1].dtype).reshape(w['output'][1].shape)
    ref = w['ref']
    err = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    tol = 2e-2 + 2e-2 * np.abs(ref.astype(np.float32))
    bad = int((err > tol).sum())
    return bad, float(err.max()) if err.size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', help='comma-separated workloads')
    ap.add_argument('--policies', help='comma-separated, from: ' + ', '.join(POLICIES))
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--json', help='write results here')
    ap.add_argument('--dram-interval', type=int, default=DRAM_INTERVAL,
                    help='cycles per DRAM request of the bench to run (default %(default)s)')
    ap.add_argument('--plan-only', action='store_true',
                    help='compile and print what the compiler registers; run nothing')
    ap.add_argument('--nodco', action='store_true',
                    help='also run round-robin on an image compiled without '
                         'registrations, to price the registration stores')
    args = ap.parse_args()
    global BENCH
    BENCH = os.path.join(BUILD, f'tb_top_dco_i{args.dram_interval}.vvp')
    if not args.plan_only and not os.path.exists(BENCH):
        sys.exit(f"{BENCH} not built; run `make dco-bench DCO_DRAM_INTERVAL={args.dram_interval}`")

    names = args.only.split(',') if args.only else list(WORKLOADS)
    pols = args.policies.split(',') if args.policies else DEFAULT_POLICIES
    report = {'knobs': KNOBS, 'dram_interval': args.dram_interval,
              'bench': os.path.basename(BENCH), 'workloads': {}}
    failed = False
    for name in names:
        w = WORKLOADS[name]()
        build = compile_workload(w)
        print(f"== {name}: {w['what']}", flush=True)
        print(f"   compiler: {build['note']}")
        if build['plan']:
            pl = build['plan']
            print(f"   TILE: default D_LSB={pl['d_lsb']}")
            for t in pl['tensors']:
                print(f"   register {t['name']:8s} nAcc={t['nacc']:<4d}"
                      f"{'bypass' if t['bypass'] else '      '} tiles={t['tiles']} "
                      f"loads={t['loads']} size={t['size']} granule={1 << t['lsb']}B")
        if args.plan_only:
            continue
        jobs = [(POLICIES[p], p, build) for p in pols]
        if args.nodco:
            jobs.append((0, 'rr/no-registration', compile_workload(w, dco=False)))
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(lambda j: run_one(w, j[2], j[0], KNOBS,
                                                    j[1].replace('/', '_')), jobs))
        for (code, p, _), r in zip(jobs, results):
            r['policy'] = p
        first = results[0]
        bad, emax = check(w, first['_out'])
        same = [r['policy'] for r in results if r['_out'] != first['_out']]
        ok = all(r['ok'] for r in results) and bad == 0 and not same
        failed |= not ok
        base = {r['policy']: r for r in results}
        rr, lru = base.get('rr'), base.get('lru')
        print(f"   numerics: {'ok' if bad == 0 else f'{bad} elements off'} "
              f"(max |err| {emax:.3g}); identical bits across policies: "
              f"{'yes' if not same else 'NO: ' + ', '.join(same)}")
        print(f"   {'policy':20s} {'cycles':>9s} {'vs rr':>6s} {'vs lru':>6s} "
              f"{'DRAM rd':>8s} {'wr':>6s} {'hit':>7s} {'miss':>6s} {'bypass':>6s} "
              f"{'dead_ev':>7s} {'tlw':>5s} {'dead':>5s} {'drop':>5s} {'sec':>5s}")
        for r in results:
            sp = lambda b: (f"{b['cycles'] / r['cycles']:.2f}x"
                            if b and b['cycles'] and r['cycles'] else '-')
            print(f"   {r['policy']:20s} {r['cycles'] or 0:9d} {sp(rr):>6s} {sp(lru):>6s} "
                  f"{r['dram_reads'] or 0:8d} {r['dram_writes'] or 0:6d} "
                  f"{r['hit'] or 0:7d} {r['miss'] or 0:6d} {r['bypass'] or 0:6d} "
                  f"{r['dead_evict'] or 0:7d} {r['tlw'] or 0:5d} {r['dead'] or 0:5d} "
                  f"{r['dropped'] or 0:5d} {r['seconds']:5.0f}"
                  + ('' if r['ok'] else '   FAIL'))
            if not r['ok']:
                print('      ' + r.get('log', '').replace('\n', '\n      '))
        report['workloads'][name] = dict(
            what=w['what'], compiler=build['plan'], note=build['note'],
            numerics_ok=bad == 0, identical=not same,
            runs=[{k: v for k, v in r.items() if not k.startswith('_')} for r in results])
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
