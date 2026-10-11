#!/usr/bin/env python3
"""R40-04: single-core HBM -- the checks that need no board.

    python3 tools/r40/hbm_tests.py [--only GROUP ...] [--evidence]

Groups:

  bridge     tb/fpga/tb_axi_bridge.v: rtl/fpga/tnca_axi_bridge.v, and the
             burst mechanism rtl/fpga/tnca_axi_bridge_lb.v in three
             geometries, under random traffic against a reference, with an
             AXI slave that answers late, out of order and under backpressure,
             that refuses part of the window (+err), with resets while
             requests are in flight (+rst), and with memory written behind the
             bridge's back followed by `inv` (+dma, the host's DMA) --
             acceptance 3's out-of-order, backpressure, same-word sub-word
             store, error and reset cases at the port
  mutations  each mutation of the bridge, in a copy, must be caught by the
             case named for it: a gate that passes a broken bridge is not one
  device     the tnca PyTorch device on the simulated /hbm memory mode
             (core-vrf-sync-v5/hbm: everything from the heap up behind the
             bridge, as on the U280 image): tools/r40/device_tests.py with the
             AXI memory at its default timing and again late, out of order and
             under backpressure; a graph whose tensors sit 24 MiB into the
             window; and host/device samples across the whole simulated window,
             written before any is read back, so an address that loses its high
             bits overwrites another sample

Gates (sim/build/r40/R40-04.json): R40-04.robustness carries these checks and
stays `not-measured` until the board's half of acceptance 3 -- data correct on
the U280 image, timing and CDC, the HBM model's fidelity -- is in; R40-04.hbm
records the simulated capacity check the same way, since acceptance 1 is a
64 MiB standard Torch graph on HBM, which is a board measurement.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BRIDGE = 'rtl/fpga/tnca_axi_bridge.v'
BRIDGE_LB = 'rtl/fpga/tnca_axi_bridge_lb.v'
BRIDGE_TB = 'tb/fpga/tb_axi_bridge.v'
BRIDGE_DEPS = ['rtl/fpga/tnca_fpga_mem.v']

# The bridges and the geometries they are tested in: name -> (source, iverilog
# defines). `one` is the single-beat bridge; the `line*` builds are the burst
# mechanism (tnca_axi_bridge_lb) with lines of 8, 2 and 4 beats, 4, 2 and 8
# buffered lines, and 1, 0 and 2 lines fetched ahead; line8w is the geometry
# chosen for the board (16 lines, 32 waiters, 16 writes in flight) and one16
# the single-beat bridge with 16 accesses in flight (the outstanding A/B).
BRIDGE_BUILDS = {
    'one': (BRIDGE, []),
    'line8': (BRIDGE_LB, ['-DBRIDGE_LB=8']),
    'line2': (BRIDGE_LB, ['-DBRIDGE_LB=2', '-DLB_SLOTS=2', '-DLB_PF=0']),
    'line4': (BRIDGE_LB, ['-DBRIDGE_LB=4', '-DLB_SLOTS=8', '-DLB_PF=2']),
    # the geometry the bandwidth sweep chose (doc/r40_hbm.md sec 6)
    'line8w': (BRIDGE_LB, ['-DBRIDGE_LB=8', '-DLB_SLOTS=16', '-DLB_WAIT=32', '-DLB_PF=1', '-DBR_OUT=16']),
    'one16': (BRIDGE, ['-DBR_OUT=16']),
}

# name -> plusargs. Latencies are the slave's, from acceptance of a channel to
# its response; bp is the percentage of cycles each READY is dropped.
BRIDGE_CASES = {
    'fast': '+seed=1 +n=20000 +lat_min=1 +lat_max=1',
    'slow': '+seed=2 +n=20000 +lat_min=20 +lat_max=200',
    'ooo-bp': '+seed=3 +n=20000 +lat_min=1 +lat_max=40 +reorder +bp=30 +dma',
    'ooo-slow-bp': '+seed=4 +n=20000 +lat_min=20 +lat_max=200 +reorder +bp=60 +dma',
    'error': '+seed=5 +n=20000 +lat_min=1 +lat_max=40 +reorder +bp=30 +err',
    'reset': '+seed=6 +n=20000 +lat_min=1 +lat_max=60 +reorder +bp=30 +rst +dma',
    'reset-error-slow': '+seed=7 +n=20000 +lat_min=100 +lat_max=400 +reorder +bp=20 +err +rst',
}

# name, build, [(text, replacement)], the case that must catch it, what it breaks
MUTATIONS = [
    ('no-hazard', 'one', [("                hazard = 1'b1;", "                hazard = 1'b0;")], 'ooo-bp',
     'a request may pass an AXI access to the same word'),
    ('no-drain', 'one', [('if (!m_arvalid && !w_hold && t_n == 0) begin',
                   'if (!m_arvalid && !w_hold) begin')], 'ooo-bp',
     'an atomic reads before the accesses ahead of it complete'),
    ('no-quiet', 'one', [('wire quiet_now = quiet != 0;', "wire quiet_now = 1'b0;")], 'reset',
     'answers to requests made before a reset are taken as new ones'),
    ('no-error', 'one', [("if (m_rresp[1]) axi_err <= 1'b1;", ''),
                  ("if (m_bresp[1]) axi_err <= 1'b1;", '')], 'error',
     'an error response is not reported'),
    ('local-not-first', 'one', [('if (l_rsp_valid) begin\n            rsp_valid = 1\'b1; rsp_tag = l_rsp_tag; '
                          'rsp_rdata = l_rsp_rdata;\n        end else if (take_null)',
                          'if (1\'b0) begin\n        end else if (take_null)')], 'fast',
     'the RAM\'s fixed-latency answers lose the response port'),
    # the burst mechanism's own rules
    ('lb-fill-past-write', 'line8',
     [('if (lb_ask[k] && !lb_dem[k] && !lwb[k])', 'if (lb_ask[k] && !lb_dem[k])'),
      ('if (lb_ask[k] && lb_dem[k] && !lwb[k])', 'if (lb_ask[k] && lb_dem[k])')], 'ooo-bp',
     'a line is fetched while a write to it is in flight'),
    ('lb-no-inv', 'line8', [('            if (inv) begin', "            if (1'b0) begin")], 'ooo-bp',
     'memory written by the host\'s DMA is not seen'),
    ('lb-write-into-fill', 'line8', [('wire can_write = axi_path_free && !l_pend && ',
                                      'wire can_write = axi_path_free && ')], 'ooo-bp',
     'a write lands in a line its fill then overwrites'),
    ('lb-write-past-waiter', 'line8', [(' && !w_line_ref && !t_word_hz', ' && !t_word_hz')], 'ooo-bp',
     'a read accepted first sees a later write'),
    ('lb-no-merge', 'line8', [('                    lb_data[{l_ent, rbeat}] <= merge(lb_data[{l_ent, rbeat}], '
                               'h_wdata, h_be);', '                    ;')], 'fast',
     'a write does not update the buffered line'),
    ('lb-amo-keeps-line', 'line8', [("                        if (lb_v[i] && lb_line[i] == a_line) "
                                     "lb_v[i] <= 1'b0;", '                        ;')], 'fast',
     'an atomic leaves its line buffered with the old value'),
]

HBM_PROFILE = 'core-vrf-sync-v5/hbm'
ADVERSE = '+axi_lat_min=20 +axi_lat_max=120 +axi_reorder +axi_bp=30 +axi_seed=7'


def _torch_python():
    return os.environ.get('TORCH_PYTHON') or sys.executable


def _env():
    env = dict(os.environ)
    paths = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'compiler' / 'python'),
             str(ROOT / 'tools' / 'asm')]
    env['PYTHONPATH'] = os.pathsep.join(paths + [env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    return env


def _build_bridge_tb(tmp, bridge_src, defines=()):
    vvp = Path(tmp) / 'tb_axi_bridge.vvp'
    r = subprocess.run(['iverilog', '-g2005'] + list(defines) +
                       ['-I', 'rtl/common', '-I', 'tb/common', '-o', str(vvp),
                        BRIDGE_TB, str(bridge_src)] + BRIDGE_DEPS,
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('the bridge bench does not build: ' + (r.stdout + r.stderr)[-400:])
    return vvp


def _run_bridge(vvp, args, timeout=3600):
    try:
        r = subprocess.run(['vvp', '-n', str(vvp)] + args.split(), cwd=ROOT,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'timed out (a hang)'
    lines = (r.stdout + r.stderr).splitlines()
    verdict = [l for l in lines if l.startswith('tb_axi_bridge')]
    errors = [l for l in lines if l.startswith('ERROR')]
    ok = r.returncode == 0 and bool(verdict) and verdict[-1].endswith(': PASS') and not errors
    detail = verdict[-1] if verdict else 'no verdict'
    if errors:
        detail += f' | {len(errors)} error line(s), first: {errors[0]}'
    return ok, detail


def group_bridge():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        for build, (src, defines) in BRIDGE_BUILDS.items():
            bdir = Path(tmp) / build
            bdir.mkdir()
            vvp = _build_bridge_tb(bdir, ROOT / src, defines)
            for name, args in BRIDGE_CASES.items():
                ok, detail = _run_bridge(vvp, args)
                checks.append(dict(name=f'bridge:{build}:{name}', ok=ok, detail=detail))
                print(f"  {'PASS' if ok else 'FAIL'}  bridge:{build}:{name:<18s} {detail[:200]}", flush=True)
    return checks


def group_mutations():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, build, edits, case, what in MUTATIONS:
            src_path, defines = BRIDGE_BUILDS[build]
            text = (ROOT / src_path).read_text()
            applied = True
            for old, new in edits:
                if text.count(old) != 1:
                    applied = False
                    break
                text = text.replace(old, new)
            if not applied:
                # A mutation whose anchor has moved tests nothing; say so as a failure.
                checks.append(dict(name=f'mutation:{name}', ok=False,
                                   detail=f'anchor not found once in {src_path}'))
                print(f'  FAIL  mutation:{name:<16s} anchor not found', flush=True)
                continue
            mdir = Path(tmp) / name
            mdir.mkdir()
            mfile = mdir / Path(src_path).name
            mfile.write_text(text)
            try:
                vvp = _build_bridge_tb(mdir, mfile, defines)
                passed, detail = _run_bridge(vvp, BRIDGE_CASES[case], timeout=1800)
            except RuntimeError as e:
                # A syntax/elaboration failure is not a killed mutation:
                # the protocol property was never exercised.
                checks.append(dict(name=f'mutation:{name}', ok=False, detail=str(e)))
                print(f'  FAIL  mutation:{name:<16s} mutated bench does not build', flush=True)
                continue
            ok = not passed
            checks.append(dict(name=f'mutation:{name}', ok=ok,
                               detail=f'{what}: {"caught" if ok else "NOT caught"} by {case} ({detail[:160]})'))
            print(f"  {'PASS' if ok else 'FAIL'}  mutation:{name:<16s} "
                  f"{'caught' if ok else 'NOT caught'} by {case}", flush=True)
    return checks


# Run in a child with torch: the samples and the capacity graph.
_CHILD = r'''
import os, sys
import torch, triton
import tnca.torch_device as D
D.register()
from tnca import device as _device
which = sys.argv[1]
limit = _device.memory_limit(os.environ['TNCA_PROFILE'])
if which == 'samples':
    # One small tensor per step through the window, each with its own values,
    # all written before any is read: a lost high address bit aliases two.
    n_samples, step = 32, (limit - 0x40000) // 32
    held, samples = [], []
    for i in range(n_samples):
        pad = step - 4096
        if pad > 0:
            held.append(torch.empty(pad, dtype=torch.uint8, device='tnca'))
        h = torch.arange(64, dtype=torch.int64) * 1000003 + i * 7919 + (1 << 40)
        samples.append((h, h.to('tnca')))
    addrs = [d.untyped_storage().data_ptr() for _, d in samples]
    bad = [i for i, (h, d) in enumerate(samples) if not torch.equal(d.cpu(), h)]
    assert not bad, f'samples {bad} read back wrong'
    print(f'OK {n_samples} samples from {addrs[0]:#x} to {addrs[-1]:#x} (window end {limit:#x})')
elif which == 'capacity':
    pad = torch.empty(24 << 20, dtype=torch.uint8, device='tnca')
    n = 1 << 15
    a, b = torch.randn(n), torch.randn(n)
    da, db = a.to('tnca'), b.to('tnca')
    f = torch.compile(lambda x, y: (x * 2 + y).sum() + x.abs().max(), fullgraph=True)
    g = torch.compile(lambda x, y: x * 2 + y, fullgraph=True)
    out, red = g(da, db), f(da, db)
    assert torch.allclose(out.cpu(), a * 2 + b), 'pointwise wrong'
    want = (a * 2 + b).sum() + a.abs().max()
    assert abs(red.cpu().item() - want.item()) <= 1e-3 * max(1.0, abs(want.item())), 'reduction wrong'
    lo = min(t.untyped_storage().data_ptr() for t in (da, db, out))
    print(f'OK 3 x {n * 4 >> 10} KiB tensors from {lo:#x}, {lo >> 20} MiB into the device; '
          'pointwise and reduction right')
'''


LINE8 = 'AXI_LINE=8'          # the burst mechanism, as an RTL override of the /hbm bench


def _child(which, plusargs='', timeout=7200, rtl=''):
    env = _env()
    env['TNCA_PROFILE'] = HBM_PROFILE
    env['CUDA_VISIBLE_DEVICES'] = ''
    if plusargs:
        env['TNCA_DEVICE_PLUSARGS'] = plusargs
    if rtl:
        env['TNCA_RTL_PARAMS'] = rtl
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / 'hbm_child.py'
        script.write_text(_CHILD)
        try:
            r = subprocess.run([_torch_python(), str(script), which], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, 'timed out'
    ok_lines = [l for l in r.stdout.splitlines() if l.startswith('OK')]
    if r.returncode == 0 and ok_lines:
        return True, ok_lines[-1][3:]
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return False, f'exit {r.returncode}: ' + ' | '.join(tail)


def _device_tests(plusargs, only=None, timeout=14400, rtl=''):
    env = _env()
    env['TNCA_PROFILE'] = HBM_PROFILE
    env['CUDA_VISIBLE_DEVICES'] = ''
    if plusargs:
        env['TNCA_DEVICE_PLUSARGS'] = plusargs
    if rtl:
        env['TNCA_RTL_PARAMS'] = rtl
    argv = [_torch_python(), str(HERE / 'device_tests.py')]
    if only:
        argv += ['--only'] + only
    try:
        r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'timed out'
    summary = [l for l in r.stdout.splitlines() if l.startswith('r40-device-tests')]
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    ok = r.returncode == 0 and bool(summary) and 'PASS' in summary[-1]
    detail = summary[-1] if summary else f'exit {r.returncode}, no summary'
    if fails:
        detail += ' | ' + ' | '.join(f[:120] for f in fails[:3])
    return ok, detail


def _prepare(rtl=''):
    """Build the /hbm bench before anything uses it: a launch never builds one,
    and the bench is keyed by the RTL it was built from (and its overrides)."""
    env = _env()
    if rtl:
        env['TNCA_RTL_PARAMS'] = rtl
    r = subprocess.run([_torch_python(), '-c', f'from tnca import device; device.prepare({HBM_PROFILE!r})'],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] or ['']


def group_device():
    checks = []
    ok, tail = _prepare()
    if ok:
        ok, tail = _prepare(LINE8)
    if not ok:
        checks.append(dict(name='device:prepare', ok=False, detail=f'the /hbm bench does not build: {tail[0]}'))
        print(f'  FAIL  device:prepare  {tail[0]}', flush=True)
        return checks
    runs = [
        ('device-tests', lambda: _device_tests('')),
        ('device-tests-adverse', lambda: _device_tests(
            ADVERSE, only=['storage', 'copies', 'module', 'compile_graphs', 'compile_views'])),
        ('capacity', lambda: _child('capacity')),
        ('samples', lambda: _child('samples')),
        ('samples-adverse', lambda: _child('samples', ADVERSE)),
        # the burst mechanism under the same adverse memory
        ('line8-device-tests-adverse', lambda: _device_tests(
            ADVERSE, only=['storage', 'copies', 'module', 'compile_graphs', 'compile_views'], rtl=LINE8)),
        ('line8-capacity', lambda: _child('capacity', rtl=LINE8)),
        ('line8-samples-adverse', lambda: _child('samples', ADVERSE, rtl=LINE8)),
    ]
    for name, fn in runs:
        ok, detail = fn()
        checks.append(dict(name=f'device:{name}', ok=ok, detail=detail))
        print(f"  {'PASS' if ok else 'FAIL'}  device:{name:<22s} {detail}", flush=True)
    return checks


GROUPS = {'bridge': group_bridge, 'mutations': group_mutations, 'device': group_device}


def group_address():
    from hbm_address import PC_BYTES, PROBE_BYTES, probe_offsets
    checks = []
    for name, base, size, expected in [
        ('sim-32MiB', 0, 32 << 20, None),
        ('one-GiB', 0, 1 << 30, list(range(4))),
        ('all-32-PCs', 0, 8 << 30, list(range(32))),
        ('shifted-window', 4096, (8 << 30) - 4096, list(range(32))),
        ('high-address', 4 << 30, 4 << 30, list(range(16, 32))),
        ('minimal-window', 0, PROBE_BYTES, None),
    ]:
        offsets = probe_offsets(base, size)
        pcs = sorted({(base + a) // PC_BYTES for a in offsets})
        ok = (all(0 <= a <= size - PROBE_BYTES and a % 64 == 0 for a in offsets)
              and len(offsets) == len(set(offsets)) and offsets == sorted(offsets)
              and (expected is None or pcs == expected))
        checks.append(dict(name=f'address:{name}', ok=ok,
                           detail=f'{len(offsets)} probes, PCs {pcs[0]}..{pcs[-1]}'))
    for name, base, size in [('too-small', 0, 64), ('unaligned', 1, 1 << 30),
                             ('short-final-PC', 0, 2 * PC_BYTES + 64)]:
        try:
            probe_offsets(base, size)
            ok = False
        except ValueError:
            ok = True
        checks.append(dict(name=f'address:{name}', ok=ok, detail='invalid window refused'))
    for c in checks:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']}  {c['detail']}", flush=True)
    return checks


GROUPS['address'] = group_address


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', choices=sorted(GROUPS))
    ap.add_argument('--evidence', action='store_true')
    args = ap.parse_args(argv)
    if not shutil.which('iverilog') and (not args.only or {'bridge', 'mutations'} & set(args.only)):
        print('r40-hbm-tests  FAIL  iverilog is not installed, so nothing was simulated')
        return 1
    checks = []
    for name, fn in GROUPS.items():
        if args.only and name not in args.only:
            continue
        checks += fn()
    passed = sum(c['ok'] for c in checks)
    print(f"r40-hbm-tests  {'PASS' if passed == len(checks) else 'FAIL'}  ({passed}/{len(checks)})")
    if args.evidence:
        import evidence as EV
        mine = {c['name'].split(':', 1)[0] for c in checks}
        try:
            import json
            prior = json.loads(EV.path_for('R40-04.robustness').read_text())['gates'][
                'R40-04.robustness']['checks']
        except Exception:                                # noqa: BLE001
            prior = []
        kept = [c for c in prior if c['name'].split(':', 1)[0] not in mine]
        allc = kept + checks
        sim_ok = all(c['ok'] for c in allc)
        board = ('the board half of acceptance 3 (data on the U280 HBM image, post-route timing '
                 'and CDC, the HBM model calibrated against the board) is not measured yet')
        EV.record('R40-04.robustness', 'not-measured' if sim_ok else 'fail', allc,
                  detail=f'simulation {sum(c["ok"] for c in allc)}/{len(allc)}; {board}')
        cap = [c for c in allc if c['name'] in ('device:capacity', 'device:samples',
                                                'device:samples-adverse')]
        if cap:
            EV.record('R40-04.hbm', 'not-measured' if all(c['ok'] for c in cap) else 'fail', cap,
                      detail='simulated /hbm window only; acceptance 1 is a >=64 MiB standard Torch '
                             'graph on the U280 HBM image with non-zero HBM traffic')
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    sys.exit(main())
