#!/usr/bin/env python3
"""Vivado-free structural check that M1/M3 cut the attempt-3 cone.

For each module on the attempt-3 worst path, elaborate it with yosys (proc,
async2sync, dffunmap, so every flop is a plain $dff) and ask whether a
combinational input cone still reaches the input that the chain ran through:

  tnca_idle_host_mux   c_ready / n_* outputs  must not reach  host_allow
  tnca_memory_mode     response/request outputs must not reach single_port
  tnca_line_axi        req_ready and the fault/slot logic must not reach
                       axi_base / axi_size / admission_limit

Each check is run on the candidate source (must pass) and on a baseline
source tree (default: git d3c215d, the attempt-3 RTL; must fail), so the
check is shown to detect the original chain. Small modules only; never
flattens a core. Needs yosys on PATH.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKS = [
    dict(name='host mux: c_ready/n_* independent of host_allow', top='tnca_idle_host_mux',
         files=['rtl/noc/tnca_idle_host_mux.v'], params={},
         outputs=['c_ready', 'n_valid', 'n_addr', 'n_tag', 'h_ready'], inputs=['host_allow']),
    dict(name='memory mode: routing/response independent of single_port', top='tnca_memory_mode',
         files=['rtl/nuca/tnca_mem_arbiter.v', 'rtl/fpga/tnca_memory_mode.v'],
         params=dict(PORTS=3, N_SRC=3, SRC_W=2, TAG_W=3),
         outputs=['s_rsp_valid', 's_rsp_src', 's_rsp_rdata', 's_req_ready', 'm_req_valid', 'm_req_addr'],
         inputs=['single_port']),
    dict(name='AXI adapter: request ready independent of aperture/limit inputs', top='tnca_line_axi',
         files=['rtl/fpga/tnca_line_axi.v'],
         params=dict(SRC_W=2, TAG_W=3, OUT_N=8, DATA_W=256, DYNAMIC_LIMIT=1, ID_W=3),
         outputs=['req_ready', 'm_araddr', 'm_awaddr'], inputs=['axi_base', 'axi_size', 'admission_limit']),
]


def script(check, root):
    lines = []
    for f in check['files']:
        lines.append(f'read_verilog -sv -I {root}/rtl/common -I {root}/rtl/noc {root}/{f}')
    chparam = ' '.join(f'-set {k} {v}' for k, v in check['params'].items())
    lines.append(f'hierarchy -top {check["top"]} {("-chparam " + " -chparam ".join(f"{k} {v}" for k, v in check["params"].items())) if check["params"] else ""}')
    lines += ['proc', 'flatten', 'opt_clean', 'async2sync', 'dffunmap', 'opt_clean']
    outs = ' '.join(f'o:{o}' for o in check['outputs'])
    ins = ' '.join(f'i:{i}' for i in check['inputs'])
    # select applies %ci to the top of its stack only: build the output set first.
    lines.append(f'select -set outs {outs}')
    lines.append('select -set cone @outs %ci*:-$dff')
    lines.append(f'select -set ins {ins}')
    lines.append('select -set hit @cone @ins %i')
    lines.append('select -count @hit')
    lines.append('select -assert-none @hit')
    return '\n'.join(lines) + '\n', chparam


def run(check, root, workdir):
    text, _ = script(check, root)
    path = Path(workdir) / f'{check["top"]}.ys'
    path.write_text(text)
    r = subprocess.run(['yosys', '-q', '-s', str(path)], capture_output=True, text=True, timeout=1800)
    return r.returncode == 0, (r.stdout + r.stderr)[-600:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--baseline', default='d3c215d', help='git revision of the attempt-3 RTL')
    ap.add_argument('--baseline-root', help='source tree of the baseline instead of git (no .git needed)')
    a = ap.parse_args()
    ok = True
    with tempfile.TemporaryDirectory(prefix='cone-cuts-') as tmp:
        base = Path(a.baseline_root) if a.baseline_root else Path(tmp) / 'base'
        if not a.baseline_root:
            base.mkdir()
        files = sorted({f for c in CHECKS for f in c['files']} |
                       {str(p.relative_to(ROOT)) for p in (ROOT / 'rtl/common').glob('*.vh')} |
                       {str(p.relative_to(ROOT)) for p in (ROOT / 'rtl/noc').glob('*.vh')})
        for f in ([] if a.baseline_root else files):
            dst = base / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob = subprocess.run(['git', 'show', f'{a.baseline}:{f}'], cwd=ROOT, capture_output=True)
            if blob.returncode:
                raise SystemExit(f'cannot read {a.baseline}:{f} (run where git history is available, '
                                 'or pass a baseline the checkout has)')
            dst.write_bytes(blob.stdout)
        for check in CHECKS:
            new_ok, new_log = run(check, ROOT, tmp)
            old_ok, old_log = run(check, base, tmp)
            verdict = 'PASS' if new_ok and not old_ok else 'FAIL'
            ok &= verdict == 'PASS'
            print(f'{verdict} {check["name"]}: candidate cut={"yes" if new_ok else "NO"}, '
                  f'{a.baseline} cut={"yes (check insensitive)" if old_ok else "no (chain present)"}', flush=True)
            if not new_ok:
                print(new_log)
    print('CONE CUTS PASS' if ok else 'CONE CUTS FAIL', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
