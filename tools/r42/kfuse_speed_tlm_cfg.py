#!/usr/bin/env python3
"""Write SystemC/TLM configuration files for the kfuse-speed study (domain: model prediction).

    python3 tools/r42/kfuse_speed_tlm_cfg.py OUTDIR

The variants are the R41 G4 memory variants (tools/r41/g4_model_suite.py configs(): `cal_line` is the model of the
R41 run-3 image: one 64 B lane, home calibrated against the RTL home pipe, line-wide tensor engine), at 100 MHz,
plus the matrix-engine keys of the run-3 profile (`m.mma.dp.async`):
  mtx_dp=1, mtx_dp_extra_latency=EXTRA (calibrated so that one K16 FP16 command occupies the engine 168 cycles, the
  single-core RTL measurement of evidence/r42/gemm/ANALYSIS.md sec 2), mtx_rf_read_cycles=2 (MTX_VRF_REG 1).
What-if keys of this study (default off; in the patched model only, see tools/r42/kfuse_speed_tlm.patch):
  mtx_dp_g_override=G   timing-only group size of every m.mma.dp.async command (numerics unchanged)
  mtx_dp_b_regs=1       one 64-byte B-register read per ceil(G*bytes/4) registers per step (wide group)
  issue_free_scalar=1   scalar/control instructions take no issue slot (bookkeeping removed, upper bound of S1)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/r41'))


def base_variants():
    import g4_model_suite
    return g4_model_suite.configs()


def variant(name='cal_line', **kw):
    values, ports = base_variants()[name]
    v = dict(values)
    v.update(mtx_dp='1', mtx_dp_extra_latency=str(kw.pop('extra', 3)))
    v.update({k: str(x) for k, x in kw.items()})
    return v, ports


def write(path, values):
    Path(path).write_text(''.join(f'{k}={v}\n' for k, v in values.items()))


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for n in ('run1', 'cal', 'cal_line'):
        v, ports = variant(n)
        write(out / f'{n}.cfg', v)
        print(n, 'ports', ports, 'outstanding', v['outstanding'])
