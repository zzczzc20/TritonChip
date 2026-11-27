#!/usr/bin/env python3
"""Configuration generator of the R42 design-space exploration (domain: SystemC/TLM model what-ifs).

    python3 tools/r42/dse_cfg.py OUTDIR [--base cal|cal_hy] [--list]

Every lever is a dict of TLM configuration keys applied to a base configuration (`cal`: the calibrated three-core run-3 model of
tools/r42/kfuse_speed_tlm_cfg.py; 100 MHz). Each entry records the PHYSICAL ASSUMPTION it stands for and the provenance of the numbers
it uses, so that the report can state exactly what was assumed. Nothing here is a board number.

Key provenance:
  recorded  = a value of the shipped configuration or an RTL parameter (cited)
  derived   = computed from recorded values by the rule written next to it
  assumed   = a design assumption of this study (named as such in the report)
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/r42'))
import kfuse_speed_tlm_cfg as K   # noqa: E402

# latency keys that are DRAM/HBM/interconnect time (ns held fixed when the core clock changes)
# hbm_latency_cycles 23 and tl_hbm_write_latency_cycles 23: FIT of the board calibration, "AXI/HBM read round trip behind an outlet (not in the RTL)"
# tl_outlet_to_slice_cycles 13: FIT; the RTL-derived value is 5 (configs/r40-board-m2.sources.json), the 8 excess cycles are physical time (assumed)
PHYS = dict(hbm_latency_cycles=23, tl_hbm_write_latency_cycles=23)
OUTLET_RTL, OUTLET_FIT_EXTRA = 5, 8
# structural cycle counts that grow when a faster clock needs more pipeline registers (variant "pipelined")
STRUCT = ['noc_stages', 'tl_noc_inject_cycles', 'tl_noc_eject_cycles', 'tl_slice_hit_latency', 'tl_slice_to_outlet_cycles',
          'branch_taken_cycles', 'park_refetch_cycles', 'vector_int_latency', 'vector_fp_latency', 'convert_latency',
          'scalar_latency', 'sfu_latency', 'lsu_complete_cycles', 'load_latency']


# The DSE base model = `cal` (calibrated three-core run-3 model, kfuse-speed) + three structural corrections calibrated on single-core RTL
# (evidence/r42/dse/REPORT.md section 2): the hybrid LSU (fast line path + 2 word units, mode drains) and a 2-cycle issue stage for
# vector-datapath instructions other than shuffles.
DSE_BASE = dict(lsu_hybrid=1, lsu_fast_slots=8, lsu_slow_units=2, issue_vector_cycles=2, issue_shuffle_cycles=1)


def base(name='cal'):
    values, ports = K.base_variants()['cal']
    v = dict(values)
    v.update(mtx_dp='1', mtx_dp_extra_latency='3')
    if name == 'dse':
        v.update(DSE_BASE)
    return {k: str(x) for k, x in v.items()}


def scale_cores(b, nx, ny, outlets=None):
    n = nx * ny
    o = outlets if outlets else n
    return dict(nx=nx, ny=ny, outlets=o, active_outlets=o)


def clock(b, mhz, pipelined=False):
    r = mhz / 100.0
    kv = dict(clock_hz=int(mhz * 1e6))
    kv['hbm_latency_cycles'] = max(1, round(PHYS['hbm_latency_cycles'] * r))
    kv['tl_hbm_write_latency_cycles'] = max(1, round(PHYS['tl_hbm_write_latency_cycles'] * r))
    kv['tl_outlet_to_slice_cycles'] = OUTLET_RTL + round(OUTLET_FIT_EXTRA * r)
    if pipelined:
        p = 1 + (r - 1) / 2.0     # assumed: half the extra clock ratio is paid in pipeline registers of on-chip paths
        for k in STRUCT:
            kv[k] = max(1, math.ceil(int(b[k]) * p))
    return kv


def levers(b):
    """name -> (keys, cores, description)"""
    L = {}

    def add(name, kv, cores=3, desc=''):
        L[name] = dict(keys={k: str(v) for k, v in kv.items()}, cores=cores, desc=desc)
    add('base', {}, 3, 'calibrated three-core run-3 model')
    # --- cores (shell, NoC mesh, NUCA home slice per core and one HBM outlet per core scale with the core count) ---
    add('c4', scale_cores(b, 2, 2), 4, '4 cores, 2x2 mesh, 1 home slice and 1 outlet per core (assumed scaling of the 3-core shell)')
    add('c6', scale_cores(b, 3, 2), 6, '6 cores, 3x2 mesh, 1 home slice and 1 outlet per core')
    add('c6o3', scale_cores(b, 3, 2, 3), 6, '6 cores, 3x2 mesh, 6 home slices, outlets stay 3 (HBM side unchanged)')
    add('c8', scale_cores(b, 4, 2), 8, '8 cores, 4x2 mesh, 1 home slice and 1 outlet per core')
    # --- core clock; HBM/AXI latency in ns held fixed ---
    for mhz in (150, 200, 250):
        add(f'clk{mhz}', clock(b, mhz), 3, f'{mhz} MHz; HBM round trip and the 8 fitted outlet cycles fixed in ns; RTL pipeline depths fixed in cycles (timing closes)')
        add(f'clk{mhz}p', clock(b, mhz, True), 3, f'{mhz} MHz as above, plus assumed extra pipeline registers on on-chip paths (x1+(r-1)/2)')
    # --- memory side at 3 cores ---
    add('o6', dict(outlets=6, active_outlets=6), 3, '6 HBM outlets for 3 cores (2 per core)')
    add('out64', dict(outstanding=64, mshrs=64, targets=128), 3, 'outstanding lines per outlet 32->64, MSHR 32->64')
    add('lat_half', dict(hbm_latency_cycles=12, tl_hbm_write_latency_cycles=12, tl_outlet_to_slice_cycles=9), 3,
        'HBM round trip halved (23->12) and outlet return 13->9: what a faster memory path would give')
    add('noc1', dict(noc_stages=1, tl_noc_inject_cycles=1, tl_noc_eject_cycles=1), 3, 'NoC hop latency 3->1')
    add('lane2', dict(tl_memory_lanes=2), 3, 'two NoC/home lanes (home engine and links duplicated)')
    add('port2', dict(core_mem_ports=2), 3, 'two core memory ports (LSU and tensor engine each get a port; tnca_mem_mux.v has one)')
    # --- core load/store ---
    add('lsuf16', dict(lsu_fast_slots=16), 3, 'hybrid LSU: line-path landing slots 8->16')
    add('lsus4', dict(lsu_slow_units=4), 3, 'hybrid LSU: word-path units 2->4 (gather/scatter/scalar)')
    add('lsuf4', dict(lsu_fast_slots=4), 3, 'hybrid LSU: line-path landing slots 8->4 (cheaper LSU)')
    # --- vector / SFU / VRF ---
    for s in (2, 4, 16):
        occ = 16 // s + 6
        add(f'sfu{s}', dict(sfu_occupancy=occ, sfu_latency=occ + 1), 3,
            f'SFU_LANES={s} (rtl/core/tnca_sfu.v: VLEN/LANES passes + LAT 6): occupancy {occ}, latency {occ + 1}')
    add('vi1', dict(issue_vector_cycles=1, issue_shuffle_cycles=1), 3, 'single-cycle issue of vector-datapath instructions (the base model uses 2; mechanism in the RTL not identified: operand fetch width is the candidate)')
    add('vrf8', dict(vrf_banks=8, vrf_bank_conflict_cycles=1), 3, 'VRF banks 4->8')
    add('vrf0', dict(vrf_bank_conflict_cycles=0), 3, 'no VRF bank conflicts (ideal register-file read ports)')
    # --- tensor engine ---
    for u in (16, 32, 64):
        add(f'te{u}', dict(dma_unit_slots=u), 3, f'tensor-engine load slots 8->{u}')
    le = dict(dma_queue=8, dma_line_engine=1, dma_line_buffers=64, dma_line_rf_cycles=1)
    add('le0', dict(le, dma_line_elig=1), 3, 'line engine, RTL eligibility rule (tnca_tensor.v le_elig)')
    add('le2', dict(le, dma_line_elig=2), 3, 'LINE_HALF: half-line (32 B) rows eligible (dma_line_elig=2; RTL in R42-lineelig)')
    add('le2x', dict(le, dma_line_elig=2, dma_xpose_line_buffer=1), 3, 'LINE_HALF + transposed line buffer')
    # --- matrix engine ---
    for g in (4, 8):
        add(f'g{g}', dict(mtx_dp_g_override=g, mtx_dp_b_regs=1), 3, f'K-fusion group G={g} (timing only; equivalent to a {g // 2}x wider lane array for engine busy time)')
    add('mpf', dict(mtx_dp_prefetch=1), 3, 'MTX_PIPE: next step B register read under the current sweep')
    # --- issue ---
    add('s1', dict(issue_free_mode=1), 3, 'scalar/branch instructions of matrix K loops free (hardware loop + hoisting upper bound)')
    add('s2', dict(issue_free_mode=2), 3, 'every instruction of matrix K loops free of the issue slot (macro-command upper bound)')
    add('hl', dict(issue_free_mode=5), 3, 'hardware loop: scalar/branch of EVERY innermost loop free and taken branch free (upper bound)')
    add('dual', dict(issue_free_mode=4), 3, '2-wide issue, upper bound: scalar/branch/special/predicate instructions never take the issue slot')
    add('br0', dict(branch_taken_cycles=0, park_refetch_cycles=1), 3, 'zero-cost taken branch and context switch')
    # --- NUCA capacity ---
    for sets in (64, 128, 512):
        add(f'sets{sets}', dict(sets=sets), 3, f'NUCA slice {sets} sets x 8 ways x 64 B = {sets * 8 * 64 // 1024} KiB per slice (was 16 KiB)')
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--base', default='cal')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    b = base(a.base)
    L = levers(b)
    if a.list:
        for n, l in L.items():
            print(f'{n:<10} cores={l["cores"]}  {l["desc"]}')
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, l in L.items():
        c = dict(b)
        c.update(l['keys'])
        (out / f'{a.base}_{n}.cfg').write_text(''.join(f'{k}={v}\n' for k, v in c.items()))
    (out / f'{a.base}_levers.json').write_text(json.dumps(L, indent=1))
    print(len(L), 'configs in', out)


if __name__ == '__main__':
    main()
