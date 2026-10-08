#!/usr/bin/env python3
"""
synth.py -- Yosys synthesis and PPA proxy report for the TNCA RTL.

What the numbers mean
---------------------
No PDK is assumed, so these are *relative* figures, not signoff data:

  gates   cell count after mapping to a standard two-level gate set. A
          reasonable area proxy for random logic; it is NOT a good proxy for
          register files, which map to flops here but belong in SRAM macros.

  depth   longest topological path between registers, in gate levels (yosys
          `ltp -noff`). A reasonable delay proxy. As a rule of thumb, a level
          costs roughly 15-25 ps in a modern node, so a 1 GHz target wants the
          depth under about 50 levels once clock overhead is allowed for.

  flops   sequential bits.

The report exists to make expensive structures visible rather than to predict
silicon. Where a module is over the depth budget, that is stated as a finding,
not hidden.

Usage:
    python3 flow/synth/synth.py                  # every module
    python3 flow/synth/synth.py tnca_core        # one module
    python3 flow/synth/synth.py --json out.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RTL = os.path.join(ROOT, 'rtl')
INC = os.path.join(RTL, 'common')

# Depth at which a module is flagged. 1 GHz with ~20 ps per level and a
# reasonable clock-overhead allowance.
#
# Read it together with the calibration block: this library maps every addition
# to ripple carry, so a bare 32-bit adder already costs 32 levels and a 64-bit
# one costs about 70. Any module containing wide arithmetic is over this budget
# by construction, and the useful question is how far above ITS OWN floor it
# sits, not whether it clears 50.
DEPTH_BUDGET = 50

# Reference structures, synthesised alongside the design so the numbers can be
# read against the library rather than in a vacuum.
CALIBRATION = ['calib_add32', 'calib_add64', 'calib_add89',
               'calib_mul24', 'calib_mul32', 'calib_xbar16']

# top -> (sources, parameter overrides)
COMMON = ['common/tnca_lzc.v', 'common/tnca_popc.v']

MODULES = [
    ('tnca_lzc',        ['common/tnca_lzc.v'], {}),
    ('tnca_popc',       ['common/tnca_popc.v'], {}),
    ('tnca_decoder',    ['core/tnca_decoder.v'], {}),
    ('tnca_mul64',      ['core/tnca_mul64.v'], {}),
    # XLS-generated. These replaced the hand-written tnca_cvt16_f32, which
    # handled FP16 and BF16 to binary32 only and was deleted when the matrix
    # unit and the vector FPU moved to the generated pair -- two conversion
    # implementations to keep in agreement is exactly what doc/xls.md argues
    # against. Its last measurement, 140 gates and 14 levels for two formats
    # one way, is recorded in doc/synthesis.md.
    ('tnca_cvt_xls',      ['gen/tnca_cvt_xls.v'], {}),
    ('tnca_cvt_comb_xls', ['gen/tnca_cvt_comb_xls.v'], {}),
    ('tnca_widen_xls',  ['gen/tnca_widen_xls.v'], {}),
    ('tnca_narrow_xls', ['gen/tnca_narrow_xls.v'], {}),
    ('tnca_widen_pair_xls', ['gen/tnca_widen_pair_xls.v'], {}),
    # The SFU. There is nothing hand-written to compare these against: ISA
    # sec 8.8 specified them and Stage 1a shipped without them, which is why
    # CAPABILITY0.SFU reads 0. These are the first implementation.
    ('tnca_sfu_exp2_xls',  ['gen/tnca_sfu_exp2_xls.v'], {}),
    ('tnca_sfu_log2_xls',  ['gen/tnca_sfu_log2_xls.v'], {}),
    ('tnca_sfu_rcp_xls',   ['gen/tnca_sfu_rcp_xls.v'], {}),
    ('tnca_sfu_rsqrt_xls', ['gen/tnca_sfu_rsqrt_xls.v'], {}),
    ('tnca_barrier',    ['core/tnca_barrier.v'], {}),
    ('tnca_mem_mux',    ['core/tnca_mem_mux.v'], {}),
    ('tnca_tensor',     ['core/tnca_tensor.v'], {}),
    ('tnca_acc_rf',     ['core/tnca_acc_rf.v'], {}),
    ('tnca_matrix',     ['core/tnca_matrix.v', 'core/tnca_acc_rf.v',
                         'core/tnca_fpu_multi.v', 'core/tnca_fp32_fma.v', 'gen/tnca_fma_xls.v',
                         'gen/tnca_widen_xls.v', 'common/tnca_lzc.v'], {}),
    ('tnca_scalar_alu', ['core/tnca_scalar_alu.v'] + COMMON, {}),
    ('tnca_fp32_fma',   ['core/tnca_fp32_fma.v', 'gen/tnca_fma_xls.v'] + COMMON, {}),
    ('tnca_vector_alu', ['core/tnca_vector_alu.v'] + COMMON, {}),
    ('tnca_sfu',        ['core/tnca_sfu.v', 'gen/tnca_sfu_exp2_xls.v',
                         'gen/tnca_sfu_log2_xls.v', 'gen/tnca_sfu_rcp_xls.v',
                         'gen/tnca_sfu_rsqrt_xls.v'], {}),
    # The vector FPU is measured in two configurations on purpose. With the
    # SFU slice in it -- sixteen FMAs, sixteen all-pairs converters and four
    # SFU pipelines, about 180 k gates of deep muxing -- yosys's mapper does
    # not finish in five and a half hours, which is not a PPA result but is a
    # fact about measuring this design. HAS_SFU=0 is a real build parameter,
    # and the SFU slice is already measured on its own as `tnca_sfu`, so the
    # two halves are covered even though the composite is not.
    ('tnca_vector_fpu@nosfu',
                        ['core/tnca_vector_fpu.v', 'core/tnca_fp32_fma.v', 'gen/tnca_fma_xls.v',
                         'gen/tnca_cvt_comb_xls.v',
                         'core/tnca_sfu.v', 'gen/tnca_sfu_exp2_xls.v',
                         'gen/tnca_sfu_log2_xls.v', 'gen/tnca_sfu_rcp_xls.v',
                         'gen/tnca_sfu_rsqrt_xls.v'] + COMMON,
                        {'HAS_SFU': 0}),
    ('tnca_vector_fpu', ['core/tnca_vector_fpu.v', 'core/tnca_fp32_fma.v', 'gen/tnca_fma_xls.v',
                         'gen/tnca_cvt_comb_xls.v',
                         'core/tnca_sfu.v', 'gen/tnca_sfu_exp2_xls.v',
                         'gen/tnca_sfu_log2_xls.v', 'gen/tnca_sfu_rcp_xls.v',
                         'gen/tnca_sfu_rsqrt_xls.v'] + COMMON, {}),
    ('tnca_lsu',        ['core/tnca_lsu.v'], {}),
    ('tnca_scalar_rf',  ['core/tnca_scalar_rf.v'], {}),
    ('tnca_pred_rf',    ['core/tnca_pred_rf.v'], {}),
    ('tnca_vector_rf',  ['core/tnca_vector_rf.v'], {}),
    ('tnca_core', [
        'core/tnca_core.v', 'core/tnca_decoder.v', 'core/tnca_scalar_alu.v',
        'core/tnca_scalar_rf.v', 'core/tnca_pred_rf.v', 'core/tnca_vector_rf.v',
        'core/tnca_vector_alu.v', 'core/tnca_vector_fpu.v', 'core/tnca_fp32_fma.v', 'gen/tnca_fma_xls.v',
        'core/tnca_lsu.v', 'core/tnca_mul64.v', 'core/tnca_matrix.v',
        'core/tnca_acc_rf.v', 'core/tnca_barrier.v',
        'core/tnca_tensor.v', 'core/tnca_mem_mux.v', 'core/tnca_sfu.v',
        'core/tnca_fpu_multi.v', 'gen/tnca_widen_xls.v',
        'gen/tnca_cvt_comb_xls.v',
        'gen/tnca_sfu_exp2_xls.v', 'gen/tnca_sfu_log2_xls.v',
        'gen/tnca_sfu_rcp_xls.v', 'gen/tnca_sfu_rsqrt_xls.v',
    ] + COMMON, {}),
    ('tnca_fifo',       ['common/tnca_fifo.v'], {}),
    ('tnca_router',     ['noc/tnca_router.v', 'common/tnca_fifo.v'], {}),
    ('tnca_mesh',       ['noc/tnca_mesh.v', 'noc/tnca_router.v',
                         'common/tnca_fifo.v'], {}),
    ('tnca_nuca_slice', ['nuca/tnca_nuca_slice.v'], {}),
    ('tnca_mem_arbiter',['nuca/tnca_mem_arbiter.v'], {}),
]

GATE_SET = "AND,NAND,OR,NOR,XOR,XNOR,MUX,AOI3,OAI3,AOI4,OAI4"


def run_one(label, sources, params, keep_dir=None):
    # A label may name a *configuration* rather than a module: everything
    # after the first '@' is the parameter set's name, so the same top can be
    # measured twice without the report calling both rows the same thing.
    top = label.split('@')[0]
    srcs = " ".join(os.path.join(RTL, s) for s in sources)
    pset = "\n".join(f"chparam -set {k} {v} {top}" for k, v in params.items())
    script = f"""
read_verilog -I {INC} -I {os.path.join(RTL, 'noc')} {srcs}
hierarchy -check -top {top}
{pset}
synth -top {top} -flatten
abc -g {GATE_SET}
opt_clean
stat -top {top}
ltp -noff
"""
    with tempfile.TemporaryDirectory() as td:
        ys = os.path.join(td, 'run.ys')
        log = os.path.join(td, 'run.log')
        with open(ys, 'w') as fh:
            fh.write(script)
        proc = subprocess.run(['yosys', '-q', '-l', log, ys],
                              capture_output=True, text=True)
        text = open(log).read() if os.path.exists(log) else ''
        if keep_dir:
            os.makedirs(keep_dir, exist_ok=True)
            with open(os.path.join(keep_dir, f'{top}.log'), 'w') as fh:
                fh.write(text)
        ok = proc.returncode == 0
        errors = [l for l in text.splitlines() if l.startswith('ERROR')]
        warns = [l for l in text.splitlines()
                 if l.startswith('Warning') and 'Yosys has only limited' not in l]

    # gate counts from the last "=== <top> ===" stat block
    gates = flops = 0
    cells = {}
    blocks = text.split(f'=== {top} ===')
    if len(blocks) > 1:
        blk = blocks[-1]
        for line in blk.splitlines():
            m = re.match(r'\s+(\$_\w+_)\s+(\d+)', line)
            if m:
                name, cnt = m.group(1), int(m.group(2))
                cells[name] = cnt
                if 'DFF' in name or 'SDFF' in name or 'LATCH' in name or 'FF' in name:
                    flops += cnt
                else:
                    gates += cnt
            m2 = re.match(r'\s+Number of cells:\s+(\d+)', line)
            if m2 and gates == 0 and flops == 0:
                pass

    depth = None
    m = re.search(r'Longest topological path in \S+ \(length=(\d+)\)', text)
    if m:
        depth = int(m.group(1))

    return {
        'top': top, 'ok': ok, 'gates': gates, 'flops': flops,
        'depth': depth, 'cells': cells,
        'errors': errors, 'warnings': warns,
    }


def run_calibration(keep_dir=None):
    out = []
    for top in CALIBRATION:
        out.append(run_one(top, ['../flow/synth/calibration.v'], {}, keep_dir))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('modules', nargs='*', help="limit to these tops")
    ap.add_argument('--no-calib', action='store_true',
                    help="skip the library calibration block")
    ap.add_argument('--json', help="write the raw results here")
    ap.add_argument('--logs', help="keep per-module yosys logs here")
    ap.add_argument('--budget', type=int, default=DEPTH_BUDGET)
    args = ap.parse_args()

    todo = [m for m in MODULES if not args.modules or m[0] in args.modules]
    if not todo:
        print(f"no such module; known: {', '.join(m[0] for m in MODULES)}",
              file=sys.stderr)
        return 2

    results = []
    failed = False
    over_budget = []

    if not args.no_calib:
        print("Library calibration -- what these structures cost on their own:")
        print(f"  {'reference':<18} {'gates':>8} {'depth':>7}")
        for r in run_calibration(args.logs):
            d = r['depth'] if r['depth'] is not None else '-'
            print(f"  {r['top']:<18} {r['gates']:>8} {str(d):>7}")
        print("  (no carry-lookahead cells in this library: adder depth is linear")
        print("   in width, so wide arithmetic is over any flat budget by")
        print("   construction. Compare each module against its own floor.)")
        print()

    print(f"{'module':<20} {'gates':>8} {'flops':>8} {'depth':>7}  status")
    print("-" * 60)
    for top, srcs, params in todo:
        r = run_one(top, srcs, params, args.logs)
        results.append(r)
        if not r['ok']:
            failed = True
            status = "SYNTH FAILED"
        elif r['depth'] is not None and r['depth'] > args.budget:
            status = f"depth > {args.budget}"
            over_budget.append(r)
        else:
            status = "ok"
        d = r['depth'] if r['depth'] is not None else '-'
        print(f"{top:<20} {r['gates']:>8} {r['flops']:>8} {str(d):>7}  {status}")
        for e in r['errors'][:5]:
            print(f"    {e}")

    print()
    if over_budget:
        print(f"Depth over {args.budget} levels (read against the calibration above):")
        for r in over_budget:
            print(f"  {r['top']}: {r['depth']} levels")
        print()

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f"raw results -> {args.json}")

    # Synthesisability is a hard gate; depth is a reported finding, not a gate.
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
