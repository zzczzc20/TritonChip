#!/usr/bin/env python3
"""R42 lineelig: what the half-line rows may do to the 113-point geometric mean on the board (MODEL: an extrapolation of rtl-3c ratios; not a board number).

    lineelig_predict.py COMPARISON.json OLD_DIR NEW_DIR [OLD_DIR NEW_DIR ...]

COMPARISON.json: a board comparison (tools/r42/board_compare.py; rows carry point, family, tier, old_cycles, new_cycles). The
cycles of its `new` side are the software being extended (here ab-int2). OLD_DIR/NEW_DIR: link_budget_rtl.py records of the
unchanged and the half-line stand-in. Each measured point gives its own ratio new/old. The other points of the same GEMM
family take the family's measured ratio, in four scenarios stated in the output:
  measured   only the board points that were also measured on the stand-in (other points: ratio 1)
  family     every fp16 point of a family with a measured ratio (matmul, linear, gemv, bmm, mlp), the family's ratio
  family/2   the same with the saving halved (in log) for tiers medium, large and hbm (a larger share of their cycles is the matrix engine)
  models     'family' plus gpt2_block, llama_decode, llama_prefill at the mlp ratio (the same halving for medium/large/hbm)
Points of any other family, and fp32 points, keep ratio 1. The result is exp(-mean(log ratio)) over the 113 points.
"""
import glob, json, math, os, sys


def records(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            s = r.get('device_cycles_samples') or []
            if s and r.get('status') == 'pass':
                out[r['point']] = s[0]
    return out


def main():
    comp = json.load(open(sys.argv[1]))
    rows = comp['rows']
    meas = {}
    for old_d, new_d in zip(sys.argv[2::2], sys.argv[3::2]):
        o, n = records(old_d), records(new_d)
        for p in o:
            if p in n:
                meas[p] = n[p] / o[p]
    fam_of = {r['point']: (r['family'], r['tier']) for r in rows}
    fam_ratio = {}
    for p, ratio in meas.items():
        if p in fam_of:
            f = fam_of[p][0]
            if 'fp32' in p:
                continue
            fam_ratio.setdefault(f, []).append(math.log(ratio))
    fam_ratio = {f: math.exp(sum(v) / len(v)) for f, v in fam_ratio.items()}
    gemm = ('matmul', 'linear', 'gemv', 'bmm', 'mlp')
    models = ('gpt2_block', 'llama_decode', 'llama_prefill')
    big = ('medium', 'large', 'hbm')
    print('measured ratios (new/old cycles, rtl-3c):')
    for p, r in sorted(meas.items()):
        print(f'  {p:56s} {r:.3f}' + ('' if p in fam_of else '   (not a point of the board table)'))
    print('family ratios (geometric mean of the measured points of the family):', {f: round(r, 3) for f, r in fam_ratio.items()})

    def scenario(name):
        logs, n_aff = [], 0
        for r in rows:
            p, f, t = r['point'], r['family'], r['tier']
            ratio = 1.0
            if 'fp32' in p:
                ratio = 1.0
            elif name == 'measured':
                ratio = meas.get(p, 1.0)
            else:
                if p in meas:
                    ratio = meas[p]
                elif f in gemm and f in fam_ratio:
                    ratio = fam_ratio[f]
                    if name in ('family/2', 'models') and t in big:
                        ratio = math.sqrt(ratio)
                elif name == 'models' and f in models and 'mlp' in fam_ratio:
                    ratio = fam_ratio['mlp']
                    if t in big:
                        ratio = math.sqrt(ratio)
            if ratio != 1.0:
                n_aff += 1
            logs.append(math.log(ratio))
        return math.exp(-sum(logs) / len(logs)), n_aff
    print('\npredicted factor on the 113-point geometric mean of device cycles (>1: faster), MODEL:')
    base = comp.get('speedup_geomean_old_over_new')
    for name in ('measured', 'family', 'family/2', 'models'):
        fac, n = scenario(name)
        extra = f'; {base:.3f} x {fac:.3f} = {base * fac:.3f} over the run-3 baseline' if base else ''
        print(f'  {name:9s} {fac:.3f}x ({(fac - 1) * 100:+.1f} %) over {n} affected points of {len(rows)}{extra}')


if __name__ == '__main__':
    main()
