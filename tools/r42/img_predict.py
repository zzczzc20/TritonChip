#!/usr/bin/env python3
"""R42-img: what the RTL items may do to the 113-point geometric mean on the board (MODEL, from rtl-3c ratios; not a board number).

    img_predict.py COMPARISON.json BASE_DIR ARM=DIR [ARM=DIR ...] [--proxy POINT=FAMILY ...] [--table OUT.json]

COMPARISON.json: a board comparison (tools/r42/board_compare.py; rows carry point, family, tier, new_cycles). BASE_DIR: link_budget_rtl.py
records of the unchanged run-3 geometry (arm A). ARM=DIR: records of an arm (L, M, F, LM, LF, LMF ...). A point measured on both gives the
ratio arm/A of device cycles. Ratios are extended to the board points that were not measured on the stand-in by (family, dtype); the rules
are the scenarios below, printed with the result:
  measured   only board points that were also measured on the stand-in take their ratio, all others 1
  family     every board point of a (family, dtype) with a measured ratio takes the geometric mean of the measured ratios of that group
  family/2   as family, with the logarithm of the ratio halved for the tiers medium, large and hbm (their cycles hold a larger share that
             no item changes: the tiny and small points are where the transfer and the sequencer bubbles are largest)
  models     family/2 plus gpt2_block, llama_decode, llama_prefill (fp16) at the ratio of the `mlp` group
--proxy POINT=FAMILY: a measured point that is not a row of the table (a reduced-shape proxy) stands for the rows of FAMILY.
Points of a group without a measured ratio keep 1. The factor is exp(-mean(log ratio)) over the 113 board points (>1: fewer cycles).
The `base` of the printed product is the board geometric mean new/old of COMPARISON.json (the software it carries is the software being
extended: ab-int2 here, not the current head).
"""
import glob, json, math, os, sys

GEMM16 = ('matmul', 'linear', 'gemv', 'bmm', 'mlp')
MODELS = ('gpt2_block', 'llama_decode', 'llama_prefill')
BIG = ('medium', 'large', 'hbm')


def records(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
        if f.endswith('digest.jsonl'):
            continue
        for line in open(f):
            r = json.loads(line)
            s = r.get('device_cycles_samples') or []
            if s and r.get('status') == 'pass':
                out[r['point']] = s[0]
    return out


def dtype_of(point):
    return 'fp32' if 'fp32' in point else 'fp16'


def main():
    argv = sys.argv[1:]
    table = None
    if '--table' in argv:
        i = argv.index('--table'); table = argv[i + 1]; del argv[i:i + 2]
    proxy = {}
    while '--proxy' in argv:
        i = argv.index('--proxy'); k, v = argv[i + 1].split('=', 1); proxy[k] = v; del argv[i:i + 2]
    comp = json.load(open(argv[0]))
    base = records(argv[1])
    arms = dict(a.split('=', 1) for a in argv[2:])
    rows = comp['rows']
    meta = {r['point']: (r['family'], r['tier']) for r in rows}
    for p, fam in proxy.items():
        meta.setdefault(p, (fam, 'proxy'))
    out = dict(base_geomean=comp.get('speedup_geomean_old_over_new'), arms={})
    for arm, d in arms.items():
        recs = records(d)
        meas = {p: recs[p] / base[p] for p in recs if p in base}
        groups = {}
        for p, ratio in meas.items():
            if p in meta:
                groups.setdefault((meta[p][0], dtype_of(p)), []).append(math.log(ratio))
        gratio = {g: math.exp(sum(v) / len(v)) for g, v in groups.items()}
        print(f'\n=== arm {arm} ({d}): {len(meas)} points measured on both sides')
        for p, ratio in sorted(meas.items()):
            print(f'  {p:60s} {base[p]:>9d} -> {recs[p]:>9d}  {ratio:.3f}' + ('' if p in meta else '   (not a board point)'))
        print('  groups (family, dtype): ' + ', '.join(f'{f}/{t} {r:.3f}' for (f, t), r in sorted(gratio.items())))

        def scenario(name):
            logs, aff = [], 0
            for r in rows:
                p, f, t = r['point'], r['family'], r['tier']
                g = (f, dtype_of(p))
                ratio = 1.0
                if p in meas:
                    ratio = meas[p]
                elif name != 'measured':
                    src = g
                    if name == 'models' and f in MODELS and dtype_of(p) == 'fp16':
                        src = ('mlp', 'fp16')
                    if src in gratio:
                        ratio = gratio[src]
                        if name in ('family/2', 'models') and t in BIG:
                            ratio = math.sqrt(ratio)
                if ratio != 1.0:
                    aff += 1
                logs.append(math.log(ratio))
            return math.exp(-sum(logs) / len(logs)), aff
        out['arms'][arm] = dict(measured={p: dict(base=base[p], arm=recs[p], ratio=meas[p]) for p in meas}, groups={f'{f}/{t}': r for (f, t), r in gratio.items()}, scenarios={})
        b = comp.get('speedup_geomean_old_over_new')
        for name in ('measured', 'family', 'family/2', 'models'):
            fac, aff = scenario(name)
            out['arms'][arm]['scenarios'][name] = dict(factor=fac, affected=aff)
            extra = f'; {b:.3f} x {fac:.3f} = {b * fac:.3f}' if b else ''
            print(f'  {name:9s} {fac:.3f}x ({(fac - 1) * 100:+.1f} %) over {aff} affected points of {len(rows)}{extra}')
    if table:
        json.dump(out, open(table, 'w'), indent=1)


if __name__ == '__main__':
    main()
