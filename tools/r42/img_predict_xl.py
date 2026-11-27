#!/usr/bin/env python3
"""R42-img: what an arm may do to the xl tier on the board (MODEL: rtl-3c ratios applied to the xl board cycles; not a board number).

    img_predict_xl.py XL_COMPARISON.json BASE_DIR ARM=DIR [ARM=DIR ...] [--fp8 PROXY_POINT]

XL_COMPARISON.json: the xl board comparison (xl-cand2: R42 ab-int2 software on the run-3 image; its `new_cycles` are the reference; the
FP8 Hub-kernel lowering of R42-fp8sw is NOT in them). Ratios (arm / BASE on the rtl-3c stand-in, device cycles of the timed call) are taken from:
  * the FP8 proxy `op.fp8_linear/xl/fp8-m16-k256-n128` (the Hub kernel at a reduced shape, tools/r42/fp8sw/points.py), for the 13 `fp8_linear` points;
  * the fp16 points of the same family measured on the stand-in (linear, gemv, attention_prefill, attention_decode), geometric mean of the family's
    measured points, extended to the xl points of the family;
  * the five FP8 composites (decoder, layer; prefill and decode) at ratio^s of the FP8 proxy for the share s of their cycles in FP8 matrix kernels:
    s = 0.5 and 0.8 are printed (the share is not measured: the composite's estimated time parts in archbench/xl.py put the FP8 projections at
    most of the prefill time and about half of the decode time);
  * every other point (fp8_act_quant, embedding, argmax, rmsnorm_residual, rope, swiglu): ratio 1.
The factor is exp(-mean(log ratio)) over the 32 points (>1: fewer cycles); the 18 FP8 points alone are printed too.
"""
import glob, json, math, os, sys


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


def main():
    argv = sys.argv[1:]
    proxy = 'op.fp8_linear/xl/fp8-m16-k256-n128'
    if '--fp8' in argv:
        i = argv.index('--fp8'); proxy = argv[i + 1]; del argv[i:i + 2]
    comp = json.load(open(argv[0]))
    base = records(argv[1])
    rows = comp['rows']
    fam_src = {'linear': 'linear', 'gemv': 'gemv', 'attention_prefill': 'attention_prefill', 'attention_decode': 'attention_decode'}
    for spec in argv[2:]:
        arm, d = spec.split('=', 1)
        recs = records(d)
        meas = {p: recs[p] / base[p] for p in recs if p in base}
        r8 = meas.get(proxy)
        fam = {}
        for p, r in meas.items():
            f = p.split('/')[0].split('.', 1)[1]
            if f in fam_src and 'fp8' not in p:
                fam.setdefault(f, []).append(math.log(r))
        fam = {f: math.exp(sum(v) / len(v)) for f, v in fam.items()}
        print(f'\n=== arm {arm}: FP8 proxy {proxy}: ' + (f'{r8:.3f}' if r8 else 'not measured') + '; fp16 family ratios ' +
              ', '.join(f'{f} {v:.3f}' for f, v in sorted(fam.items())))
        for share in (0.5, 0.8):
            logs, logs8, n8 = [], [], 0
            for r in rows:
                f = r['family']
                ratio = 1.0
                if f == 'fp8_linear' and r8:
                    ratio = r8
                elif f in ('fp8_decoder_decode', 'fp8_decoder_prefill', 'fp8_layer_decode', 'fp8_layer_prefill') and r8:
                    ratio = r8 ** share
                elif f in fam:
                    ratio = fam[f]
                logs.append(math.log(ratio))
                if r.get('fp8'):
                    logs8.append(math.log(ratio)); n8 += 1
            print(f'  composite FP8 share {share}: xl 32 points {math.exp(-sum(logs) / len(logs)):.3f}x ({(math.exp(-sum(logs) / len(logs)) - 1) * 100:+.1f} %),'
                  f' the {n8} FP8 points {math.exp(-sum(logs8) / len(logs8)):.3f}x')


if __name__ == '__main__':
    main()
