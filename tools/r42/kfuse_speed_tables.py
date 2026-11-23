#!/usr/bin/env python3
"""Text tables of the kfuse-speed prediction (domain: model prediction; see evidence/r42/kfuse_speed/REPORT.md).

    python3 tools/r42/kfuse_speed_tables.py PREDICT_113.json [XL.json] > tables.txt

Per family: geometric-mean predicted speed-up of the modelled points for the nine {G} x {S0,S1,S2} scenarios; the 113-point
geometric mean (non-modelled points held constant); the per-point predicted board cycles of every modelled point; the xl tier.
"""
import json
import math
import sys
from pathlib import Path

MAIN = ['cal', 'cal_s1', 'cal_s2', 'cal_g8', 'cal_g8_s1', 'cal_g8_s2', 'cal_g16', 'cal_g16_s1', 'cal_g16_s2']
LABEL = {'cal': 'today S0', 'cal_s1': 'today S1', 'cal_s2': 'today S2', 'cal_g8': 'G8 S0', 'cal_g8_s1': 'G8 S1',
         'cal_g8_s2': 'G8 S2', 'cal_g16': 'G16 S0', 'cal_g16_s1': 'G16 S1', 'cal_g16_s2': 'G16 S2'}


def geo(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')


def main():
    d = json.loads(Path(sys.argv[1]).read_text())
    rows = d['rows']
    print('MODEL PREDICTION (SystemC/TLM `cal`, applied as a ratio to the ab-int2 board cycles); not a board result\n')
    print('1. geometric-mean speed-up over the 113 points (points without a model run held constant)')
    print(f'{"scenario":<12}{"113 pts":>10}{"modelled pts":>14}{"n":>5}{"geo (modelled)":>16}')
    for v in MAIN:
        s = d['summary'].get(v)
        if s:
            print(f'{LABEL[v]:<12}{s["geo_speedup_113"]:>10.4f}{s["n_with_model"]:>14}{"":>5}{(s["geo_speedup_modelled"] or float("nan")):>16.4f}')
    print('\n2. per family: geometric-mean speed-up of the family\'s modelled points')
    fams = d['by_family']
    print(f'{"family":<20}{"n":>3}' + ''.join(f'{LABEL[v]:>10}' for v in MAIN[1:]))
    for f, vs in sorted(fams.items()):
        n = vs.get('cal_s1', vs.get('cal_g8', {})).get('n', 0)
        print(f'{f:<20}{n:>3}' + ''.join(f'{vs[v]["geo"]:>10.3f}' if v in vs else f'{"-":>10}' for v in MAIN[1:]))
    print('\n3. per point: board cycles (ab-int2) and predicted board cycles')
    print(f'{"point":<58}{"src":>7}{"board":>14}' + ''.join(f'{LABEL[v]:>13}' for v in MAIN[1:]))
    for r in rows:
        if not r['factor'] or 'cal_s1' not in r['factor'] and 'cal_g8' not in r['factor']:
            continue
        b = r['board']
        print(f'{r["point"]:<58}{r["source"][:6]:>7}{b:>14,}' + ''.join(
            f'{b * r["factor"][v]:>13,.0f}' if v in r['factor'] else f'{"-":>13}' for v in MAIN[1:]))
    if len(sys.argv) > 2:
        x = json.loads(Path(sys.argv[2]).read_text())
        print('\n4. xl tier: predicted device cycles per point (S0 = Hub kernel as compiled, today G=4) and speed-up per scenario')
        vs = x['variants']
        print(f'{"point":<58}{"S0 cycles":>14}' + ''.join(f'{v:>12}' for v in vs))
        for pid, r in x['points'].items():
            if not r['variants'] or 'cal' not in r['variants']:
                continue
            print(f'{pid:<58}{r["variants"]["cal"]["s0"]:>14,.0f}' + ''.join(
                f'{1 / r["variants"][v]["factor"]:>12.3f}' if v in r['variants'] else f'{"-":>12}' for v in vs))
        print('geometric-mean speed-up over the xl points: ' + ', '.join(f'{v} {s:.4f}' for v, s in x['geo_speedup'].items() if s))


if __name__ == '__main__':
    main()
