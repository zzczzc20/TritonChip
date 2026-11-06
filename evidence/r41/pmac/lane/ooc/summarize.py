#!/usr/bin/env python3
"""summarize.py OUTROOT -> one JSON row per OOC run (lane-only area from the
hierarchical report; CARRY8 from the flat report; WNS and per-boundary worst
paths post-synthesis and post-route)."""
import json, os, re, sys, glob

def hier(path):
    for ln in open(path):
        if re.match(r'^\|\s+u_lane\b', ln.replace('(', ' (')) or '|   u_lane' in ln or '| u_lane' in ln:
            f = [x.strip() for x in ln.strip().strip('|').split('|')]
            return dict(luts=int(f[2]), logic=int(f[3]), lutram=int(f[4]), srl=int(f[5]), ff=int(f[6]), dsp=int(f[-1]))
    for ln in open(path):
        if 'u_lane' in ln:
            f = [x.strip() for x in ln.strip().strip('|').split('|')]
            return dict(luts=int(f[2]), logic=int(f[3]), lutram=int(f[4]), srl=int(f[5]), ff=int(f[6]), dsp=int(f[-1]))

def flat(path, name):
    m = re.search(r'^\|\s*' + name + r'\s*\|\s*(\d+)', open(path).read(), re.M)
    return int(m.group(1)) if m else None

def wns(path):
    t = open(path).read()
    m = re.search(r'WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)', t)
    return float(m.group(1)) if m else None

def stages(path):
    rows = {}
    for ln in open(path).read().splitlines()[1:]:
        f = ln.split('\t')
        rows[f[0]] = dict(n=int(f[1]), slack=f[2], dp=f[3], logic=f[4], net=f[5], lv=f[6], start=f[7], end=f[8])
    return rows

out = []
for d in sorted(glob.glob(os.path.join(sys.argv[1], '*_p*'))):
    r = dict(tag=os.path.basename(d))
    try:
        for ph in ('synth', 'route'):
            r[ph] = dict(area=hier(f'{d}/{ph}_util_hier.rpt'), carry8=flat(f'{d}/{ph}_util.rpt', 'CARRY8'),
                         wns=wns(f'{d}/{ph}_summary.rpt'), stages=stages(f'{d}/{ph}_stages.tsv'))
    except Exception as e:
        r['error'] = repr(e)
    out.append(r)
if len(sys.argv) > 2 and sys.argv[2] == '--table':
    for r in out:
        if 'error' in r:
            print(r['tag'], r['error']); continue
        s, t = r['synth'], r['route']; a = t['area']
        st = ' '.join('%s:%s/%s/%s' % (k, v['slack'], v['dp'], v['lv']) for k, v in t['stages'].items() if v['n'])
        print('%-14s LUT %5d FF %4d DSP %d C8 %s | WNS syn %s rt %s | %s' % (r['tag'], a['luts'], a['ff'], a['dsp'], t['carry8'], s['wns'], t['wns'], st))
else:
    json.dump(out, sys.stdout, indent=1)
