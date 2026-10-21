#!/usr/bin/env python3
"""project.py BASE_DIR AFTER_DIR [AFTER_DIR ...] -- the F03/F08/F09 projection table.

Each DIR is one run_ooc.sh output tree (out/<tag>/) holding
<unit>/<unit>_families.tsv and <unit>/<unit>_paths.csv for tnca_vector_fpu,
tnca_matrix and tnca_fp32_fma. For every census family the unit path is
mapped twice: by name in the baseline (the XLS-scheduled registers the census
names) and structurally in the candidate (the s0_q / s1_q cut registers).

    projected routed = after-synth x (census routed / baseline synth)

with census routed the worst endpoint's data-path delay in the attempt-3 routed
census (census_families.json) and baseline synth the same path's delay in the
baseline unit OOC. A path that is absent from a run's family report is taken
from its paths.csv (every endpoint whose worst slack is under GROUP_SLACK);
absent from both, it is reported as an upper bound (period - GROUP_SLACK).
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERIOD, GROUP_SLACK = 10.0, 5.0

# family: census key, baseline (unit, family-or-csv-match), after (unit, family-or-csv-match)
ROWS = [
    ('F03', 'red_busy -> VFPU stage 0 (p0_small_wins)', 'F03 red_busy->p0_small_wins',
     ('tnca_vector_fpu', 'fam:s0_red_small_wins'), ('tnca_vector_fpu', 'csv:red_busy_reg>s0_q_reg')),
    ('F03', 'red_busy -> p0_res_sign_mag', 'F03 red_busy->p0_res_sign_mag',
     ('tnca_vector_fpu', 'fam:s0_red_res_sign_mag'), ('tnca_vector_fpu', 'csv:red_busy_reg>s0_q_reg')),
    ('F03', 'red_busy -> p0_sub_671', 'F03 red_busy->p0_sub_671',
     ('tnca_vector_fpu', 'fam:s0_red_sub'), ('tnca_vector_fpu', 'csv:red_busy_reg>s0_q_reg')),
    ('F08', 'VFPU p0_sticky_al -> p1_sh_mag (stage 1)', 'F08 VFPU p0_sticky_al->p1_sh_mag',
     ('tnca_vector_fpu', 'fam:s1_sticky_to_sh_mag'), ('tnca_fp32_fma', 'fam:c1_s0q_to_s1q')),
    ('F08', 'MTX stage 1 (p0 -> p1)', 'F08 MTX p0->p1',
     ('tnca_matrix', 'fam:s1_p0_to_p1'), ('tnca_fp32_fma', 'fam:c1_s0q_to_s1q')),
    ('F09', 'MTX p1_sh_mag -> accumulator lane_mem', 'F09 MTX p1_sh_mag->lane_mem',
     ('tnca_matrix', 'fam:sL_sh_mag_to_lane_mem'), ('tnca_matrix', 'fam:any_to_lane_mem')),
    ('F09', 'VFPU p1 -> reduction acc (neighbour)', 'F09-adj VFPU p1->acc (reduction)',
     ('tnca_vector_fpu', 'fam:sL_last_to_acc'), ('tnca_vector_fpu', 'fam:c2_s1q_to_acc')),
    ('-', 'MTX operand regs -> stage 0 (neighbour)', 'MTX a_q->p0 (stage 0)',
     ('tnca_matrix', 'fam:s0_a_q_small_wins'), ('tnca_matrix', 'fam:c0_inreg_to_s0q')),
]


def families(d, unit):
    p = d / unit / f'{unit}_families.tsv'
    if not p.exists():
        return {}
    return {r['family']: r for r in csv.DictReader(p.open(), delimiter='\t')}


def csv_worst(d, unit, start, end):
    p = d / unit / f'{unit}_paths.csv'
    best = None
    if p.exists():
        for r in csv.DictReader(p.open()):
            if start in r['start'] and end in r['end']:
                if best is None or float(r['delay']) > float(best['delay']):
                    best = r
    return best


def lookup(d, unit, spec):
    kind, key = spec.split(':', 1)
    if kind == 'fam':
        r = families(d, unit).get(key)
        if r and r['datapath_ns'] != 'NA':
            return dict(delay=float(r['datapath_ns']), levels=int(r['levels']),
                        start=r['start'], end=r['end'], bound=False)
        # fall back to the paths CSV for a family whose report is missing
        start, end = {'c2_s1q_to_acc': ('s1_q_reg', 'acc_reg'), 'c0_inreg_to_s0q': ('g_in_reg', 's0_q_reg'),
                      'c1_s0q_to_s1q': ('s0_q_reg', 's1_q_reg')}.get(key, (None, None))
        if start is None:
            return None
    else:
        start, end = key.split('>')
    r = csv_worst(d, unit, start, end)
    if r:
        return dict(delay=float(r['delay']), levels=int(r['levels']), start=r['start'], end=r['end'], bound=False)
    return dict(delay=PERIOD - GROUP_SLACK, levels=None, start=start, end=end, bound=True)


def main():
    base = Path(sys.argv[1])
    afters = [Path(a) for a in sys.argv[2:]]
    census = json.load((HERE / 'census_families.json').open())
    out = []
    for cand, label, ckey, bspec, aspec in ROWS:
        c = census[ckey]
        b = lookup(base, *bspec)
        row = dict(candidate=cand, family=label, census_endpoints=c['endpoints'], census_tns_ns=c['tns'],
                   census_worst_slack_ns=c['worst_slack'], census_routed_ns=c['worst_datapath'],
                   baseline_synth_ns=b['delay'] if b else None)
        ratio = c['worst_datapath'] / b['delay'] if b else None
        row['ratio'] = round(ratio, 3) if ratio else None
        row['after'] = []
        for a in afters:
            x = lookup(a, *aspec)
            if x is None:
                row['after'].append(dict(state=a.name, synth_ns=None))
                continue
            proj = x['delay'] * ratio if ratio else None
            row['after'].append(dict(state=a.name, synth_ns=x['delay'], upper_bound=x['bound'], levels=x['levels'],
                                     start=x['start'], end=x['end'],
                                     projected_routed_ns=round(proj, 2) if proj else None,
                                     meets_9ns=(proj is not None and proj <= 9.0)))
        out.append(row)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
