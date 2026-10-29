#!/usr/bin/env python3
"""R41 run 2: check a routed image's reports against the G4 screens and the
pass/fail thresholds (no card, no Vivado: it only reads report files).

    python3 tools/r41/run2_postlink/check_run2.py ANALYSIS_DIR [--json OUT]

ANALYSIS_DIR is the output of analyze_run2.tcl on the routed checkpoint
(timing.rpt, route_status.rpt, utilization_slr.rpt, cdc.rpt, ...). Run it on
run 1's reports (evidence/r41/stage2/run1/run1-reports.tgz, analysis/) to see
the reference values.

Hard gates (FAIL): design and kernel-clock setup/hold slack below zero, any
failing endpoint, any routing error or unrouted net, and Critical CDC findings
other than the 56 platform findings reviewed for attempt 3 / run 1.
Reported against the G4 estimate (no gate): SLR CLB packing (SLR0 estimate
95.5 +/- 2.5 %, run 1 96.74 %), SLL use, endpoints below +0.5 ns.
Domain of every number: the routed image's own reports.
"""
import argparse, json, re, sys
from pathlib import Path

KERNEL_CLOCK = 'clk_kernel_00_unbuffered_net'
CDC_REVIEWED = 56          # platform-only Critical CDC findings (run 1 = attempt 3)
SLR0_ESTIMATE = (95.5, 2.5)
SCREENS = {'tile0_wns_ns': 0.600, 'tile1_wns_ns': 0.641, 'tile0_clb': 43345, 'tile1_clb': 42369}


def design_summary(text):
    i = text.index('Design Timing Summary')
    nums = re.findall(r'^\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)',
                      text[i:], re.M)[0]
    return dict(wns=float(nums[0]), tns=float(nums[1]), setup_failing=int(nums[2]), setup_endpoints=int(nums[3]),
                whs=float(nums[4]), ths=float(nums[5]), hold_failing=int(nums[6]), hold_endpoints=int(nums[7]))


def clock_row(text, clock):
    i = text.index('Intra Clock Table')
    for line in text[i:].splitlines():
        f = line.split()
        if f and f[0] == clock and len(f) >= 9:
            try:
                return dict(wns=float(f[1]), tns=float(f[2]), setup_failing=int(f[3]),
                            whs=float(f[5]), ths=float(f[6]), hold_failing=int(f[7]))
            except ValueError:
                continue
    return None


def route_status(text):
    def n(label):
        m = re.search(re.escape(label) + r'[^0-9]*?:\s*([\d,]+)', text)
        return int(m.group(1).replace(',', '')) if m else None
    return dict(routable=n('# of routable nets'), fully_routed=n('# of fully routed nets'),
                errors=n('# of nets with routing errors'),
                unrouted=n('# of unrouted nets'))


def slr_util(text):
    out = {}
    for name in ('CLB', 'CLB LUTs', 'CLB Registers'):
        m = re.search(r'^\|\s*' + re.escape(name) + r'\s*\|' + r'\s*([\d.]+)\s*\|' * 6, text, re.M)
        if m:
            v = [float(x) for x in m.groups()]
            out[name] = dict(SLR0=int(v[0]), SLR1=int(v[1]), SLR2=int(v[2]),
                             SLR0_pct=v[3], SLR1_pct=v[4], SLR2_pct=v[5])
    m = re.search(r'^\|\s*SLR1 <-> SLR0\s*\|\s*(\d+)\s*\|\s*\|\s*(\d+)\s*\|\s*([\d.]+)', text, re.M)
    if m:
        out['SLL_SLR1_SLR0'] = dict(used=int(m.group(1)), available=int(m.group(2)), pct=float(m.group(3)))
    m = re.search(r'^\|\s*SLR2 <-> SLR1\s*\|\s*(\d+)\s*\|\s*\|\s*(\d+)\s*\|\s*([\d.]+)', text, re.M)
    if m:
        out['SLL_SLR2_SLR1'] = dict(used=int(m.group(1)), available=int(m.group(2)), pct=float(m.group(3)))
    return out


def cdc_critical(text):
    rules = {}
    for m in re.finditer(r'^\s*(CDC-\d+)\s+Critical\s+(\d+)', text, re.M):
        rules[m.group(1)] = rules.get(m.group(1), 0) + int(m.group(2))
    return rules


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('analysis')
    ap.add_argument('--json')
    a = ap.parse_args()
    d = Path(a.analysis)
    timing = (d / 'timing.rpt').read_text()
    res = dict(design=design_summary(timing), kernel=clock_row(timing, KERNEL_CLOCK),
               route=route_status((d / 'route_status.rpt').read_text()),
               slr=slr_util((d / 'utilization_slr.rpt').read_text()),
               cdc_critical=cdc_critical((d / 'cdc.rpt').read_text()) if (d / 'cdc.rpt').exists() else None)
    near = d / 'near_endpoints_0p5.tsv'
    if near.exists():
        res['kernel_endpoints_below_0p5ns'] = max(0, len(near.read_text().splitlines()) - 1)
    fails = []
    ds, k, r = res['design'], res['kernel'], res['route']
    if ds['wns'] < 0 or ds['setup_failing']:
        fails.append(f"design setup WNS {ds['wns']} ns, {ds['setup_failing']} failing endpoints")
    if ds['whs'] < 0 or ds['hold_failing']:
        fails.append(f"design hold WHS {ds['whs']} ns, {ds['hold_failing']} failing endpoints")
    if k is None:
        fails.append(f'no {KERNEL_CLOCK} row in the intra-clock table')
    elif k['wns'] < 0 or k['whs'] < 0:
        fails.append(f"{KERNEL_CLOCK} WNS {k['wns']} / WHS {k['whs']} ns")
    if r['errors'] or (r['unrouted'] or 0) or (r['routable'] and r['fully_routed'] != r['routable']):
        fails.append(f'routing: {r}')
    if res['cdc_critical'] is not None and sum(res['cdc_critical'].values()) != CDC_REVIEWED:
        fails.append(f"Critical CDC findings {sum(res['cdc_critical'].values())} != {CDC_REVIEWED} reviewed "
                     '(a new review is needed)')
    clb = res['slr'].get('CLB', {})
    est, err = SLR0_ESTIMATE
    if clb:
        p = clb['SLR0_pct']
        res['slr0_vs_estimate'] = ('inside' if abs(p - est) <= err else 'above' if p > est else 'below') + \
            f' the G4 estimate {est} +/- {err} % (routed {p} %)'
    res['screens'] = SCREENS
    res['verdict'] = 'FAIL' if fails else 'PASS'
    res['failures'] = fails
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + '\n')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
