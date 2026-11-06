#!/usr/bin/env python3
"""Review the corrected candidate's original timing and xclbin metadata."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/r41/run2_postlink'))
sys.path.insert(0, str(ROOT/'tools/performance_harness'))
import auto
import check_run2
import clock_qualify

FILES = ('timing.rpt', 'route_status.rpt', 'bus_skew.rpt', 'failing_setup_all_groups.tsv')


def review(image, reports, xclbinutil, waiver=None):
    image, reports = Path(image), Path(reports)
    image_sha = auto.sha(image)
    hashes = {n: auto.sha(reports/n) for n in FILES}
    info = subprocess.run([xclbinutil, '--info', '--input', str(image)],
                          capture_output=True, text=True, check=True).stdout
    u = re.search(r'UUID\s*\(xclbin\):\s*(\S+)', info, re.I)
    clocks = {}
    for name in ('hbm_aclk', 'DATA_CLK'):
        m = re.search(r'Name:\s*'+name+r'\s*\n\s*Index:.*\n\s*Type:.*\n\s*Frequency:\s*(\d+) MHz', info)
        if not m:
            raise ValueError('xclbin clock metadata missing: '+name)
        clocks[name] = int(m.group(1))
    if not u or clocks['DATA_CLK'] != 100:
        raise ValueError('candidate UUID or DATA_CLK missing or incorrect')
    text = (reports/'timing.rpt').read_text()
    design = check_run2.design_summary(text)
    pulse = clock_qualify.design_timing(text)
    kernel = check_run2.clock_row(text, check_run2.KERNEL_CLOCK)
    route = check_run2.route_status((reports/'route_status.rpt').read_text())
    if (not kernel or kernel['wns'] < 0 or kernel['whs'] < 0 or kernel['setup_failing'] or kernel['hold_failing']
            or design['whs'] < 0 or design['hold_failing'] or pulse['wpws'] < 0 or pulse['tpws_failing']):
        raise ValueError('kernel, hold, or pulse width timing not met')
    if not route['routable'] or route['fully_routed'] != route['routable'] or route['errors'] != 0 or route['unrouted']:
        raise ValueError('candidate routing incomplete')
    skew = (reports/'bus_skew.rpt').read_text()
    if 'Slack (MET)' not in skew or 'Slack (VIOLATED)' in skew:
        raise ValueError('candidate bus skew check failed')
    with (reports/'failing_setup_all_groups.tsv').open() as f:
        failures = list(csv.DictReader(f, delimiter='\t'))
    if len(failures) != design['setup_failing']:
        raise ValueError('setup failure census differs from timing summary')
    exception = design['wns'] < 0 or design['setup_failing'] or clocks['hbm_aclk'] != 450
    if exception:
        if waiver is None:
            raise ValueError('candidate timing or HBM frequency requires a new image-specific author decision')
        decision = json.loads(Path(waiver).read_text())
        if (decision.get('status') != 'accepted' or decision.get('image_sha256') != image_sha
                or decision.get('hbm_clock_mhz') != clocks['hbm_aclk'] or not decision.get('author_decision')):
            raise ValueError('candidate author decision does not match the image and HBM clock')
        if any(r['group'] != 'hbm_aclk' or '/hmss_0/' not in r['startpoint'] or '/hmss_0/' not in r['endpoint'] for r in failures):
            raise ValueError('candidate setup failure extends beyond the HBM platform')
    else:
        decision = None
    return dict(schema='tnca-candidate-clock-review/1', kind='clock-review', status='pass',
                image_sha256=image_sha, image_uuid=u.group(1).lower(), clocks_mhz=clocks,
                original_design_timing=design, kernel_timing=kernel, pulse_width_timing=pulse,
                route=route, setup_failures=failures, author_exception=decision, report_sha256=hashes,
                scope='original routed reports and xclbin metadata; CDC, placement and measured board clock remain separate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--xclbinutil', default='xclbinutil')
    p.add_argument('--waiver', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = review(a.image, a.reports, a.xclbinutil, a.waiver)
    with a.out.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
