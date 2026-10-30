#!/usr/bin/env python3
"""Pre-load review of the author's specific run-2 HBM-clock exception.

This checks original, hash-pinned implementation reports and xclbin metadata.
It does not claim the entire design closed at the original 450 MHz and does not
substitute the separately recovered checkpoint or a board clock measurement.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/r41/run2_postlink'))
sys.path.insert(0, str(ROOT / 'tools/performance_harness'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto
import check_run2
import clock_qualify

REPORTS = {
    'timing.rpt': 'cd4c6254c960f5f368ea38f4a9e86b606747fba5d467fc4ac07c00f17971f577',
    'route_status.rpt': 'a9f8adc3cd14885894ad79341aa1ba3673432cd5fac136d778c8e2fd141c4980',
    'bus_skew.rpt': '483df731682afdece87a52a0e6aacdf01be7f9bd90bb348109bc2fdf803ddfad',
    'failing_setup_all_groups.tsv': 'c6f98a3decef43a23db6de79bc9d47e2d1e78394b2ae70c40b6c1ef09d3cc6de',
}


def review(image, reports, xclbinutil):
    image, reports = Path(image), Path(reports)
    if auto.sha(image) != auto.IMAGE_SHA:
        raise ValueError('not the explicitly approved image')
    for name, expected in REPORTS.items():
        if auto.sha(reports / name) != expected:
            raise ValueError('original report differs: ' + name)
    info = subprocess.run([xclbinutil, '--info', '--input', str(image)],
                          capture_output=True, text=True, check=True).stdout
    if auto.IMAGE_UUID not in info.lower():
        raise ValueError('xclbin UUID mismatch')
    for name, mhz in (('hbm_aclk', 441), ('DATA_CLK', 100)):
        match = re.search(r'Name:\s*' + name + r'\s*\n\s*Index:.*\n\s*Type:.*\n\s*Frequency:\s*(\d+) MHz', info)
        if not match or int(match.group(1)) != mhz:
            raise ValueError('xclbin clock metadata differs: ' + name)
    timing = (reports / 'timing.rpt').read_text()
    design = check_run2.design_summary(timing)
    pulse = clock_qualify.design_timing(timing)
    kernel = check_run2.clock_row(timing, check_run2.KERNEL_CLOCK)
    if (design['wns'], design['setup_failing']) != (-0.045, 2):
        raise ValueError('original full-design setup failure is not the reviewed exception')
    if design['whs'] < 0 or design['hold_failing'] or not kernel or kernel['wns'] < 0 or kernel['whs'] < 0 or kernel['setup_failing'] or kernel['hold_failing']:
        raise ValueError('kernel or full-design hold timing not met')
    if pulse['wpws'] < 0 or pulse['tpws_failing']:
        raise ValueError('full-design pulse width timing not met')
    with (reports / 'failing_setup_all_groups.tsv').open() as f:
        paths = list(csv.DictReader(f, delimiter='\t'))
    if len(paths) != 2 or any(row['group'] != 'hbm_aclk' or '/hmss_0/' not in row['startpoint'] or '/hmss_0/' not in row['endpoint'] for row in paths):
        raise ValueError('failing setup paths are not the two reviewed HBM paths')
    route = check_run2.route_status((reports / 'route_status.rpt').read_text())
    if not route['routable'] or route['routable'] != route['fully_routed'] or route['errors'] != 0:
        raise ValueError('route is incomplete')
    skew = (reports / 'bus_skew.rpt').read_text()
    if 'Slack (MET)' not in skew or 'Slack (VIOLATED)' in skew:
        raise ValueError('bus skew report missing or contains violations')
    return dict(schema='tnca-run2-clock-review/1', kind='clock-review', status='pass',
                image_sha256=auto.IMAGE_SHA, image_uuid=auto.IMAGE_UUID,
                decision='author accepts hbm_aclk automatic reduction 450 to 441 MHz for this image',
                clock_scope='xclbin metadata, not measured board counter', kernel_clock_mhz=100,
                hbm_clock_mhz=441, original_450mhz_design=design, kernel_timing=kernel,
                pulse_width_timing=pulse,
                exception_paths=paths, route=route, report_sha256=REPORTS,
                remaining='independent CDC review, placement, board counter and runtime gates')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--xclbinutil', default='xclbinutil')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = review(a.image, a.reports, a.xclbinutil)
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
