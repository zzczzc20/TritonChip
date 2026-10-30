#!/usr/bin/env python3
"""Capture and qualify the approved run-2 clock, then execute identity smoke."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/performance_harness'))
sys.path.insert(0, str(ROOT/'python'))
import auto
import clock_qualify
import clock_review
import stage
from tnca.operating_clock import qualified_clock

PLACEMENT_SHA = 'eb5a884a26b7d2cab5fed2b26121aa044a257bcdb3551aaa94e5f29e02fd657f'
CDC_SHA = '35617dbb86a0ddb86718ea622492634bf91ff3c51effe52e19b7a299841785c5'


def capture_and_qualify(image, reports, cdc_review, capture_path, qualified_path, sysfs, xclbinutil):
    image, reports, cdc_review = Path(image), Path(reports), Path(cdc_review)
    capture_path, qualified_path = Path(capture_path), Path(qualified_path)
    if capture_path.exists() or qualified_path.exists():
        raise ValueError('clock evidence already exists; preserve it')
    static = clock_review.review(image, reports, xclbinutil)
    review = json.loads(cdc_review.read_text())
    if review.get('status') != 'pass' or review.get('kind') != 'cdc-review' or review.get('image_sha256') != auto.IMAGE_SHA:
        raise ValueError('individual run-2 CDC review is not complete')
    if auto.sha(reports/'actual_cores.rpt') != PLACEMENT_SHA or auto.sha(reports/'cdc.rpt') != CDC_SHA:
        raise ValueError('original placement or CDC report changed')
    slrs = clock_qualify.actual_cores((reports/'actual_cores.rpt').read_text())
    if slrs != {'0': 0, '1': 1, '2': 2}:
        raise ValueError('original core placement is not the reviewed three-SLR layout')
    capture = clock_qualify.capture(sysfs, image, auto.IMAGE_UUID)
    if capture['clock_hz'] != 100_000_000:
        raise ValueError('measured board clock is not 100 MHz')
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    with capture_path.open('x') as f:
        f.write(json.dumps(capture, indent=2)+'\n')
    files = {name: dict(path=str((reports/file).resolve()), sha256=auto.sha(reports/file)) for name, file in
             (('timing','timing.rpt'), ('route','route_status.rpt'), ('cdc','cdc.rpt'),
              ('placement','actual_cores.rpt'))}
    files['board_clock'] = dict(path=str(capture_path.resolve()), sha256=auto.sha(capture_path))
    files['cdc_review'] = dict(path=str(cdc_review.resolve()), sha256=auto.sha(cdc_review))
    row = dict(schema='tnca-qualified-operating-clock/1', image_sha256=auto.IMAGE_SHA,
               qualification='r41-run2-100mhz-hbm441', clock_hz=capture['clock_hz'],
               clock_source='measured-board-counter', physical_core_slrs=slrs,
               setup_scope='kernel-clock', setup_wns_ns=static['kernel_timing']['wns'],
               hold_whs_ns=static['original_450mhz_design']['whs'],
               pulse_width_wpws_ns=static['pulse_width_timing']['wpws'],
               original_design_setup_wns_ns=static['original_450mhz_design']['wns'],
               hbm_metadata_mhz=static['hbm_clock_mhz'], route_legal=True, cdc_review='passed',
               cdc_review_scope='individual platform-scoped review; original Critical severity retained',
               exception=static['decision'], clock_capture=capture, evidence_files=files)
    tmp = qualified_path.with_name(qualified_path.name + '.tmp')
    with tmp.open('x') as f:
        f.write(json.dumps(row, indent=2)+'\n')
    qualified_clock(tmp, auto.IMAGE_SHA)
    tmp.replace(qualified_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--cdc-review', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--qualified', type=Path, required=True)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--xclbinutil', default='xclbinutil')
    a = p.parse_args()
    capture_and_qualify(a.image, a.reports, a.cdc_review, a.capture, a.qualified, a.sysfs, a.xclbinutil)
    stage.run(a.root, a.out, a.python, json.loads(a.registration.read_text()))


if __name__ == '__main__':
    main()
