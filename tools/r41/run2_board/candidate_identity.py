#!/usr/bin/env python3
"""Capture the corrected candidate's board clock and run identity smoke."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'python'))
sys.path.insert(0, str(ROOT/'tools/performance_harness'))
import auto
import candidate_clock
import clock_qualify
import stage
from tnca.operating_clock import qualified_clock, validate_candidate_cdc


def qualify(image, reports, clock_review_path, cdc_review, capture_path, qualified_path, sysfs, xclbinutil, waiver=None):
    image, reports, clock_review_path, cdc_review = map(Path, (image,reports,clock_review_path,cdc_review))
    capture_path, qualified_path = Path(capture_path), Path(qualified_path)
    if capture_path.exists() or qualified_path.exists():
        raise ValueError('candidate clock evidence already exists')
    static = candidate_clock.review(image, reports, xclbinutil, waiver)
    if json.loads(clock_review_path.read_text()) != static:
        raise ValueError('candidate clock review differs from the current reports')
    image_sha = auto.sha(image)
    validate_candidate_cdc(cdc_review, reports/'cdc.rpt', image_sha)
    slrs = clock_qualify.actual_cores((reports/'actual_cores.rpt').read_text())
    if slrs != {'0':0,'1':1,'2':2}:
        raise ValueError('candidate physical core placement is not the reviewed layout')
    capture = clock_qualify.capture(sysfs, image, static['image_uuid'])
    if capture['clock_hz'] != 100_000_000:
        raise ValueError('candidate board clock is not 100 MHz')
    with capture_path.open('x') as f:
        f.write(json.dumps(capture, indent=2)+'\n')
    files = {name: dict(path=str((reports/file).resolve()), sha256=auto.sha(reports/file)) for name,file in (
        ('timing','timing.rpt'), ('route','route_status.rpt'), ('cdc','cdc.rpt'),
        ('placement','actual_cores.rpt'), ('bus_skew','bus_skew.rpt'),
        ('failing_setup','failing_setup_all_groups.tsv'))}
    for name,path in (('board_clock',capture_path), ('clock_review',clock_review_path), ('cdc_review',cdc_review)):
        files[name] = dict(path=str(path.resolve()), sha256=auto.sha(path))
    design, kernel = static['original_design_timing'], static['kernel_timing']
    excepted = bool(design['wns'] < 0 or design['setup_failing'] or static['clocks_mhz']['hbm_aclk'] != 450)
    row = dict(schema='tnca-qualified-operating-clock/1', image_sha256=image_sha,
               qualification='r41-corrected-candidate-100mhz', clock_hz=100_000_000,
               clock_source='measured-board-counter', physical_core_slrs=slrs,
               setup_scope='kernel-clock' if excepted else 'full-design',
               setup_wns_ns=kernel['wns'] if excepted else design['wns'],
               hold_whs_ns=design['whs'], pulse_width_wpws_ns=static['pulse_width_timing']['wpws'],
               route_legal=True, cdc_review='passed', clock_capture=capture, evidence_files=files)
    tmp = qualified_path.with_name(qualified_path.name+'.tmp')
    with tmp.open('x') as f:
        f.write(json.dumps(row, indent=2)+'\n')
    qualified_clock(tmp, image_sha)
    tmp.replace(qualified_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','out','python','registration','image','reports','clock-review','cdc-review','capture','qualified'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--xclbinutil', default='xclbinutil')
    p.add_argument('--waiver', type=Path)
    a = p.parse_args()
    qualify(a.image,a.reports,a.clock_review,a.cdc_review,a.capture,a.qualified,a.sysfs,a.xclbinutil,a.waiver)
    stage.run(a.root,a.out,a.python,json.loads(a.registration.read_text()))


if __name__ == '__main__':
    main()
