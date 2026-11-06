#!/usr/bin/env python3
"""Publish a complete guarded run-2 board plan after all preflight reviews."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'python'))
import auto
import clock_review
import hbm_review
import manifests
import registrations
import source_review
import stage
from tnca.operating_clock import validate_run2_cdc


def make_plan(root, work, python, image, xclbinutil):
    root, work, python, image = map(lambda p: Path(p).resolve(), (root, work, python, image))
    if root != ROOT or not python.is_file() or not (root/'flow/fpga/program.sh').is_file():
        raise ValueError('use the immutable runtime tree and the staged board interpreter')
    if auto.sha(image) != auto.IMAGE_SHA:
        raise ValueError('not the approved image')
    reports = work/'original_reports'
    evidence = {'source': work/'source_review_fullflow.json', 'clock-review': work/'clock_review_pulse.json',
                'hbm-map': work/'hbm_review.json', 'cdc-review': work/'cdc_review.json'}
    source = source_review.review(work/'src_1f0a5aff.tgz', work/'src_1f0a5aff')
    if json.loads(evidence['source'].read_text()) != source:
        raise ValueError('staged frozen source changed after its review')
    clock = clock_review.review(image, reports, xclbinutil)
    if json.loads(evidence['clock-review'].read_text()) != clock:
        raise ValueError('original clock review changed')
    hbm = hbm_review.review(root/manifests.MAP)
    if json.loads(evidence['hbm-map'].read_text()) != hbm:
        raise ValueError('original HBM mapping review changed')
    validate_run2_cdc(evidence['cdc-review'], reports/'cdc.rpt', auto.IMAGE_SHA)
    if (work/'qualified_clock.json').exists() or (work/'board_clock_after_load.json').exists():
        raise ValueError('existing board clock evidence must be preserved')
    stages = []
    for name in auto.REQUIRED:
        reg = root/registrations.OUT/(name+'.json')
        registered = json.loads(reg.read_text())
        if registered != registrations.group(name):
            raise ValueError('stage registration differs from its generator: ' + name)
        for row in registered['experiments']:
            if stage.sha(root/row['manifest']) != row['sha256']:
                raise ValueError('registered manifest changed: ' + row['id'])
        if registered.get('coverage'):
            stage.check_coverage(root, registered['experiments'], registered['coverage'])
        output = work/'results'/name
        if output.exists():
            raise ValueError('stage output already exists: ' + name)
        command = [str(python), str(root/'tools/r41/run2_board'/('identity.py' if name == 'identity' else 'stage.py')),
                   '--root', str(root), '--out', str(output), '--python', str(python), '--registration', str(reg)]
        if name == 'identity':
            command += ['--image', str(image), '--reports', str(reports), '--cdc-review', str(evidence['cdc-review']),
                        '--capture', str(work/'board_clock_after_load.json'), '--qualified', str(work/'qualified_clock.json'),
                        '--xclbinutil', xclbinutil]
        stages.append(dict(id=name, argv=command, timeout_seconds=259200,
                           registration=dict(path=str(reg), sha256=stage.sha(reg)),
                           checks=[dict(path=str(output/'stage.json'), field=['status'], equals='pass')]))
    plan = dict(schema='tnca-r41-run2-board-plan/1', root=str(root), reports=str(reports), stages=stages,
                preflight_evidence=[dict(kind=k, path=str(v), sha256=auto.sha(v)) for k,v in evidence.items()])
    auto.validate_plan(plan, image)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--xclbinutil', default='/opt/xilinx/xrt/bin/xclbinutil')
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('plan already exists; preserve it')
    plan = make_plan(ROOT, a.work, a.python, a.image, a.xclbinutil)
    tmp = a.out.with_name(a.out.name+'.tmp')
    with tmp.open('x') as f:
        f.write(json.dumps(plan, indent=2)+'\n')
    tmp.replace(a.out)


if __name__ == '__main__':
    main()
