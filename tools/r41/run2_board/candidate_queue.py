#!/usr/bin/env python3
"""Run the corrected image's complete board stages after explicit approval."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'python'))
sys.path.insert(0, str(ROOT/'tools/performance_harness'))
import auto
import candidate_clock
import candidate_map
import candidate_source
import clock_qualify
import stage
from tnca.operating_clock import validate_candidate_cdc


def approval(path, image_sha, image_uuid):
    decision = json.loads(Path(path).read_text())
    if (decision.get('status') != 'accepted' or decision.get('image_sha256') != image_sha
            or decision.get('image_uuid') != image_uuid or not decision.get('author_decision')):
        raise ValueError('explicit author approval for this exact candidate image is required')
    return decision


def preflight(root, work, image, approval_path, waiver, xclbinutil):
    root, work, image = map(Path, (root,work,image))
    digest = auto.sha(image)
    decision = json.loads(Path(approval_path).read_text())
    uid = decision.get('image_uuid')
    approval(approval_path, digest, uid)
    static = candidate_clock.review(image, work/'reports', xclbinutil, waiver)
    if static['image_uuid'] != uid or json.loads((work/'clock_review.json').read_text()) != static:
        raise ValueError('candidate clock review or UUID differs')
    validate_candidate_cdc(work/'cdc_review.json', work/'reports/cdc.rpt', digest)
    if clock_qualify.actual_cores((work/'reports/actual_cores.rpt').read_text()) != {'0':0,'1':1,'2':2}:
        raise ValueError('candidate physical core placement differs')
    source = candidate_source.review(work/'src_8d237223.tgz', work/'src_8d237223', image, digest)
    if json.loads((work/'source_review.json').read_text()) != source:
        raise ValueError('candidate source review differs')
    mapping = candidate_map.read_map(image)
    if json.loads((work/'xclbin_hbm_map.json').read_text()) != mapping:
        raise ValueError('candidate HBM map differs')
    if auto.sha(root/'spec/r41/run3/xclbin_hbm_map.json') != auto.sha(work/'xclbin_hbm_map.json'):
        raise ValueError('registered HBM map differs')
    regs = []
    for name in auto.REQUIRED:
        path = root/'spec/r41/run3/stages'/(name+'.json')
        reg = json.loads(path.read_text())
        if reg.get('id') != name or reg.get('image_identity') != {'sha256':digest,'uuid':uid}:
            raise ValueError('stage registration belongs to another image: '+name)
        for row in reg['experiments']:
            if auto.sha(root/row['manifest']) != row['sha256'] or not row.get('checks'):
                raise ValueError('stage manifest changed or lacks result checks: '+name)
            spec=json.loads((root/row['manifest']).read_text());board=spec.get('board',{});args=board.get('backend_args',{})
            if (spec.get('id')!=row['id'] or board.get('image_sha256')!=digest or board.get('image_uuid')!=uid
                    or board.get('clock_hz')!=100_000_000 or board.get('profile')!=stage.PROFILE
                    or args.get('image_profile')!=stage.PROFILE or args.get('core_count')!=3
                    or args.get('memory_ports')!=3 or args.get('hbm_bytes')!=8<<30 or args.get('hbm_dma') is not True):
                raise ValueError('stage manifest has the wrong candidate hardware: '+row['id'])
        if reg.get('coverage'):
            stage.check_coverage(root, reg['experiments'], reg['coverage'])
        regs.append(dict(id=name,path=str(path),sha256=auto.sha(path)))
    if len({r['id'] for r in regs}) != len(auto.REQUIRED):
        raise ValueError('incomplete candidate stages')
    if (work/'qualified_clock.json').exists() or (work/'board_clock_after_load.json').exists():
        raise ValueError('prior candidate clock evidence must be preserved')
    files=[Path(approval_path),work/'clock_review.json',work/'cdc_review.json',work/'source_review.json',
           work/'xclbin_hbm_map.json',root/'spec/r41/run3/xclbin_hbm_map.json']
    files += [work/'reports'/n for n in ('timing.rpt','route_status.rpt','bus_skew.rpt','cdc.rpt',
              'actual_cores.rpt','failing_setup_all_groups.tsv')]
    if waiver:files.append(Path(waiver))
    return dict(image_sha256=digest,image_uuid=uid,registrations=regs,
                evidence={str(p):auto.sha(p) for p in files})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','work','image','approval','out','python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--waiver', type=Path)
    p.add_argument('--xclbinutil', default='xclbinutil')
    p.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    p.add_argument('--poll', type=float, default=5)
    a = p.parse_args()
    if not 0 < a.poll <= 10:
        raise ValueError('poll must be in (0,10] seconds')
    root,work,image,out = (x.resolve() for x in (a.root,a.work,a.image,a.out))
    state = preflight(root,work,image,a.approval,a.waiver,a.xclbinutil)
    if out.exists():
        raise ValueError('candidate queue output already exists')
    env = auto.board_environment()
    out.mkdir(parents=True)
    with (work/'candidate_queue.lock').open('a') as lock, (out/'queue.jsonl').open('a', buffering=1) as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def record(kind,**kw):
            log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind=kind,**kw))+'\n')
        tracker = auto.IdleTracker(max_gap=max(15,a.poll*2.5))
        while True:
            try:
                row=auto.probe(a.sysfs);ready=tracker.observe(time.monotonic(),row)
                record('idle-probe',**row)
                if ready:break
            except (OSError,ValueError) as e:
                tracker=auto.IdleTracker(max_gap=max(15,a.poll*2.5));record('probe-error',error=str(e))
            time.sleep(a.poll)
        for mode in ('--check','--go'):
            if auto.sha(image)!=state['image_sha256'] or any(auto.sha(Path(k))!=v for k,v in state['evidence'].items()):
                raise RuntimeError('candidate image or preflight evidence changed')
            row=auto.probe(a.sysfs)
            if (time.monotonic()-tracker.last>tracker.max_gap or not auto.idle(row)
                    or (row['uuid'],row['total execs'])!=tracker.identity):
                raise RuntimeError('card changed or became busy before programming')
            tracker.last=time.monotonic()
            if row['uuid']==state['image_uuid']:
                record('already-loaded',uuid=row['uuid']);break
            record('program',mode=mode,previous=row['uuid'])
            with (out/'program.log').open('a') as f:
                subprocess.run(['bash',str(root/'flow/fpga/program.sh'),mode,str(image)],env=env,
                               stdout=f,stderr=subprocess.STDOUT,check=True)
        for item in state['registrations']:
            row=auto.probe(a.sysfs)
            if not auto.idle(row) or row['uuid']!=state['image_uuid'] or auto.clock_mhz(a.sysfs)!=100:
                raise RuntimeError('card busy or image/clock changed before '+item['id'])
            reg=Path(item['path'])
            if auto.sha(reg)!=item['sha256']:
                raise RuntimeError('candidate registration changed before '+item['id'])
            target=out/'results'/item['id']
            script='candidate_identity.py' if item['id']=='identity' else 'stage.py'
            command=[str(a.python),str(root/'tools/r41/run2_board'/script),'--root',str(root),'--out',str(target),
                     '--python',str(a.python),'--registration',str(reg)]
            if item['id']=='identity':
                command += ['--image',str(image),'--reports',str(work/'reports'),'--clock-review',str(work/'clock_review.json'),
                            '--cdc-review',str(work/'cdc_review.json'),'--capture',str(work/'board_clock_after_load.json'),
                            '--qualified',str(work/'qualified_clock.json'),'--sysfs',str(a.sysfs),'--xclbinutil',a.xclbinutil]
                if a.waiver:command += ['--waiver',str(a.waiver)]
            record('stage-start',stage=item['id'])
            with (out/(item['id']+'.log')).open('x') as f:
                result=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=259200)
            record('stage-end',stage=item['id'],returncode=result.returncode)
            if result.returncode or json.loads((target/'stage.json').read_text()).get('status')!='pass':
                raise RuntimeError('candidate stage failed: '+item['id'])
            row=auto.probe(a.sysfs)
            if not auto.idle(row) or row['uuid']!=state['image_uuid'] or auto.clock_mhz(a.sysfs)!=100:
                raise RuntimeError('card busy or image/clock changed after '+item['id'])
        record('completed')


if __name__ == '__main__':
    main()
