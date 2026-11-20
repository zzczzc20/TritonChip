#!/usr/bin/env python3
"""R42 board A/B: register and queue the 113 original ArchBench points on the R41 run-3 image.

Local (lead machine; seconds, no torch, no board):

    board_ab.py register --name NAME --commit COMMIT --src-dir DIR [--knob TNCA_X=V ...]

writes spec/r42/board/NAME/{NAME-identity.json, NAME.json, queue.json}: the run-3 registrations
(evidence/r42/board/run3_registration/, fetched from the FPGA host, sha256 equal to the manifest hashes in the
run-3 records) with only the fields listed in CHANGES below changed. COMMIT names the measured software; its
software-tree identity (archbench/provenance.py of that commit) is computed from git and registered, and DIR is
the directory under ~/r42 on the FPGA host that holds `git archive COMMIT` (without evidence/).

FPGA host (run from the staged registration directory ~/r42/reg/NAME, detached):

    board_ab.py run --reg ~/r42/reg/NAME --out ~/r42/q/NAME [--check-only]

1. verifies the registration, the staged software tree identity, the image file and the author's approval record;
2. `acceptance_queue.py --check-only` and `preflight_board_manifest.py` (no card access); `--check-only` stops here;
3. when the card does not hold the registered image: waits with the R41 queue's own idle tracker
   (tools/r41/run2_board/auto.py: 15 consecutive idle minutes, same image and exec count, 5 s poll), takes the
   shared board lease, and loads the image with flow/fpga/program.sh --check, then --go (stdout in OUT/program.log).
   It never retries a load, never resets, and stops on any change;
4. runs tools/performance_harness/launches/acceptance_queue.py on the two manifests (identity first). That queue
   waits again for 15 idle minutes on the registered image, checks image and clock around every step, and never
   programs the card.

The guard code (auto.py, acceptance_queue.py, program.sh) is always taken from the registered HARNESS tree
(~/r42/src_base, unchanged main); the measured software tree supplies experiment.py, board_batch.py, ArchBench,
the compiler and the runtime, as in run 3.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
IMAGE_SHA = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
IMAGE_UUID = 'ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'
IMAGE = '{home}/r41run3/tnca_u280.xclbin'
APPROVAL = '{home}/r41run3/author_approval.json'
HBM_MAP = '{home}/r41run3/xclbin_hbm_map.json'
PYTHON = '{home}/tc_env/torchenv/bin/python'
HARNESS_DIR = 'src_base'
DEVICE = '0000:04:00.1'
R3 = 'evidence/r42/board/run3_registration'
R3_SHA = {'r3-s2-original-suite.json': '1696ea1b9ce6d00d3c6a39a96477c46f6a5773dad22d8c49c1b65e46c35a4d77',
          'r3-s2-identity.json': '1c15dbbc8563e0f1922634e791db3a6d6c41f81966a421efd4d639ccf649e7aa'}
RUN3_KNOBS = {'TNCA_ACTIVE_CORES': '3', 'TNCA_TORCH_DEVICE': 'native', 'TNCA_TRANSFER_POLICY': 'resident'}
TREE_DIRS = ('archbench', 'python', 'compiler', 'tools/asm', 'spec', 'rtl', 'tb/device')
CHANGES = [
    'id, hypothesis, competing_explanations, expected_*: R42 wording',
    'registration: R42 block (software commit, tree identity, knobs); the run-3 block is kept as derived_from',
    'steps[].argv --manifest and sources: the manifest is read from {home}/r42/reg/NAME/ (outside the measured '
    'tree, so the tree stays byte-equal to its commit) instead of spec/r41/run3/board/',
    'board.memory_map.source and sources: {home}/r41run3/xclbin_hbm_map.json (same file, same sha256) instead of '
    'spec/r41/run3/xclbin_hbm_map.json, which is not in the repository',
    'estimates: removed (only --dry-run reads them; evidence/ is not staged)',
    'identity manifest: the `source` step (review of the frozen RTL image source archive, an R41 delivery check) '
    'and its two tool sources are removed; the clock verification and the three identity points are unchanged',
    'board.load_policy: R42 wording',
    'optional --knob entries are added to every configuration (none for the baseline)',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def tree_identity(root):
    """archbench.provenance.software_tree of ROOT, by ROOT's own definition."""
    spec = importlib.util.spec_from_file_location('_r42_provenance', Path(root)/'archbench/provenance.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.software_tree(Path(root), refresh=True)


# ---------------------------------------------------------------------------------------------- register (local)

def commit_tree_identity(repo, commit):
    with tempfile.TemporaryDirectory(prefix='r42tree_') as tmp:
        archive = subprocess.Popen(['git', '-C', str(repo), 'archive', commit, '--', *TREE_DIRS],
                                   stdout=subprocess.PIPE)
        subprocess.run(['tar', 'xf', '-', '-C', tmp], stdin=archive.stdout, check=True)
        if archive.wait():
            raise SystemExit('git archive failed')
        return tree_identity(tmp)


def derive(old, name, kind, knobs, registration):
    spec = copy.deepcopy(old)
    old_manifest = f'spec/r41/run3/board/{old["id"]}.json'
    new_manifest = f'{{home}}/r42/reg/{name}/{name}{"-identity" if kind == "identity" else ""}.json'
    spec['id'] = name + ('-identity' if kind == 'identity' else '')
    spec['hypothesis'] = (f'R42 {name}: the 113 original ArchBench points '
                          f'({"three identity points first" if kind == "identity" else "full suite"}) pass their '
                          'numerical and structural checks on the R41 run-3 image with the registered software '
                          'tree, and give its serial-grid device cycles at 100 MHz.')
    spec['competing_explanations'] = [
        'the software tree differs from run 3 (compiler, runtime, harness) and changes cycles or numerics',
        'a workload fails, falls back, times out, or is omitted from its registered checks',
        'the card identity or clock changes during the run (shared card)']
    spec['expected_counter_changes'] = [
        'baseline tree (main 4b61abd8): serial-grid cycles equal to the run-3 records within run-to-run noise',
        'candidate tree: cycles change only where the candidate changes generated code or the runtime',
        'no host fallback and no upload/download inside resident calls']
    spec['expected_observations'] = [
        'every registered configuration completes with its numerical and structural checks passing',
        'the experiment records the observed image, clock, software tree and raw results']
    spec['board']['memory_map']['source'] = HBM_MAP
    spec['board']['load_policy'] = ('approved run-3 image; loaded only by tools/r42/board_ab.py after 15 consecutive '
                                    'idle minutes through program.sh --check/--go; every FPGA step checks identity')
    spec['sources'] = [HBM_MAP if s == 'spec/r41/run3/xclbin_hbm_map.json' else
                       new_manifest if s == old_manifest else s for s in spec['sources']
                       if s not in ('tools/r41/run2_board/source_review.py',
                                    'tools/r41/run2_board/candidate_source.py')]
    spec.pop('estimates', None)
    steps = []
    for step in spec['steps']:
        if step['id'] == 'source':
            continue
        step['argv'] = [new_manifest if a == old_manifest else a for a in step['argv']]
        steps.append(step)
    spec['steps'] = steps
    for batch in spec['board_batches'].values():
        for row in (batch.get('expand') or []) + (batch.get('configurations') or []):
            if row['knobs'] != RUN3_KNOBS:
                raise SystemExit('run-3 registration has unexpected knobs')
            row['knobs'] = dict(row['knobs'], **knobs)
    text = json.dumps({k: v for k, v in spec.items() if k != 'registration'})
    if 'spec/r41/run3' in text or 'candidate_source' in text:
        raise SystemExit('a run-3 path survived the derivation')
    spec['registration'] = dict(registration, derived_from=f'{R3}/{old["id"]}.json',
                                derived_from_sha256=R3_SHA[old['id']+'.json'],
                                run3_registration=old['registration'])
    return spec


def register(a):
    repo = HERE.parents[1]
    name = a.name
    if not name.replace('-', '').replace('_', '').isalnum():
        raise SystemExit('NAME must be a simple name')
    knobs = dict(k.split('=', 1) for k in a.knob)
    if any(not k.startswith('TNCA_') for k in knobs):
        raise SystemExit('knobs are TNCA_* variables')
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', a.commit + '^{commit}'], capture_output=True,
                            text=True, check=True).stdout.strip()
    out = repo/'spec/r42/board'/name
    if out.exists():
        raise SystemExit(f'{out} exists: a registration is never rewritten; choose a new NAME')
    for f, digest in R3_SHA.items():
        if sha(repo/R3/f) != digest:
            raise SystemExit('run-3 registration copy changed: ' + f)
    registration = dict(status='registered-not-run', registered_utc=utc(), round='R42', name=name,
                        software=dict(commit=commit, host_dir='{home}/r42/' + a.src_dir,
                                      software_tree=commit_tree_identity(repo, commit),
                                      staging=f"git archive {commit} -- . ':!evidence' ':!e7c' | tar x"),
                        knobs=dict(RUN3_KNOBS, **knobs), extra_knobs=knobs,
                        image=dict(sha256=IMAGE_SHA, uuid=IMAGE_UUID),
                        generator='tools/r42/board_ab.py register', changes_against_run3=CHANGES,
                        sampling='unchanged: spec/r41/stage1/plan.json measurement.long_device_tasks (qualification '
                                 'record 0 warm-up + 1 timed call; a point at <= 1 s device time then gets a formal '
                                 'record of 3 timed calls in a new process, else the qualification record is formal)',
                        boundary='unchanged: ArchBench protocol 2, device-cycles-serial-grid at the qualified 100 MHz')
    out.mkdir(parents=True)
    manifests = {}
    for kind, source in (('identity', 'r3-s2-identity.json'), ('suite', 'r3-s2-original-suite.json')):
        spec = derive(json.loads((repo/R3/source).read_text()), name, kind, knobs, registration)
        path = out/(spec['id'] + '.json')
        path.write_text(json.dumps(spec, indent=1) + '\n')
        manifests[kind] = dict(file=path.name, id=spec['id'], sha256=sha(path))
    queue = dict(schema='tnca-r42-board-ab/1', name=name, registration=registration, manifests=manifests,
                 harness=dict(host_dir='{home}/r42/' + HARNESS_DIR, commit_note='unchanged main 4b61abd8 tree'),
                 image=dict(path=IMAGE, sha256=IMAGE_SHA, uuid=IMAGE_UUID, approval=APPROVAL, clock_mhz=100,
                            device=DEVICE),
                 python=PYTHON, budget_seconds=10800,
                 checks=[dict(manifest='identity', path='identity/batch.json', field=['status'], equals='complete'),
                         dict(manifest='identity', path='identity/batch.json', field=['configurations'], equals=3),
                         dict(manifest='identity', path='clock.json', field=['status'], equals='pass'),
                         dict(manifest='suite', path='original/batch.json', field=['configurations'], equals=113)])
    (out/'queue.json').write_text(json.dumps(queue, indent=1) + '\n')
    print(json.dumps(dict(registered=str(out), software_tree=registration['software']['software_tree'],
                          commit=commit, manifests=manifests), indent=1))


# ------------------------------------------------------------------------------------------------ run (FPGA host)

def load_module(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(reg_dir, home):
    queue = json.loads((reg_dir/'queue.json').read_text())
    if queue.get('schema') != 'tnca-r42-board-ab/1' or queue['name'] != reg_dir.name:
        raise ValueError('not an R42 board A/B registration')
    expand = lambda s: Path(s.replace('{home}', str(home)))
    manifests = {}
    for kind in ('identity', 'suite'):
        row = queue['manifests'][kind]
        path = reg_dir/row['file']
        if sha(path) != row['sha256'] or json.loads(path.read_text())['id'] != row['id']:
            raise ValueError('registered manifest missing or changed: ' + str(path))
        manifests[kind] = path
    image = queue['image']
    if (image['sha256'], image['uuid']) != (IMAGE_SHA, IMAGE_UUID):
        raise ValueError('only the approved run-3 image is admitted')
    image_path = expand(image['path'])
    if sha(image_path) != IMAGE_SHA:
        raise ValueError('image file is not the approved run-3 image')
    approval = json.loads(expand(image['approval']).read_text())
    if (approval.get('status') != 'accepted' or approval.get('image_sha256') != IMAGE_SHA
            or approval.get('image_uuid') != IMAGE_UUID or not approval.get('author_decision')):
        raise ValueError('the author approval record does not cover this image')
    root = expand(queue['registration']['software']['host_dir'])
    harness = expand(queue['harness']['host_dir'])
    observed = tree_identity(root)
    if observed != queue['registration']['software']['software_tree']:
        raise ValueError(f'staged software tree {observed} is not the registered tree')
    for needed in ('tools/r41/run2_board/auto.py', 'tools/performance_harness/launches/acceptance_queue.py',
                   'tools/performance_harness/preflight_board_manifest.py', 'flow/fpga/program.sh'):
        if not (harness/needed).is_file():
            raise ValueError('harness tree lacks ' + needed)
    return dict(queue=queue, manifests=manifests, image=image_path, root=root, harness=harness,
                python=expand(queue['python']), approval=expand(image['approval']),
                identity=dict(software_tree=observed, harness_tree=tree_identity(harness),
                              image_sha256=IMAGE_SHA, approval_sha256=sha(expand(image['approval'])),
                              runner_sha256=sha(Path(__file__).resolve())))


def guarded_load(state, out, record, env, auto, lease_path, sysfs, poll=5):
    """Load the registered image, only after the R41 idle rule; never retried."""
    tracker = auto.IdleTracker(max_gap=max(15, poll*2.5))
    last_logged = 0
    with lease_path.open('a') as lease:
        while True:
            try:
                row = auto.probe(sysfs)
            except (OSError, ValueError) as error:
                # As auto.py: an unreadable card is not idle; start the interval again.
                tracker = auto.IdleTracker(max_gap=max(15, poll*2.5))
                record('probe-error', error=str(error))
                time.sleep(poll)
                continue
            if row['uuid'] == IMAGE_UUID:
                record('already-loaded', **row)
                return False
            ready = tracker.observe(time.monotonic(), row)
            if time.monotonic() - last_logged >= 60 or not auto.idle(row):
                record('idle-probe', idle_seconds=(None if tracker.start is None else round(tracker.last - tracker.start)),
                       **row)
                last_logged = time.monotonic()
            if ready and auto.claim_idle_lease(tracker, lease, lambda: auto.probe(sysfs)):
                break
            time.sleep(poll)
        try:
            record('idle-rule-met', idle_seconds=round(tracker.last - tracker.start), identity=list(tracker.identity))
            previous = state['image'].parent/'program.log'
            if previous.is_file():
                (out/'program.log.previous_load').write_bytes(previous.read_bytes())
            for mode in ('--check', '--go'):
                if sha(state['image']) != IMAGE_SHA or sha(state['approval']) != state['identity']['approval_sha256']:
                    raise RuntimeError('image or approval changed before programming')
                now = auto.probe(sysfs)
                if not auto.idle(now) or (now['uuid'], now['total execs']) != tracker.identity:
                    raise RuntimeError('card changed or became busy before programming')
                record('program', mode=mode, previous=now['uuid'])
                with (out/'program.log').open('a') as f:
                    subprocess.run(['bash', str(state['harness']/'flow/fpga/program.sh'), mode, str(state['image'])],
                                   stdout=f, stderr=subprocess.STDOUT, env=env, check=True)
            row = auto.probe(sysfs)
            mhz = auto.clock_mhz(sysfs)
            record('loaded', clock_mhz=mhz, **row)
            if row['uuid'] != IMAGE_UUID or mhz != 100:
                raise RuntimeError('loaded image UUID or clock differs; stopping')
            if previous.is_file():
                (out/'program.log.loader').write_bytes(previous.read_bytes())
            return True
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def run(a):
    home = Path.home()
    reg_dir, out = a.reg.resolve(), a.out.resolve()
    state = verify(reg_dir, home)
    queue = state['queue']
    harness, root, python = state['harness'], state['root'], state['python']
    out.mkdir(parents=True, exist_ok=True)
    log = (out/'events.jsonl').open('a', buffering=1)

    def record(kind, **kw):
        log.write(json.dumps(dict(utc=utc(), kind=kind, **kw)) + '\n')

    record('verified', check_only=a.check_only, **state['identity'])
    base = [str(python), str(harness/'tools/performance_harness/launches/acceptance_queue.py'), '--root', str(root),
            '--out', str(out/'queue'), '--python', str(python), '--budget-seconds', str(queue['budget_seconds'])]
    for kind in ('identity', 'suite'):
        base += ['--manifest', str(state['manifests'][kind])]
    check = subprocess.run(base + ['--check-only'], cwd=root, capture_output=True, text=True)
    record('queue-check-only', returncode=check.returncode, stdout=check.stdout.strip(), stderr=check.stderr.strip()[-2000:])
    if check.returncode or json.loads(check.stdout).get('status') != 'prepared-only':
        raise SystemExit('acceptance_queue --check-only failed')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    pre = subprocess.run([str(python), str(harness/'tools/performance_harness/preflight_board_manifest.py'),
                          '--root', str(root), '--manifest', str(state['manifests']['suite']),
                          '--out', str(out/('preflight_' + stamp))], cwd=root, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(root)))
    record('preflight', returncode=pre.returncode, stdout=pre.stdout.strip()[-1000:], stderr=pre.stderr.strip()[-2000:])
    if pre.returncode:
        raise SystemExit('preflight_board_manifest failed')
    if a.check_only:
        print(json.dumps(dict(status='prepared-only', card_access=False, **state['identity']), indent=1))
        return
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            auto = load_module(harness/'tools/r41/run2_board/auto.py', '_r42_auto')
            lease_mod = load_module(harness/'tools/tritonchip_fpga_harness/board_lease.py', '_r42_lease')
            env = auto.board_environment()
            sysfs = Path('/sys/bus/pci/devices')
            guarded_load(state, out, record, env, auto, lease_mod.lease_path(DEVICE), sysfs)
            record('queue-start', command=base)
            with (out/'queue.log').open('a') as f:
                result = subprocess.run(base, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
            record('queue-end', returncode=result.returncode)
            if result.returncode:
                raise RuntimeError('acceptance queue stopped; evidence preserved under ' + str(out/'queue'))
            failed = []
            for c in queue['checks']:
                value = json.loads((out/'queue'/queue['manifests'][c['manifest']]['id']/c['path']).read_text())
                for key in c['field']:
                    value = value[key]
                if value != c['equals']:
                    failed.append(dict(c, observed=value))
            suite = json.loads((out/'queue'/queue['manifests']['suite']['id']/'original/batch.json').read_text())
            record('completed', failed_checks=failed, suite_status=suite['status'],
                   failed_configurations=suite['failed_configurations'], missing_formal=suite['missing_formal'])
        except BaseException as error:
            record('stopped', error=f'{type(error).__name__}: {error}')
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('register')
    r.add_argument('--name', required=True)
    r.add_argument('--commit', required=True)
    r.add_argument('--src-dir', required=True, help='directory name under ~/r42 on the FPGA host')
    r.add_argument('--knob', action='append', default=[], help='TNCA_NAME=VALUE added to every configuration')
    q = sub.add_parser('run')
    q.add_argument('--reg', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    register(a) if a.cmd == 'register' else run(a)


if __name__ == '__main__':
    main()
