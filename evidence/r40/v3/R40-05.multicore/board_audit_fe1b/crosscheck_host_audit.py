"""Independent cross-check of the on-host R40-05.multicore audit against the preserved fe1b raw evidence.

    python3 evidence/r40/v3/R40-05.multicore/board_audit_fe1b/crosscheck_host_audit.py [XCLBIN]

tools/r40/multicore_board_acceptance.py can only run on the FPGA host: it hashes
the xclbin, re-reads the qualified clock's raw timing/route/CDC files and the
compiler bundles at their absolute host paths. The recollect_x2 queue ran it there
(stage physical-multicore-capability-audit). This script re-checks that run from
the copies in evidence/r40/v3 and re-runs the auditor's checkpoint and inventory
checks locally. Read-only; it writes nothing.
"""
import gzip, hashlib, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(R), str(R / 'python'), str(R / 'tools/r40')]
import contract_v2, native_inventory, multicore_board_acceptance as mba
E = R / 'evidence/r40/v3'
A = E / 'R40-05.multicore/board_audit_fe1b'
B = E / 'R40-04.hbm/board_run_fe1b'
Q = E / 'R40-02.zero-transfer/board_queue_fe1b'
CLK = E / 'R40-08.decode/board_model_fe1b/qualified_clock.json'
TREE = 'files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75'
IMAGE = 'a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083'
H_RUN = '/home/zzhouch/tnca_runs_r40_v3x2'
H_REC = '/home/zzhouch/tnca_v3_recollect_x2'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = []
def ok(name, cond, **kw):
    out.append(dict(name=name, ok=bool(cond), **kw)); print(('PASS ' if cond else 'FAIL ') + name, kw if not cond else '')
rep = json.loads((A / 'multicore_capability.json').read_text())
gate = json.loads((A / 'host_gate_evidence/R40-05.json').read_text())['gates']['R40-05.multicore']
ok('host report capability-pass with every check ok', rep['status'] == 'capability-pass' and rep['checks'] and all(c['ok'] for c in rep['checks']), n=len(rep['checks']))
ok('host auditor = repository tools/r40/multicore_board_acceptance.py', rep['harness_sha256'] == sha(R / 'tools/r40/multicore_board_acceptance.py') == gate['harness_sha256'])
ok('host report identity = delivery tree/image, v2 lock, clock evidence 8ad4b3bd', rep['identity']['software_tree'] == TREE and rep['identity']['image_sha256'] == IMAGE and rep['identity']['contract_sha256'] == contract_v2.LOCK_SHA256 and rep['identity']['operating_clock_evidence_sha256'] == sha(CLK) and rep['identity']['clock_hz'] == 90000000)
ok('host gate entry = host report (status, checks, inputs, identity) under the v2 lock', gate['status'] == rep['status'] and gate['checks'] == rep['checks'] and gate['inputs'] == rep['inputs'] and gate['identity'] == rep['identity'] and gate['contract_sha256'] == contract_v2.LOCK_SHA256)
local = {f'{H_RUN}/checkpoint/report.json': B / 'checkpoint/report.json',
         '/home/zzhouch/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json': CLK,
         **{f'{H_RUN}/native/u280_{t}.jsonl': B / f'native/u280_{t}.jsonl' for t in ('tiny', 'small', 'medium')}}
ok('every host audit input equals the preserved fe1b copy', set(rep['inputs']) == set(local) and all(rep['inputs'][k] == sha(v) for k, v in local.items()))
state = json.loads((Q / 'state.json').read_text()); cmds = dict(json.loads((Q / 'commands.json').read_text()))
st = [c for c in state['completed'] if c['name'] == 'physical-multicore-capability-audit']
argv = cmds['physical-multicore-capability-audit']
ok('queue stage physical-multicore-capability-audit exited 0, exact command', len(st) == 1 and st[0]['exit_code'] == 0 and argv[1].endswith('/tools/r40/multicore_board_acceptance.py') and argv[argv.index('--native') + 1] == f'{H_RUN}/native' and argv[argv.index('--checkpoint') + 1] == f'{H_RUN}/checkpoint/report.json' and argv[argv.index('--image-sha256') + 1] == IMAGE and argv[argv.index('--software-tree-sha256') + 1] == TREE and argv[argv.index('--out') + 1] == f'{H_REC}/multicore_capability.json' and '--evidence' in argv and json.loads((A / 'command.json').read_text()) == argv)
ok('stage log', (A / 'physical-multicore-capability-audit.log').read_text().strip() == 'R40-05.multicore: capability-pass')
ok('queue state complete on the delivery identity', state['status'] == 'complete-review-required' and state['software_tree'] == TREE and state['image_sha256'] == IMAGE and all(c['exit_code'] == 0 for c in state['completed']))
# bundles: every file each native77 record names is present on the host with the recorded hash (listing taken 2026-10-20)
listing = {}
for line in gzip.decompress((A / 'bundles_sha256.txt.gz').read_bytes()).decode().splitlines():
    if line and not line.startswith('#'):
        h, p = line.split('  ', 1); listing[p] = h
missing = changed = files = 0
for t in ('tiny', 'small', 'medium'):
    for line in (B / f'native/u280_{t}.jsonl').read_text().splitlines():
        r = json.loads(line); b = r['provenance']['bundle']
        for rel, h in b['files'].items():
            files += 1; p = f"{b['dir']}/{rel}"
            if p not in listing: missing += 1
            elif listing[p] != h: changed += 1
ok('native77 compiler bundles intact on the host (sha256 listing vs record hashes)', files and not missing and not changed, files=files, missing=missing, changed=changed)
# re-run the auditor's own checkpoint and inventory checks locally (no host paths needed)
clock = json.loads(CLK.read_text())
ck = mba.checkpoint_checks(json.loads((B / 'checkpoint/report.json').read_text()), tree=TREE, image=IMAGE,
                           clock=dict(clock_hz=clock['clock_hz'], operating_clock_evidence_sha256=sha(CLK)))
ok('local re-run of checkpoint_checks', all(c['ok'] for c in ck), n=len(ck))
inv = native_inventory.audit([B / f'native/u280_{t}.jsonl' for t in ('tiny', 'small', 'medium')])
ok('local re-run of native_inventory.audit', all(c['ok'] for c in inv), n=len(inv))
ok('local checks equal the host report checks', ck + inv == rep['checks'][:len(ck) + len(inv)])
xcl = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if xcl: ok('xclbin fetched from the host hashes to the delivery image', sha(xcl) == IMAGE)
print(json.dumps(dict(passed=sum(c['ok'] for c in out), total=len(out))))
sys.exit(0 if all(c['ok'] for c in out) else 1)
