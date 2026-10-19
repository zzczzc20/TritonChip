#!/usr/bin/env python3
"""R40 v3 audit of the memory-system capability gates.

    R40_CONTRACT=v3 python3 tools/r40/v3_memsys_gates.py [--dry-run]
        [--board-run DIR] [--window-samples FILE] [--model-report FILE] [--large24 FILE]
        [--ci-dir DIR] [--semantics-dir DIR] [--preserve]
        [--hbm-model-fidelity strict|uncalibrated-declared]

Gates: R40-04.hbm, R40-04.robustness, R40-05.dispatch, R40-06.nuca,
R40-06.semantics, R40-09.suites. Performance gates (R40-04.bandwidth,
R40-05.scaling, R40-06.streaming-scaling) and R40-06.locality (deferred to
R41) are out of scope.

Identity: contract_v3.delivery(plan) -- product tree (after the identity
amendment), image a88ebdf5, measured 90 MHz. Board evidence must be bound to
exactly that tree; records on an earlier tree are early signals only and leave
the gate not-measured. Pure RTL/simulation evidence may come from an earlier
tree when every differing product file is listed by the identity amendment and
the executed harness does not enable the changed behaviour (TNCA_MAX_BLOCK);
the auditor recomputes both trees and records the differing files.

Inputs (defaults are the preserved copies; arguments point at new runs):

  --board-run DIR       a functional delivery run: checkpoint/report.json,
                        native/u280_{tiny,small,medium}.jsonl, hbm14.jsonl
  --window-samples FILE tools/r40/hbm_board.py --only window samples --json
  --model-report FILE   tools/r40/native_model.py report (.json or .json.gz)
  --large24 FILE        archbench records of the original large tier
  --ci-dir DIR          run.json (gh run view --json headSha,status,conclusion,jobs)
                        and <lane>.log job logs of one CI run
  --semantics-dir DIR   tools/r40/v3_semantics_matrix.py results and
                        journal_replay/report.json
  --preserve            copy the given board inputs into evidence/r40/v3/<gate>/
                        (large sets as tar.gz) before judging them

Only raw evidence is judged, never a producer's verdict. Missing evidence is
not-measured; contradicting evidence is fail.

HBM model fidelity (R40-04 acceptance 3, sec 3.4) needs the frozen calibration
(R40-04.calibration, 17 points) and holdout (R40-04.model-holdout, 19 points)
compared with the board. They are large/hbm points whose RTL simulation is
infeasible, so no calibrated model exists. By default (`strict`) the gate stays
not-measured. `--hbm-model-fidelity uncalibrated-declared` records the explicit
sec 3.4 branch -- model marked uncalibrated, its cycles never used for
acceptance -- as a reviewer decision in the check; the fixed-input replay
must still pass.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python'), str(ROOT / 'tools/r40')]

EVIDENCE = ROOT / 'evidence/r40/v3'
GIB = 1 << 30
PC_BYTES = 256 << 20
CLOCK_EVIDENCE_SHA = '8ad4b3bd30a3f5fe8819fe59ef8ad160fd0de1478aaa506dc33d82785fb9f487'
V2_LOCK = '49fc0e02733139707ae2292eff2ef893e76e73f0723d0d002b2e7e1864af71c8'
DELIVERY_GIT = '0958db2'   # last commit whose harness files the delivery runs use
LANES = {
    'check': 'lint, unit, core, system, compiler',
    'transport': 'R40 grid and pipeline (transport)',
    'native': 'R40 grid and pipeline (native)',
    'chunks': 'R40 grid and pipeline (chunks)',
    'memory': 'R40 grid and pipeline (memory)',
    'mmio-checks': 'R40 multicore MMIO (checks)',
    'registers-1': 'R40 multicore MMIO (registers-1)',
    'registers-3': 'R40 multicore MMIO (registers-3)',
    'registers-board': 'R40 multicore MMIO (registers-board)',
    'native-ports': 'R40 multicore MMIO (native-ports)',
    'native-features': 'R40 multicore MMIO (native-features)',
    'native-pair': 'R40 multicore MMIO (native-pair)',
}
# Everything outside the hashed product tree that the CI lanes execute.
# The delivery commit's Makefile only adds these two board-harness unit tests to
# r40-multicore-mmio-checks; every other recipe is compared line for line.
MAKEFILE_ADDED_OK = ('\t@$(PYTHON) tools/r40/conditional_delivery_tests.py',
                     '\t@$(TRITON_PYTHON) tools/r40/bucket_decode_tests.py')
HARNESS = ('.github/workflows/ci.yml', 'tb', 'tools/r1', 'tools/r36/conformance.py',
           'tools/run_test.sh') + tuple(f'tools/r40/{n}.py' for n in (
    'grid_dispatch_tests', 'coherent_host_tests', 'multicore_shell_tests', 'memory_control_tests',
    'physical_grid_option_tests', 'multicore_transport_tests', 'large_grid_tests',
    'physical_grid_runtime_tests', 'native_multicore_tests', 'physical_chunk_tests',
    'physical_chunk_runtime_tests', 'native_multicore_features', 'memory_port_option_tests',
    'memory_ports_tests', 'native_memory_ports_tests', 'line_axi_tests', 'multicore_axi_tests',
    'cache_maintenance_tests', 'multicore_mmio_option_tests', 'dma_handoff_guard_tests',
    'multicore_dma_tests', 'recovery_tests', 'multicore_mmio_tests', 'mesh_pipeline_tests',
    'code_link_tests', 'reserved_memory_tests', 'storage_owner_tests', 'copy_range_tests',
    'multicore_mmio_synth_tests'))
SEM_CONFIGS = ('mc1', 'mc2', 'mc3p3', 'mc4', 'mc8', 'bt3p3s2', 'regs3p1s0')


class Missing(Exception):
    """Evidence not (yet) present for the delivery identity: not-measured."""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def need(path):
    path = Path(path)
    if not path.exists():
        raise Missing(f'missing evidence {rel(path)}')
    return path


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def check(name, ok, **detail):
    return dict(name=name, ok=bool(ok), **detail)


def read_json(path):
    path = need(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


# --------------------------------------------------------------- identity ---
class Identity:
    def __init__(self):
        import contract_v3
        contract_v3.verify()
        self.contract_v3 = contract_v3
        plan = json.loads((ROOT / 'doc/r40_v3/plan.json').read_text())
        d = contract_v3.delivery(plan)
        self.tree, self.image, self.clock = d['software_tree'], d['image_sha256'], d['clock_hz']
        self.backend, self.shape = d['backend'], d['backend_args_shape']
        self.amended = {}
        path = ROOT / contract_v3.IDENTITY_AMENDMENT
        if path.exists():
            a = json.loads(path.read_text())
            self.amended = dict(from_tree=a['from_software_tree'], files=sorted(a['files']))
        self.locks = {V2_LOCK, contract_v3.LOCK_SHA256}
        from archbench import provenance
        self.provenance = provenance
        self.root_tree = provenance.software_tree(ROOT, refresh=True)
        self._files = {}

    def files(self, root):
        root = Path(root)
        return {str(f.relative_to(root)): file_sha(f) for f in self.provenance._tree_files(root)}

    def commit_files(self, commit):
        if commit not in self._files:
            with tempfile.TemporaryDirectory(prefix='r40-tree-') as tmp:
                archive = subprocess.run(['git', 'archive', commit, *self.provenance.TREE_DIRS], cwd=ROOT,
                                         capture_output=True, check=True).stdout
                with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                    tar.extractall(tmp, filter='data')
                self._files[commit] = (self.provenance.software_tree(Path(tmp), refresh=True), self.files(tmp))
        return self._files[commit]

    def simulation_relation(self, evidence_tree, commit):
        """Check: simulation evidence on `evidence_tree` (the product tree of git
        `commit`) is valid for the delivery tree."""
        if evidence_tree == self.tree:
            return check('simulation evidence measured on the delivery tree', True, tree=evidence_tree)
        if not commit:
            return check('simulation evidence names the commit of its product tree', False, tree=evidence_tree)
        tree, files = self.commit_files(commit)
        if tree != evidence_tree:
            return check('simulation evidence tree matches its recorded commit', False,
                         evidence_tree=evidence_tree, commit=commit, commit_tree=tree)
        if self.root_tree != self.tree:
            raise Missing(f'this checkout is not the delivery tree ({self.root_tree}); cannot diff')
        here = self.files(ROOT)
        differ = sorted(p for p in set(files) | set(here) if files.get(p) != here.get(p))
        return check('simulation evidence tree differs from the delivery tree only in amendment files',
                     differ and set(differ) <= set(self.amended.get('files', [])) and
                     evidence_tree == self.amended.get('from_tree'),
                     evidence_tree=evidence_tree, delivery_tree=self.tree, differing_files=differ,
                     amendment_files=self.amended.get('files'),
                     effect='scan/sort lane cap only when TNCA_MAX_BLOCK > 16; generated source at the '
                            'default cap is unchanged by construction')

    def board(self, ident, what):
        args = ident.get('backend_args') or {}
        ok = (ident.get('software_tree') == self.tree and ident.get('image_sha256') == self.image
              and ident.get('clock_hz') == self.clock and ident.get('backend') == self.backend
              and ident.get('operating_clock_evidence_sha256') == CLOCK_EVIDENCE_SHA
              and {k: args.get(k) for k in self.shape} == self.shape
              and ident.get('contract_sha256') in self.locks)
        if ident.get('software_tree') != self.tree:
            raise Missing(f'{what} measured on {ident.get("software_tree")}, not the delivery tree {self.tree}')
        return check(f'{what} bound to the delivery tree, image, measured 90 MHz and the contract lock', ok,
                     software_tree=ident.get('software_tree'), image=ident.get('image_sha256'),
                     clock_hz=ident.get('clock_hz'), contract_sha256=ident.get('contract_sha256'))

    def record(self, r):
        prov, p = r.get('provenance') or {}, r.get('platform') or {}
        ev = p.get('operating_clock_evidence') or {}
        return (prov.get('software_tree') == self.tree and prov.get('image') == 'xclbin:' + self.image[:16]
                and prov.get('clock_hz') == self.clock and p.get('backend') == self.backend
                and {k: (p.get('backend_args') or {}).get(k) for k in self.shape} == self.shape
                and ev.get('image_sha256') == self.image and ev.get('clock_hz') == self.clock)


# ------------------------------------------------------------------ inputs ---
class Inputs:
    """Board inputs from arguments or the preserved copies."""
    HBM = EVIDENCE / 'R40-04.hbm'
    SUITES = EVIDENCE / 'R40-09.suites'

    def __init__(self, args):
        self.args = args
        self.sources = {}
        if args.preserve:
            self.preserve()

    def preserve(self):
        a = self.args
        self.HBM.mkdir(parents=True, exist_ok=True)
        self.SUITES.mkdir(parents=True, exist_ok=True)
        sources = {}
        for src, dst in ((a.window_samples, self.HBM / 'window_samples.json'),
                         (a.board_run and a.board_run / 'checkpoint/report.json', self.HBM / 'checkpoint_report.json')):
            if src:
                shutil.copyfile(need(src), dst)
                sources[rel(dst)] = dict(source=str(src), sha256=file_sha(dst))
        if a.model_report:
            raw = need(a.model_report).read_bytes()
            if a.model_report.suffix != '.gz':
                raw = gzip.compress(raw, 9, mtime=0)
            (self.HBM / 'model_report.json.gz').write_bytes(raw)
            sources[rel(self.HBM / 'model_report.json.gz')] = dict(
                source=str(a.model_report), sha256=file_sha(self.HBM / 'model_report.json.gz'),
                uncompressed_sha256=sha(gzip.decompress(raw)))
        if a.board_run or a.large24:
            members = {}
            if a.board_run:
                for t in ('tiny', 'small', 'medium'):
                    members[f'native/u280_{t}.jsonl'] = need(a.board_run / f'native/u280_{t}.jsonl')
                members['hbm14.jsonl'] = need(a.board_run / 'hbm14.jsonl')
            if a.large24:
                members['large24.jsonl'] = need(a.large24)
            out = self.SUITES / 'board_records.tar.gz'
            with tarfile.open(out, 'w:gz') as tar:
                for name, path in sorted(members.items()):
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, '', ''
                    with open(path, 'rb') as fh:
                        tar.addfile(info, fh)
            sources[rel(out)] = dict(sha256=file_sha(out),
                                     members={n: dict(source=str(p), sha256=file_sha(p)) for n, p in members.items()})
        old = {}
        if (self.SUITES / 'sources.json').exists():
            old = json.loads((self.SUITES / 'sources.json').read_text())
        old.update(sources)
        (self.SUITES / 'sources.json').write_text(json.dumps(old, indent=1, sort_keys=True) + '\n')

    def _path(self, given, preserved):
        return given if (given and not self.args.preserve) else preserved

    def window_samples(self):
        return read_json(self._path(self.args.window_samples, self.HBM / 'window_samples.json'))

    def checkpoint(self):
        given = self.args.board_run and self.args.board_run / 'checkpoint/report.json'
        return read_json(self._path(given, self.HBM / 'checkpoint_report.json'))

    def model(self):
        return read_json(self._path(self.args.model_report, self.HBM / 'model_report.json.gz'))

    def records(self, name):
        files = dict(native77=[f'native/u280_{t}.jsonl' for t in ('tiny', 'small', 'medium')],
                     hbm14=['hbm14.jsonl'], large24=['large24.jsonl'])[name]
        a = self.args
        direct = None
        if not a.preserve:
            if name == 'large24' and a.large24:
                direct = [a.large24]
            elif name != 'large24' and a.board_run:
                direct = [a.board_run / f for f in files]
        texts = []
        if direct:
            texts = [need(p).read_text() for p in direct]
        else:
            archive = need(self.SUITES / 'board_records.tar.gz')
            with tarfile.open(archive) as tar:
                names = set(tar.getnames())
                for f in files:
                    if f not in names:
                        raise Missing(f'{name}: {f} not in {rel(archive)}')
                    texts.append(tar.extractfile(f).read().decode())
        return [json.loads(line) for text in texts for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------- CI ---
class CI:
    def __init__(self, directory, ident):
        self.dir = Path(directory)
        self.ident = ident
        self._identity = None

    def identity(self):
        if self._identity is None:
            run = read_json(self.dir / 'run.json')
            self.run = run
            head = run.get('headSha')
            tree, _ = self.ident.commit_files(head)
            checks = [check(f'CI run {run.get("databaseId", self.dir.name)} head {head}', bool(head), head=head),
                      self.ident.simulation_relation(tree, head)]
            differ = [p for p in HARNESS
                      if git('rev-parse', f'{head}:{p}').strip() != git('rev-parse', f'{DELIVERY_GIT}:{p}').strip()]
            checks.append(check('every executed harness/testbench path is the same git object at the delivery '
                                'commit', not differ, differ=differ))
            diff = git('diff', '-U0', head, DELIVERY_GIT, '--', 'Makefile').splitlines()
            changed = [l for l in diff if l[:1] in '+-' and not l.startswith(('+++', '---'))]
            checks.append(check('Makefile recipes equal the delivery commit except two added unrelated unit tests',
                                all(l[:1] == '+' and l[1:] in MAKEFILE_ADDED_OK for l in changed),
                                changed_lines=changed))
            text = git('show', f'{head}:.github/workflows/ci.yml') + git('show', f'{head}:Makefile')
            checks.append(check('the CI lanes never enable the amended behaviour (TNCA_MAX_BLOCK unset)',
                                'TNCA_MAX_BLOCK' not in text))
            self._identity = checks
        return self._identity

    def lane(self, name):
        self.identity()
        jobs = [j for j in self.run.get('jobs', []) if j.get('name') == LANES[name]]
        if not jobs or jobs[0].get('status') != 'completed':
            raise Missing(f'CI lane {name} not completed')
        path = need(self.dir / f'{name}.log')
        return path.read_text(errors='replace'), check(
            f'CI lane {name} succeeded', jobs[0].get('conclusion') == 'success',
            job=jobs[0].get('databaseId'), log=rel(path), log_sha256=file_sha(path))


def harness_files(src, what):
    """The non-product files a simulation executed (testbenches, harness) must be
    the delivery commit's blobs, or this checkout's file for a new v3 harness."""
    differ, absent = [], []
    for path, digest in sorted((src.get('files') or {}).items()):
        try:
            want = sha(subprocess.run(['git', 'show', f'{DELIVERY_GIT}:{path}'], cwd=ROOT,
                                      capture_output=True, check=True).stdout)
        except subprocess.CalledProcessError:
            local = ROOT / path
            want = file_sha(local) if local.is_file() else None
            if want is None:
                absent.append(path)
        if want != digest:
            differ.append(path)
    return check(f'{what}: every executed testbench/harness file equals the delivery commit '
                 f'({DELIVERY_GIT}) or this checkout', src.get('files') and not differ,
                 files=len(src.get('files') or {}), differ=differ, absent=absent,
                 host=src.get('host'), path=src.get('path'))


def markers(name, text, patterns):
    """Each (regex, minimum count) must occur in the log text."""
    found = {p: len(re.findall(p, text, re.M)) for p, _ in patterns}
    bad = {p: found[p] for p, n in patterns if found[p] < n}
    return check(name, not bad, counts=found, missing=bad)


# ------------------------------------------------------------------ board ---
def memory_peak(native_memory):
    s = native_memory['session']
    data = max(s['heap_peak_bytes'], native_memory.get('tensor_peak_bytes', 0))
    return dict(data_peak_bytes=data, tensor_peak_bytes=native_memory.get('tensor_peak_bytes', 0),
                code_peak_bytes=s['code_peak_bytes'], total_peak_bytes=data + s['code_peak_bytes'],
                heap_capacity_bytes=s['heap_capacity_bytes'],
                reserved_bytes=sum(hi - lo for lo, hi in s.get('heap_reserved_ranges', [])))


def pcs_of(addresses, window):
    banks = window['allocation']['physical_banks']
    local, base = window['local_limit'], window['base']
    hit = set()
    for address, size in addresses:
        lo = address - local + base
        hit |= {b['pc'] for b in banks
                if lo < b['allocated_base'] + b['allocated_size'] and lo + size > b['allocated_base']}
    return sorted(hit)


def per_port_probes(samples, window):
    """Which outlet served each probe line. On the board top (NX=3, NY=1) a
    line's home is the XOR of its line-address bytes mod 3 (rtl/noc/
    tnca_core_ni.v) and outlet = home mod MEM_PORTS = home (rtl/nuca/
    tnca_mem_ports.v); both are read from this tree, not assumed."""
    ni = (ROOT / 'rtl/noc/tnca_core_ni.v').read_text()
    xo = (ROOT / 'rtl/board/tnca_multicore_hbm_xo.v').read_text()
    ports = (ROOT / 'rtl/nuca/tnca_mem_ports.v').read_text()
    rtl = ('line[7:0] ^ line[15:8] ^ line[23:16] ^ line[31:24]' in ni and ': hash % (NX*NY);' in ni
           and '.NX(3),.NY(1)' in xo and 'source%PORTS==p' in ports)

    def outlet(address):
        line, h = address >> 6, 0
        for i in range(8):
            h ^= (line >> (8 * i)) & 0xff
        return h % 3
    local, pc = int(window['local_limit'], 16), PC_BYTES
    per = {p: dict(probes=0, pc_boundary=0, high_address=0) for p in range(3)}
    for e in samples.get('probe_storage') or []:
        a = local + e['offset']
        served = {outlet(a + 64 * k) for k in range(512 // 64)}
        for p in served:
            per[p]['probes'] += 1
            per[p]['pc_boundary'] += int(e['offset'] // pc != (e['offset'] + 511) // pc)
            per[p]['high_address'] += int(a + 511 >= 1 << 32)
    return check('each of the 3 active outlets served PC-boundary, >=2^32-address and alias-pattern probe lines',
                 rtl and all(v['probes'] and v['pc_boundary'] and v['high_address'] for v in per.values()),
                 per_outlet=per, rule='outlet = (XOR of line-address bytes) mod 3, from this tree\'s RTL')


def capacity_plan(peak, window_bytes, connected_bytes):
    """8/16/32-core capacity planning from the measured peak (analysis).

    Per-core state is the per-launch argument/descriptor/scratch table the
    physical-grid runtime allocates once per core; everything the heap holds
    beyond tensors is scaled with the core count (conservative)."""
    per_core = max(0, peak['data_peak_bytes'] - peak['tensor_peak_bytes']) / 3
    rows = []
    for cores in (1, 2, 3, 4, 8, 16, 32):
        need_bytes = peak['tensor_peak_bytes'] + per_core * cores + peak['code_peak_bytes'] + peak['reserved_bytes']
        required = 1.25 * need_bytes
        rows.append(dict(cores=cores, peak_bytes=int(need_bytes), required_window_bytes=int(required),
                         required_pcs=math.ceil(required / PC_BYTES),
                         fits_exposed_window_with_25pct=required <= window_bytes,
                         fits_full_connected_hbm_with_25pct=required <= connected_bytes,
                         status='measured image' if cores == 3 else
                                ('functional RTL (8 cores)' if cores in (1, 2, 4, 8) else 'analysis')))
    return dict(schema='tnca-r40-v3-capacity-plan/1', basis_peak_point=peak['point'],
                basis='measured complete-suite/model peak on the delivery image; per-core non-tensor heap '
                      'scaled linearly; 25 % headroom; 256 MiB per HBM PC', window_bytes=window_bytes,
                connected_bytes=connected_bytes, per_core_bytes=per_core, cores=rows)


# ------------------------------------------------------------------ gates ---
def gate_suites(inp, ident):
    import ablate
    import native_inventory
    checks = []
    records = {n: inp.records(n) for n in ('native77', 'hbm14', 'large24')}
    for name, recs in records.items():
        wrong = sorted({(r.get('provenance') or {}).get('software_tree') for r in recs} - {ident.tree})
        if wrong:
            raise Missing(f'{name} measured on {wrong}, not the delivery tree {ident.tree}')
    with tempfile.TemporaryDirectory(prefix='r40-suites-') as tmp:
        paths = []
        for tier in ('tiny', 'small', 'medium'):
            p = Path(tmp) / f'u280_{tier}.jsonl'
            p.write_text(''.join(json.dumps(r) + '\n' for r in records['native77'] if r['tier'] == tier))
            paths.append(p)
        inv = native_inventory.audit(paths)
    checks.append(check('native77: complete frozen tiny/small/medium inventory, every point pass, resident, '
                        'native, no fallback, one configuration', all(c['ok'] for c in inv),
                        points=len(inv) - 1, failed=[c['name'] for c in inv if not c['ok']]))
    checks.append(check('native77: every record on the delivery tree, image and clock',
                        all(ident.record(r) for r in records['native77'])))
    for name, subset_file in (('hbm14', 'hbm-core.json'), ('large24', 'large.json')):
        subset, _ = ablate.load_subset(ROOT / 'spec/r40/suites' / subset_file)
        recs = records[name]
        ids = [r['point_id'] for r in recs]
        bad = [r['point_id'] for r in recs
               if not (r.get('status') == 'pass' and r.get('point_sha256') == subset.get(r['point_id'])
                       and not r.get('fallbacks') and (r.get('host') or {}).get('upload_bytes') == 0
                       and (r.get('host') or {}).get('download_bytes') == 0
                       and (r.get('platform') or {}).get('knobs', {}).get('TNCA_TORCH_DEVICE') == 'native'
                       and ((r.get('time') or {}).get('device') or 0) > 0 and r.get('native_memory'))]
        unbound = [r['point_id'] for r in recs if not ident.record(r)]
        configs = {ablate.digest(ablate.config_of(r, [])) for r in recs}
        checks.append(check(f'{name}: complete frozen subset {subset_file} ({len(subset)} points) on the board',
                            len(ids) == len(set(ids)) and set(ids) == set(subset), points=len(ids),
                            missing=sorted(set(subset) - set(ids)), extra=sorted(set(ids) - set(subset))))
        checks.append(check(f'{name}: every point numerically correct (frozen tolerance), native resident, '
                            'zero staging, measured device time', not bad, failed=bad,
                            max_rel_err={r['point_id']: (r.get('correctness') or {}).get('max_rel_err') for r in recs}))
        checks.append(check(f'{name}: every record bound to the delivery identity, one configuration',
                            not unbound and len(configs) == 1, unbound=unbound, configurations=len(configs)))
    fams = {r['family'] for r in records['hbm14']}
    cover = dict(streaming={'add', 'copy', 'silu'}, gemm={'matmul', 'linear', 'gemv'},
                 transpose={'transpose'}, gather={'embedding'}, norm={'rmsnorm', 'softmax'},
                 sdpa={'attention_prefill', 'attention_decode'}, decoder={'llama_prefill', 'llama_decode'})
    checks.append(check('HBM core suite covers streaming, GEMM, transpose, gather, norm, SDPA, decoder',
                        all(f & fams for f in cover.values()),
                        coverage={k: sorted(v & fams) for k, v in cover.items()}))
    return ('capability-pass' if all(c['ok'] for c in checks) else 'fail'), checks


def gate_hbm(inp, ident, ci, write_derived):
    checks = []
    probe = inp.window_samples()
    checks.append(ident.board(probe['identity'], 'window/PC probe'))
    res = {r['name']: r for r in probe['results']}
    win, smp = res.get('window', {}), res.get('samples', {})
    w, s = win.get('detail') or {}, smp.get('detail') or {}
    alloc = w.get('allocation') or {}
    banks = alloc.get('physical_banks') or []
    checks.append(check('declared and actually exposed XRT window >= 1 GiB', win.get('ok') and
                        alloc.get('requested_bytes') == alloc.get('exposed_bytes') == w.get('size') >= GIB,
                        exposed_bytes=alloc.get('exposed_bytes'), bank=w.get('bank')))
    contiguous = (all(b['used'] and b['allocated_size'] == b['size'] and b['tag'] == f"HBM[{b['pc']}]" for b in banks)
                  and sum(b['allocated_size'] for b in banks) == w.get('size'))
    checks.append(check('PC map from the actual XRT BO physical ranges and platform topology, >= 2 PCs',
                        len(banks) >= 2 and contiguous and alloc.get('topology_sha256'),
                        physical_pcs=[b['pc'] for b in banks], topology_sha256=alloc.get('topology_sha256')))
    conn = alloc.get('connected_memory') or {}
    checks.append(check('unexposed HBM marked uncertified/unallocatable; no full 8 GiB claim',
                        (conn.get('size') or 0) >= (w.get('size') or 0) and
                        w.get('heap_end') == hex(int(w.get('local_limit'), 16) + w.get('size')),
                        connected=conn.get('tag'), connected_bytes=conn.get('size'), exposed_bytes=w.get('size'),
                        uncertified=f"HBM[{len(banks)}:31]: outside the {w.get('size')}-byte BO, never allocated, "
                                    'not certified', heap_capacity_bytes=w.get('heap_capacity_bytes'),
                        reserved_virtual_ranges=w.get('reserved_virtual_ranges')))
    crossings = s.get('boundary_crossings') or []
    checks.append(check('every allocated PC device-incremented and host-verified after all were written '
                        '(alias-detecting), every PC boundary straddled, >= 2^32 device address',
                        smp.get('ok') and s.get('configured_pc_coverage') and
                        s.get('physical_pcs') == [b['pc'] for b in banks] and len(crossings) == len(banks) - 1
                        and s.get('high_32bit_virtual_address_tested')
                        and s.get('topology_sha256') == alloc.get('topology_sha256'),
                        samples=s.get('samples'), crossings=len(crossings),
                        last_virtual_address=hex(s.get('last_virtual_address') or 0)))
    checks.append(per_port_probes(s, w))
    cp = inp.checkpoint()
    checks.append(ident.board(cp['identity'], '>=64 MiB checkpoint'))
    outlets = cp.get('outlets') or []
    mc = (cp.get('capabilities') or {}).get('memory_configuration') or {}
    checks.append(check('standard Torch graph with >= 64 MiB inputs correct on HBM with nonzero traffic',
                        cp.get('status') == 'checkpoint-pass' and cp['correctness']['ok'] and
                        cp['warm_correctness']['ok'] and cp.get('input_bytes', 0) >= 64 << 20 and
                        sum(o['read_lines'] for o in outlets) * 64 >= cp.get('input_bytes', 0),
                        point=cp['point']['id'], input_bytes=cp.get('input_bytes'),
                        max_rel_err=cp['correctness'].get('max_rel_err')))
    checks.append(check('every active port carries actual reads and writes (3 of 3 outlets)',
                        len(outlets) == mc.get('physical_ports') == mc.get('active_ports') == 3 and
                        all(o['read_lines'] > 0 and o['write_lines'] > 0 for o in outlets),
                        outlets=outlets, memory_configuration=mc))
    hbm14 = inp.records('hbm14')
    if any((r.get('provenance') or {}).get('software_tree') != ident.tree for r in hbm14):
        raise Missing('HBM14 not on the delivery tree')
    live = {r['point_id']: pcs_of([(b['address'], b['bytes']) for b in r['native_memory'].get('live_buffers', [])],
                                  r['native_memory']['window']) for r in hbm14}
    checks.append(check('at least two actual physical PCs covered by real workload tensors',
                        len(cp.get('tensor_physical_pcs') or []) >= 2 and max(map(len, live.values())) >= 2,
                        checkpoint=cp.get('tensor_physical_pcs'), hbm14_live_pcs=live))
    model = inp.model()
    mi = dict(model['identity'])
    checks.append(ident.board(mi, 'complete frozen model'))
    checks.append(check('complete frozen model (prefill512 + 64 decode) functional on the board',
                        model.get('status') == 'functional-native-model-pass' and len(model.get('steps', [])) == 65
                        and model.get('tokens_match_independent_reference') is True,
                        steps=len(model.get('steps', [])), tokens_match=model.get('tokens_match_independent_reference')))
    rows = []
    for name in ('native77', 'hbm14', 'large24'):
        recs = inp.records(name)
        if any((r.get('provenance') or {}).get('software_tree') != ident.tree for r in recs):
            raise Missing(f'{name} not on the delivery tree')
        rows += [dict(source=name, point=r['point_id'], status=r['status'], **memory_peak(r['native_memory']))
                 for r in recs]
    rows.append(dict(source='checkpoint', point=cp['point']['id'], status=cp['status'], **memory_peak(cp['memory'])))
    rows.append(dict(source='model', point=mi.get('point'), status=model['status'], **memory_peak(model['memory'])))
    peak = max(rows, key=lambda x: x['total_peak_bytes'])
    capacity = min(x['heap_capacity_bytes'] for x in rows)
    window = w.get('size')
    ratio = min(capacity / peak['data_peak_bytes'], window / (peak['total_peak_bytes'] + peak['reserved_bytes']))
    checks.append(check('exposed window >= 1.25 x measured complete-suite/model peak (real allocator: tensors, '
                        'scratch, argument/descriptor tables, KV, code; reserved page excluded from capacity)',
                        ratio >= 1.25 and len(rows) == 77 + 14 + 24 + 2, peak_point=peak['point'],
                        peak_total_bytes=peak['total_peak_bytes'], heap_capacity_bytes=capacity,
                        window_bytes=window, headroom_ratio=round(ratio, 4), rows=len(rows)))
    plan = capacity_plan(peak, window, conn.get('size') or window)
    checks.append(check('8/16/32-core capacity plan from the measured peak (analysis beyond 8 cores)',
                        all(r['fits_exposed_window_with_25pct'] for r in plan['cores'] if r['cores'] <= 8),
                        cores={r['cores']: dict(required_window_bytes=r['required_window_bytes'],
                                                fits_exposed=r['fits_exposed_window_with_25pct'],
                                                fits_connected=r['fits_full_connected_hbm_with_25pct'])
                               for r in plan['cores']}))
    if write_derived:
        d = EVIDENCE / 'R40-04.hbm'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'memory_peaks.json').write_text(json.dumps(dict(
            schema='tnca-r40-v3-memory-peaks/1', delivery_tree=ident.tree, rows=rows, peak=peak,
            heap_capacity_bytes=capacity, window_bytes=window, headroom_ratio=ratio), indent=1) + '\n')
        (d / 'capacity_plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    mem, mem_ok = ci.lane('memory')
    tr, tr_ok = ci.lane('transport')
    checks += ci.identity() + [mem_ok, tr_ok]
    checks.append(markers('out-of-window/unmapped line refused without AXI side effects (line adapter with a '
                          '64-bit 0x100000000 window base)', mem, [(r'LINE AXI PASS width=(64|256) slots=3 mode=3 ', 2)]))
    checks.append(markers('capacity safety: transfers past the end of device memory refused', tr,
                          [(r'a transfer past the end of memory: status 3', 1),
                           (r'conformance\s+PASS\s+1 of 1 backend', 1)]))
    return ('capability-pass' if all(c['ok'] for c in checks) else 'fail'), checks


def gate_robustness(inp, ident, ci, sem_dir, gb10, fidelity):
    d = EVIDENCE / 'R40-04.robustness'
    checks = list(ci.identity())
    mem, mem_ok = ci.lane('memory')
    chk, chk_ok = ci.lane('check')
    reg, reg_ok = ci.lane('mmio-checks')
    checks += [mem_ok, chk_ok, reg_ok]
    checks.append(markers('out-of-order responses and random backpressure: line adapter over all widths/slots/stages',
                          mem, [(r'LINE AXI PASS width=\d+ slots=(3|8) mode=0 ', 16),
                                (r'LINE AXI MATRIX PASS 35 configurations', 1)]))
    checks.append(markers('NUCA slice against reordering/stalling memory; mesh under random backpressure', chk,
                          [(r'BASE-NUCA-ooo-seed\d+\s+PASS', 3), (r'BASE-NUCA-slow-seed\d+\s+PASS', 3),
                           (r'BASE-MESH-2x2-seed\d+\s+PASS', 3)]))
    checks.append(markers('elastic links under random backpressure conserve every packet', reg,
                          [(r'sent=20000 received=20000', 4), (r'MESH PIPELINE PASS', 1)]))
    checks.append(markers('same-line sub-line stores from several cores, masked host writes, partial-line DMA', mem,
                          [(r'GRID REAL CORES PASS N=(2|3|4|8) memory=1 ports=\d phase=1', 6),
                           (r'GRID COHERENT HOST PASS N=\d downloads=457 uploads=2', 6),
                           (r'partial-line preservation, zero/range guards PASS', 2)]))
    checks.append(markers('AXI read/write errors, stray, unmapped and malformed responses latched as faults', mem,
                          [(r'LINE AXI PASS width=(64|256) slots=3 mode=1 ', 2),
                           (r'LINE AXI PASS width=(64|256) slots=3 mode=2 ', 2),
                           (r'LINE AXI PASS width=(64|256) slots=3 mode=3 ', 2),
                           (r'LINE AXI PASS width=(64|256) slots=3 mode=4 ', 2),
                           (r'SINGLE MEMORY PROTOCOL FAULT PASS', 1),
                           (r'MULTICORE SHELL GRID PASS N=\d id=14 status=1 ', 4)]))
    checks.append(markers('timeout (global deadline) and reset with transactions in flight', mem + chk,
                          [(r'GRID DEADLINE PASS', 1), (r'MULTICORE SHELL GRID PASS N=\d id=13 status=2 ', 6),
                           (r'LINE AXI PASS width=(64|256) slots=3 mode=5 ', 2),
                           (r'r40-recovery-tests PASS \(3/3\) multicore_dma=True', 1)]))
    log = need(gb10 / 'r40_hbm.log')
    judged = log.read_text(errors='replace')
    src = read_json(gb10 / 'sources.json')
    checks.append(ident.simulation_relation(src.get('software_tree'), src.get('commit')))
    checks.append(harness_files(src, 'GB10 make r40-hbm / r40-core-parameters'))
    checks.append(markers('make r40-hbm: bridge cases (ooo, backpressure, sub-word, error, reset, DMA), every '
                          'mutation caught, simulated-/hbm device tests', log.read_text(errors='replace'),
                          [(r'PASS  bridge:\S+:(ooo-bp|ooo-slow-bp) ', 20), (r'PASS  bridge:\S+:error ', 10),
                           (r'PASS  bridge:\S+:(reset|reset-error-slow) ', 20), (r'PASS  mutation:', 16),
                           (r'r40-hbm-tests  PASS', 1), (r'^EXIT=0$', 1)]))
    # Every other preserved attempt stays visible: its failing cases are listed.
    others = {}
    for other in sorted(gb10.glob('r40_hbm_*.log')):
        text = other.read_text(errors='replace')
        others[other.name] = dict(sha256=file_sha(other), failed=re.findall(r'^\s*FAIL\s+(\S+)', text, re.M),
                                  completed='r40-hbm-tests' in text)
    if others:
        checks.append(check('earlier make r40-hbm attempts preserved with their failures (a failure is '
                            'acceptable here only when the judged complete run passes the same case)',
                            all(re.search(rf'^\s*PASS\s+{re.escape(f)}\s', judged, re.M)
                                for o in others.values() for f in o['failed']), attempts=others,
                            diagnostics=sorted(str(rel(p)) for p in (gb10 / 'diagnostics').glob('*'))
                            if (gb10 / 'diagnostics').is_dir() else []))
    probe = inp.window_samples()
    checks.append(ident.board(probe['identity'], 'window/PC probe'))
    smp = {r['name']: r for r in probe['results']}.get('samples', {})
    checks.append(check('board: transactions straddling every PC boundary of the window are correct',
                        smp.get('ok') and len((smp.get('detail') or {}).get('boundary_crossings', [])) >= 1,
                        crossings=len((smp.get('detail') or {}).get('boundary_crossings', []))))
    cp = inp.checkpoint()
    checks.append(ident.board(cp['identity'], '>=64 MiB checkpoint'))
    checks.append(check('board: data correct after drained completion and ownership handoff',
                        cp.get('status') == 'checkpoint-pass' and cp['correctness']['ok'], visibility=cp.get('visibility')))
    clock = read_json(d / 'qualified_clock.json')
    checks.append(check('qualified clock evidence is the file every board record is bound to',
                        file_sha(d / 'qualified_clock.json') == CLOCK_EVIDENCE_SHA))
    checks.append(check('timing/CDC gate of the delivery image (v3 change 1: 90 MHz, routed setup/hold met, CDC '
                        'reviewed)', clock.get('image_sha256') == ident.image and clock.get('clock_hz') == ident.clock
                        and clock.get('setup_wns_ns', -1) >= 0 and clock.get('hold_whs_ns', -1) >= 0
                        and clock.get('route_legal') is True and clock.get('cdc_review') == 'passed',
                        setup_wns_ns=clock.get('setup_wns_ns'), hold_whs_ns=clock.get('hold_whs_ns'),
                        cdc_scope=clock.get('cdc_review_scope') or clock.get('scope')))
    rep = read_json(sem_dir / 'journal_replay' / 'report.json')
    src = read_json(sem_dir / 'journal_replay' / 'sources.json')
    checks.append(ident.simulation_relation(src.get('software_tree'), src.get('commit')))
    checks.append(harness_files(src, 'journal capture/replay'))
    checks.append(check('sec 3.4 fixed-input replay: the captured mailbox journal of a resident Torch run on the '
                        'board XO top (NoC, homes, three line-AXI adapters, AXI memory models) replays with '
                        'identical response bytes and completion times', rep.get('status') == 'replay-pass' and
                        not rep.get('diagnostic_override') and 'MMIO_JOURNAL_REPLAY_PASS' in
                        need(sem_dir / 'journal_replay' / 'replay.log').read_text(errors='replace'),
                        backend=(rep.get('capture') or {}).get('backend'),
                        harness_sha256=rep.get('replay_harness_sha256')))
    cal = d / 'hbm_model_calibration.json'
    if cal.exists():
        c = read_json(cal)
        checks.append(check('HBM model calibrated: frozen holdout median error <= 10 %, p95 <= 20 %',
                            c.get('holdout_median_rel_err', 1) <= 0.10 and c.get('holdout_p95_rel_err', 1) <= 0.20))
    elif fidelity == 'uncalibrated-declared':
        checks.append(check('HBM model fidelity: sec 3.4 uncalibrated branch declared (reviewer decision); model '
                            'cycles never substitute for board acceptance', True, decision='uncalibrated-declared',
                            reason='frozen calibration/holdout are large/hbm points (17/19); RTL simulation is '
                                   'infeasible; every v3 timing gate uses board device cycles'))
    else:
        if all(c['ok'] for c in checks):
            return 'not-measured', checks + [check(
                'HBM model calibration against the board (frozen 17 calibration / 19 holdout points)', False,
                status='not-measured', subsets=['spec/r40/suites/R40-04.calibration.json',
                                                'spec/r40/suites/R40-04.model-holdout.json'])]
    return ('capability-pass' if all(c['ok'] for c in checks) else 'fail'), checks


def semantics_results(sem_dir):
    return {name: read_json(sem_dir / f'{name}.json') for name in SEM_CONFIGS}


def semantics_identity(res, ident, commit):
    trees = {r['software_tree'] for r in res.values()}
    if len(trees) != 1:
        return check('one product tree across the semantics matrix', False, trees=sorted(trees))
    return ident.simulation_relation(trees.pop(), commit)


def gate_semantics(ident, ci, sem_dir):
    res = semantics_results(sem_dir)
    src = read_json(sem_dir / 'sources.json')
    checks = [semantics_identity(res, ident, src.get('commit')),
              harness_files(src, 'semantics matrix')]
    for name, r in res.items():
        checks.append(check(f'{name}: {r["backend"]} {json.dumps(r["backend_args"], sort_keys=True)}: both graphs '
                            'correct, resident, one whole-grid launch, every program once, every outlet used',
                            r['ok'] and r['software_tree_unchanged'],
                            graphs={g: v['checks'] for g, v in r['graphs'].items()},
                            peak_active={g: v['physical_grid'].get('peak_active') for g, v in r['graphs'].items()},
                            outlets={g: v['outlet_traffic'] for g, v in r['graphs'].items()}))
    harness = {r['harness_sha256'] for r in res.values()}
    env = {json.dumps({k: v for k, v in r['env'].items() if k not in ('TNCA_BACKEND', 'TNCA_BACKEND_ARGS')},
                      sort_keys=True) for r in res.values()}
    checks.append(check('same model source, harness, product tree and compile options in every configuration; '
                        'only the backend selection differs (no model, ISA or frontend change)',
                        len(harness) == 1 and len(env) == 1 and not any('TNCA_MAX_BLOCK' in e for e in env),
                        harness_sha256=sorted(harness), env=sorted(env)))
    for g in ('pointwise', 'softmax'):
        outs = {r['graphs'][g]['output_sha256'] for r in res.values()}
        arts = {json.dumps(r['graphs'][g]['artifacts']) for r in res.values()}
        checks.append(check(f'{g}: bit-identical outputs across every core count, outlet count and memory model',
                            len(outs) == 1, output_sha256=sorted(outs)))
        checks.append(check(f'{g}: the same compiled kernel artifacts in every configuration (no model, frontend '
                            'or ISA change; only the runtime/backend parameters differ)', len(arts) == 1,
                            artifacts=sorted(arts)))
    cores = {r['backend_args']['core_count'] for r in res.values()}
    ports = {r['backend_args'].get('memory_ports', 1) for r in res.values()}
    models = {r['backend'] for r in res.values()}
    checks.append(check('matrix covers 1/2/3/4/8 cores, several outlet counts and two memory-latency models '
                        '(DRAM model 11+7p cycles with stalls/reordering; AXI line model 25+19p cycles behind '
                        '0- or 2-stage links)', {1, 2, 3, 4, 8} <= cores and len(ports) >= 2 and len(models) >= 2,
                        cores=sorted(cores), ports=sorted(ports), memory_models=sorted(models)))
    checks += ci.identity()
    nat, nat_ok = ci.lane('native')
    mem, mem_ok = ci.lane('memory')
    np_, np_ok = ci.lane('native-ports')
    checks += [nat_ok, mem_ok, np_ok]
    checks.append(markers('CI: the same 4097-element graph on 3/8 cores and 2/3 outlets, with DMA ownership',
                          nat + mem + np_, [(r'NATIVE MULTICORE TORCH PASS cores=3 active=3', 1),
                                            (r'NATIVE MULTICORE TORCH PASS cores=8 active=8', 1),
                                            (r'NATIVE MEMORY PORTS PASS', 4)]))
    ch, ch_ok = ci.lane('chunks')
    checks.append(ch_ok)
    checks.append(markers('CI: role-chunk SPMD variant of the same graph on 3/8 cores', ch,
                          [(r'NATIVE MULTICORE TORCH PASS cores=3', 1), (r'NATIVE MULTICORE TORCH PASS cores=8', 1)]))
    return ('capability-pass' if all(c['ok'] for c in checks) else 'fail'), checks


def gate_dispatch(inp, ident, ci, sem_dir, gb10):
    checks = list(ci.identity())
    chk, chk_ok = ci.lane('check')
    tr, tr_ok = ci.lane('transport')
    nat, nat_ok = ci.lane('native')
    mem, mem_ok = ci.lane('memory')
    regs = [ci.lane(n) for n in ('registers-1', 'registers-3', 'registers-board')]
    checks += [chk_ok, tr_ok, nat_ok, mem_ok] + [ok for _, ok in regs]
    per_n = []
    for n in (1, 2, 3, 4, 8):
        per_n += [(rf'PASS N={n} invalid grid refused', 3),
                  (rf'PASS N={n} grid=19x1x1 issued=19 fault=-1', 1),          # 1-D, ragged
                  (rf'PASS N={n} grid=23x2x1 issued=46 fault=-1', 1),          # 2-D
                  (rf'PASS N={n} grid=13x7x5 issued=455 fault=-1', 1),         # 3-D
                  (rf'PASS N={n} grid=6145x1x1 issued=6145 fault=-1', 2),      # >4096, plain and chunked
                  (rf'PASS N={n} grid=0x3x2 issued=0 fault=-1', 1),            # zero work
                  (rf'PASS N={n} grid=(5x2x3|17x3x2) issued=\d+ fault=-1', 2),  # partial chunk batches
                  (rf'PASS N={n} grid=101x3x2 issued=\d+ fault=-2', 1)]        # global abort, relaunch
        per_n += [(rf'PASS N={n} grid=101x3x2 issued=\d+ fault={f}\b', 1) for f in range(n)]
    checks.append(markers('dispatcher on 1/2/3/4/8 cores: 1-D/2-D/3-D, non-divisible, partial batches, >4096, '
                          'zero work, invalid input, every core fault drains, abort, relaunch', chk,
                          per_n + [(r'GRID DISPATCH PASS', 5)]))
    tb = git('show', f'{ci.run["headSha"]}:tb/tb_grid_dispatch.v')
    checks.append(check('mixed fast/slow programs: each program has its own latency (7 + PID mod 11) under '
                        'random issue backpressure', 'timer[i]=7+(key%11);' in tb and 'ready[i]=((tick+i*3)%7<3);' in tb,
                        testbench='tb/tb_grid_dispatch.v'))
    checks.append(markers('real cores 1/2/3/4/8 without and with memory, two dependent grids', chk,
                          [(rf'GRID REAL CORES PASS N={n} memory=0 ', 1) for n in (1, 2, 3, 4, 8)] +
                          [(rf'GRID REAL CORES PASS N={n} memory=1 ports=1 phase=1', 1) for n in (1, 2, 3, 4, 8)]))
    checks.append(markers('persistent shell 1/3/8 cores: refusal, deadline, completion', chk,
                          [(rf'MULTICORE PERSISTENT SHELL PASS N={n}', 1) for n in (1, 3, 8)] +
                          [(r'GRID DEADLINE PASS', 1), (r'r40-multicore-shell-tests PASS', 1)]))
    checks.append(markers('whole-grid mailbox, >4096 streaming admission, exactly-once atomics', tr,
                          [(r'MULTICORE TRANSPORT PASS programs=455 host_launches=1', 1),
                           (r'PHYSICAL GRID RUNTIME PASS programs=455 exact_atomic_count=455 host_launches=1', 2),
                           (r'physical-grid-option-tests PASS', 1)]))
    checks.append(markers('core fault, then the same session runs again (backend conformance)', tr,
                          [(r'the session still works after a fault', 1)]))
    checks.append(markers('recovery epochs with physical address reuse; protocol fault stops dispatch', mem,
                          [(r'r40-recovery-tests PASS \(3/3\)', 1), (r'SINGLE MEMORY PROTOCOL FAULT PASS', 1)]))
    checks.append(markers('register path on 1 and 3 cores and the board top', '\n'.join(t for t, _ in regs),
                          [(r'MULTICORE MMIO PASS cores=1 ports=1 programs=455', 1),
                           (r'MULTICORE MMIO PASS cores=3 ports=3 programs=455', 2)]))
    ch, ch_ok = ci.lane('chunks')
    checks.append(ch_ok)
    checks.append(markers('bounded role chunks: IR cases, 3-D chunked runtime grid, chunked native Torch on 3/8 '
                          'cores, descriptor and producer/consumer pair kernels', ch,
                          [(r'PHYSICAL CHUNK IR PASS 40 cases', 1),
                           (r'PHYSICAL CHUNK RUNTIME PASS cores=3 width=4 grid=13x7x5 count=455 host_launches=1', 1),
                           (r'NATIVE MULTICORE TORCH PASS cores=3 active=3', 1),
                           (r'NATIVE MULTICORE TORCH PASS cores=8 active=8', 1), (r'NATIVE MULTICORE FEATURES PASS', 2)]))
    checks.append(markers('8-core standard Torch smoke executes (not elaboration-only)', nat,
                          [(r'NATIVE MULTICORE TORCH PASS cores=8 active=8 elements=4097', 1),
                           (r'NATIVE MULTICORE TORCH PASS cores=3 active=3 elements=4097', 1)]))
    res = semantics_results(sem_dir)
    src = read_json(sem_dir / 'sources.json')
    checks.append(semantics_identity(res, ident, src.get('commit')))
    checks.append(check('standard Torch regression on 1/2/3/4/8 cores',
                        all(res[c]['ok'] for c in ('mc1', 'mc2', 'mc3p3', 'mc4', 'mc8')),
                        configs={c: res[c]['ok'] for c in ('mc1', 'mc2', 'mc3p3', 'mc4', 'mc8')}))
    src = read_json(gb10 / 'sources.json')
    checks.append(ident.simulation_relation(src.get('software_tree'), src.get('commit')))
    checks.append(markers('make r40-core-parameters (1/2/3/4/8-core v5 parameter propagation)',
                          need(gb10 / 'r40_core_parameters.log').read_text(),
                          [(r'r40-core-parameter-tests PASS \(6/6\)', 1), (r'^EXIT=0$', 1)]))
    recs = {n: inp.records(n) for n in ('native77', 'hbm14')}
    for n, rs in recs.items():
        if any((r.get('provenance') or {}).get('software_tree') != ident.tree for r in rs):
            raise Missing(f'{n} not on the delivery tree')
    grids = [tuple(e['grid']) for rs in recs.values() for r in rs for e in r['provenance']['launch_map']['launches']]
    on3 = all(r['provenance']['launch_map']['cores'] == [0, 1, 2] and ident.record(r) for rs in recs.values() for r in rs)
    big = [g for g in grids if g[0] * g[1] * g[2] > 4096]
    two_d = [g for g in grids if g[1] > 1 or g[2] > 1]
    ragged = [g for g in grids if (g[0] * g[1] * g[2]) % 3]
    checks.append(check('board: every native77/HBM14 launch on the three physical cores, including 2-D, '
                        '>4096-program and non-divisible grids; all points correct',
                        on3 and big and two_d and ragged and all(r['status'] == 'pass' for rs in recs.values() for r in rs),
                        launches=len(grids), over_4096=len(big), two_d=len(two_d), not_divisible_by_3=len(ragged),
                        largest=max(g[0] * g[1] * g[2] for g in grids)))
    return ('capability-pass' if all(c['ok'] for c in checks) else 'fail'), checks


def gate_nuca(inp, ident, ci):
    checks = list(ci.identity())
    chk, chk_ok = ci.lane('check')
    tr, tr_ok = ci.lane('transport')
    mem, mem_ok = ci.lane('memory')
    reg, reg_ok = ci.lane('mmio-checks')
    np_, np_ok = ci.lane('native-ports')
    checks += [chk_ok, tr_ok, mem_ok, reg_ok, np_ok]
    checks.append(markers('cross-core producer/consumer through shared lines (a second grid reads every word the '
                          'first wrote) on 1/2/3/4/8 cores and 1/2/3/4/8 outlets', mem,
                          [(rf'GRID REAL CORES PASS N={n} memory=1 ports={p} phase=1', 1)
                           for n, p in ((1, 1), (2, 2), (3, 2), (3, 3), (4, 4), (8, 3), (8, 8))]))
    checks.append(markers('same-line masked stores from several cores and masked coherent host writes', mem,
                          [(r'GRID COHERENT HOST PASS N=(2|3|4|8) downloads=457 uploads=2', 5)]))
    checks.append(markers('atomic reduction across three physical cores, exactly once', tr,
                          [(r'exact_atomic_count=455', 2)]))
    checks.append(markers('dirty eviction then coherent host copy; ownership handoff writes back all dirty results; '
                          'partial DMA range preserved', mem,
                          [(r'GLOBAL CACHE HANDOFF PASS N=3 dirty_results=455 partial_dma=1', 2),
                           (r'GRID COHERENT HOST PASS N=\d downloads=457', 7)]))
    checks.append(markers('DMA updates cache: range writeback/invalidate at every home; board-top DMA ownership with a '
                          'resident Torch graph', mem,
                          [(r'NUCA MAINTENANCE PASS', 6), (r'CACHE HANDOFF PASS homes=(1|3|8)', 3),
                           (r'r40-cache-maintenance PASS \(19/19\)', 1), (r'DMA HANDOFF GUARDS PASS \(8/8\)', 1),
                           (r'MMIO CACHE HANDOFF PASS snapshot drain range faults reset', 1),
                           (r'MULTICORE DMA OWNERSHIP PASS cores=3 ports=3 pipeline=2 board=True', 1),
                           (r'"dma_ownership": true', 1)]))
    checks.append(markers('alias and dead/bypass/replacement hints keep every read equal to shadow memory', chk + mem,
                          [(r'DCO-NUCA-policy(1|2|46|62|74)-seed\d+\s+PASS', 10),
                           (r'PRESSURE PASS maintenance=1 policy=74', 2), (r'TMU REGISTERED RANGES PASS ports=(3|8)', 2)]))
    checks.append(markers('saturated reads/writes: late write responses, 8 outlets, 8 outstanding lines', mem,
                          [(r'MULTICORE AXI PERSISTENT PASS N=3 ports=3 stages=2 write_delay=300', 1),
                           (r'MEMORY PORTS PASS N=8 P=8 responses=128', 1), (r'LINE AXI PASS width=\d+ slots=8 mode=0 ', 4),
                           (r'PRESSURE PASS maintenance=\d policy=\d+ seed=\d+ requests=20000', 8)]))
    checks.append(markers('fault drain: AXI faults latched, completion withheld until drain, recovery', mem,
                          [(r'LINE AXI PASS width=(64|256) slots=3 mode=(1|2|4) ', 6),
                           (r'MULTICORE SHELL GRID PASS N=\d id=14 status=1 ', 4), (r'r40-recovery-tests PASS', 1)]))
    checks.append(markers('no deadlock, lost response or starvation: arbitration fairness, exact response counts, '
                          'packet conservation', mem + chk + reg,
                          [(r'MEMORY PORTS PASS N=\d P=\d responses=\d+', 15), (r'R40 MEMORY ARBITER PASS N=\d', 5),
                           (r'LINE AXI MATRIX PASS 35 configurations', 1), (r'sent=20000 received=20000', 4)]))
    checks.append(markers('RTL: every outlet of a native Torch graph carries reads and writes', mem + np_,
                          [(r'NATIVE MEMORY PORTS PASS \{"controller_traffic"', 4)]))
    cp = inp.checkpoint()
    checks.append(ident.board(cp['identity'], 'board checkpoint'))
    outlets = cp.get('outlets') or []
    checks.append(check('board: every active port and the PC group carry actual traffic on the delivery image',
                        len(outlets) == 3 and all(o['read_lines'] > 0 and o['write_lines'] > 0 for o in outlets)
                        and cp['cores']['peak_active'] == 3 and len(cp.get('tensor_physical_pcs') or []) >= 2,
                        outlets=outlets, pc_group=cp['window']['allocation']['connected_memory'].get('tag'),
                        tensor_pcs=cp.get('tensor_physical_pcs')))
    return ('capability-pass' if all(c['ok'] for c in checks) else 'fail'), checks


def judge(fn, *args):
    try:
        status, checks = fn(*args)
    except Missing as e:
        return 'not-measured', [check('complete evidence for the delivery identity', False, why=str(e))]
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError, subprocess.CalledProcessError,
            tarfile.TarError) as e:
        return 'fail', [check('evidence well formed and consistent', False, why=f'{type(e).__name__}: {e}')]
    return status, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--board-run', type=Path)
    ap.add_argument('--window-samples', type=Path)
    ap.add_argument('--model-report', type=Path)
    ap.add_argument('--large24', type=Path)
    ap.add_argument('--ci-dir', type=Path, default=EVIDENCE / 'ci_39183153944')
    ap.add_argument('--semantics-dir', type=Path, default=EVIDENCE / 'R40-06.semantics')
    ap.add_argument('--gb10-dir', type=Path, default=EVIDENCE / 'R40-04.robustness' / 'gb10',
                    help='r40_hbm.log, r40_core_parameters.log and sources.json of the GB10 runs')
    ap.add_argument('--preserve', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='print the records instead of writing them')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--hbm-model-fidelity', choices=('strict', 'uncalibrated-declared'), default='strict')
    args = ap.parse_args()
    if not args.dry_run and os.environ.get('R40_CONTRACT') != 'v3':
        ap.error('set R40_CONTRACT=v3 (or --dry-run)')
    if args.preserve and args.dry_run:
        ap.error('--preserve writes evidence; not with --dry-run')
    ident = Identity()
    inp = Inputs(args)
    ci = CI(args.ci_dir, ident)
    gates = {
        'R40-09.suites': (gate_suites, (inp, ident)),
        'R40-04.hbm': (gate_hbm, (inp, ident, ci, not args.dry_run)),
        'R40-04.robustness': (gate_robustness, (inp, ident, ci, args.semantics_dir, args.gb10_dir,
                                                args.hbm_model_fidelity)),
        'R40-05.dispatch': (gate_dispatch, (inp, ident, ci, args.semantics_dir, args.gb10_dir)),
        'R40-06.nuca': (gate_nuca, (inp, ident, ci)),
        'R40-06.semantics': (gate_semantics, (ident, ci, args.semantics_dir)),
    }
    me = file_sha(Path(__file__))
    for gate, (fn, fargs) in gates.items():
        if args.only and gate not in args.only:
            continue
        status, checks = judge(fn, *fargs)
        inputs = {}
        for base in (args.ci_dir, args.semantics_dir, args.gb10_dir, EVIDENCE / gate):
            if base and Path(base).is_dir():
                inputs.update({rel(p): file_sha(p) for p in sorted(Path(base).rglob('*')) if p.is_file()})
        for given in (args.board_run, args.window_samples, args.model_report, args.large24):
            if given and Path(given).is_file():
                inputs[str(given)] = file_sha(given)
        extra = dict(auditor=rel(Path(__file__)), auditor_sha256=me, delivery_tree=ident.tree,
                     image_sha256=ident.image, clock_hz=ident.clock, inputs=inputs,
                     board_run=str(args.board_run) if args.board_run else None)
        if gate == 'R40-04.robustness':
            extra['hbm_model_fidelity'] = args.hbm_model_fidelity
        detail = f'v3 memory-system capability audit: {sum(c["ok"] for c in checks)}/{len(checks)} checks pass'
        if args.dry_run:
            print(json.dumps(dict(gate=gate, status=status, detail=detail,
                                  failed=[c for c in checks if not c['ok']],
                                  passed=[c['name'] for c in checks if c['ok']]), indent=1, default=str)[:20000])
        else:
            import evidence
            evidence.record(gate, status, checks, detail=detail, **extra)
        print(f'{gate}: {status}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
