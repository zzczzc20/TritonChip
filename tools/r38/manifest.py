"""Candidate manifests and measurement records (S1-03, roadmap sections 3.1, 13.2).

A candidate is V = (C, I, H, R, W, M, P, E): compiler, ISA/ABI contract,
hardware, runtime, workload, cost model, profiler schema, environment.
Each component is a content hash over the files that define it -- not a Git
SHA, because an uncommitted patch changes a result as surely as a commit
does -- and the candidate hash is the hash of the eight.

Paths are hashed relative to the tree root, so the same content in two
directories gives the same candidate.
"""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / 'spec' / 'r38' / 'measurement.schema.json').read_text())

# Which files define each layer. Generated and cached output is excluded.
COMPONENTS = {
    'C': ['compiler/python/**/*.py', 'compiler/backend/**/*.py', 'compiler/lib/**/*.cpp',
          'compiler/include/**/*.td', 'compiler/include/**/*.h', 'python/tnca/aot.py'],
    'I': ['isa/*.json', 'compiler/targets/*.json', 'spec/r2_abi_v1.json', 'spec/r1_profiles.json'],
    'H': ['rtl/**/*.v', 'rtl/**/*.vh', 'tb/device/*.v', 'tb/common/*.v'],
    'R': ['python/tnca/*.py'],
    'W': ['spec/r38/suites/*.json', 'tools/r37/upstream_cases.py', 'tools/r38/archbench/r37_workloads.py'],
    'M': ['tools/r38/costmodel/**/*.py'],
    'P': ['spec/r38/counters.json', 'spec/r3_counters.json'],
}
EXCLUDE = re.compile(r'(__pycache__|/build/|\.egg-info|/data/)')


def files_of(root, globs, exclude_aot_from_runtime=False):
    root = Path(root)
    out = set()
    for g in globs:
        for p in root.glob(g):
            rel = p.relative_to(root).as_posix()
            if p.is_file() and not EXCLUDE.search('/' + rel):
                out.add(rel)
    return sorted(out)


def tree_hash(root, rels):
    h = hashlib.sha256()
    for rel in rels:
        h.update(rel.encode() + b'\0')
        h.update(hashlib.sha256((Path(root) / rel).read_bytes()).digest())
    return h.hexdigest()


def environment():
    def first(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (r.stdout or r.stderr).strip().splitlines()[0]
        except (OSError, IndexError, subprocess.TimeoutExpired):
            return None
    try:
        import triton
        tv = triton.__version__
    except ImportError:
        tv = None
    return {'iverilog': first(['iverilog', '-V']), 'python': platform.python_version(), 'triton': tv}


def candidate(root=ROOT, *, env=None, extra_workload=None, hardware_root=None):
    """The eight component hashes and the candidate hash.

    `hardware_root` measures a candidate whose rtl/ and tb/ live elsewhere
    (a mutation lab, a feature branch's tree) with this tree's other layers.
    """
    root = Path(root)
    comp = {}
    for key, globs in COMPONENTS.items():
        base = Path(hardware_root) if key == 'H' and hardware_root else root
        rels = files_of(base, globs)
        if key == 'H':
            comp[key] = tree_hash(base, rels)
            continue
        if key == 'R':
            rels = [r for r in rels if r != 'python/tnca/aot.py']      # it is in C
        comp[key] = tree_hash(root, rels)
    if extra_workload:
        comp['W'] = hashlib.sha256((comp['W'] + json.dumps(extra_workload, sort_keys=True)).encode()).hexdigest()
    env = environment() if env is None else env
    comp['E'] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
    whole = hashlib.sha256(json.dumps(comp, sort_keys=True).encode()).hexdigest()
    return {'schema': 'r38-candidate/1', 'components': comp, 'candidate_hash': whole, 'environment': env}


# ---------------------------------------------------------------------------
# measurement records
# ---------------------------------------------------------------------------

HEX64 = re.compile(r'^[0-9a-f]{64}$')
TYPES = {'str': str, 'int': int, 'float': (int, float), 'bool': bool, 'object': dict, 'list': list}


def _get(rec, dotted):
    cur = rec
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return KeyError
        cur = cur[part]
    return cur


def validate(rec, *, stage=1):
    """Every problem with one record; empty means valid."""
    bad = []
    for name, f in SCHEMA['fields'].items():
        v = _get(rec, name)
        if v is KeyError:
            bad.append(f'{name}: missing')
            continue
        if v is None:
            if not f.get('nullable'):
                bad.append(f'{name}: null is not allowed')
            continue
        if f.get('stage1_null') and stage == 1:
            bad.append(f'{name}: must be null in Stage 1 (no board to measure it)')
        t = f['type']
        if t == 'hash':
            if not (isinstance(v, str) and HEX64.match(v)):
                bad.append(f'{name}: not a sha256')
        elif isinstance(v, bool) and t != 'bool' or not isinstance(v, TYPES[t]):
            bad.append(f'{name}: expected {t}, got {type(v).__name__}')
            continue
        if 'enum' in f and v not in f['enum']:
            bad.append(f'{name}: {v!r} not in {f["enum"]}')
        if f.get('positive_or_null') and isinstance(v, (int, float)) and v <= 0:
            bad.append(f'{name}: {v} is not a measurement; an unknown is null, never zero')
    status = _get(rec, 'status')
    cyc = _get(rec, 'time.device_cycles')
    if status == 'pass' and _get(rec, 'evidence_level') == 'measured-rtl' and cyc in (None, KeyError):
        bad.append('status pass without device cycles')
    eta = _get(rec, 'bounds.eta')
    if eta not in (None, KeyError):
        L = _get(rec, 'bounds.strict_cycles')
        if eta > 1:
            bad.append(f'bounds.eta {eta} > 1: audit, never clip')
        if isinstance(L, (int, float)) and isinstance(cyc, int) and cyc > 0 and abs(eta - L / cyc) > 1e-9:
            bad.append('bounds.eta is not strict_cycles / device_cycles')
    return bad


def compatible(records, *, bound_version, timing):
    """What a scorer may accept: one bound version, one timing protocol."""
    out = []
    for r in records:
        if r['bounds']['version'] != bound_version:
            raise ValueError(f'{r["workload_id"]}: bound version {r["bounds"]["version"]} != {bound_version}')
        if r['time']['timing'] != timing:
            raise ValueError(f'{r["workload_id"]}: timing {r["time"]["timing"]} != {timing}')
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# design changes
# ---------------------------------------------------------------------------

DC_SCHEMA = json.loads((ROOT / 'spec' / 'r38' / 'design_change.schema.json').read_text())


def validate_design_change(rec, root=ROOT):
    bad = [f'{k}: missing' for k in DC_SCHEMA['required'] if k not in rec]
    if rec.get('schema') != 'r38-design-change/1':
        bad.append('schema is not r38-design-change/1')
    for layer in rec.get('touched_layers', []):
        if layer not in DC_SCHEMA['enums']['touched_layers']:
            bad.append(f'touched_layers: {layer!r} unknown')
    if len(set(rec.get('touched_layers', [])) & {'compiler', 'isa', 'microarch'}) < 3 \
            and rec.get('claims_three_layer'):
        bad.append('claims a three-layer change without touching compiler, isa and microarch')
    if rec.get('status') not in DC_SCHEMA['enums']['status']:
        bad.append(f'status {rec.get("status")!r} unknown')
    for p in rec.get('evidence', []):
        path = p.split(' ')[0]
        if not (Path(root) / path).exists():
            bad.append(f'evidence {path} does not exist')
    for layer, files in rec.get('layers', {}).items():
        for f in files.get('files', []):
            if not (Path(root) / f).exists():
                bad.append(f'layers.{layer}: {f} does not exist')
    return bad
