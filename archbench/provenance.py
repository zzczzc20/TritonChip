"""Provenance of an ArchBench record (archbench/PROTOCOL.md, protocol 2).

doc/r40_tasks_and_acceptance.md sec 3.3: every performance point can be
traced back through the chain that produced it -- the PyTorch graph, the
Inductor wrapper and its Triton source, the TTIR, the TNCA IR and assembly,
the artifact, the profile and capabilities, the software tree, the image and
clock, the launch map and the raw samples. This module names the software by
content and writes the per-point bundle the record points to.

software_tree() is a digest of the files that decide what is compiled and
how it runs, computed the same way in a git checkout and in a copy without
.git (the board host runs from an rsync'd tree), so two runs of the same
sources carry the same name wherever they ran.
"""
import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What decides an artifact or a measurement. Staged data (python/tnca/data,
# python/_staged) is generated from these by `make r2-stage-data`.
TREE_DIRS = ('archbench', 'python', 'compiler', 'tools/asm', 'spec', 'rtl', 'tb/device')
TREE_SKIP_DIRS = {'__pycache__', '.pytest_cache'}
# Generated, ignored by git: staged copies of spec/ and the toolchain data.
TREE_SKIP_PATHS = {Path('python/_staged'), Path('python/tnca/data')}
TREE_SKIP_SUFFIX = ('.pyc', '.pyo', '.egg-info')


def _tree_files(root):
    for d in TREE_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            rel = Path(dirpath).relative_to(root)
            dirnames[:] = sorted(n for n in dirnames
                                 if n not in TREE_SKIP_DIRS and not n.endswith('.egg-info')
                                 and rel / n not in TREE_SKIP_PATHS)
            for f in sorted(filenames):
                if f.endswith(TREE_SKIP_SUFFIX):
                    continue
                yield Path(dirpath) / f


_tree_cache = {}


def software_tree(root=ROOT):
    """`files:<sha256>` over (path, content sha256) of TREE_DIRS."""
    root = Path(root)
    if root in _tree_cache:
        return _tree_cache[root]
    h = hashlib.sha256()
    for f in _tree_files(root):
        h.update(str(f.relative_to(root)).encode() + b'\0')
        h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode() + b'\n')
    _tree_cache[root] = 'files:' + h.hexdigest()
    return _tree_cache[root]


def git_commit(root=ROOT):
    """The checkout's commit and whether it has local changes, or None
    outside git (the digest above is the name that counts)."""
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
        return dict(commit=head, dirty=bool(dirty))
    except Exception:                                        # noqa: BLE001
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class GraphCapture:
    """Dynamo's graph code, captured while open: the PyTorch graph the
    compiler was given, as `TORCH_LOGS=graph_code` would print it."""

    def __init__(self):
        self.text = []

    def __enter__(self):
        import torch
        self._handler = logging.Handler()
        self._handler.emit = lambda r: self.text.append(r.getMessage())
        # torch._logging.getArtifactLogger('torch._dynamo.output_graph', 'graph_code')
        self._logger = logging.getLogger('torch._dynamo.output_graph.__graph_code')
        try:
            torch._logging.set_logs(graph_code=True)
        except Exception:                                    # noqa: BLE001
            pass
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        import torch
        self._logger.removeHandler(self._handler)
        try:
            torch._logging.set_logs()
        except Exception:                                    # noqa: BLE001
            pass
        return False


def write_bundle(bundle_dir, *, point_id, graph_text, wrapper_code, launches, dump_dir):
    """Write a point's provenance bundle and return the record's view of it:
    the directory and every file with its sha256. `launches` are launch-log
    entries (their artifact paths are copied in); `dump_dir` is where the
    TNCA kernel dump (TTIR, TNCA IR, assembly) was written."""
    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'graph.txt').write_text('\n\n'.join(graph_text) + '\n')
    for i, src in enumerate(wrapper_code):
        (d / f'wrapper_{i}.py').write_text(src)
    art = d / 'artifacts'
    art.mkdir(exist_ok=True)
    for e in launches:
        p, h = e.get('artifact_path'), e.get('artifact')
        if p and h and os.path.exists(p) and not (art / f'{h}.tnca').exists():
            shutil.copyfile(p, art / f'{h}.tnca')
    if dump_dir and Path(dump_dir).is_dir() and Path(dump_dir).resolve() != (d / 'kernels').resolve():
        shutil.copytree(dump_dir, d / 'kernels', dirs_exist_ok=True)
    files = {}
    for f in sorted(d.rglob('*')):
        if f.is_file() and f.name != 'bundle.json':
            files[str(f.relative_to(d))] = sha256_file(f)
    doc = dict(schema='archbench-bundle/1', point_id=point_id, files=files)
    (d / 'bundle.json').write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    return dict(dir=str(d), files=files)


def check_bundle(bundle, root=None):
    """Problems with a record's bundle: a file missing or changed, or an
    artifact the launch map names that is not in it."""
    if not bundle or not bundle.get('dir'):
        return ['no bundle']
    d = Path(bundle['dir'])
    if root is not None and not d.is_absolute():
        d = Path(root) / d
    if not d.is_dir():
        return [f'bundle directory {d} does not exist']
    bad = []
    for rel, h in (bundle.get('files') or {}).items():
        f = d / rel
        if not f.is_file():
            bad.append(f'{rel}: missing')
        elif sha256_file(f) != h:
            bad.append(f'{rel}: changed since the record was written')
    return bad
