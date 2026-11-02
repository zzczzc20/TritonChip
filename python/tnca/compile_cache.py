"""Namespace persistent Inductor graphs by TNCA compilation semantics.

The upstream FX cache keys each graph, input signature and guards.  Its key
cannot see TNCA's registered decompositions, so the local cache directory is
additionally partitioned by their source, toolchain versions and target knobs.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform

# Execution-only settings cannot alter a compiled graph and often contain
# per-run paths. Keep every other TNCA setting, including unknown future knobs.
_RUNTIME_ONLY = frozenset({'TNCA_COMPILE_CACHE_DIR', 'TNCA_TLM_OUT', 'TNCA_DUMP_DIR',
                           'TNCA_SUBMIT', 'TNCA_QUEUE_DEPTH', 'TNCA_MMIO_DESCRIPTOR_CACHE', 'TNCA_CODE_REUSE', 'TNCA_DEVICE_ZERO', 'TNCA_QUEUE_STEPS',
                           'TNCA_REUSE_WARM_PREFILL', 'TNCA_DEFER_CACHE_RESET',
                           'TNCA_REUSE_HYBRID_PREFILL', 'TNCA_REUSE_FIRST_DECODE', 'TNCA_DEVICE_EOS_GUARD',
                           'TNCA_DEADLINE', 'TNCA_HOST_TIMEOUT'})
# These select a runtime device or its measured clock. The ordinary compiler
# does not use them, but autotuning observes the device and therefore must
# retain both in its cache identity.
_UNTUNED_RUNTIME_ONLY = frozenset({'TNCA_BACKEND_ARGS', 'TNCA_OPERATING_CLOCK_EVIDENCE', 'TNCA_COUNTERS',
                                  'TNCA_DEVICE_EOS_EARLY_STOP', 'TNCA_HYBRID_PREFIX_CHECKPOINT',
                                  'TNCA_REUSE_GUARDED_HYBRID', 'TNCA_STATIC_PREFIX_REWIND'})
_SOURCE_SUFFIXES = frozenset({'.py', '.c', '.cpp', '.cc', '.h', '.hpp', '.td',
                              '.def', '.inc', '.cmake', '.json', '.toml', '.yaml', '.yml'})
_UNTUNED_RUNTIME_FILES = frozenset({'runtime.py', 'transport.py'})
# Cache-key bookkeeping does not change an untuned generated graph. Preserve
# the previous bookkeeping file's digest so this fix can use already-built
# artifacts immediately. Bump only if this module starts changing codegen.
_UNTUNED_CACHE_KEY_IMPLEMENTATION = bytes.fromhex(
    '59172d7585c1c40304c4f515c13abb89b1066569dd9faf9876788bbecac8eb46')


def semantic_environment(environ):
    runtime_only = _RUNTIME_ONLY
    if environ.get('TNCA_TUNE', '0') != '1':
        runtime_only = runtime_only | _UNTUNED_RUNTIME_ONLY
    return {key: value for key, value in sorted(environ.items())
            if key.startswith('TNCA_') and key not in runtime_only}


def source_fingerprint(trees, files=None, *, include_runtime=True):
    """Hash content and relative names, independently of checkout location."""
    digest = hashlib.sha256()
    for label, root in sorted(trees.items()):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f'compile cache source tree missing: {root}')
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            # Tests cannot change compiler semantics. Including them makes a
            # new test or a test-only edit invalidate every persistent graph,
            # forcing minutes of avoidable recompilation for a user prompt.
            test_file = (label in ('tnca', 'tnca_cc') and
                         (path.name.startswith('test_') or path.name.endswith('_test.py')
                          or 'tests' in relative.parts))
            # Untuned FX graphs do not embed the launch queue or the board's
            # register transport. Hashing them forced several minutes of
            # recompilation after a runtime-only deployment. Autotuning does
            # depend on measured runtime behavior, so it retains these files.
            runtime_file = (label == 'tnca' and not include_runtime and
                            (relative.as_posix() in _UNTUNED_RUNTIME_FILES or
                             (relative.parts and relative.parts[0] == 'backends')))
            if (path.is_file() and path.suffix in _SOURCE_SUFFIXES
                    and '__pycache__' not in path.parts and not test_file and not runtime_file):
                name = f'{label}/{path.relative_to(root).as_posix()}'
                digest.update(name.encode() + b'\0')
                if label == 'tnca' and relative.as_posix() == 'compile_cache.py' and not include_runtime:
                    digest.update(_UNTUNED_CACHE_KEY_IMPLEMENTATION)
                else:
                    digest.update(hashlib.sha256(path.read_bytes()).digest())
    for label, path in sorted((files or {}).items()):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'compile cache dependency missing: {path}')
        digest.update(label.encode() + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def namespace(source_hash, environ, versions, options):
    value = dict(schema='tnca-fx-cache/1', source=source_hash,
                 environment=semantic_environment(environ), versions=versions,
                 options=options)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dependency_paths(package, compiler_package, data_path, assembler_files, extra_data_roots=()):
    """Use the data actually selected by TNCA, including wheel-staged data."""
    package = Path(package)
    root = package.parent.parent
    trees = {'tnca': package, 'tnca_cc': Path(compiler_package),
             'targets': Path(data_path('compiler', 'targets')),
             'isa': Path(data_path('isa')), 'spec': Path(data_path('spec'))}
    files = {'assembler': Path(assembler_files[0]), 'assembler_isa': Path(assembler_files[1]),
             'abi_override': Path(data_path('spec', 'r2_abi_v1.json', env='TNCA_ABI_SPEC')),
             'manifest_override': Path(data_path('spec', 'r2_manifest_v1.schema.json',
                                                 env='TNCA_MANIFEST_SCHEMA'))}
    # The compiler can select TNCA_DATA_DIR independently of the runtime's
    # staged data; it can also fall back file-by-file to checkout sources.
    for index, data_root in enumerate((root, *extra_data_roots)):
        for label, relative in (('spec', 'spec'), ('targets', 'compiler/targets'), ('isa', 'isa')):
            candidate = Path(data_root) / relative
            if candidate.is_dir():
                trees[f'extra_{index}_{label}'] = candidate
    # The Python compiler is shipped in wheels; these source trees only
    # exist in a full checkout. Hash them as well when present.
    for label, relative in (('compiler_include', 'compiler/include'),
                            ('compiler_lib', 'compiler/lib'), ('compiler_tools', 'compiler/tools'),
                            ('compiler_pin', 'compiler/pin')):
        if (root / relative).is_dir():
            trees[label] = root / relative
    if (root / 'compiler/CMakeLists.txt').is_file():
        files['compiler_cmake'] = root / 'compiler/CMakeLists.txt'
    return trees, files


def configure_persistent_cache(options):
    """Select a private persistent namespace and return its path and identity."""
    import torch
    import triton
    import tnca_cc
    from . import _data
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    assembler_files = []
    for name in ('tnca_asm', 'tnca_isa'):
        module = importlib.util.find_spec(name)
        candidate = Path(module.origin) if module is not None and module.origin else root / 'tools/asm' / f'{name}.py'
        assembler_files.append(candidate)
    override = os.environ.get('TNCA_DATA_DIR')
    trees, files = dependency_paths(package, Path(tnca_cc.__file__).resolve().parent,
                                    _data.path, assembler_files, (override,) if override else ())
    source = source_fingerprint(trees, files, include_runtime=os.environ.get('TNCA_TUNE', '0') == '1')
    versions = dict(python=platform.python_version(), torch=torch.__version__,
                    triton=triton.__version__, machine=platform.machine())
    identity = namespace(source, os.environ, versions, options)
    base = Path(os.environ.get('TNCA_COMPILE_CACHE_DIR', Path.home() / '.cache/tnca/inductor')).expanduser()
    path = base.resolve() / identity
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(path)
    return dict(enabled=True, path=str(path), identity=identity, source=source)
