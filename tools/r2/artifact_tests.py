#!/usr/bin/env python3
"""
ART-02's schema half -- doc/phase2_tasks_and_acceptance.md R2-05.

    "code/constant 改一字节、截断、段重叠、entry 越界、重复 JSON key、
     未知版本/必需字段各拒绝"

The cases that need a device -- loading, executing, checking against real
readback -- belong to a later batch. These are the ones that are decided by
the manifest alone, and they are worth having now because the schema is the
contract every later batch is written against: a validator that accepts a
manifest with a missing `capabilities` is a loader that runs code on a machine
nobody checked.

Every negative here has a positive twin. A validator that rejects everything
passes a list of negatives and is useless, and the twin is what says the
refusal was about the thing being tested.
"""
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                    # noqa: E402

from tnca import manifest as M                       # noqa: E402
from tnca.errors import (ArtifactError, TncaError,        # noqa: E402
                         VersionError)


def good_manifest():
    """A complete, valid v1 manifest.

    Written out in full rather than generated, so that if the schema grows a
    required field this fixture stops being valid and somebody has to decide
    what the field means here.
    """
    return M.seal({
        'identity': {
            'format_version': list(M.FORMAT_VERSION),
            'abi_version': [1, 0],
            'isa_version': '0.1',
            'kernel_name': 'add_kernel',
            'source_kind': 'triton',
            'compiler_hash': 'c' * 64,
            'source_hash': 's' * 64,
            'toolchain_lock_hash': 't' * 64,
            'constexprs': {'BLOCK': 16},
            'options': {'opt_level': 1},
        },
        'target': {
            'profile': 'core-sim',
            'config_hash': 'f' * 64,
            'implementation_fingerprint': 'i' * 64,
            'capabilities': ['tensor_dma', 'mbarrier'],
        },
        'code': {
            'segments': [
                {'name': 'text', 'offset': 0, 'length': 256,
                 'sha256': 'a' * 64, 'alignment': 8}],
            'entry_points': {'role0': 0},
            'constant_segments': [],
        },
        'parameters': {
            'signature': [
                {'name': 'x', 'type': 'device_ptr64', 'index': 0,
                 'offset': 0, 'size': 8, 'alignment': 8, 'direction': 'in'},
                {'name': 'n', 'type': 'i32', 'index': 1,
                 'offset': 8, 'size': 4, 'alignment': 8, 'direction': 'in'}],
            'buffer_extents': {'x': 'n * 4'},
            'alias_allowed': [],
        },
        'execution': {
            'role_mask': 1,
            'program_to_role': {'0': ['role0']},
            'grid_dims_supported': [True, False, False],
            'dynamic_constraints': {},
        },
        'resources': {
            'rf_private_per_role': {'role0': 16},
            'shared_windows': [],
            'accumulators': 1,
            'barriers': 2,
            'descriptors': 2,
            'commands_outstanding': 4,
            'scratch_bytes': 0,
        },
        'semantics': {
            'completion_scope': 'device',
            'error_contract': 'outputs invalid on failure; no rollback',
            'cross_program_dependency': False,
            'ownership_check_version': 'r1/1',
            'ownership_check_result': 'clean',
        },
    })


def refuses(log, label, mutate, *, expect=ArtifactError):
    """Apply `mutate` to a good manifest and require a refusal."""
    m = mutate(copy.deepcopy(good_manifest()))
    try:
        M.require(m)
    except expect as e:
        log.append(f"    {label:<34s} -> {e.message[:78]}")
        return True
    except Exception as e:
        log.append(f"    {label:<34s} -> {type(e).__name__} (wrong kind): {e}")
        return False
    log.append(f"    {label:<34s} -> ACCEPTED (bad)")
    return False


def art02_schema(log):
    """The manifest half of ART-02, plus the twin that must still pass."""
    ok = True

    # The twin. If this fails, every refusal below means nothing.
    try:
        M.require(good_manifest())
        log.append("    a complete manifest                -> accepted")
    except Exception as e:
        log.append(f"    a complete manifest                -> REFUSED: {e}")
        return False, 'the valid fixture is not valid'

    def drop_field(section, field):
        def f(m):
            m[section].pop(field)
            return M.seal(m)
        return f

    ok &= refuses(log, 'no target.capabilities',
                  drop_field('target', 'capabilities'))
    ok &= refuses(log, 'no parameters.alias_allowed',
                  drop_field('parameters', 'alias_allowed'))
    ok &= refuses(log, 'no semantics.error_contract',
                  drop_field('semantics', 'error_contract'))
    ok &= refuses(log, 'a required field set to null',
                  lambda m: M.seal({**m, 'target': {**m['target'],
                                                    'config_hash': None}}))
    # A typo is the realistic case, and the one an "ignore unknown keys"
    # validator turns into an artifact with no capability requirements.
    ok &= refuses(log, 'capabilties (a typo) instead',
                  lambda m: M.seal({**m, 'target': {
                      **{k: v for k, v in m['target'].items()
                         if k != 'capabilities'},
                      'capabilties': ['tensor_dma']}}))
    ok &= refuses(log, 'an unknown top-level section',
                  lambda m: M.seal({**m, 'extras': {'x': 1}}))
    ok &= refuses(log, 'a higher major',
                  lambda m: M.seal({**m, 'identity': {
                      **m['identity'], 'format_version': [2, 0]}}),
                  expect=VersionError)
    ok &= refuses(log, 'a higher minor',
                  lambda m: M.seal({**m, 'identity': {
                      **m['identity'],
                      'format_version': [M.FORMAT_VERSION[0],
                                         M.FORMAT_VERSION[1] + 1]}}),
                  expect=VersionError)

    # optional_metadata is the one place an extension may live, and it must
    # NOT be refused -- otherwise the escape hatch the contract provides does
    # not exist.
    m = copy.deepcopy(good_manifest())
    m['optional_metadata'] = {'built_by': 'whatever', 'nested': {'a': [1, 2]}}
    try:
        M.require(M.seal(m))
        log.append("    optional_metadata, free-form       -> accepted")
    except Exception as e:
        log.append(f"    optional_metadata                  -> REFUSED: {e}")
        ok = False

    return ok, 'missing, unknown, null, typo and version, each refused'


def art02_duplicate_key(log):
    """A duplicate JSON key, which json.load resolves in silence."""
    text = ('{"identity":{"kernel_name":"a","kernel_name":"b"},'
            '"target":{}}')
    plain = json.loads(text)
    log.append(f"    json.loads keeps {plain['identity']['kernel_name']!r} "
               f"and never mentions the other")
    try:
        M.loads(text)
        log.append("    the manifest reader accepted it (bad)")
        return False, 'a duplicate key was resolved silently'
    except ArtifactError as e:
        log.append(f"    the manifest reader              -> {e.message}")
    return True, 'a duplicate key is two answers to one question'


def art02_integrity(log):
    """One changed byte of the manifest, after it was sealed."""
    ok = True
    m = good_manifest()
    try:
        M.check_integrity(m)
        log.append(f"    a sealed manifest verifies         -> "
                   f"{m['manifest_hash'][:16]}")
    except Exception as e:
        log.append(f"    a sealed manifest                  -> REFUSED: {e}")
        ok = False

    tampered = copy.deepcopy(m)
    tampered['resources']['barriers'] = 3     # one field, one digit
    try:
        M.check_integrity(tampered)
        log.append("    barriers 2 -> 3 verified anyway (bad)")
        ok = False
    except ArtifactError as e:
        log.append(f"    barriers 2 -> 3                    -> {e.message}")

    # The hash is over the canonical form, so reordering the keys must NOT
    # change it: two environments that build the same artifact have to agree,
    # and dict order is a Python detail.
    shuffled = {k: m[k] for k in reversed(list(m))}
    if M.manifest_hash(shuffled) != M.manifest_hash(m):
        log.append("    key order changed the hash (bad)")
        ok = False
    else:
        log.append("    key order does not change the hash -> "
                   "two environments can agree")
    return ok, 'one changed field caught; key order irrelevant'


def _write_good(tmp, name='k.tnca', segments=None):
    from tnca import artifact as A
    path = os.path.join(tmp, name)
    A.write(path, good_manifest(),
            segments or {'text': bytes(range(64)), 'constants': b'\xaa' * 32})
    return path


def art02_container(log):
    """One changed byte, a truncation, an overlap, a trailing byte."""
    import tempfile
    from tnca import artifact as A
    tmp = tempfile.mkdtemp(prefix='r2-art-')
    ok = True

    path = _write_good(tmp)
    man, segs = A.load(path)
    log.append(f"    a good artifact loads: {len(os.path.getsize(path) * [0])}"
               f" bytes, segments "
               f"{ {k: len(v) for k, v in segs.items()} }")

    def corrupt(label, mutate, expect_text=None):
        nonlocal ok
        p2 = os.path.join(tmp, f"c-{abs(hash(label)) % 10 ** 8}.tnca")
        with open(path, 'rb') as fh:
            data = bytearray(fh.read())
        data = mutate(data)
        with open(p2, 'wb') as fh:
            fh.write(bytes(data))
        try:
            A.load(p2)
            log.append(f"    {label:<34s} -> ACCEPTED (bad)")
            ok = False
        except TncaError as e:
            # TncaError, not ArtifactError: a version refusal is a correct
            # refusal with its own code, and the contract lists "unknown
            # version" as its own case. Catching only one subclass marked a
            # right answer wrong.
            log.append(f"    {label:<34s} -> [{e.code}] {e.message[:62]}")
            if expect_text and expect_text not in e.message:
                log.append(f"        expected the message to mention "
                           f"{expect_text!r}")
                ok = False
        except Exception as e:
            log.append(f"    {label:<34s} -> {type(e).__name__}: {e}")
            ok = False

    # One byte inside the CODE segment specifically, so the label is true and
    # the failure names `text`. The last byte of the file is in `constants`,
    # which tests the same mechanism about a different segment.
    _m, payload_at = A.read_header(path)
    text_seg = next(s for s in A.segment_table(_m) if s['name'] == 'text')

    def flip_code(d):
        d[payload_at + text_seg['offset'] + 3] ^= 0x01
        return d
    corrupt('one flipped byte of code', flip_code, "'text'")

    def flip_last(d):
        d[-1] ^= 0x01
        return d
    corrupt('one flipped byte of a constant', flip_last, "'constants'")

    corrupt('truncated by 16 bytes', lambda d: d[:-16])
    corrupt('the magic changed', lambda d: bytearray(b'NOTATNCA') + d[8:])

    def bump_major(d):
        d[8:10] = (A.FORMAT_VERSION[0] + 1).to_bytes(2, 'little')
        return d
    corrupt('a higher container major', bump_major)

    def huge_header(d):
        d[12:16] = (2 ** 31).to_bytes(4, 'little')
        return d
    corrupt('a header length of 2 GiB', huge_header)

    def trailing(d):
        return d + bytearray(b'\x00' * 8)
    corrupt('8 trailing bytes', trailing)

    # A segment from another artifact: same length, different content and a
    # manifest that still claims the original hash.
    other = _write_good(tmp, 'other.tnca',
                        {'text': bytes(reversed(range(64))),
                         'constants': b'\xbb' * 32})
    with open(path, 'rb') as fh:
        mine = bytearray(fh.read())
    with open(other, 'rb') as fh:
        theirs = bytearray(fh.read())
    _m, at = A.read_header(path)
    _m2, at2 = A.read_header(other)
    mixed = mine[:at] + theirs[at2:at2 + (len(mine) - at)]
    p3 = os.path.join(tmp, 'mixed.tnca')
    with open(p3, 'wb') as fh:
        fh.write(bytes(mixed))
    try:
        A.load(p3)
        log.append("    another artifact's payload        -> ACCEPTED (bad)")
        ok = False
    except ArtifactError as e:
        log.append(f"    another artifact's payload         -> "
                   f"{e.message[:70]}")

    return ok, 'byte, truncation, magic, version, length, trailing, mixture'


def art04_no_paths(log):
    """A segment name is a label, and the format has no paths to traverse."""
    import tempfile
    from tnca import artifact as A
    tmp = tempfile.mkdtemp(prefix='r2-art-')
    ok = True
    for name in ('../../etc/passwd', '/abs/path', 'a/b', 'a\\b', '..',
                 'x' * 65, ''):
        try:
            A.write(os.path.join(tmp, 'p.tnca'), good_manifest(),
                    {name: b'\x00'})
            log.append(f"    segment {name!r:<24s} -> ACCEPTED (bad)")
            ok = False
        except ArtifactError as e:
            log.append(f"    segment {name!r:<24s} -> {e.message[:52]}")
    log.append("    and the container stores no path at all, so there is "
               "nothing for a traversal to traverse")
    return ok, 'names are labels; the format cannot express a path'


def art_atomic_write(log):
    """A failed write leaves no partial file, and does not damage the old one."""
    import tempfile
    from tnca import artifact as A
    tmp = tempfile.mkdtemp(prefix='r2-art-')
    ok = True

    path = _write_good(tmp, 'k.tnca', {'text': b'\x11' * 32})
    before = open(path, 'rb').read()

    # A manifest that will fail validation part-way through writing: the
    # write must not have touched the artifact already at that path.
    bad = good_manifest()
    bad['target'].pop('capabilities')
    try:
        A.write(path, bad, {'text': b'\x22' * 32})
        log.append("    an invalid manifest was written (bad)")
        ok = False
    except ArtifactError as e:
        log.append(f"    a compile that fails validation    -> "
                   f"{e.message[:60]}")

    after = open(path, 'rb').read()
    if after != before:
        log.append("    the existing artifact was damaged (bad)")
        ok = False
    else:
        log.append("    the artifact already at that path  -> unchanged")

    leftovers = [f for f in os.listdir(tmp) if f.startswith('.tnca-')]
    if leftovers:
        log.append(f"    partial files left behind: {leftovers} (bad)")
        ok = False
    else:
        log.append("    no partial file a loader could find")
    return ok, 'a failed compile leaves nothing to mistake for an artifact'


TESTS = [('ART-02-schema', art02_schema),
         ('ART-02-container', art02_container),
         ('ART-04-paths', art04_no_paths),
         ('ART-04-atomic', art_atomic_write),
         ('ART-02-dupkey', art02_duplicate_key),
         ('ART-02-integrity', art02_integrity)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()

    print("R2-05 manifest v1 schema and integrity")
    results = []
    for cid, fn in TESTS:
        if args.only and cid not in args.only:
            continue
        log = []
        try:
            ok, what = fn(log)
        except Exception as e:
            import traceback
            ok, what = False, f"{type(e).__name__}: {e}"
            log.append('    ' + traceback.format_exc()[-600:].replace(
                '\n', '\n    '))
        results.append(dict(id=cid, passed=bool(ok), detail=what))
        print(f"  {cid}  {'PASS' if ok else 'FAIL':<5s} {what}")
        for line in log:
            print(line)
    if args.json:
        R.write_json(args.json, results)
    bad = [r['id'] for r in results if not r['passed']]
    print()
    if bad:
        print(f"  r2-artifact-tests      FAIL   {', '.join(bad)}")
        return 1
    print(f"  r2-artifact-tests      PASS   {len(results)} check(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
