#!/usr/bin/env python3
"""
Build fingerprints -- doc/phase1_tasks_and_acceptance.md R1-03.

A make prerequisite only covers files make was told about. The audit (F04)
found `tb_core.vvp` with neither `rtl/gen` nor a single `.vh` in its
prerequisites, so editing either left a stale simulator in place and the next
run "verified" RTL that no longer existed.

File lists fix that half. The other half is everything make cannot see at all:
the iverilog version, the parameter overrides a bench is elaborated with, the
seed and vector count a generator was run with. Those change what the output
*is* without changing any file's timestamp, and a build system that ignores
them will happily reuse a 2-RCTX simulator to test a 4-RCTX profile.

So each output gets a stamp file holding a hash of both. The stamp is rewritten
only when the hash changes, which is what makes it usable as a prerequisite:
an unchanged input leaves the mtime alone and make does not rebuild.

    fingerprint.py stamp <stamp-file> KEY=VALUE ... -- FILE ...

Printing (`fingerprint.py show ...`) is for the BUILD-0x experiments, which
need to compare two fingerprints without touching the tree.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
NORM = os.path.join(ROOT, 'spec', 'r1_normalization.json')


def normalized_digest(path):
    """A digest of an artefact with the listed volatile fields removed.

    Icarus names its symbols after the address of the object that created
    them, so two builds of identical input are never byte-identical. The rules
    live in spec/r1_normalization.json, in advance, because a normalisation
    invented after a comparison fails is not a normalisation.
    """
    with open(NORM) as fh:
        rules = json.load(fh)['artifacts']['*.vvp']['removed']
    with open(path, 'rb') as fh:
        text = fh.read().decode('utf-8', 'replace')
    for r in rules:
        text = re.sub(r['pattern'], r['replacement'], text,
                      flags=re.M)
    return hashlib.sha256(text.encode()).hexdigest()


def artifact_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 16), b''):
            h.update(c)
    return h.hexdigest()


def provenance_path(artefact):
    return artefact + '.prov'


def record_provenance(artefact, stamp):
    """Tie an artefact to the fingerprint of the inputs it was built from.

    make can only ask "is the target newer than its prerequisites". An
    artefact copied in from another profile is newer than everything, so make
    accepts it. This is the record that lets the runner notice, before it runs
    anything, that the file on disk is not the file those inputs produce.
    """
    with open(stamp) as fh:
        fp = fh.read().strip()
    data = dict(artifact=os.path.relpath(artefact, ROOT),
                sha256=artifact_digest(artefact),
                normalized=normalized_digest(artefact),
                fingerprint=fp)
    with open(provenance_path(artefact), 'w') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return data


def check_provenance(artefact, stamp):
    """Non-zero if the artefact on disk is not the one those inputs built."""
    pp = provenance_path(artefact)
    if not os.path.exists(pp):
        return [f"{os.path.relpath(artefact, ROOT)}: no provenance record"]
    with open(pp) as fh:
        rec = json.load(fh)
    with open(stamp) as fh:
        fp = fh.read().strip()
    problems = []
    if rec.get('fingerprint') != fp:
        problems.append(f"{rec['artifact']}: built from fingerprint "
                        f"{rec.get('fingerprint', '?')[:12]}, inputs now hash "
                        f"to {fp[:12]}")
    actual = artifact_digest(artefact)
    if rec.get('sha256') != actual:
        problems.append(f"{rec['artifact']}: content is {actual[:12]}, "
                        f"provenance says {rec.get('sha256', '?')[:12]} "
                        f"(an artefact from elsewhere?)")
    return problems


def tool_versions():
    """Versions of the tools whose output ends up in the artefact."""
    out = {}
    for name, argv in (('iverilog', ['iverilog', '-V']),
                       ('python3', [sys.executable, '--version'])):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            out[name] = (r.stdout or r.stderr).strip().splitlines()[0]
        except Exception:
            out[name] = 'MISSING'
    return out


def compute(params, files):
    h = hashlib.sha256()
    h.update(b'r1-fingerprint/1\n')
    for k in sorted(tool_versions()):
        h.update(f"tool {k}={tool_versions()[k]}\n".encode())
    for kv in sorted(params):
        h.update(f"param {kv}\n".encode())
    for rel in sorted(set(files)):
        p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        h.update(f"file {rel}\n".encode())
        try:
            with open(p, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b''):
                    h.update(chunk)
        except OSError:
            # A prerequisite that does not exist is part of the fingerprint
            # too: it must not look the same as one that does.
            h.update(b'ABSENT\n')
    return h.hexdigest()


def main(argv):
    if len(argv) >= 4 and argv[1] in ('provenance', 'check'):
        artefact, stamp = argv[2], argv[3]
        if argv[1] == 'provenance':
            record_provenance(artefact, stamp)
            return 0
        problems = check_provenance(artefact, stamp)
        for p in problems:
            print('  ' + p)
        return 1 if problems else 0
    if len(argv) >= 3 and argv[1] == 'normalized':
        print(normalized_digest(argv[2]))
        return 0
    if len(argv) < 3 or argv[1] not in ('stamp', 'show'):
        print(__doc__.strip())
        return 2
    mode = argv[1]
    rest = argv[2:]
    if mode == 'stamp':
        stamp, rest = rest[0], rest[1:]
    sep = rest.index('--') if '--' in rest else len(rest)
    params, files = rest[:sep], rest[sep + 1:]
    digest = compute(params, files)

    if mode == 'show':
        print(digest)
        return 0

    os.makedirs(os.path.dirname(stamp) or '.', exist_ok=True)
    old = None
    if os.path.exists(stamp):
        with open(stamp) as fh:
            old = fh.read().strip()
    if old != digest:
        with open(stamp, 'w') as fh:
            fh.write(digest + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
