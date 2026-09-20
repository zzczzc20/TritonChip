#!/usr/bin/env python3
"""
Profiles, manifests and the launcher check -- R1-04.

One configuration source (spec/r1_profiles.json) decides three things that had
been decided separately and had drifted apart: the elaboration parameters the
RTL is built with, the target description the compiler reads, and the
capabilities a program is allowed to use. The audit's F03 is what drift looks
like -- a four-tile target advertising an SFU that the tiles did not have.

A compiled program carries a manifest naming the profile it was compiled for
and the hash of that configuration. `verify_manifest` is what a launcher runs
*before* the first instruction: a binary built for another profile, or for a
configuration that has since changed, is refused rather than executed into a
fault somewhere in the middle.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SPEC = os.path.join(ROOT, 'spec', 'r1_profiles.json')
TARGETS = os.path.join(ROOT, 'compiler', 'targets')
SCHEMA = os.path.join(ROOT, 'isa', 'tnca-isa-v0.1.json')


def load():
    with open(SPEC) as fh:
        return json.load(fh)


def profile(name, spec=None):
    spec = spec or load()
    if name not in spec['profiles']:
        raise KeyError(f"no profile {name!r}; known: {sorted(spec['profiles'])}")
    return spec['profiles'][name]


def target_of(name, spec=None, targets_dir=None):
    with open(os.path.join(targets_dir or TARGETS,
                           profile(name, spec)['target_file'])) as fh:
        return json.load(fh)


def config_hash(name, spec=None, targets_dir=None):
    """A hash of everything that decides what this profile IS.

    Deliberately over the *elaboration* and the *capabilities*, not over the
    whole target file: a comment or a description changing must not invalidate
    a binary, and a parameter or a capability changing must.

    `spec` and `targets_dir` let an archived copy be re-derived from the files
    beside it, which is how tools/r1/verify_evidence.py notices a manifest
    lifted out of another candidate's evidence: the hashes were recomputed
    along with it, so only re-deriving catches it.
    """
    p = profile(name, spec)
    t = target_of(name, spec, targets_dir)
    blob = json.dumps({
        'profile': name,
        'elaboration': p['elaboration'],
        'num_cores': p['num_cores'],
        'topology': p['topology'],
        'capabilities': t['capabilities'],
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def isa_version():
    with open(SCHEMA) as fh:
        return json.load(fh).get('isa_version', 'unknown')


def make_manifest(name, *, requires=()):
    """What a compiled binary carries so a launcher can check it."""
    return {
        'schema': 'tnca-manifest/1',
        'profile': name,
        'config_hash': config_hash(name),
        'isa_version': isa_version(),
        'requires_capabilities': sorted(requires),
    }


def verify_manifest(manifest, name):
    """Refuse before execution, with a reason. Returns a list of problems."""
    problems = []
    if manifest.get('schema') != 'tnca-manifest/1':
        problems.append(f"unknown manifest schema {manifest.get('schema')!r}")
    if manifest.get('profile') != name:
        problems.append(f"binary was compiled for profile "
                        f"{manifest.get('profile')!r}, this machine is {name!r}")
    want = config_hash(name)
    if manifest.get('config_hash') != want:
        problems.append(f"configuration hash {str(manifest.get('config_hash'))[:12]} "
                        f"does not match this machine's {want[:12]} -- the "
                        f"profile changed since the binary was built")
    if manifest.get('isa_version') != isa_version():
        problems.append(f"binary targets ISA {manifest.get('isa_version')!r}, "
                        f"this machine implements {isa_version()!r}")
    caps = target_of(name)['capabilities']
    for c in manifest.get('requires_capabilities', []):
        if not caps.get(c):
            problems.append(f"binary requires capability {c!r}, which this "
                            f"profile does not have")
    return problems


def elaboration_argv(name):
    """The -P overrides this profile's benches are elaborated with.

    CFG-02: a configuration field with no read-back register is checked
    against this, which is derived from the same source the RTL is built from
    -- not by reading the target JSON a second time and calling the agreement
    a verification.
    """
    return [f"-P tb_tnca_core.{k}={v}"
            for k, v in sorted(profile(name)['elaboration'].items())]


if __name__ == '__main__':
    for n in load()['profiles']:
        t = target_of(n)
        print(f"{n:12s} cores={t['num_cores']} rctx={t['num_rctx']} "
              f"sfu={t['capabilities']['sfu']} hash={config_hash(n)[:12]}")
