"""
Profiles, read by the product rather than borrowed from the test tools.

`tnca.adapters.r1` used to import `tools/r1/profiles.py` for this. That is
verification code, and doc/r2_interfaces.md says in as many words that the
product compiler and runtime do not depend on the verification modules -- a
rule PKG-02 checks by installing the package and looking. It also could not
survive packaging: `tools/` is not in a wheel, and `profiles` is not a name
anything should claim at the top level of an installed environment.

So the data is read here, from files the wheel ships:

    spec/r1_profiles.json      what each profile IS
    compiler/targets/*.json    the target description generated from the RTL

The hash is computed the same way R1 computes it -- over the elaboration
parameters and the capabilities, not over the whole file -- because an
artifact bound to a fingerprint must be rejected by a machine with different
parameters and accepted by one whose comment changed.
"""
import hashlib
import json
import os

from .errors import ProfileMismatch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _search_paths(*parts):
    """Where a data file lives, installed or in a checkout.

    Installed, the data sits beside the package. In a checkout it is at the
    repository root. Both are tried, in that order, so an installed package
    never silently reads a developer's tree.
    """
    return [
        os.path.join(_HERE, 'data', *parts),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
                     *parts),
        os.path.join(os.path.dirname(os.path.dirname(_HERE)), *parts),
    ]


def _read(*parts):
    tried = _search_paths(*parts)
    for path in tried:
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh), path
    raise ProfileMismatch(
        f"cannot find {os.path.join(*parts)}",
        detail={'looked in': tried},
        remedy='the package must ship its profile data; reinstall it')


def spec():
    data, _path = _read('spec', 'r1_profiles.json')
    return data


def names():
    return sorted(spec()['profiles'])


def profile(name):
    data = spec()
    if name not in data['profiles']:
        raise ProfileMismatch(
            f"no profile {name!r}",
            detail={'known': ', '.join(sorted(data['profiles']))})
    return data['profiles'][name]


def target(name):
    p = profile(name)
    data, _path = _read('compiler', 'targets', p['target_file'])
    return data


def config_hash(name):
    """What this profile IS: elaboration parameters and capabilities.

    Deliberately not the whole target file. A description or a comment
    changing must not invalidate an artifact; a parameter or a capability
    changing must.
    """
    p, t = profile(name), target(name)
    blob = json.dumps({
        'profile': name,
        'elaboration': p['elaboration'],
        'num_cores': p['num_cores'],
        'topology': p['topology'],
        'capabilities': t['capabilities'],
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def info(name):
    p = profile(name)
    return dict(name=name, spec=p, config_hash=config_hash(name),
                target_file=p['target_file'])
