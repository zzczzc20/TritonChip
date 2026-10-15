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


# The parameters that shape the device's ports (rtl/common/tnca_uarch_defaults.vh).
# A platform reads these; every other parameter is the device's own business.
INTERFACE_PARAMETERS = ('NUM_RCTX', 'RCTX_W', 'LSU_TAG_W')


# What the RTL computes from other parameters (rtl/common/tnca_uarch_defaults.vh).
DERIVED = {
    'PHYS_VREGS': lambda p: p.get('NUM_RCTX', 2) * p.get('VREGS_PER_RCTX', 64),
}


def rtl_parameters(name):
    """Every RTL parameter the profile states: its elaboration and its
    implementation's rtl_parameters. A parameter it does not state takes the
    RTL default, which is off."""
    p = profile(name)
    params = dict(p['elaboration'])
    params.update((p.get('implementation') or {}).get('rtl_parameters') or {})
    return params


def uarch_header(name, overrides=()):
    """The profile as one Verilog header: `define TNCA_IF_<K> / TNCA_UARCH_<K>.

    Compiled before the platform and the device, it is how a profile reaches
    the RTL; no platform passes a microarchitectural parameter by name. A
    new parameter is an entry in the profile and a parameter in the device.
    """
    params = rtl_parameters(name)
    params.update(dict(overrides))
    # A value the RTL derives itself is left to it when the profile states the
    # same value: the device is then elaborated exactly as it always was.
    for key, derive in DERIVED.items():
        if key in params and params[key] == derive(params):
            del params[key]
    lines = [f'// GENERATED from spec/r1_profiles.json, profile {name}, by '
             'tnca.profiles.uarch_header. Do not edit: change the profile.']
    for key in sorted(params):
        value = params[key]
        if not key.isidentifier() or not key.isupper() or isinstance(value, bool) \
                or not isinstance(value, int):
            raise ValueError(f'profile {name}: {key}={value!r} is not NAME=<integer>')
        kind = 'IF' if key in INTERFACE_PARAMETERS else 'UARCH'
        lines.append(f'`define TNCA_{kind}_{key} {value}')
    return '\n'.join(lines) + '\n'


def check_executes(machine, manifest, *, where=''):
    """Refuse an artifact this machine cannot run, before any code is placed.

    Same profile: accepted, as before. A profile may also declare
    `executes_artifacts_of` -- profiles it is a strict superset of, with the
    same elaboration apart from added capabilities -- and runs their
    artifacts when the artifact's configuration hash is that profile's and
    every capability it requires is present here. An artifact from a
    machine with a capability this one lacks is refused as CAPABILITY_MISSING
    naming it, rather than as an unexplained profile difference.
    """
    from .errors import CapabilityError
    art = manifest['target']['profile']
    if art == machine:
        return
    here = {c for c, v in target(machine)['capabilities'].items() if v}
    needed = set(manifest['target'].get('capabilities', []))
    missing = sorted(needed - here)
    if art in profile(machine).get('executes_artifacts_of', []):
        if manifest['target'].get('config_hash') != config_hash(art):
            raise ProfileMismatch(
                f"artifact claims profile {art!r} with configuration "
                f"{str(manifest['target'].get('config_hash'))[:12]}, which is not "
                f"{art!r} as this runtime knows it", where=where,
                remedy='recompile; the profile changed since the artifact was built')
        if missing:
            raise CapabilityError(
                f"artifact needs {missing}, which {machine!r} does not have",
                where=where, remedy=f'recompile for {machine!r}')
        return
    if art in names() and machine in profile(art).get('executes_artifacts_of', []) and missing:
        raise CapabilityError(
            f"artifact is for {art!r} and needs {missing}, which {machine!r} "
            f"does not have", where=where,
            detail={'artifact_profile': art, 'machine': machine, 'missing': missing},
            remedy=f'recompile for {machine!r}; the compiler then lowers '
                   f'without the capability')
    raise ProfileMismatch(
        f"artifact is for profile {art!r}, this session is {machine!r}",
        where=where,
        remedy='recompile for this profile; v1 artifacts are not '
               'portable between them')


def info(name):
    p = profile(name)
    return dict(name=name, spec=p, config_hash=config_hash(name),
                target_file=p['target_file'])
