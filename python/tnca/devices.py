"""A multicore device as data: spec/devices/<name>.json.

A profile (spec/r1_profiles.json) describes one core, and that is all the
compiler is told: a grid's programs are dispatched to the cores by the
hardware. What makes the three-core U280 machine a machine -- how many
cores, the NUCA geometry and its pipelines, the memory outlets, the clock --
used to live only in a hand-written image.tcl under evidence/ (R40's attempt 3)
and in the defaults of whichever wrapper elaborated it. A device description
names the per-core profile and states the rest, and everything that builds or
models the machine is generated from it:

    image_tcl()          TNCA_IMAGE_CONFIG for flow/fpga/package_xo.tcl
    image_params()       the same parameters, for the board-top simulation
    backend_args()       the fpga-u280-multicore / fpga-multicore-sim arguments
    systemc_overrides()  the uncore keys of the SystemC model's configuration

Three kinds of field:

    platform  fixed by the card and the flow (core count, AXI data width, the
              clock, the controls the runtime requires). A task or a target
              owns them: check_platform() refuses a description that differs.
    uncore    the machine's own design, open to change within RTL limits.
    derived   computed here, never written: SET_W, WAY_W, AXI_ID_W from the
              uncore, PHYS_VREGS and PHYS_W from the core's contexts.
"""
import hashlib
import json
from pathlib import Path

from . import profiles

ROOT = Path(__file__).resolve().parents[2]
DEVICES = ROOT / 'spec' / 'devices'
SCHEMA = 'tnca-device/1'


class DeviceError(ValueError):
    """A description that cannot be built: every problem, one per line."""


# The fields a target or a task fixes. Values are checked against theirs.
PLATFORM_FIELDS = ('board', 'variant', 'core_count', 'kernel_clock_mhz', 'AXI_DATA_W', 'hbm_bytes',
                   'MEMORY_CONTROLS', 'CACHE_MAINTENANCE')

# The uncore fields and their legal values. The limits are the RTL's own:
# rtl/board/tnca_multicore_hbm_xo.v (MEM_PORTS 1..3), rtl/nuca/tnca_tmu.v
# (TLW 0..2), tnca_elastic_link (stages >= 0), the set and way indices
# ($clog2, so powers of two), and the outstanding slots the native line
# adapter takes (1..16, tools/r40/image_profile.py).
_POW2 = lambda lo, hi: tuple(1 << i for i in range(lo.bit_length() - 1, hi.bit_length()) if lo <= 1 << i <= hi)
UNCORE = {
    'MEM_PORTS': tuple(range(1, 4)),
    'AXI_OUT_N': tuple(range(1, 17)),
    'SETS': _POW2(4, 512),
    'WAYS': _POW2(2, 16),
    'CODE_LINK_STAGES': tuple(range(0, 5)),
    'LINK_STAGES': tuple(range(0, 5)),
    'NOC_LINK_STAGES': tuple(range(0, 5)),
    'DISPATCH_STAGES': tuple(range(0, 5)),
    'MEM_HOME_STAGES': tuple(range(0, 5)),
    'DCO_REGISTERED_RANGES': (0, 1),
    'DCO_REGISTER_WRITES': (0, 1),
    'DCO_LOCAL_REPLICAS': (0, 1),
    'DCO_TLW_PIPE_STAGES': (0, 1, 2),
    'NUCA_CACHE_DATA_PIPELINE': (0, 1),
    'NUCA_REGISTER_VICTIM_RANK': (0, 1),
    'NUCA_REGISTER_VICTIM_SELECT': (0, 1),
}
DERIVED = ('SET_W', 'WAY_W', 'AXI_ID_W', 'PHYS_VREGS', 'PHYS_W')

# The image parameters a multicore description sets from its platform.
_PLATFORM_PARAMS = ('AXI_DATA_W', 'MEMORY_CONTROLS', 'CACHE_MAINTENANCE')

# The per-core parameters an image states, from the profile. Explicit zeros:
# a wrapper's defaults may enable the banked v2 implementation (the same rule
# as tools/r40/image_profile.py, which tests/test_devices.py holds this to).
_CORE_DEFAULTS = dict(USE_BANKED_VRF=0, VRF_BYPASS=0, FE_NOBUBBLE=0, VRF_PREFETCH=0, RCTX_SWITCH=0,
                      LSU_INFLIGHT=1, FPU_OVERLAP=0, RSLICE=0, RPAIR=0, MTX_ABUF=0, HAS_TENSOR_XPOSE=0)


def path(name):
    """The file of a named description (a path is taken as it is)."""
    p = Path(name)
    if p.suffix == '.json' or p.is_absolute() or len(p.parts) > 1:
        return p
    return DEVICES / f'{name}.json'


def load(name):
    """A description, validated."""
    p = path(name)
    try:
        desc = json.loads(p.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise DeviceError(f'no device description {p}') from None
    except json.JSONDecodeError as e:
        raise DeviceError(f'{p}: not JSON: {e}') from None
    validate(desc)
    if desc['name'] != p.stem:
        # Records name a machine by its file; the two must not disagree.
        raise DeviceError(f'{p}: names itself {desc["name"]!r}; a description is named for its file')
    return desc


def core_params(profile_name):
    """The per-core image parameters of a single-core profile."""
    spec = profiles.profile(profile_name)
    if spec.get('num_cores', 1) != 1:
        raise DeviceError(f'per-core profile {profile_name!r} has {spec["num_cores"]} cores; '
                          'a device names one core and states the count itself')
    params = dict(_CORE_DEFAULTS)
    params.update({k: v for k, v in spec['elaboration'].items() if k != 'PHYS_VREGS'})
    impl = spec.get('implementation') or {}
    if impl.get('vector_register_file') == 'banked':
        params['USE_BANKED_VRF'] = 1
    params.update(impl.get('rtl_parameters') or {})
    return params


def core_derived(profile_name):
    """PHYS_VREGS and PHYS_W of a per-core profile: its contexts times their
    registers. The single-core wrapper derives them; the multicore top takes
    them as parameters (rtl/board/tnca_multicore_hbm_xo.v, 256 and 8), so an
    image that left them out would build every core with the top's register
    file whatever the profile -- right for v5's four contexts of 64, wrong for
    a two-context core."""
    core = core_params(profile_name)
    try:
        n = core['NUM_RCTX'] * core['VREGS_PER_RCTX']
    except KeyError as e:
        raise DeviceError(f'per-core profile {profile_name!r} states no {e.args[0]}') from None
    stated = profiles.profile(profile_name)['elaboration'].get('PHYS_VREGS', n)
    if stated != n:
        raise DeviceError(f'per-core profile {profile_name!r} sets PHYS_VREGS {stated}; its '
                          f'{core["NUM_RCTX"]} contexts of {core["VREGS_PER_RCTX"]} registers make {n}')
    return dict(PHYS_VREGS=n, PHYS_W=max(1, (n - 1).bit_length()))


def _log2(n):
    return n.bit_length() - 1


def derived(uncore):
    """SET_W, WAY_W and AXI_ID_W, from the fields they follow."""
    return dict(SET_W=_log2(uncore['SETS']), WAY_W=_log2(uncore['WAYS']),
                AXI_ID_W=max(1, (uncore['AXI_OUT_N'] - 1).bit_length()))


def validate(desc):
    """Refuse a description that is not buildable, naming every problem."""
    problems = []
    if not isinstance(desc, dict):
        raise DeviceError('a device description is a JSON object')
    if desc.get('schema') != SCHEMA:
        problems.append(f'schema must be {SCHEMA!r}, not {desc.get("schema")!r}')
    allowed = {'schema', 'name', 'description', 'per_core_profile', 'platform', 'uncore'}
    for key in sorted(set(desc) - allowed):
        problems.append(f'unknown field {key!r}')
    if not isinstance(desc.get('name'), str) or not desc.get('name'):
        problems.append('name must be a non-empty string')
    try:
        core_derived(desc.get('per_core_profile'))
    except (KeyError, TypeError, profiles.ProfileMismatch):
        problems.append(f'per_core_profile {desc.get("per_core_profile")!r} is not a profile '
                        'in spec/r1_profiles.json')
    except DeviceError as e:
        problems.append(str(e))
    platform = desc.get('platform')
    if not isinstance(platform, dict):
        problems.append('platform must be an object')
        platform = {}
    for key in PLATFORM_FIELDS:
        if key not in platform:
            problems.append(f'platform.{key} is missing')
    for key in sorted(set(platform) - set(PLATFORM_FIELDS)):
        problems.append(f'platform.{key} is not a platform field')
    for key in ('core_count', 'kernel_clock_mhz', 'AXI_DATA_W', 'hbm_bytes', 'MEMORY_CONTROLS',
                'CACHE_MAINTENANCE'):
        v = platform.get(key)
        if key in platform and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            problems.append(f'platform.{key}={v!r} is not a non-negative integer')
    uncore = desc.get('uncore')
    if not isinstance(uncore, dict):
        problems.append('uncore must be an object')
        uncore = {}
    for key in sorted(set(uncore) - set(UNCORE)):
        if key in DERIVED:
            problems.append(f'uncore.{key} is derived ({", ".join(DERIVED)} are computed), not written')
        else:
            problems.append(f'uncore.{key} is not an uncore field')
    for key, legal in UNCORE.items():
        if key not in uncore:
            problems.append(f'uncore.{key} is missing')
            continue
        v = uncore[key]
        if isinstance(v, bool) or v not in legal:
            problems.append(f'uncore.{key}={v!r} is outside {legal[0]}..{legal[-1]}'
                            + (' (a power of two)' if key in ('SETS', 'WAYS') else ''))
    cores = platform.get('core_count')
    if isinstance(cores, int) and isinstance(uncore.get('MEM_PORTS'), int) and uncore['MEM_PORTS'] > cores:
        problems.append(f'uncore.MEM_PORTS={uncore["MEM_PORTS"]} exceeds platform.core_count={cores}')
    if problems:
        raise DeviceError('\n'.join(problems))
    return desc


def check_platform(desc, locked):
    """The fields where a description differs from a target's or a task's
    platform: [] when every locked field matches."""
    return [f'platform.{k}: {desc["platform"].get(k)!r}, fixed at {v!r}'
            for k, v in sorted(locked.items()) if desc['platform'].get(k) != v]


def image_params(desc):
    """Every parameter the image's wrapper is packaged with, by name."""
    params = core_params(desc['per_core_profile'])
    params.update(core_derived(desc['per_core_profile']))
    params.update({k: desc['platform'][k] for k in _PLATFORM_PARAMS})
    params.update(desc['uncore'])
    params.update(derived(desc['uncore']))
    return params


def image_tcl(desc):
    """TNCA_IMAGE_CONFIG for flow/fpga/package_xo.tcl, in image_profile.py's form."""
    params = image_params(desc)
    return ('set TNCA_IMAGE_PARAMS [dict create '
            + ' '.join(f'{k} {v}' for k, v in sorted(params.items())) + ']\n')


def fingerprint(desc):
    """sha256 of what the description builds: the per-core profile, the
    platform and every image parameter. Two descriptions with one fingerprint
    are the same machine."""
    blob = json.dumps(dict(profile=desc['per_core_profile'], platform=desc['platform'],
                           params=image_params(desc)), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def backend_args(desc, *, xclbin=None):
    """The arguments of the board backend (fpga-u280-multicore)."""
    args = dict(image_profile=desc['per_core_profile'], core_count=desc['platform']['core_count'],
                memory_ports=desc['uncore']['MEM_PORTS'], hbm_dma=True,
                hbm_bytes=desc['platform']['hbm_bytes'], device_clock=True)
    if xclbin is not None:
        args['xclbin'] = str(xclbin)
    return args


def systemc_overrides(desc):
    """The SystemC model's configuration keys a description decides
    (tools/performance_harness/systemc/configs); the rest stay the base
    configuration's. The model routes to one outlet or to all of them, so a
    description that uses some of several cannot be modelled."""
    cores = desc['platform']['core_count']
    ports = desc['uncore']['MEM_PORTS']
    if ports not in (1, cores):
        raise DeviceError(f'the SystemC model routes to one outlet or to all {cores}; '
                          f'MEM_PORTS={ports} cannot be modelled')
    return dict(nx=cores, ny=1, sets=desc['uncore']['SETS'], ways=desc['uncore']['WAYS'],
                outlets=cores, active_outlets=ports, outstanding=desc['uncore']['AXI_OUT_N'],
                clock_hz=desc['platform']['kernel_clock_mhz'] * 1_000_000)


def parse_image_tcl(text):
    """{name: int} from a TNCA_IMAGE_CONFIG file (image_tcl()'s or R40's)."""
    head = 'set TNCA_IMAGE_PARAMS [dict create '
    line = text.strip()
    if not line.startswith(head) or not line.endswith(']'):
        raise DeviceError('not a TNCA_IMAGE_PARAMS dict')
    words = line[len(head):-1].split()
    if len(words) % 2:
        raise DeviceError('TNCA_IMAGE_PARAMS has an odd number of words')
    return {k: int(v) for k, v in zip(words[0::2], words[1::2])}
