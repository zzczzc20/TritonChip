"""Which upstream is present, and whether it is the pinned one.

R35-UP-02: a version mismatch, a missing helper or a bad protocol
version must each be a stable error. In particular O2 must not quietly
fall back to running without LinearLayout -- a stage whose headline
claim is "we reuse upstream" cannot answer a missing upstream by
silently not reusing it.
"""
import json

# Through _datapath, not four directories up: installed from the wheel, four
# up from site-packages/tnca_cc/upstream is <prefix>/lib, and the lock was
# looked for at lib/spec/r1_toolchain.lock.json. _datapath reads the copy the
# wheel ships (tnca/data/spec/), then the checkout.
from .._datapath import find

LOCK = find('spec', 'r1_toolchain.lock.json')


class UpstreamError(Exception):
    """The upstream is absent, the wrong version, or missing a symbol."""


def pinned_version():
    with open(LOCK) as fh:
        return json.load(fh)['triton']['version']


def describe():
    """What is actually importable here. Never raises."""
    out = {'pinned': pinned_version(), 'present': None, 'matches': False,
           'linear_layout': False, 'passes': False, 'why': None}
    try:
        import triton
        out['present'] = triton.__version__
    except Exception as e:                                # noqa: BLE001
        out['why'] = f"triton does not import: {type(e).__name__}: {e}"
        return out
    out['matches'] = out['present'] == out['pinned']
    try:
        from triton._C import libtriton
        out['linear_layout'] = hasattr(libtriton, 'linear_layout') and \
            hasattr(libtriton.linear_layout, 'LinearLayout')
        out['passes'] = (hasattr(libtriton, 'passes') and
                         hasattr(libtriton.passes, 'common') and
                         hasattr(libtriton.passes.common, 'add_canonicalizer')
                         and hasattr(libtriton.passes.common, 'add_cse'))
    except Exception as e:                                # noqa: BLE001
        out['why'] = f"libtriton does not import: {type(e).__name__}: {e}"
    if out['why'] is None and not out['matches']:
        out['why'] = (f"triton {out['present']} is installed and "
                      f"{out['pinned']} is pinned")
    return out


def require(what):
    """Raise unless `what` ('layout' or 'passes') is really available.

    The error names the pinned version and what was found, because the
    usual cause is a second environment that installed whatever pip
    offered.
    """
    d = describe()
    if d['present'] is None:
        raise UpstreamError(
            f"{what}: triton is not importable, and the pinned "
            f"{d['pinned']} is what this stage reuses. {d['why']}")
    if not d['matches']:
        raise UpstreamError(
            f"{what}: {d['why']}. Section 4.2 pins the version; it is not "
            f"tracked, and a near-enough one is not substituted.")
    if not d[{'layout': 'linear_layout', 'passes': 'passes'}[what]]:
        raise UpstreamError(
            f"{what}: triton {d['present']} imports but does not export "
            f"the symbol this needs. {d['why'] or ''}".strip())
    return d
