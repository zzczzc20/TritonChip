"""
Where the package's own data lives.

Installed, the spec files sit in `tnca/data/`; in a checkout they are at the
repository root. Each module used to compute its own path relative to
`__file__`, which worked in a checkout and produced

    FileNotFoundError: .../site-packages/tnca/../../spec/r2_abi_v1.json

the first time the wheel was installed. One function, tried in one order, so
an installed package reads what it shipped and never a developer's tree.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Installed first. A checkout that has both -- because `make r2-stage-data`
# has run -- should still read the staged copy, which is the one the wheel
# would carry.
_ROOTS = (
    os.path.join(_HERE, 'data'),
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
)


def path(*parts, env=None):
    """The first existing path for `parts`, or the last candidate tried.

    `env` names an environment variable that overrides everything, which the
    tests use to point at a fixture without rewriting a package.
    """
    if env:
        override = os.environ.get(env)
        if override:
            return override
    tried = [os.path.join(root, *parts) for root in _ROOTS]
    for candidate in tried:
        if os.path.exists(candidate):
            return candidate
    return tried[0]


def roots():
    return list(_ROOTS)
