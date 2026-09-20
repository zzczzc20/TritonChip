"""
Where the compiler's data files are, in a checkout and in an installed wheel.

`isa.py` and `target.py` each computed a path three directories above
themselves, which is the repository root in a checkout and

    .../site-packages/isa/tnca-isa-v0.1.json

in an installed package -- a directory that does not exist. R1 never noticed
because R1 runs from a checkout; R2 ships a wheel (PKG-01 installs it and then
moves the repository away), and the first thing the installed compiler did was
fail to find the ISA.

The order is: an explicit override, then the data the package shipped, then
the checkout. An installed package reads what it shipped.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# The checkout root, for development.
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))


def _candidates(*parts):
    out = []
    env = os.environ.get('TNCA_DATA_DIR')
    if env:
        out.append(os.path.join(env, *parts))
    # Installed beside the product package, which is where `make
    # r2-stage-data` puts it.
    for up in (os.path.dirname(_HERE),):
        out.append(os.path.join(up, 'tnca', 'data', *parts))
    out.append(os.path.join(ROOT, *parts))
    return out


def find(*parts):
    tried = _candidates(*parts)
    for path in tried:
        if os.path.exists(path):
            return path
    return tried[-1]
