"""
tnca_cc -- the TNCA compiler backend.

Structure follows compiler_overview.md:

  target.py       the target contract of sec 15 C0, read from
                  compiler/targets/*.json (generated from the RTL)
  isa.py          the single ISA source of sec 11.3, read from
                  isa/tnca-isa-v0.1.json (generated from the assembler tables)
  diagnostics.py  the capability refusals of sec 12 -- an absent feature is an
                  error, never a quiet substitution

The MLIR half (the `tnca` dialect, tnca-opt, the Triton plugin) lives beside
this in C++ and needs the pinned LLVM; see compiler/README.md. This package is
the bring-up path of sec 11.1, which reaches RTL through the existing
assembler and therefore runs with nothing installed.
"""

from .diagnostics import CapabilityError, Diagnostic, ResourceError  # noqa: F401
from .isa import ISA, Instruction  # noqa: F401
from .target import TargetInfo  # noqa: F401

__all__ = ['TargetInfo', 'ISA', 'Instruction', 'CapabilityError',
           'ResourceError', 'Diagnostic']
