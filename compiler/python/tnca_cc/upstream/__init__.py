"""The pinned upstream components, and the boundary they live behind.

doc/phase3_5_tasks_and_acceptance.md section 4.2 and R35-UP-01/02.

Section 4.2 is mostly about a hazard that turns out not to arise here.
It warns against linking the distribution MLIR 20 that the R1 toolchain
pins against a second, separately pinned LLVM inside one process, and
prefers a separate `tnca-layout-opt` tool "或等价隔离 adapter" so that
only versioned text or a versioned data protocol crosses the boundary.

The locked Triton wheel already is that isolated adapter. It exports
`libtriton.linear_layout.LinearLayout` and
`libtriton.passes.common.add_canonicalizer` / `add_cse` as public
submodules of one self-contained extension, built against one matched
LLVM that never meets ours. What crosses the boundary is MLIR *text*
through `parse_mlir_module`, and plain Python dicts and lists for
layouts. No MLIR object crosses an ABI, no second LLVM is linked, and
nothing here reaches for a private symbol.

So this module does not reimplement anything. It is the thin place where
the real upstream code is called, plus the two pieces of defensive work
that calling it requires:

  * upstream aborts the process on a domain error. Composing layouts
    whose dimensions do not match prints `LLVM ERROR: Dimensions must
    match` and calls exit -- there is no exception to catch. Every entry
    point here validates first, so a caller gets a TncaError instead of
    losing the process.
  * `pass_manager.run` takes a name as well as a module.
"""
from .layout import (LinearLayoutAdapter, LayoutError, available as
                     layout_available)
from .passes import PureValueOptimizer, PassError, available as passes_available
from .version import describe, require

__all__ = ['LinearLayoutAdapter', 'LayoutError', 'layout_available',
           'PureValueOptimizer', 'PassError', 'passes_available',
           'describe', 'require']
