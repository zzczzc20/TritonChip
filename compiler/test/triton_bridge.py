"""
Compile a @triton.jit kernel to TTIR with the real Triton frontend.

compiler_overview.md sec 3.1: the frontend is reused unmodified. This wraps
the few calls needed to get a TTIR module out of it, so the rest of the
compiler never touches Triton's internals.

Returns None when Triton is not importable, so the test suite still runs on a
machine without it.
"""
import os
import sys


def available():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def make_ttir(fn, signature, constexprs, *, backend_name='tnca'):
    """Run the Triton frontend and return (module, context)."""
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    from tnca_cc.triton_options import CODEGEN_FNS, frontend_options

    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    module = src.make_ir(GPUTarget(backend_name, 0, 32),
                         frontend_options(backend_name), CODEGEN_FNS, {},
                         ctx)
    # Inline Triton's library functions. `tl.exp`, `tl.sum` and friends
    # are @triton.jit functions and arrive as `tt.call`, which no
    # backend can lower -- so this is legalisation, not optimisation,
    # and it happens on every path that produces TTIR for this backend.
    from tnca_cc.upstream import PureValueOptimizer
    PureValueOptimizer.legalize_module(module, ctx)
    # Keep the context alive: the module borrows from it.
    return module, ctx


def param_names(fn):
    """The kernel's declared parameter order, minus the constexprs.

    The authority on which value binds to which parameter. Passing a dict and
    letting the importer walk it in insertion order made two calls with
    identical arguments compile to two different kernels (audit F07), so the
    order comes from the @triton.jit function itself.
    """
    const = set(getattr(fn, 'constexprs', ()) or ())
    return [n for i, n in enumerate(fn.arg_names) if i not in const]
