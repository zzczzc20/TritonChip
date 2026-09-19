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

    class _Options:
        num_warps = 1
        num_ctas = 1
        num_stages = 1
        debug = False
        sanitize_overflow = False
        arch = None
        allowed_dot_input_precisions = ()
        launch_cooperative_grid = False

    _Options.backend_name = backend_name

    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    ctx = ir.context()
    ir.load_dialects(ctx)
    module = src.make_ir(GPUTarget(backend_name, 0, 32), _Options(), {}, {},
                         ctx)
    # Keep the context alive: the module borrows from it.
    return module, ctx
