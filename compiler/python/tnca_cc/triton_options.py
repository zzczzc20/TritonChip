"""What this backend tells the Triton frontend about itself.

The frontend asks the target two questions before it will build a
`tt.dot` at all, and until R3.7 both were answered with an empty
placeholder in a shim that nobody had revisited:

    allowed_dot_input_precisions = ()      -> "input_precision must be
                                              one of (). Got ieee"
    codegen_fns = {}                       -> "target doesn't provide
                                              lower shape bounds for dot"

Both read like a missing hardware feature and were a missing callback.
There were two copies of that shim -- one in the product compile driver
and one in the test bridge -- so fixing the first left `tl.dot` refused
on the second, which is what this module exists to stop.
"""
# ISA sec 9C fixes the MMA fragment at 16x16 with K in {16, 32, 64}, so
# the smallest product the hardware performs is 16x16x16. Triton pads an
# operand up to these bounds rather than refusing it, which is what lets
# a kernel written for a smaller tile compile here unchanged.
MMA_MIN_M = MMA_MIN_N = MMA_MIN_K = 16

# Reduced input precision modes are accepted as permission to approximate.
# TNCA computes the stronger IEEE product with full FP32 vector FMAs when
# there is no native matrix input format; no input bits are discarded.
DOT_INPUT_PRECISIONS = ('ieee', 'tf32', 'tf32x3')


def min_dot_size(lhs_type, rhs_type):
    return (MMA_MIN_M, MMA_MIN_N, MMA_MIN_K)


CODEGEN_FNS = {'min_dot_size': min_dot_size}


def frontend_options(backend_name='tnca', **overrides):
    """The options object `ASTSource.make_ir` expects."""
    fields = dict(
        num_warps=1, num_ctas=1, num_stages=1,
        debug=False, sanitize_overflow=False, arch=None,
        allowed_dot_input_precisions=DOT_INPUT_PRECISIONS,
        supported_fp8_dtypes=('fp8e4nv', 'fp8e5'),
        deprecated_fp8_dot_operand_dtypes=(),
        max_num_imprecise_acc_default=0,
        launch_cooperative_grid=False, backend_name=backend_name,
    )
    fields.update(overrides)
    return type('TncaFrontendOptions', (), fields)()
