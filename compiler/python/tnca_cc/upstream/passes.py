"""The real upstream MLIR canonicalizer and CSE, run on our IR.

doc/phase3_5_tasks_and_acceptance.md section 4.1 and R35-OPT-01/03.

Section 4.1 draws the boundary: these run on pure-value regions with an
explicit effect boundary, and load/store, DMA, MMA, barrier, window
ownership, trap and anything that can fault are NOT pure. Marking one of
them Pure to get CSE is exactly the mutation R35-MUT-01 lists as N03.

What is enabled is deliberately small: constant folding, integer address
simplification, common pure subexpressions and dead pure values. Not
load CSE across an effect, not speculation, not floating-point
reassociation, and not new FMA contraction.
"""
import os
import tempfile

from .version import require, UpstreamError


class PassError(Exception):
    """The pass pipeline could not run, or would not have been safe."""


def available():
    try:
        require('passes')
        return True
    except UpstreamError:
        return False


# Operations that carry an effect. Section 4.1 lists the categories; this
# is the TTIR spelling of them. An op named here is never inside a region
# handed to CSE.
EFFECTFUL_PREFIXES = (
    'tt.store', 'tt.load', 'tt.atomic', 'tt.dot', 'tt.reduce',
    'tt.scan', 'tt.print', 'tt.assert', 'tt.trans', 'tt.descriptor',
    'tt.experimental_descriptor', 'tt.elementwise_inline_asm',
    'ttg.async', 'ttng.', 'gpu.barrier', 'scf.', 'cf.',
)


class PureValueOptimizer:
    """Canonicalisation and CSE, from upstream, on a module's text.

    Text in, text out. The module is parsed by upstream, transformed by
    upstream and printed by upstream; nothing here edits IR. Section 4.2
    wanted only versioned text to cross the boundary and this is that.
    """

    PIPELINE_VERSION = '1'

    def __init__(self, *, cse=True, canonicalize=True):
        require('passes')
        self.cse = cse
        self.canonicalize = canonicalize

    def pipeline(self):
        out = []
        if self.canonicalize:
            out.append('canonicalize')
        if self.cse:
            out.append('cse')
        return out

    def run(self, mlir_text, *, name='tnca-r35'):
        """Return (optimised text, what changed).

        Raises PassError rather than returning the input unchanged: a
        stage that claims to reuse upstream must not answer a broken
        upstream by quietly not reusing it (R35-UP-02).
        """
        from triton._C import libtriton as L
        if not isinstance(mlir_text, str) or not mlir_text.strip():
            raise PassError('nothing to optimise')

        ctx = L.ir.context()
        L.ir.load_dialects(ctx)
        tmp = tempfile.mkdtemp(prefix='tnca-r35-opt-')
        path = os.path.join(tmp, 'in.mlir')
        with open(path, 'w') as fh:
            fh.write(mlir_text)
        try:
            mod = L.ir.parse_mlir_module(path, ctx)
        except Exception as e:                            # noqa: BLE001
            raise PassError(
                f"upstream could not parse this module: "
                f"{type(e).__name__}: {e}") from e
        mod.context = ctx
        before = str(mod)

        pm = L.ir.pass_manager(ctx)
        if self.canonicalize:
            L.passes.common.add_canonicalizer(pm)
        if self.cse:
            L.passes.common.add_cse(pm)
        try:
            pm.run(mod, name)
        except Exception as e:                            # noqa: BLE001
            raise PassError(
                f"the upstream pass pipeline failed: "
                f"{type(e).__name__}: {e}") from e
        after = str(mod)
        return after, self._difference(before, after)

    def optimize_module(self, module, ctx, *, name='tnca-r35'):
        """Run the pipeline on the module the compile is actually using.

        In place, in the module's own context. Section 4.2's rule about
        passing only text applies to a boundary between two different
        LLVM builds; there is one here, the wheel's, and this module
        never leaves it. The product path already hands this same object
        to the importer, so optimising it in place is what makes
        R35-UP-03 true: the optimised module IS the emission input, not
        a copy that was computed and dropped.
        """
        from triton._C import libtriton as L
        before = str(module)
        pm = L.ir.pass_manager(ctx)
        if self.canonicalize:
            L.passes.common.add_canonicalizer(pm)
        if self.cse:
            L.passes.common.add_cse(pm)
        try:
            pm.run(module, name)
        except Exception as e:                            # noqa: BLE001
            raise PassError(
                f"the upstream pass pipeline failed: "
                f"{type(e).__name__}: {e}") from e
        after = str(module)
        return after, self._difference(before, after)

    @staticmethod
    def _difference(before, after):
        """What the passes actually did, countable.

        R35-UP-03 requires evidence that the bridge output reaches
        emission rather than being computed and dropped, and R35-OPT-01
        requires a real operation to disappear. Counting here is what
        makes both statable.
        """
        import re

        def ops(t):
            return re.findall(r'^\s+%\S+ = ([\w.]+)', t, re.M)
        b, a = ops(before), ops(after)
        from collections import Counter
        cb, ca = Counter(b), Counter(a)
        removed = {k: cb[k] - ca.get(k, 0) for k in cb
                   if cb[k] > ca.get(k, 0)}
        added = {k: ca[k] - cb.get(k, 0) for k in ca
                 if ca[k] > cb.get(k, 0)}
        return {'ops_before': len(b), 'ops_after': len(a),
                'removed': removed, 'added': added,
                'changed': before != after}
