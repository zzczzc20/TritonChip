"""TNCA IR pure-value regions, optimised by the upstream passes.

doc/phase3_5_tasks_and_acceptance.md section 4.1:

    TNCA IR pure regions -> small MLIR bridge -> canonicalize/CSE -> TNCA IR

A TNCA IR source never had a TTIR, so the canonicalisation and CSE that
the Triton path gets for free never reach it. The GEMM is the reason
this matters: it materialises the same transfer coordinates for every
descriptor, and each one is a separate SSA value that nothing merges.

    s.movz s2, #0 ; s.movz s3, #0 ; s.mov s4, s2 ; s.mov s5, s3 ; t.ld.async
    s.movz s2, #0 ; s.movz s3, #0 ; s.mov s4, s2 ; s.mov s5, s3 ; t.ld.async

What crosses the bridge is a scalar expression forest: constants and
integer arithmetic over them, with no memory, no effect and no control
flow. That restriction is not a simplification, it is section 4.1's
rule. An operation that loads, stores, transfers, multiplies matrices,
arms a barrier, takes ownership of a window or can fault is never
emitted here, so upstream is never in a position to move one. Marking
one Pure to get it included is the mutation R35-MUT-01 calls N03.

The result is applied as an equivalence: upstream tells us which of our
values are the same value, and we rewrite the operands of everything
else to agree. No effectful operation is reordered, removed or
duplicated by this; the only thing that changes is which SSA value an
operand names, and afterwards a pure definition that nothing names is
dropped.
"""
from .version import require


class BridgeError(Exception):
    """The region could not be expressed, so it was left alone."""


# The pure scalar operations this bridge can express as arith ops.
# Anything else in a role's region is a barrier: values that flow into it
# are still defined, but it is not itself sent across.
#
# This is a narrower idea than "has no effect": a splat has no effect and
# still cannot cross, because it produces a vector and there is no
# arith op for it. The two lists are kept apart in optimization.py for
# exactly that reason.
PURE_KINDS = ('const', 'sbin', 'sbini')
BRIDGEABLE_KINDS = PURE_KINDS

# TNCA's scalar binary operations, and the arith op each one is.
_ARITH = {
    'add': 'arith.addi', 'sub': 'arith.subi', 'mul': 'arith.muli',
    'and': 'arith.andi', 'or': 'arith.ori', 'xor': 'arith.xori',
    'shl': 'arith.shli', 'shr': 'arith.shrui', 'sar': 'arith.shrsi',
}


def _is_pure(op):
    return op.kind in PURE_KINDS


class PureScalarBridge:
    """Ask upstream which of a region's pure scalar values coincide.

    The answer has to come from upstream, not from a Python function that
    happens to agree with it. Section 4.2 makes that point about
    LinearLayout -- an independent implementation is for checking, never
    for quietly standing in -- and the same applies to claiming CSE.

    So the forest is emitted as a function that RETURNS every one of our
    values. CSE merges the duplicate definitions and rewrites the uses,
    and the return operand list is then upstream's own statement of which
    of our values are the same value. Reading that list is reading the
    decision; nothing here re-derives it.
    """

    PROTOCOL_VERSION = '2'

    def __init__(self):
        require('passes')

    def _emit(self, ops):
        """The pure forest as a function returning all of it.

        i64 throughout: TNCA scalars are 64-bit, and narrowing here to
        make the text shorter would change what upstream folds.

        Every value is returned, which keeps them all live. Without that,
        canonicalisation would delete the lot as dead and the answer
        would be an empty module.
        """
        body, name, order = [], {}, []
        for op in ops:
            r = op.results[0]
            v = f"%v{len(order)}"
            if op.kind == 'const':
                body.append(f"    {v} = arith.constant "
                            f"{int(op.attrs['value'])} : i64")
            elif op.kind == 'sbin':
                a, b = op.operands
                mn = _ARITH.get(op.attrs.get('op'))
                if mn is None or id(a) not in name or id(b) not in name:
                    raise BridgeError(
                        f"cannot express sbin {op.attrs.get('op')!r}")
                body.append(f"    {v} = {mn} {name[id(a)]}, "
                            f"{name[id(b)]} : i64")
            elif op.kind == 'sbini':
                a, = op.operands
                mn = _ARITH.get(op.attrs.get('op'))
                if mn is None or id(a) not in name:
                    raise BridgeError(
                        f"cannot express sbini {op.attrs.get('op')!r}")
                c = f"%k{len(order)}"
                body.append(f"    {c} = arith.constant "
                            f"{int(op.attrs['imm'])} : i64")
                body.append(f"    {v} = {mn} {name[id(a)]}, {c} : i64")
            else:
                raise BridgeError(f"{op.kind} is not a pure scalar operation")
            name[id(r)] = v
            order.append(r)

        # tt.func, not func.func: the context these passes run in is
        # Triton's, and it registers the Triton dialect. `func` is not
        # loaded there and parsing fails on it.
        types = ', '.join(['i64'] * len(order))
        rets = ', '.join(name[id(r)] for r in order)
        text = ('module {\n'
                f"  tt.func public @pure() -> ({types}) {{\n"
                + '\n'.join(body) + '\n'
                f"    tt.return {rets} : {types}\n"
                '  }\n'
                '}\n')
        return text, order

    @staticmethod
    def _returned(text):
        """The operand list of the optimised function's return."""
        import re
        m = re.search(r'\breturn\s+([^:\n]*?)\s*:', text)
        if not m:
            raise BridgeError('the optimised module has no return to read')
        return [x.strip() for x in m.group(1).split(',') if x.strip()]

    def equivalence(self, ops):
        """{id(value): the value upstream says it is}, and what changed.

        Two of our values that upstream proved equal come back as the
        same SSA name in the return list, so they map to one
        representative -- the first of them, which is the definition CSE
        kept.
        """
        text, order = self._emit(ops)
        from .passes import PureValueOptimizer
        opt = PureValueOptimizer()
        after, diff = opt.run(text, name='tnca-pure')
        names = self._returned(after)
        if len(names) != len(order):
            raise BridgeError(
                f"upstream returned {len(names)} value(s) for "
                f"{len(order)} -- the bridge protocol and the module "
                f"disagree")
        rep, out = {}, {}
        for value, mlir_name in zip(order, names):
            rep.setdefault(mlir_name, value)
            out[id(value)] = rep[mlir_name]
        merged = sum(1 for v in order if out[id(v)] is not v)
        return out, {'values': len(order), 'merged': merged,
                     'distinct': len(rep),
                     'ops_before': diff['ops_before'],
                     'ops_after': diff['ops_after'],
                     'removed': diff['removed'],
                     'protocol': self.PROTOCOL_VERSION,
                     'pipeline': opt.pipeline()}
