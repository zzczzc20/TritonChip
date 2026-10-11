"""Split wide one-dimensional TTIR values into VLEN register chunks.

Views preserve scalar SSA identities and give every vector chunk its own
identity and type. Existing typed lowering handles each chunk, including
masked memory accesses and checked i64 addressing. This is a shared
compiler transformation; the incoming standard-path TTIR stays unchanged.
"""
import re


def _width(value):
    match = re.fullmatch(r'tensor<(\d+)x([^\d].*)>', str(value.get_type()))
    return int(match.group(1)) if match else None


class _ValueView:
    def __init__(self, value, chunk, vlen):
        self.value, self.chunk, self.vlen = value, chunk, vlen

    def id(self):
        width = _width(self.value)
        return ('register-chunk', self.value.id(), self.chunk) if width and width > self.vlen else self.value.id()

    def get_type(self):
        text = str(self.value.get_type())
        width = _width(self.value)
        if width and width > self.vlen:
            return re.sub(r'^tensor<\d+x', f'tensor<{self.vlen}x', text)
        return self.value.get_type()

    def __getattr__(self, name):
        return getattr(self.value, name)


class _OpView:
    def __init__(self, op, chunk, vlen):
        self.op, self.chunk, self.vlen = op, chunk, vlen

    def get_operand(self, i):
        return _ValueView(self.op.get_operand(i), self.chunk, self.vlen)

    def get_result(self, i):
        return _ValueView(self.op.get_result(i), self.chunk, self.vlen)

    def get_int_attr(self, name):
        value = self.op.get_int_attr(name)
        if self.op.get_name() == 'tt.make_range' and name in ('start', 'end'):
            start = self.op.get_int_attr('start') or 0
            return start + self.chunk * self.vlen + (self.vlen if name == 'end' else 0)
        return value

    def __getattr__(self, name):
        return getattr(self.op, name)

    def __str__(self):
        return str(self.op)


# Side effects and control flow require explicit semantics. Unrecognised
# operations fail rather than accidentally executing a region once per chunk.
_ALLOWED = {'tt.make_range', 'tt.splat', 'tt.addptr', 'tt.load', 'tt.store',
            'tt.broadcast', 'tt.bitcast', 'tt.fp_to_fp', 'tt.clamp', 'tt.reduce'}


def lower(imp, op):
    if getattr(imp, '_in_register_chunk', False):
        return False
    values = ([op.get_operand(i) for i in range(op.get_num_operands())] +
              [op.get_result(i) for i in range(op.get_num_results())])
    widths = {_width(v) for v in values if _width(v) and _width(v) > imp.target.vlen}
    if not widths:
        return False
    from .frontend import Unsupported
    if len(widths) != 1:
        raise Unsupported('register-block operation with mismatched widths')
    width = widths.pop()
    if width % imp.target.vlen or width > 128:
        raise Unsupported(f'register block width {width}; expected a VLEN multiple up to 128')
    widths_seen = getattr(imp, '_register_block_widths', set())
    widths_seen.add(width)
    imp._register_block_widths = widths_seen
    name = op.get_name()
    if name not in _ALLOWED and not name.startswith(('arith.', 'math.')):
        raise Unsupported(f'wide register block operation {name}')
    if name == 'tt.reduce':
        # Combining chunks must use the original region's exact operation;
        # defer unsupported folds and mixed outputs instead of guessing.
        raise Unsupported('wide register reduction; cross-chunk combinator is not implemented yet')
    imp._in_register_chunk = True
    try:
        for chunk in range(width // imp.target.vlen):
            view = _OpView(op, chunk, imp.target.vlen)
            # Decimal/hex float literals are indexed during the original
            # module walk; the virtual SSA identities retain that literal.
            for i in range(op.get_num_results()):
                original = op.get_result(i).id()
                if original in (imp._printed_consts or {}):
                    imp._printed_consts[view.get_result(i).id()] = imp._printed_consts[original]
            pure_predicate = (width // imp.target.vlen >= max(1, imp.target.pregs_per_rctx // 2) and
                              name.startswith('arith.') and
                              name != 'arith.constant' and op.get_num_results() == 1 and
                              str(op.get_result(0).get_type()).endswith('xi1>'))
            if pure_predicate:
                # Recompute pure lane masks at each use. Original operand
                # SSA values remain live; memory accesses keep their order.
                # This avoids carrying both eight boundary predicates and
                # eight select predicates in a sixteen-register predicate file.
                remat = getattr(imp, '_register_predicates', {})
                remat[view.get_result(0).id()] = view
                imp._register_predicates = remat
            else:
                imp._dispatch(view)
    finally:
        imp._in_register_chunk = False
    return True
