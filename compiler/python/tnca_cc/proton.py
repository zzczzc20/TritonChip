"""Proton record semantics using existing TNCA instructions, no RTL changes.

Single-execution markers only (loops may live inside a scope). Each
(program, role) owns a fixed record; no
atomic append or HBM store inside scopes. Timestamps stay in scalar
registers until the epilogue. Register exhaustion is an explicit refusal.
Invoke after normal optimization, before CodeGen. Input is never mutated.
"""
import copy
import math

from .diagnostics import Diagnostic
from .ir import Builder, SCALAR

SCHEMA = 'tnca-proton-record/1'


def lower_records(kernel, *, buffer_base, buffer_bytes, grid=(1, 1, 1), enabled=True):
    """Return (kernel, layout). Caller supplies disjoint fresh zeroed memory.

    Launch exactly one context per declared role per logical program and
    exactly this grid. Caller owns completion wait and CPU ownership handoff.
    Core clocks are reset-relative; no inter-core alignment is assumed.
    """
    out = copy.deepcopy(kernel)
    if not enabled:
        for role in out.roles:
            role.region.ops = [o for o in role.region.ops if o.kind != 'proton.record']
        return out, None
    if len(grid) != 3 or any(type(n) is not int or n < 1 for n in grid):
        raise ValueError('grid must contain three positive integers')
    if (type(buffer_base) is not int or buffer_base < 0 or buffer_base % 8
            or type(buffer_bytes) is not int or buffer_bytes < 0
            or buffer_base + buffer_bytes > 1 << 64):
        raise ValueError('capture buffer must be aligned and inside uint64 address space')
    layouts = []
    if not out.roles:
        raise Diagnostic('Proton capture needs at least one role')
    for role in out.roles:
        stack, seen, endpoints = [], set(), []
        ops = role.region.ops
        # Importer preserves tt.return and a conservative final exit. Drop
        # only unreachable trailing exits in the instrumented copy.
        while len(ops) > 1 and ops[-1].kind == ops[-2].kind == 'exit':
            ops.pop()
        if not ops or ops[-1].kind != 'exit' or any(o.kind == 'exit' for o in ops[:-1]):
            raise Diagnostic('Proton capture requires a single final exit')
        labels = {o.attrs['name']: i for i, o in enumerate(ops) if o.kind == 'label'}
        if (len(labels) != sum(o.kind == 'label' for o in ops)
                or any(n.startswith('__proton_') for n in labels)):
            raise Diagnostic('duplicate or reserved Proton label')
        for i, op in enumerate(ops):
            if op.kind == 'branch_nz':
                dest = labels.get(op.attrs['target'])
                if dest is None:
                    raise Diagnostic('Proton capture has an unknown branch target')
                if any(o.kind == 'proton.record' for o in ops[min(i, dest):max(i, dest) + 1]):
                    raise Diagnostic('branch skips or repeats proton.record; use coarser scopes '
                                     'outside the loop or a bounded dynamic event backend')
        for op in ops:
            if op.kind != 'proton.record':
                continue
            name, start = op.attrs['name'], op.attrs['isStart']
            if not isinstance(name, str) or not name or type(start) is not bool:
                raise Diagnostic('invalid proton.record attributes')
            if start:
                if name in seen:
                    raise Diagnostic(f'Proton scope name reused: {name!r}')
                stack.append(name)
                seen.add(name)
            elif not stack or stack.pop() != name:
                raise Diagnostic(f'Proton scope is not properly nested: {name!r}')
            endpoints.append(dict(name=name, start=start))
        if stack or not endpoints:
            raise Diagnostic('Proton capture requires nonempty balanced scopes in every role')
        layouts.append(dict(role=role.index, endpoints=endpoints))
    stride = 8 * (2 + max(len(r['endpoints']) for r in layouts))
    required = math.prod(grid) * len(layouts) * stride
    if required > buffer_bytes:
        raise Diagnostic(f'Proton buffer needs {required} bytes, has {buffer_bytes}; no truncation')
    for slot, (role, layout) in enumerate(zip(out.roles, layouts)):
        original = role.region.ops
        role.region.ops = []
        b = Builder(out, role.region)
        samples = []
        for op in original[:-1]:
            if op.kind != 'proton.record':
                role.region.ops.append(op)
                continue
            # Scheduler block boundaries prevent pure ops crossing the sample.
            b.label(f'__proton_{slot}_{len(samples)}_before')
            stamp = b._value(SCALAR, 'i64', hint='proton_clock')
            b._emit('proton_timestamp', [stamp])
            samples.append(stamp)
            b.label(f'__proton_{slot}_{len(samples)}_after')
        x, y, z = [b.program_id(i) for i in range(3)]
        # Refuse a mismatched launch before any capture memory is touched.
        # Additional contexts with the same role/program would race one slot.
        b.assert_eq(b.special('role_id'), role.index, 0x5052)
        for dim, expected in enumerate(grid):
            b.assert_eq(b.num_programs(dim), expected, 0x5053 + dim)
        linear = b.sbin('add', x, b.sbini('mul',
                        b.sbin('add', y, b.sbini('mul', z, grid[1])), grid[0]))
        address = b.sbin('add', b.const(buffer_base + slot * stride),
                        b.sbini('mul', linear, len(layouts) * stride))
        b.sstore(address, b.special('core_id'), offset=8)
        for i, stamp in enumerate(samples):
            b.sstore(address, stamp, offset=16 + i * 8)
        b.fence(order='release', scope='device')
        b.sstore(address, b.const(1), offset=0)
        b.exit()
    return out, dict(schema=SCHEMA, grid=list(grid), roles=layouts,
                     stride_bytes=stride, required_bytes=required, buffer_base=buffer_base,
                     clock='per-core-reset-relative-uint64',
                     timestamp_semantics='instruction-issue; async completion needs explicit wait',
                     store_policy='deferred-epilogue',
                     performance_status='diagnostic-only; overhead not-measured')
