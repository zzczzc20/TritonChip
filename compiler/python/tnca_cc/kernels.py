"""
Kernel constructors: the C1 and C2 cases, expressed in the TNCA IR.

compiler_overview.md sec 15 lists what each milestone must cover. These build
the IR for those cases; tnca_cc.codegen turns it into assembly, and
compiler/test/test_rtl_closure.py runs the result on the RTL.

They take an explicit description rather than TTIR because the Triton frontend
and the TNCA lowering fail differently, and sec 11.1 keeps them separable on
purpose.
"""
from .diagnostics import Diagnostic
from .ir import Builder, make_kernel


def elementwise(target, *, name='elementwise', inputs=('x', 'y'), output='out',
                op='add', dtype='f32', n_elements, addresses, masked=True,
                other=0, num_programs=None):
    """C1: vector add and masked elementwise."""
    vlen = target.vlen
    num_programs = num_programs or (n_elements + vlen - 1) // vlen

    k = make_kernel(name)
    b = Builder(k, k.role(0).region)

    b.comment(f"{name}: {op} over BLOCK={vlen}"
              + (", masked" if masked else ""))
    s_n = b.const(n_elements)
    ptrs = {a: b.const(addresses[a]) for a in list(inputs) + [output]}
    pid = b.const(0)

    with b.loop(pid, num_programs):
        b.comment("offs = pid * BLOCK + arange(0, BLOCK)")
        block = b.const(vlen)
        base = b.sbin('mul', pid, block, comment='pid * BLOCK')
        offs = b.vbin('add', b.arange(), b.splat(base), comment='offs')

        mask = None
        if masked:
            b.comment("mask = offs < n")
            mask = b.vcmp('lt', offs, b.splat(s_n))

        b.comment("The offsets are contiguous, so a plain vector load reaches\n"
                  "them: ISA sec 9.1 gives lane i the address base + i*sz. No\n"
                  "gather is needed, and emitting one would cost per-lane\n"
                  "addressing for nothing.")
        byte = b.sbini('shl', base, 2, comment='byte offset = pid*BLOCK*4')
        addrs = {a: b.sbin('add', ptrs[a], byte, comment=f"&{a}[pid*BLOCK]")
                 for a in list(inputs) + [output]}

        if masked:
            b.comment("other= is a value, not an absence (sec 12). The\n"
                      "masked-off lanes keep what the register already held,\n"
                      "so it has to be set first.")
        loaded = [b.load(addrs[a], mask=mask, other=other, dtype=dtype)
                  for a in inputs]
        acc = loaded[0]
        for v in loaded[1:]:
            acc = b.vbin(op, acc, v, dtype=dtype)
        b.store(addrs[output], acc, mask=mask)

    return k, b


def reduction(target, *, name='reduce', src='x', output='out', op='add',
              dtype='f32', n_elements, addresses, masked=True):
    """C1: integer and FP32 reduction.

    One block, reduced to a scalar. ISA sec 8.7 fixes the order as a balanced
    binary tree over lane index and makes inactive lanes contribute the
    identity, so the masked tail needs no pre-select -- which is the part a
    lowering is most likely to get wrong.
    """
    vlen = target.vlen
    if n_elements > vlen:
        raise Diagnostic(
            f"reducing {n_elements} elements needs "
            f"{(n_elements + vlen - 1) // vlen} vector registers",
            because='multi-register reductions are not lowered yet; the '
                    'cross-register combine has to keep the same fixed order '
                    'ISA sec 8.7 gives the in-register one, or the result '
                    'stops being bit-reproducible',
            remedy=f"use n <= {vlen} for now",
            source='compiler_overview.md sec 12')

    k = make_kernel(name)
    b = Builder(k, k.role(0).region)

    b.comment(f"{name}: {op} reduction over {n_elements} elements")
    s_n = b.const(n_elements)
    s_src = b.const(addresses[src])
    s_out = b.const(addresses[output])

    mask = None
    if masked:
        b.comment("mask = arange(0, BLOCK) < n")
        mask = b.vcmp('lt', b.arange(), b.splat(s_n))

    identity = 0 if op in ('add', 'or', 'xor') else 0
    v = b.load(s_src, mask=mask, other=identity, dtype=dtype)
    b.comment("ISA sec 8.7: inactive lanes contribute the identity, so the\n"
              "masked tail needs no pre-select, and the order is a fixed\n"
              "balanced tree so the result is bit-reproducible.")
    total = b.reduce(op, v, mask=mask, dtype=dtype)
    # Store the scalar result through a one-lane vector store.
    b.comment("store the scalar result")
    vt = b.splat(total, dtype=dtype)
    one = b.const(1)
    lane0 = b.vcmp('lt', b.arange(), b.splat(one))
    b.store(s_out, vt, mask=lane0)
    return k, b


def gather_kernel(target, *, name='gather', src='x', idx='i', output='out',
                  n_elements, addresses, dtype='f32', masked=True):
    """C1: the gather/scatter subset.

    ISA sec 9.1 scales a gather index by the access size, so the indices are
    element indices. Computing byte offsets is wrong for every lane but lane
    0 -- which is exactly the shape of bug a careless test survives.
    """
    vlen = target.vlen
    if n_elements > vlen:
        raise Diagnostic(f"gather of {n_elements} exceeds one register")

    k = make_kernel(name)
    b = Builder(k, k.role(0).region)
    b.comment(f"{name}: gather {n_elements} elements by index")
    s_n = b.const(n_elements)
    s_src = b.const(addresses[src])
    s_idx = b.const(addresses[idx])
    s_out = b.const(addresses[output])

    mask = b.vcmp('lt', b.arange(), b.splat(s_n)) if masked else None
    indices = b.load(s_idx, mask=mask, other=0, dtype='i32')
    values = b.gather(s_src, indices, mask=mask, other=0, dtype=dtype)
    b.store(s_out, values, mask=mask)
    return k, b
