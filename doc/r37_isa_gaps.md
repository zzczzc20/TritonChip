# Gaps between the ISA and what is built

R3.7 compiled and ran all forty-five TileBench operators, and doing that
walked over parts of the ISA nothing had exercised before. Four of them
turned out not to say the same thing in every place they are written
down. Each is recorded here with what the specification says, what the
implementation does, and what the compiler does about it — because a
compiler that quietly works around a gap makes the gap permanent.

## 1. `t.ld.async` `mod[5]`: the transposed write is not implemented

**ISA sec 9E** — "`mod[2:0]` is the transfer's element-size log2;
`mod[4:3]` selects the window layout (`0` = flat row-major); `mod[5]`
requests a transposed write."

**`rtl/core/tnca_core.v`** passes `x_mod[2:0]` to the transfer engine
and nothing else:

```verilog
tnca_tensor #( ... ) u_tensor (
    ...
    .cmd_esz(x_mod[2:0]),
```

`rtl/core/tnca_tensor.v` has no port for a layout or a transpose and no
signal named for either. An emitted `t.ld.async.w.t` therefore assembles
cleanly, executes, and transfers the tile **untransposed**. It is not a
fault and there is no diagnostic: it is a wrong answer.

**What the compiler does.** It does not emit the bit. `tl.trans` is
lowered by reading the source down its columns with `v.ld.gather` — one
instruction per output row, addressing lane `i` at `Sa + Vb[i]*sz` per
sec 9.1 — which is correct on the machine that exists. The transposed
transfer would be cheaper and the lowering is ready for it; what is
missing is the hardware.

**What would close it.** Either the engine decodes `mod[5]`, or sec 9E
stops promising it and the bit becomes Reserved. The present state —
specified, encodable, ignored — is the one state that produces silently
wrong results.

`mod[4:3]`, the window layout selector, is in the same position: only
layout `0` exists, so nothing is lost today, but the field is not
checked either and a non-zero value would be accepted and ignored.

## 2. `fence` was marked absent everywhere while the machine ran it

**Fixed in this change.** Recorded because the shape of the mistake is
worth keeping.

`tools/isa/gen_target.py` carried:

```python
# Not a CAPABILITY0 bit: no fence encoding exists in v0.1 at all.
'fence': False,
```

while `tools/asm/tnca_isa.py` held the encoding (ATOM `0x20`),
`rtl/common/tnca_defs.vh` defined `TNCA_AT_FENCE`, `tnca_core.v`
decoded it into `stall_fence`, and `tb/core/progs/t14_caps.s` and
`t22_atomic.s` executed `fence.acq_rel.device` on the RTL. The
generated support matrix printed the contradiction on one line — every
profile `no`, machine-verified `yes` — and nothing read it.

`fence` has no CAPABILITY0 bit because it is **baseline**, the same
thing `s.add` is. The special case is gone and the ATOM family gate
applies.

## 3. There is no `dtype` code for a 32-bit integer

**ISA sec 9D.3** — "`dtype`: `0` = FP16, `1` = BF16, `2` = FP32, `3` =
INT8, others reserved."

A tile of `i32` — a block-sparse attention mask, an index map — cannot
be described by a TensorMap at all. There is no code to write in byte
109 and no honest way to borrow one: FP32 has the right width and says
something false about the contents.

**What the compiler does.** A tile whose element type is not one the
descriptor can name is loaded a row at a time instead. That needs no
descriptor and no unpacking: sec 9.1 gives lane `i` of a vector load
the address `base + i*sz`, so one `v.ld.w` is one row of sixteen. It
costs sixteen instructions where a transfer would have been one.

The same route carries INT8, for a different reason: INT8 *has* a code,
but a packed INT8 fragment cannot be read back apart because sec 8.4
reserves `v.unpack`.

**What would close it.** Allocating `4` = INT32 in sec 9D.3. The engine
already takes its element size from the instruction's `mod[2:0]` rather
than from the descriptor, so nothing in `tnca_tensor.v` would change.

## 4. Same-address store-then-load ordering is not stated

A kernel that exchanges values between lanes through memory — a bitonic
sort's compare-exchange partner, `x[i ^ span]` — stores a vector and
then loads from an address inside it. Nothing in the ISA says that load
observes the store.

sec 9.2 defines `fence` as waiting "until every access this core has
issued is complete at its home slice", which is sufficient, and
`tl.debug_barrier()` now lowers to it. But a fence is a heavy answer to
a question the memory model should answer directly, and today a kernel
that omits it is relying on something unwritten.

**What would close it.** A sentence in sec 9.1 saying whether the LSU
keeps program order between accesses to the same address from one RCTX.
Either answer is workable; the absence is not.

---

## Not a gap: things the compiler works around on purpose

For contrast, because these look similar and are not:

- **`v.pack.f16` / `v.unpack.f16` are Reserved** (sec 8.4). That is a
  stated deferral, not a contradiction, and the compiler repacks
  through memory — a per-lane 16-bit store writes 32 contiguous bytes
  a row, and sixteen of those are exactly the fragment of sec 9C.2.
- **There is no integer MMA** (sec 9C `mod[3:2]`). The compiler uses
  the FP16 unit and the result is *exactly* the integer product for
  K ≤ 1024, which is a proof rather than an approximation; above that
  it refuses.
- **There is no divide and no scalar floating point.** Both are
  deliberate, both are stated, and `mathlib.py` says what it does
  about the first and where a float lives for the second.
