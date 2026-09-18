# TNCA ISA — Version 0.1 (Stage 1 Frozen Subset)

**Status:** Stage-1 normative. Families marked *Reserved* are allocated but not
yet defined; encoding into them raises `ILLEGAL_INSTRUCTION`.

This document is the single source of truth for instruction encoding. RTL
(`rtl/common/tnca_defs.vh`), the assembler (`tools/asm/tnca_asm.py`) and the
architectural model (`tools/model/tnca_model.py`) are all derived from it. If
they disagree, this document wins and the others are bugs.

Derived from `doc/architecture_overview.md` §11–§12. Section references below
point at that document.

---

## 1. Architectural state

### 1.1 Per Role Context (RCTX)

| State | Count | Width | Notes |
|---|---:|---:|---|
| `PC` | 1 | 64 | 8-byte aligned; low 3 bits RAZ/WI |
| `S0..S63` | 64 | 64 | `S0` reads as zero, writes discarded |
| `P0..P15` | 16 | 16 | `P0` reads as all-ones, writes discarded |
| Predicate-negate | — | — | encoded per instruction, not state |

`V` (vector) and `A` (accumulator) registers are **core-wide physical
resources** with a per-RCTX logical view established at launch (§11.2). In
Stage 1 the mapping is the identity over a statically partitioned range; the
window/rename machinery arrives in Stage 2.

### 1.2 Per core

| State | Count | Width | Notes |
|---|---:|---:|---|
| `V0..V255` (logical) | ≤256/RCTX | 16 lanes × 32 b | physical pool is shared |
| `A0..A15` | 16 | 16×16 × FP32 | Stage 2 |
| `W0..W31` | 32 | — | RF window handles, Stage 2 |
| `D0..D15` | 16 | — | TensorMap snapshots, Stage 2 |
| `B0..B31` | 32 | — | mbarriers, Stage 2 |

**Vector lane count `VLEN` is 16 and is architecturally visible** via the
`NUM_LANES` special register. Element width for `V` operations is 32 bits in
Stage 1; packed FP16/BF16 views arrive with the convert family.

---

## 2. Instruction format

All instructions are 64 bits, stored little-endian, 8-byte aligned. An
instruction with `X = 1` is followed by one additional 64-bit **extension
word**; the pair is a single 128-bit instruction and a branch may only target
its first byte (§11.4).

```
 63      56 55      48 47      40 39      32
+----------+----------+----------+----------+
|  major   |   func   |    d     |    a     |
+----------+----------+----------+----------+
 31      24 23      16 15 12 11 10 9        0
+----------+----------+-----+--+--+----------+
|    b     |    c     |pred |X |PN|   mod    |
+----------+----------+-----+--+--+----------+
```

| Field | Bits | Meaning |
|---|---|---|
| `major` | 63:56 | instruction family (§3) |
| `func` | 55:48 | operation within the family |
| `d` | 47:40 | destination register index |
| `a` | 39:32 | source register index 0 |
| `b` | 31:24 | source register index 1 |
| `c` | 23:16 | source register index 2 |
| `pred` | 15:12 | predicate register selector |
| `X` | 11 | 1 ⇒ a 64-bit extension word follows |
| `PN` | 10 | 1 ⇒ use `~P[pred]` instead of `P[pred]` |
| `mod` | 9:0 | family-specific modifier |

The register *class* of `d`/`a`/`b`/`c` is determined entirely by
(`major`, `func`) — there is no type field (§11.4).

### 2.1 Immediate assembly

| Name | Source bits | Value |
|---|---|---|
| `imm16` | `{b, c}` = 31:16 | as specified per instruction (sign- or zero-extended) |
| `imm24` | `{a, b, c}` = 39:16 | sign-extended to 64 |
| `imm64` | extension word | requires `X = 1` |

### 2.2 Predication rule (normative)

* **Scalar (`0x01`), control (`0x02`) and system (`0x00`) instructions are
  unpredicated.** They must encode `pred = 0` and `PN = 0`. Any other encoding
  raises `ILLEGAL_INSTRUCTION`. This keeps uniform control flow explicitly
  scalar, per §12.1 ("不建议实现通用 SIMT divergence stack").
* **Vector, predicate, memory and tensor instructions** use `pred`/`PN` as a
  16-bit lane mask. `P0` is architecturally all-ones, so `pred = 0, PN = 0` is
  the unpredicated form and `pred = 0, PN = 1` is the never-execute form.
* **Masked write semantics:** inactive lanes *preserve the previous value* of
  the destination register (§12.3). Loads are the exception: an inactive lane
  performs **no memory access at all** and its destination lane keeps its old
  value; materialising Triton's `other=` operand is the compiler's job, via an
  explicit `v.sel` (§12.3).
* A fully-masked-off instruction still retires and still writes its completion
  event, if it has one (§13.4).

### 2.3 Reserved-bit rule

Every field not assigned a meaning by the (`major`, `func`) pair **must be
zero**. Non-zero reserved bits raise `ILLEGAL_INSTRUCTION`. This preserves the
whole encoding space for later versions instead of letting assemblers bake in
don't-cares.

---

## 3. Major opcode map

| `major` | Family | Stage |
|---|---|---|
| `0x00` | System / special registers | 1 |
| `0x01` | Scalar integer | 1 |
| `0x02` | Control flow | 1 |
| `0x10` | Vector integer | 1 |
| `0x11` | Vector floating point | 1 |
| `0x12` | Convert / pack | 1 (partial) |
| `0x13` | Predicate | 1 |
| `0x14` | Shuffle / permute | 1 |
| `0x15` | Reduction | 1 |
| `0x16` | SFU | 1 |
| `0x20` | Load / store | 1 |
| `0x21` | Atomic / fence | 2 *(Reserved)* |
| `0x30` | Tensor descriptor | 2 *(Reserved)* |
| `0x31` | Tensor transfer | 2 *(Reserved)* |
| `0x32` | Tensor / cache / flow hints | 4 *(Reserved)* |
| `0x40` | Matrix | 2 *(Reserved)* |
| `0x41` | Accumulator access | 2 *(Reserved)* |
| `0x50` | Mbarrier | 2 *(Reserved)* |
| `0x51` | Role control | 2 *(Reserved)* |
| `0x60` | Debug / performance | 2 *(Reserved)* |

All other `major` values are *Reserved*.

---

## 4. Condition codes

Used by `s.cmp`, `s.cmpi`, `v.cmp` in `mod[3:0]`, and by `vf.cmp` with the
floating-point interpretation.

| Code | Mnemonic | Integer | Floating point |
|---:|---|---|---|
| `0x0` | `eq` | `a == b` | ordered and equal |
| `0x1` | `ne` | `a != b` | unordered or not equal |
| `0x2` | `lt` | signed `<` | ordered and `<` |
| `0x3` | `le` | signed `<=` | ordered and `<=` |
| `0x4` | `gt` | signed `>` | ordered and `>` |
| `0x5` | `ge` | signed `>=` | ordered and `>=` |
| `0x6` | `ltu` | unsigned `<` | — |
| `0x7` | `leu` | unsigned `<=` | — |
| `0x8` | `gtu` | unsigned `>` | — |
| `0x9` | `geu` | unsigned `>=` | — |
| `0xa` | `o` | — | ordered (neither is NaN) |
| `0xb` | `uo` | — | unordered (either is NaN) |

Codes `0x6..0x9` are illegal for `vf.cmp`; `0xa..0xb` are illegal for integer
compares.

---

## 5. `major = 0x00` — System

`d`/`a` are scalar registers. `mod[9:0]` carries the special-register id.

| `func` | Mnemonic | Semantics |
|---:|---|---|
| `0x00` | `nop` | no architectural effect; all other fields must be 0 |
| `0x01` | `s.rdspecial Sd, #id` | `Sd = SPECIAL[id]` |
| `0x02` | `s.wrspecial #id, Sa` | `SPECIAL[id] = Sa`, writable ids only |

### 5.1 Special register ids

| id | Name | R/W | Meaning |
|---:|---|---|---|
| `0x000` | `PROGRAM_ID_X` | R | Triton `program_id(0)` |
| `0x001` | `PROGRAM_ID_Y` | R | |
| `0x002` | `PROGRAM_ID_Z` | R | |
| `0x004` | `NUM_PROGRAMS_X` | R | Triton `num_programs(0)` |
| `0x005` | `NUM_PROGRAMS_Y` | R | |
| `0x006` | `NUM_PROGRAMS_Z` | R | |
| `0x008` | `ROLE_ID` | R | this RCTX's index, `0..NUM_ROLES-1` |
| `0x009` | `NUM_ROLES` | R | active RCTX count for this launch |
| `0x00a` | `CORE_ID` | R | tile index |
| `0x00b` | `NUM_CORES` | R | |
| `0x00c` | `NUM_LANES` | R | `VLEN`, architecturally 16 in v0.1 |
| `0x00d` | `NUM_VREGS` | R | logical `V` registers this implementation gives each RCTX; naming one beyond it raises `ILLEGAL_INSTRUCTION` |
| `0x010` | `CLOCK` | R | free-running core cycle counter |
| `0x011` | `INSTRET` | R | instructions retired by this RCTX |
| `0x020` | `ARG_BASE` | R | kernel argument buffer base address |
| `0x030` | `ISA_VERSION` | R | `{major[15:0], minor[15:0]}` = `0x0000_0001` |
| `0x031` | `CAPABILITY0` | R | feature bitmap, §11 |
| `0x040` | `TRAP_PC` | R | PC of the faulting instruction |
| `0x041` | `TRAP_CODE` | R | trap cause, §10 |

Reads of unimplemented ids return 0; writes to read-only ids raise
`ILLEGAL_INSTRUCTION` rather than being silently dropped.

---

## 6. `major = 0x01` — Scalar integer

`d`, `a`, `b`, `c` are scalar register indices (`[5:0]` significant, `[7:6]`
must be zero). All operations are on 64-bit values unless stated.

| `func` | Mnemonic | Semantics |
|---:|---|---|
| `0x00` | `s.mov Sd, Sa` | `Sd = Sa` |
| `0x01` | `s.add Sd, Sa, Sb` | `Sd = Sa + Sb` (wrapping) |
| `0x02` | `s.sub Sd, Sa, Sb` | `Sd = Sa - Sb` |
| `0x03` | `s.mul Sd, Sa, Sb` | `Sd = (Sa * Sb) mod 2^64` |
| `0x04` | `s.mulh.s Sd, Sa, Sb` | high 64 of signed 128-bit product |
| `0x05` | `s.mulh.u Sd, Sa, Sb` | high 64 of unsigned 128-bit product |
| `0x08` | `s.and` | |
| `0x09` | `s.or` | |
| `0x0a` | `s.xor` | |
| `0x0b` | `s.andn Sd, Sa, Sb` | `Sd = Sa & ~Sb` |
| `0x0c` | `s.shl Sd, Sa, Sb` | shift amount `Sb[5:0]` |
| `0x0d` | `s.lshr` | logical right |
| `0x0e` | `s.ashr` | arithmetic right |
| `0x10` | `s.min.s` | |
| `0x11` | `s.max.s` | |
| `0x12` | `s.min.u` | |
| `0x13` | `s.max.u` | |
| `0x14` | `s.clz Sd, Sa` | count leading zeros; `clz(0) = 64` |
| `0x15` | `s.popc Sd, Sa` | population count |
| `0x16` | `s.sext.w Sd, Sa` | sign-extend `Sa[31:0]` |
| `0x17` | `s.zext.w Sd, Sa` | zero-extend `Sa[31:0]` |
| `0x18` | `s.cmp Sd, Sa, Sb` | `Sd = (Sa cc Sb) ? 1 : 0`, `cc = mod[3:0]` |
| `0x19` | `s.sel Sd, Sa, Sb, Sc` | `Sd = (Sc != 0) ? Sa : Sb` |
| `0x20` | `s.addi Sd, Sa, #imm16` | `Sd = Sa + sext(imm16)` |
| `0x21` | `s.subi Sd, Sa, #imm16` | `Sd = Sa - sext(imm16)` |
| `0x22` | `s.muli Sd, Sa, #imm16` | `Sd = Sa * sext(imm16)` |
| `0x28` | `s.andi Sd, Sa, #imm16` | `Sd = Sa & zext(imm16)` |
| `0x29` | `s.ori` | zero-extended immediate |
| `0x2a` | `s.xori` | zero-extended immediate |
| `0x2c` | `s.shli Sd, Sa, #imm16` | shift amount `imm16[5:0]` |
| `0x2d` | `s.lshri` | |
| `0x2e` | `s.ashri` | |
| `0x38` | `s.cmpi Sd, Sa, #imm16` | `cc = mod[3:0]`, immediate sign-extended |
| `0x40` | `s.movz Sd, #imm16, lsl` | `Sd = zext(imm16) << (16 * mod[1:0])` |
| `0x41` | `s.movk Sd, #imm16, lsl` | replace that 16-bit field of `Sd`, keep the rest |
| `0x42` | `s.movn Sd, #imm16, lsl` | `Sd = ~(zext(imm16) << (16 * mod[1:0]))` |

`s.div` / `s.rem` are deliberately absent: expanded by the compiler (§12.1, "L").

---

## 7. `major = 0x02` — Control flow

Branch displacements are in **instruction words**: the target is
`PC + 8 * sext(offset)`, where `PC` is the address of the branch itself.

| `func` | Mnemonic | Fields | Semantics |
|---:|---|---|---|
| `0x00` | `br #off24` | `off24 = {a,b,c}` | `PC += 8*sext(off24)` |
| `0x01` | `br.nz Sa, #off16` | `a`, `off16 = {b,c}` | if `Sa != 0` |
| `0x02` | `br.z Sa, #off16` | `a`, `off16 = {b,c}` | if `Sa == 0` |
| `0x04` | `br.r Sa` | `a` | `PC = Sa & ~7` |
| `0x08` | `call Sd, #off24` | `d`, `off24` | `Sd = PC + 8`; `PC += 8*sext(off24)` |
| `0x09` | `ret Sa` | `a` | `PC = Sa & ~7` |
| `0x10` | `yield` | — | relinquish the issue slot for ≥1 cycle; no architectural effect |
| `0x18` | `trap #imm24` | `off24` | raise `TRAP_USER` with code `imm24` |
| `0x1f` | `exit` | — | terminate this RCTX (§18.3) |

`exit` does **not** release resources; a program is complete only when every
role has exited *and* every outstanding asynchronous command has completed
(§18.3).

---

## 8. Vector families

`d`, `a`, `b`, `c` index `V` registers (8 bits) except where noted. All Stage-1
vector operations are 16 lanes × 32 bits. `pred`/`PN` select the lane mask;
inactive lanes preserve the destination's previous value (§2.2).

### 8.1 `major = 0x10` — Vector integer

| `func` | Mnemonic | Semantics |
|---:|---|---|
| `0x00` | `v.mov Vd, Va` | |
| `0x01` | `v.add` | |
| `0x02` | `v.sub` | |
| `0x03` | `v.mul` | low 32 bits |
| `0x04` | `v.mulh.s` | high 32 of signed product |
| `0x05` | `v.mulh.u` | high 32 of unsigned product |
| `0x08` | `v.and` | |
| `0x09` | `v.or` | |
| `0x0a` | `v.xor` | |
| `0x0b` | `v.andn` | |
| `0x0c` | `v.shl` | shift amount `Vb[4:0]` |
| `0x0d` | `v.lshr` | |
| `0x0e` | `v.ashr` | |
| `0x10` | `v.min.s` | |
| `0x11` | `v.max.s` | |
| `0x12` | `v.min.u` | |
| `0x13` | `v.max.u` | |
| `0x14` | `v.clz` | |
| `0x15` | `v.popc` | |
| `0x18` | `v.cmp Pd, Va, Vb` | **`d` is a predicate index.** `Pd[i] = mask[i] & (Va[i] cc Vb[i])`; inactive lanes yield 0 |
| `0x19` | `v.sel Vd, Va, Vb` | `Vd[i] = mask[i] ? Va[i] : Vb[i]`, **all** lanes written |
| `0x1a` | `v.splat Vd, Sa` | `Vd[i] = Sa[31:0]` |
| `0x1b` | `v.iota Vd` | `Vd[i] = i` |
| `0x20` | `v.addi Vd, Va, #imm16` | `sext(imm16)` |
| `0x21` | `v.subi` | |
| `0x22` | `v.muli` | |
| `0x28` | `v.andi` | `zext(imm16)` |
| `0x29` | `v.ori` | |
| `0x2a` | `v.xori` | |
| `0x2c` | `v.shli Vd, Va, #imm16` | `imm16[4:0]` |
| `0x2d` | `v.lshri` | |
| `0x2e` | `v.ashri` | |
| `0x38` | `v.cmpi Pd, Va, #imm16` | `cc = mod[3:0]` |
| `0x40` | `v.movi Vd, #imm16` | `Vd[i] = sext(imm16)` |

`v.sel` is deliberately exempt from the masked-write rule: it *is* the select,
so masking its own write would make it useless.

### 8.2 `major = 0x11` — Vector floating point (binary32)

Default rounding is round-to-nearest-even. Subnormals are handled (no flush to
zero) unless `mod[8]` (`.ftz`) is set. FTZ and approximation are **explicit
modes, never implicit implementation choices** (§12.4).

| `func` | Mnemonic | Semantics |
|---:|---|---|
| `0x01` | `vf.add Vd, Va, Vb` | |
| `0x02` | `vf.sub` | |
| `0x03` | `vf.mul` | |
| `0x06` | `vf.fma Vd, Va, Vb, Vc` | `Vd = fma(Va, Vb, Vc)`, single rounding |
| `0x07` | `vf.fms Vd, Va, Vb, Vc` | `Vd = fma(Va, Vb, -Vc)` |
| `0x10` | `vf.min Vd, Va, Vb` | NaN behaviour from `mod[1:0]`, §8.3 |
| `0x11` | `vf.max` | |
| `0x14` | `vf.neg Vd, Va` | sign bit flip; NaN payload preserved |
| `0x15` | `vf.abs Vd, Va` | sign bit clear |
| `0x18` | `vf.cmp Pd, Va, Vb` | `cc = mod[3:0]`, FP interpretation, §4 |

### 8.3 Floating-point min/max modes (`mod[1:0]`)

Stated explicitly because leaving it to lowering is a known correctness hazard
(§12.2).

| `mod[1:0]` | Name | NaN | Signed zero |
|---:|---|---|---|
| `0` | `.ieee` | IEEE 754-2019 `minimumNumber`: NaN operand ignored, both NaN ⇒ qNaN | `min(+0,-0) = -0`, `max(+0,-0) = +0` |
| `1` | `.prop` | any NaN operand ⇒ qNaN | as `.ieee` |
| `2` | `.fast` | unspecified for NaN | unspecified for ±0 |
| `3` | — | *Reserved* | |

### 8.4 `major = 0x12` — Convert / pack

| `func` | Mnemonic | Semantics | Stage |
|---:|---|---|---|
| `0x00` | `vf.cvt.s32.f32 Vd, Va` | signed int → FP32, RNE | 1 |
| `0x01` | `vf.cvt.u32.f32` | unsigned int → FP32, RNE | 1 |
| `0x02` | `vf.cvt.f32.s32` | FP32 → signed int, round mode `mod[2:0]` | 1 |
| `0x03` | `vf.cvt.f32.u32` | FP32 → unsigned int | 1 |
| `0x08` | `v.pack.f16` | *Reserved* | 2 |
| `0x09` | `v.unpack.f16` | *Reserved* | 2 |
| `0x0a` | `v.pack.bf16` | *Reserved* | 2 |
| `0x0b` | `v.unpack.bf16` | *Reserved* | 2 |

FP→int rounding modes in `mod[2:0]`: `0` = toward zero, `1` = RNE,
`2` = toward −∞, `3` = toward +∞. Out-of-range and NaN inputs saturate to the
integer extremes; NaN maps to 0.

### 8.5 `major = 0x13` — Predicate

`d`, `a`, `b` index predicate registers except where noted. These instructions
are themselves **unpredicated** (`pred = 0`, `PN = 0`) to avoid a mask needing a
mask.

| `func` | Mnemonic | Semantics |
|---:|---|---|
| `0x00` | `p.mov Pd, Pa` | |
| `0x01` | `p.and Pd, Pa, Pb` | |
| `0x02` | `p.or` | |
| `0x03` | `p.xor` | |
| `0x04` | `p.not Pd, Pa` | |
| `0x05` | `p.andn` | |
| `0x10` | `p.any Sd, Pa` | **`d` is scalar.** `Sd = |Pa` |
| `0x11` | `p.all Sd, Pa` | `Sd = &Pa` |
| `0x12` | `p.ballot Sd, Pa` | `Sd = zext(Pa[15:0])` |
| `0x13` | `p.frombits Pd, Sa` | **`a` is scalar.** `Pd = Sa[15:0]` |
| `0x14` | `p.popc Sd, Pa` | `Sd = popcount(Pa)` |

### 8.6 `major = 0x14` — Shuffle / permute

Lane indices are taken modulo `VLEN`.

| `func` | Mnemonic | Semantics |
|---:|---|---|
| `0x00` | `v.shuffle.xor Vd, Va, Sb` | `Vd[i] = Va[i ^ Sb[3:0]]` |
| `0x01` | `v.shuffle.up Vd, Va, Sb` | `Vd[i] = (i >= n) ? Va[i-n] : Va[i]`, `n = Sb[3:0]` |
| `0x02` | `v.shuffle.down Vd, Va, Sb` | `Vd[i] = (i+n < VLEN) ? Va[i+n] : Va[i]` |
| `0x03` | `v.permute Vd, Va, Vb` | `Vd[i] = Va[Vb[i][3:0]]` |
| `0x04` | `v.rotate Vd, Va, Sb` | `Vd[i] = Va[(i + Sb[3:0]) mod VLEN]` |
| `0x08` | `v.extract Sd, Va, #imm` | **`d` scalar.** `Sd = zext(Va[mod[3:0]])` |
| `0x09` | `v.insert Vd, Sa, #imm` | `Vd[mod[3:0]] = Sa[31:0]`, other lanes preserved |

Shuffles read **all** source lanes regardless of the mask; the mask applies
only to the destination write.

### 8.7 `major = 0x15` — Reduction

`mod[0] = 0` ⇒ destination is scalar `Sd`; `mod[0] = 1` ⇒ destination is a
vector register `Vd` with the result broadcast to every lane (§12.6).

Inactive lanes contribute the operation's identity element.

| `func` | Mnemonic | Identity |
|---:|---|---|
| `0x00` | `v.reduce.add` | `0` |
| `0x01` | `v.reduce.max.s` | `INT32_MIN` |
| `0x02` | `v.reduce.min.s` | `INT32_MAX` |
| `0x03` | `v.reduce.and` | all-ones |
| `0x04` | `v.reduce.or` | `0` |
| `0x05` | `v.reduce.xor` | `0` |
| `0x06` | `v.reduce.max.u` | `0` |
| `0x07` | `v.reduce.min.u` | `UINT32_MAX` |
| `0x08` | `vf.reduce.add` | `+0.0` |
| `0x09` | `vf.reduce.max` | `-inf` |
| `0x0a` | `vf.reduce.min` | `+inf` |

**`vf.reduce.add` reduction order is architecturally fixed** as a balanced
binary tree over lane index — `((l0+l1)+(l2+l3)) + ...` — so the result is
bit-reproducible across implementations. An implementation may not substitute a
different association order.

### 8.8 `major = 0x16` — SFU

Approximate, single-precision. Accuracy is a published contract, not an
implementation detail (§12.7).

**Not implemented in Stage 1a.** `CAPABILITY0.SFU` reads 0 and these encodings
raise `ILLEGAL_INSTRUCTION`; the error bounds below are the target the Stage-1b
implementation must be measured against before the bit may be set. A capability
bit is a promise about a native datapath, never a claim that something "works,
but slowly, by some other route" (§11).

| `func` | Mnemonic | Max relative error | Special values |
|---:|---|---|---|
| `0x00` | `v.exp2.approx` | ≤ 2⁻²² over `[-126, 126]` | `exp2(-inf)=+0`, `exp2(+inf)=+inf`, `exp2(NaN)=qNaN` |
| `0x01` | `v.rcp.approx` | ≤ 2⁻²² | `rcp(±0)=±inf`, `rcp(±inf)=±0` |
| `0x02` | `v.rsqrt.approx` | ≤ 2⁻²² | `rsqrt(+0)=+inf`, `rsqrt(x<0)=qNaN`, `rsqrt(+inf)=+0` |
| `0x03` | `v.log2.approx` | ≤ 2⁻²² abs. for `[0.5,2]`, else rel. | `log2(±0)=-inf`, `log2(x<0)=qNaN` |

Subnormal inputs are flushed to zero of the same sign by all SFU operations;
subnormal results are flushed to zero. This is a *stated* deviation, not a
silent one, and the strict paths live in the math library.

---

## 9. Memory families

### 9.1 `major = 0x20` — Load / store

`mod` layout:

| bits | Meaning |
|---|---|
| `2:0` | access size: `0`=1 B, `1`=2 B, `2`=4 B, `3`=8 B, `4..7` *Reserved* |
| `3` | `1` ⇒ sign-extend loaded value (integer loads only) |
| `5:4` | cache hint: `0` default, `1` streaming/no-allocate, `2` high priority, `3` *Reserved* |
| `9:6` | *Reserved*, must be 0 |

Cache hints are **performance-only**: no hint may change the address accessed,
the value returned, or the ordering observed (§1.4).

| `func` | Mnemonic | Fields | Address |
|---:|---|---|---|
| `0x00` | `s.ld Sd, [Sa, #imm16]` | `d`=dest, `a`=base, `{b,c}`=imm16 | `Sa + sext(imm16)` |
| `0x08` | `s.st [Sa, #imm16], Sd` | `d`=**data**, `a`=base, `{b,c}`=imm16 | `Sa + sext(imm16)` |
| `0x10` | `v.ld Vd, [Sa, #imm16]` | `d`=dest, `a`=base, `{b,c}`=imm16 | lane `i` ⇒ `… + i*sz` |
| `0x18` | `v.st [Sa, #imm16], Vd` | `d`=**data**, `a`=base, `{b,c}`=imm16 | lane `i` ⇒ `… + i*sz` |
| `0x20` | `v.ld.gather Vd, [Sa, Vb]` | `d`=dest, `a`=base, `b`=index | lane `i` ⇒ `Sa + sext(Vb[i]) * sz` |
| `0x28` | `v.st.scatter [Sa, Vb], Vd` | `d`=**data**, `a`=base, `b`=index | same |

**Stores put their data register in field `d`, not field `b`.** Field `b` is
part of the 16-bit displacement, so a store cannot name its source there; `d` is
the only field a store leaves free. The assembler still writes the data operand
last (`s.st.d [s1, #8], s2`), matching the load form's left-to-right reading.

Vector loads narrower than 32 bits write the low bits of each lane and extend
per `mod[3]`. Sizes wider than 32 bits are *Reserved* for vector forms in v0.1.

Unaligned accesses are split by the LSU and are architecturally legal but may
be slow. A store that partially covers a word must leave the uncovered bytes
unchanged (§12.8).

### 9.2 `major = 0x21` — Atomic / fence *(Reserved until Stage 2)*

The encoding below is allocated and normative, but no Stage-1 implementation
decodes it: `CAPABILITY0.ATOMICS_DEVICE` reads 0 and every encoding in this
family raises `ILLEGAL_INSTRUCTION`. Atomics execute at the home slice of the
physical address (§12.9), which does not exist until the NUCA slices land in
Stage 3; shipping a single-core stand-in would have committed the ISA to
semantics the real implementation could not keep. `mod`:

| bits | Meaning |
|---|---|
| `2:0` | size: `2` = 4 B, `3` = 8 B; others *Reserved* |
| `4:3` | memory order: `0` relaxed, `1` acquire, `2` release, `3` acq_rel |
| `6:5` | scope: `0` core, `1` device, `2` system, `3` *Reserved* |
| `9:7` | *Reserved* |

`system` scope is only legal when `CAPABILITY0.SYSTEM_ATOMICS` is set;
otherwise it raises `ILLEGAL_INSTRUCTION` rather than silently degrading
(§12.9).

| `func` | Mnemonic | Semantics |
|---:|---|---|
| `0x00` | `atom.add Sd, [Sa], Sb` | `Sd` = old value |
| `0x01` | `atom.and` | |
| `0x02` | `atom.or` | |
| `0x03` | `atom.xor` | |
| `0x04` | `atom.xchg` | |
| `0x05` | `atom.min.s` | |
| `0x06` | `atom.max.s` | |
| `0x07` | `atom.min.u` | |
| `0x08` | `atom.max.u` | |
| `0x10` | `atom.cas Sd, [Sa], Sb, Sc` | compare `Sb`, swap `Sc`, return old |
| `0x18` | `atom.ld Sd, [Sa]` | atomic load with the encoded order |
| `0x19` | `atom.st [Sa], Sb` | atomic store |
| `0x20` | `fence` | order per `mod[4:3]`, scope per `mod[6:5]` |

FP32 `atom.add` is expanded to a `atom.cas` loop by the compiler in v0.1.

---

## 10. Traps

| Code | Name | Cause |
|---:|---|---|
| `0x00` | `NONE` | |
| `0x01` | `ILLEGAL_INSTRUCTION` | unknown encoding, reserved bit set, illegal field |
| `0x02` | `MISALIGNED_PC` | branch target not 8-byte aligned |
| `0x03` | `ACCESS_FAULT` | translation or permission failure |
| `0x04` | `DESCRIPTOR_FAULT` | Stage 2 |
| `0x05` | `BARRIER_FAULT` | Stage 2 |
| `0x06` | `RESOURCE_EXHAUSTED` | descriptor/window/barrier table full |
| `0x40` | `TRAP_USER` | `trap #imm24`, code in `TRAP_CODE[23:0]` |

A trap stops the raising RCTX, records `TRAP_PC`/`TRAP_CODE`, and **wakes every
waiter that could otherwise block forever on this role** (§13.4). It must never
turn into a silent hang.

---

## 11. `CAPABILITY0` bitmap

| bit | Name | Meaning |
|---:|---|---|
| 0 | `VECTOR_FP32` | native FP32 vector datapath |
| 1 | `SFU` | native `v.*.approx` |
| 2 | `GATHER_SCATTER` | native `v.ld.gather` / `v.st.scatter` |
| 3 | `ATOMICS_DEVICE` | device-scope atomics |
| 4 | `SYSTEM_ATOMICS` | system-scope atomics |
| 5 | `MATRIX_F16` | Stage 2 |
| 6 | `MATRIX_BF16` | Stage 2 |
| 7 | `TENSOR_DMA` | Stage 2 |
| 8 | `MBARRIER` | Stage 2 |
| 9 | `PACKED_F16` | Stage 2 |
| 10 | `CACHE_HINTS` | Stage 4 |

A capability bit being clear means the corresponding encodings raise
`ILLEGAL_INSTRUCTION`. It never means "works, but slowly, by some other route" —
that distinction is the whole point of §6.3.

---

## 12. Encoding invariants checked by the toolchain

1. Every instruction's reserved fields are zero.
2. Scalar register indices fit in 6 bits; predicate indices in 4 bits.
3. Scalar/control/predicate/system instructions encode `pred = 0, PN = 0`.
4. `X = 1` only for instructions defined to take an extension word (none in
   v0.1; the bit is reserved and must be 0).
5. Branch displacements are expressible in the encoded field width.
6. Writes to `S0` and `P0` assemble fine but have no architectural effect.
