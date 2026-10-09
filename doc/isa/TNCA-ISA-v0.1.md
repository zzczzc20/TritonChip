# TNCA ISA — Version 0.1 (Stage 1 Frozen Subset)

**Status:** Stage-1 normative. Families marked *Reserved* are allocated but not
yet defined; encoding into them raises `ILLEGAL_INSTRUCTION`.

This document is the single source of truth for instruction encoding. RTL
(`rtl/common/tnca_defs.vh`), the assembler (`tools/asm/tnca_asm.py`) and the
architectural model (`tools/model/tnca_model.py`) are all derived from it. If
they disagree, this document wins and the others are bugs.

Derived from the TNCA-v1 design draft, `doc/design/tnca_v1_architecture_draft.md`
(until 2026-10-08 `doc/architecture_overview.md`) §11–§12. Section references
below point at that document.

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
| `0x30` | Tensor descriptor / RF window | 2 |
| `0x31` | Tensor transfer | 2 |
| `0x32` | Tensor / cache / flow hints | 4 *(Reserved)* |
| `0x40` | Matrix | 2 |
| `0x41` | Accumulator access | 2 |
| `0x50` | Mbarrier | 2 |
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
| `0x009` | `NUM_ROLES` | R | active RCTX count for **this launch** — the number of Role Contexts started, not the core's architectural maximum. A launch may activate a subset, and a program that synchronises across roles has to know how many there actually are. |
| `0x00a` | `CORE_ID` | R | tile index |
| `0x00b` | `NUM_CORES` | R | |
| `0x00c` | `NUM_LANES` | R | `VLEN`, architecturally 16 in v0.1 |
| `0x00d` | `NUM_VREGS` | R | logical `V` registers this implementation gives each RCTX; naming one beyond it raises `ILLEGAL_INSTRUCTION` |
| `0x00e` | `TENSOR_MAX_RANK` | R | highest descriptor rank the transfer engine executes; a higher rank raises `DESCRIPTOR_FAULT` rather than being approximated |
| `0x00f` | `NUM_ACC` | R | accumulator slots |
| `0x010` | `CLOCK` | R | free-running core cycle counter |
| `0x011` | `INSTRET` | R | instructions retired by this RCTX |
| `0x012` | `NUM_BARRIERS` | R | mbarrier slots |
| `0x013` | `NUM_WINDOWS` | R | RF window handles |
| `0x014` | `NUM_DESCRIPTORS` | R | tensor descriptor slots |
| `0x015` | `NUM_PHYS_VREGS` | R | physical vector registers in the core. This is *not* `NUM_VREGS * NUM_ROLES`; the difference is what lets one window be visible to two Role Contexts (§9D.2). |
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
| `0x04` | `vf.cvt.<src>.<dst> Vd, Va` | float → float, RNE; `mod[2:0]` source format, `mod[5:3]` destination | 1 |
| `0x08` | `v.pack.f16` | *Reserved* | 2 |
| `0x09` | `v.unpack.f16` | *Reserved* | 2 |
| `0x0a` | `v.pack.bf16` | *Reserved* | 2 |
| `0x0b` | `v.unpack.bf16` | *Reserved* | 2 |

FP→int rounding modes in `mod[2:0]`: `0` = toward zero, `1` = RNE,
`2` = toward −∞, `3` = toward +∞. Out-of-range and NaN inputs saturate to the
integer extremes; NaN maps to 0.

#### Float formats (`func = 0x04`)

| code | Name | Layout | Infinity |
|---:|---|---|---|
| `0` | `f32` | binary32, 1.8.23 | yes |
| `1` | `bf16` | bfloat16, 1.8.7 | yes |
| `2` | `f16` | binary16, 1.5.10 | yes |
| `3` | `e5m2` | OCP FP8 E5M2, 1.5.2 | yes |
| `4` | `e4m3` | OCP FP8 E4M3, 1.4.3 | **no** |

Codes `5`–`7` are unallocated in either field and raise `ILLEGAL_INSTRUCTION`,
as do any non-zero bits in `mod[9:6]`. All twenty-five pairs are legal,
including the five identities, which are moves.

**A narrow value occupies the low bits of its lane and the rest read zero.**
This instruction is the *unpacked* form: one element per lane whatever the
format. Packing two or four elements into a lane is `v.pack.*`, `func` `0x08`
and up, which is why that space is reserved separately. Fragments for the
matrix unit are packed, and the transfer engine writes them that way (§9C.2).

Rounding is RNE. Every widening is exact — binary32 represents every value of
all four narrow formats — so `narrow → f32 → narrow` is the identity, and only
the narrowing direction rounds. **E4M3 overflow saturates to ±448**, which the
OCP specification permits as an alternative to producing a NaN; the choice is
stated here rather than left to be discovered.

E4M3 is not an IEEE interchange format and must not be treated as one: it has
no infinity, its all-ones exponent field is mostly ordinary numbers, and only
`S.1111.111` is a NaN. Reading it as IEEE-like caps its range at 240 instead
of 448.

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

**Implemented.** `CAPABILITY0.SFU` reads 1 in the reference build. The bounds
below were the target the implementation had to be measured against before the
bit could be set, and it has been: `tb_sfu_xls` checks 16,740 vectors against a
60-digit-decimal reference, and `tb_sfu_core` checks the same vectors through
the core's shared slice. A build with `HAS_SFU = 0` still clears the bit and
raises `ILLEGAL_INSTRUCTION` for these encodings — a capability bit is a promise
about a native datapath, never a claim that something "works, but slowly, by
some other route" (§11).

| `func` | Mnemonic | Max relative error | Special values |
|---:|---|---|---|
| `0x00` | `v.exp2.approx` | ≤ 2⁻²² over `[-126, 126]` | `exp2(-inf)=+0`, `exp2(+inf)=+inf`, `exp2(NaN)=qNaN` |
| `0x01` | `v.rcp.approx` | ≤ 2⁻²² | `rcp(±0)=±inf`, `rcp(±inf)=±0` |
| `0x02` | `v.rsqrt.approx` | ≤ 2⁻²² | `rsqrt(+0)=+inf`, `rsqrt(x<0)=qNaN`, `rsqrt(+inf)=+0` |
| `0x03` | `v.log2.approx` | ≤ 2⁻²² abs. for `[0.5,2]`, else rel. | `log2(±0)=-inf`, `log2(x<0)=qNaN` |

Subnormal inputs are flushed to zero of the same sign by all SFU operations;
subnormal results are flushed to zero. This is a *stated* deviation, not a
silent one, and the strict paths live in the math library.

The flush happens *before* the special-value table is consulted, which is
visible in one place: `rsqrt` of a negative **normal** is a qNaN, while `rsqrt`
of a negative **subnormal** is −∞, because by then it is −0. Both are checked.

These operations are architecturally multicycle. The reference implementation
has one four-function slice shared by all lanes, so a `VLEN`-wide operation
occupies it for `VLEN + 8` cycles and no other FP instruction may issue in the
meantime; the slice itself is fully pipelined, so back-to-back SFU instructions
cost one pass each rather than one pass per lane. Nothing in this section fixes
that ratio — an implementation may widen the slice — but §11's promise is about
a native datapath, and a per-lane unit is not required to make it true.

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

### 9.2 `major = 0x21` — Atomic / fence

Atomics execute **at the home slice of the physical address** (§12.9). That is
not an implementation note: it is the reason the family could not exist before
the NUCA slices did, and it is what makes a read-modify-write indivisible
without a lock. The slice's arbiter already refuses to start a second request
to a line while one is in flight (§14.1), so the order between cores is the
order in which their requests arrive at one slice, and the RMW happens inside
a step that order already protects.

`CAPABILITY0.ATOMICS_DEVICE` reads 1. `SYSTEM_ATOMICS` reads 0 — there is one
device, so system scope has nothing wider to mean — and `mod[6:5] = 2` raises
`ILLEGAL_INSTRUCTION` rather than silently executing at device scope (§12.9).

`mod`:

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

**An atomic must be naturally aligned** — 4 bytes at a 4-byte boundary, 8 at an
8-byte boundary — and raises `ACCESS_FAULT` otherwise. An unaligned *ordinary*
access is legal and merely slow, because the LSU splits it across two words
(§9.1); that split is exactly what an atomic cannot do, so the alternative to
faulting would be two separate read-modify-writes that no other core could tell
from one.

1- and 2-byte atomics are not allocated: `mod[2:0]` accepts only `2` and `3`.
The memory's word is 8 bytes and a narrower RMW would need a sub-word read path
that nothing else in the design has.

`fence` waits until every access this core has issued is complete at its home
slice. The LSU waits for a response to a store as well as to a load, so
draining the LSU *is* the fence at core and device scope. Asynchronous
transfers are deliberately outside it: `t.*.async` completes against an
mbarrier (§9E.1 step 8), and a fence that also waited for them would make the
barrier redundant and the fence unboundedly long.

---

## 9A. `major = 0x50` — Mbarrier

Barriers are **per core**, shared by every RCTX on it, which is what makes them
the synchronisation between roles (§14.3). There are 32 of them, `B0..B31`.

### 9A.1 Why a barrier is not a counter

§13.1 is explicit that three completions must not be confused:

| Event | Meaning |
|---|---|
| **load-ready** | the data is in the destination registers; a consumer may read it |
| **compute-done** | a result is finished; its destination may be read |
| **source-reusable** | an operation has stopped reading its sources; a producer may overwrite them |

A single "done" bit cannot express all three, and a double-buffered pipeline
that only implements load-ready is the most common way to get this wrong
(§13.5): *a transfer completing is not the same as the consumer having finished
reading it.* The barrier object below carries arrivals and operations
separately for exactly this reason — arrivals are what software signals, and
operations are what asynchronous commands signal.

### 9A.2 Barrier state

| Field | Width | Meaning |
|---|---:|---|
| `current_epoch` | 48 | generation currently armed |
| `completed_epoch` | 48 | highest generation that has completed |
| `pending_arrivals` | 8 | software arrivals still owed |
| `pending_operations` | 8 | asynchronous completions still owed |
| `waiter_mask` | NUM_RCTX | RCTXs asleep on this barrier |
| `error` | 1 | set by `mb.inval` or by a faulting command |

A **token** identifies one generation and fits in a scalar register:

```
token = { epoch[47:0], barrier_id[15:0] }
```

Epoch wrap is only legal when the barrier is quiescent.

### 9A.3 Instructions

| `func` | Mnemonic | Fields | Semantics |
|---:|---|---|---|
| `0x00` | `mb.init Bd` | `d`=barrier | epoch 0, counts 0, no waiters, error clear |
| `0x01` | `mb.arm Sd, Ba, #arr, #ops` | `d`=Sd, `a`=barrier, `b`=arrivals, `c`=operations | `current_epoch++`; set both counts; `Sd` ← token |
| `0x02` | `mb.arrive Ba` | `a`=barrier | release, then `pending_arrivals--` — see §9A.5 |
| `0x03` | `mb.test Sd, Sa` | `d`=Sd, `a`=token | `Sd` = 1 if that generation has completed |
| `0x04` | `mb.wait Sa` | `a`=token | acquire; sleep this RCTX until that generation completes |
| `0x05` | `mb.inval Ba` | `a`=barrier | set error and wake every waiter |

`mb.arm` fixes both counts for the generation in one instruction. §13.3 prefers
this to letting an expected byte count grow at arbitrary times, because a count
that is only correct if every producer ran is not a count you can verify.

**Counting unit: one complete asynchronous command** — not a cache line, and
not a byte count (§13.3). Two `t.ld.async` are `operations=2` however many line
transactions they expand into; one `m.mma.async` is `operations=1`.

**A fully predicated-off asynchronous command still completes once** (§13.4), so
a static operation count stays correct under predication. The compiler still
has to get the consumer's control flow right.

### 9A.7 Barriers are quiescent at kernel entry

Every barrier starts at epoch 0, with both counts zero, no error and no
waiter. A kernel does not have to run `mb.init` to reach that state, and a
role must not be made to wait until another role has.

This is not a convenience. There is no race-free way to express "wait until
role 0 has initialised the barriers" using barriers, because waiting means
sleeping on one and §9A.6 makes an `mb.init` under a sleeping waiter a
`BARRIER_FAULT`. Any handshake built around that has to leave the barrier
mechanism entirely — in practice, polling a memory flag — and a polling
consumer competes with its own producer for the memory port. A compiled
two-role GEMM deadlocked exactly this way: the consumer's poll starved the
transfer engine, so the flag it was waiting for was never written.

Guaranteeing the state at entry removes the problem rather than working
around it. A consumer may sleep on generation 1 of a barrier before the
producer has executed a single instruction.

`mb.init` remains, for re-initialising a barrier *within* a kernel, where the
program already knows nothing is waiting on it.

### 9A.5 What `release` covers

`mb.arrive` carries release semantics, and that has to mean something a
consumer can rely on. Scalar instructions retire in order, but a vector or FP
result is still in flight for several cycles after its instruction has left the
execute stage, and an asynchronous command may run for hundreds. An arrive that
retired before those landed would make "everything before it is visible" false
in exactly the cases a consumer cares about.

> **`mb.arrive` does not retire until every register write this RCTX has issued
> has landed, and no engine is mid-command.**

An implementation may be more precise about which engine commands belong to
which context; it may not be less precise about the register writes.

This covers what the *arriving context* has done. It deliberately does **not**
cover asynchronous commands as a class — those are counted as `operations`, not
signalled by arrivals, precisely so that "the transfer finished" and "the
producer reached this point" stay distinguishable (§13.1).

### 9A.6 Re-initialising a barrier with a waiter

`mb.init` resets `completed_epoch`, so a context already asleep on that barrier
could never see its generation complete. Doing it is a program bug, and the
hardware raises `BARRIER_FAULT` rather than inventing an ordering: no choice the
hardware could make would be the one the program meant.

### 9A.4 Completion

A generation completes when `pending_arrivals == 0 && pending_operations == 0`.
Required properties, all of them safety rather than performance:

- completion happens **exactly once** per generation;
- neither count may underflow — an extra arrival or completion is a
  `BARRIER_FAULT`, not a wrap;
- an error wakes every waiter and is reported, so a faulting command can never
  turn into an unbounded wait;
- `mb.wait` **sleeps** the RCTX. It does not spin (§13.4): a blocked RCTX stops
  fetching entirely and is woken by the completion.

---

## 9B. `major = 0x41` — Accumulator access

`A0..A15`, each 16 rows × 16 lanes of binary32.

Accumulators are reachable from the vector unit, and §5.4 explains why this is
not a convenience: FlashAttention rescales `O` by `exp(m - m')` on every key
block. If that round trip had to go accumulator → cache → vector → cache →
accumulator, a scratchpad-free design would lose the argument for existing.

| `func` | Mnemonic | Fields | Semantics |
|---:|---|---|---|
| `0x00` | `a.vread Vd, Aa, Sb` | `d`=Vd, `a`=A, `b`=row | `Vd[n] = A[row][n]` |
| `0x01` | `a.vwrite Ad, Sa, Vb` | `d`=A, `a`=row, `b`=Vs | `A[row][n] = Vs[n]` |
| `0x02` | `a.mul.row Ad, Sa, Sb` | `d`=A, `a`=row, `b`=scale | `A[row][n] *= f32(Sb[31:0])` |
| `0x03` | `a.fma.row Ad, Sa, Vb, Sc` | `d`=A, `a`=row, `b`=Vx, `c`=scale | `A[row][n] = fma(A[row][n], f32(Sc), Vx[n])` |
| `0x08` | `a.zero Ad` | `d`=A | every element `+0.0` |

`row` is `Sa[3:0]`. All five are lane-maskable; an inactive lane leaves that
element unchanged. `a.mul.row` and `a.fma.row` are specified as a single
rounding per element, so they are `tnca_fp32_fma` operations and not a
multiply followed by an add.

---

## 9C. `major = 0x40` — Matrix

| `func` | Mnemonic | Fields | `mod` |
|---:|---|---|---|
| `0x00` | `m.zero Ad` | `d`=A | — |
| `0x01` | `m.mma.async Ad, Va, Vb, Sc` | `d`=A, `a`=A-fragment base vreg, `b`=B-fragment base vreg, `c`=token | `[1:0]` K, `[3:2]` dtype, `[4]` source-free |

```
C[16x16] <- C[16x16] + A[16xK] . B[Kx16]
```

`mod[1:0]`: `0` = K16, `1` = K32, `2` = K64, `3` reserved.
`mod[3:2]`: `0` = FP16, `1` = BF16, `2` = FP8 E5M2, `3` = FP8 E4M3 inputs.

The two FP8 encodings are gated together by `CAPABILITY0.MATRIX_FP8` (§11).
They differ by one parameter in one widening step and share every other gate,
so no implementation can reasonably have one without the other — unlike FP16
and BF16, which have separate bits because an implementation could plausibly
build a datapath for one exponent range and not the other.

The accumulator is binary32 for every input format; §9C.1 applies unchanged.

#### `mod[4]` — the second completion event

A `m.mma.async` finishes twice, and the two are not the same moment:

| event | means | signalled on |
|---|---|---|
| **result** | `C` has landed in the accumulator | the barrier token in `Sc` |
| **source-free** | the A and B fragments have been read for the last time | the barrier token in `Sc+1`, when `mod[4]` is set |

Both are *operations*, counted by `mb.arm`'s operation field, and they must go
to different barriers — two events on one barrier cannot be waited on
separately, which is the entire reason for having two.

`Sc = S63` with `mod[4]` set raises `ILLEGAL_INSTRUCTION`, for the same reason
§9E refuses it for a transfer's coordinates: the next index is `S0`, which
reads as zero and discards writes, so the second token could not be written at
all and the command would signal a barrier the program never armed.

**Why this is in the ISA and not left to the compiler.** A double-buffered
producer wants to refill the fragment registers as early as it legally can.
Without a source-free event its only safe choice is to wait for the *result*,
which is later — architecture overview §13.1 names confusing "the data
arrived", "the computation finished" and "the source may be reused" as a known
hazard, and permits the conservative proxy only for a first version. This is
the version that stops needing it. An implementation may signal source-free at
any point after the last fragment read and no later than the result; the
reference implementation signals it `FMA_LAT` cycles earlier, at the end of the
MAC sweep.

`Sc` holds an mbarrier token; the command signals one operation completion on
that barrier's generation when it finishes.

### 9C.1 The accumulation contract (§6.2)

Precision and order are architectural, not an implementation detail:

> ```
> C[k+1] = RNE_f32( exact(A[m][k]) * exact(B[k][n]) + C[k] )
> ```
>
> The product is formed at **full width and is never rounded on its own**; there
> is exactly **one rounding per accumulation step**, applied to the sum, in
> **increasing `k` order**.

So the result is bit-reproducible across implementations, and an implementation
may not reassociate for speed. An implementation that wants a faster reduction
order must expose it as a different opcode, not as this one being faster
somewhere.

**Why this is stated as a fused operation rather than "the product is exact".**
An earlier wording of this section claimed an FP16×FP16 *or BF16×BF16* product
is exactly representable in binary32. That is true for FP16 and **false for
BF16**: BF16 has the same 8-bit exponent field as binary32, so while its
significand always fits, the *exponent* of a product need not. Two BF16 values
near 3.4×10³⁸ multiply to about 10⁷⁷, which binary32 cannot represent at all.

The distinction is observable, not pedantic:

| | `A = 2^127` (BF16), `B = 2.0`, `C = -2^127` |
|---|---|
| fused, as specified | `2^127` |
| product rounded first | `+inf` — because `2^128` overflows binary32 |

A widened-input FP32 FMA gives the first answer. An implementation that
converts to binary32, multiplies with rounding, then adds, gives the second and
is **not** conforming. `tb_fp32_fma` carries these cases permanently.

### 9C.2 Fragment layout

A fragment is a window's registers read as a **flat row-major byte array**: this
is exactly what the transfer engine writes, so no repacking sits between a
`t.ld.async` and the `m.mma.async` that consumes it.

For a 16×K A fragment of elements `E` bytes wide (`E = 2` for FP16 and BF16,
`E = 1` for both FP8 encodings), element `(m, k)` lives at byte `(m*K + k)*E`
of the array, that is at

```
vreg   = base + (m*K + k)*E / 64
byte   = (m*K + k)*E % 64
```

and `B[k][n]` follows the same rule with `(k, n)` over a K×16 tile. A vector
register is 64 bytes, so a 16×16 FP16 fragment is eight registers and the same
fragment in FP8 is four.

**The element width is the only thing the dtype changes about the layout.**
There is no separate FP8 packing rule to get wrong, and an implementation that
kept the 2-byte stride for an FP8 fragment would read the right registers and
the wrong halves of them — a wrong `C`, not a fault. `t11_matrix` runs the
same product through all four dtypes for that reason.

### 9C.3 What the ISA does not promise

A `m.mma.async` that has not completed leaves three things undefined, and the
compiler owns all three (§12.12):

- its source registers may not be overwritten;
- its destination accumulator may not be read by the vector unit;
- that accumulator may not be reassigned to another program.

---

## 9D. `major = 0x30` — Tensor descriptor and RF window

### 9D.1 Two objects, deliberately separate (§7.1)

A **tensor descriptor** says what the data *is* — address, shape, stride,
dtype, boundary behaviour. An **RF window** says where a transfer may land —
a range of physical vector registers. They are separate because they have
different lifetimes: a descriptor outlives many transfers, and a window is
reused every pipeline stage.

### 9D.2 An RF window is register aliasing, not a scratchpad (§5.2)

A window names a contiguous range of vector registers and nothing else. It
deliberately does **not** provide:

- an arbitrary byte pointer;
- a general `load [window + arbitrary_byte_offset]`;
- residency for arbitrary software objects;
- any storage of its own.

Without those restrictions it would simply be a scratchpad with a new name, and
removing the scratchpad is the entire point of the design.

The window is also the transfer engine's **bounds check**: a transfer that
would write past `len` registers raises `DESCRIPTOR_FAULT` instead of
corrupting a neighbouring allocation.

| `func` | Mnemonic | Fields | Semantics |
|---:|---|---|---|
| `0x00` | `td.load Dd, [Sa]` | `d`=D, `a`=address | snapshot a 256-byte TensorMap into descriptor slot `D` |
| `0x02` | `td.release Dd` | `d`=D | release the slot; illegal while a transfer still references it |
| `0x10` | `w.def Wd, #base, #len` | `d`=W, `{b,c}`=imm16 | window `W` = vector registers `[base, base+len)`; `base`=`imm16[15:8]`, `len`=`imm16[7:0]` |
| `0x11` | `w.release Wd` | `d`=W | |

### 9D.3 TensorMap layout (§7.2)

256 bytes. The first 128 are **semantics that must be executed exactly**; the
second 128 are **hints that may be ignored entirely**. Nothing in the second
half may change an address, a value, an access, or an order (§1.4).

| Bytes | Field |
|---|---|
| 0–7 | base address |
| 8–47 | `extent[5]`, 64-bit each |
| 48–87 | `byte_stride[5]`, signed 64-bit each |
| 88–107 | default `block_shape[5]`, 32-bit each |
| 108 | rank |
| 109 | dtype |
| 110 | padding mode |
| 111 | semantic flags |
| 112–119 | storage object UID |
| 120–127 | descriptor version |
| 128–191 | cache/flow hints — **ignorable** |
| 192–255 | reserved |

```
addr(i0..i_{r-1}) = base + sum_d  i_d * byte_stride_d
```

`dtype`: `0` = FP16, `1` = BF16, `2` = FP32, `3` = INT8, others reserved.
`padding mode`: `0` = zero-fill out-of-bounds elements, `1` = clamp the
coordinate, `2` = raise `DESCRIPTOR_FAULT`.

### 9D.4 Snapshot semantics (§7.4)

`td.load` takes a **versioned snapshot**. Once a transfer has been issued
against descriptor `D`:

- a later `td.load` or update to `D` does not change that transfer;
- no command ever observes a half-updated descriptor;
- `td.release` on a slot with outstanding references raises
  `RESOURCE_EXHAUSTED` rather than silently recycling it.

§7.4 puts this above replacement policy in importance, and it is why the engine
copies the descriptor rather than re-reading it per line.

---

## 9E. `major = 0x31` — Tensor transfer

| `func` | Mnemonic | Fields | Semantics |
|---:|---|---|---|
| `0x00` | `t.ld.async Wd, Da, Sb, Sc` | `d`=W dest, `a`=D, `b`=coord base, `c`=token | global → window |
| `0x08` | `t.st.async Wd, Da, Sb, Sc` | `d`=W **source**, `a`=D, `b`=coord base, `c`=token | window → global |
| `0x10` | `t.prefetch Da, Sb` | `a`=D, `b`=coord base | performance only; guarantees nothing |

`mod[2:0]` is the transfer's element-size log2; `mod[4:3]` selects the window
layout (`0` = flat row-major); `mod[5]` requests a transposed write.

A non-zero `mod[4:3]` raises `ILLEGAL_INSTRUCTION`: layout `0` is the only one
that exists, and accepting another value would transfer the tile in a shape the
program did not ask for.

**The transposed write (`mod[5]`, capability `tensor_transpose`).** For a
descriptor with block shape `R × C`, the window receives the tile as `C` rows of
`R` elements, flat row-major: window element `(c, r)` is source element
`(coord0 + r, coord1 + c)`, and zero wherever either index falls outside the
descriptor's extents (§9E.1 step 7). Everything else — snapshot, bounds, one
completion — is as for the untransposed transfer. The implementation
requires, in addition to §9E.2:

| Condition | Otherwise |
|---|---|
| the engine was built with the capability | `ILLEGAL_INSTRUCTION` |
| rank 2 | `DESCRIPTOR_FAULT` |
| 16-bit elements (`mod[2:0] = 1`) | `DESCRIPTOR_FAULT` |
| `R` is 4, 8 or 16 | `DESCRIPTOR_FAULT` |

The last two keep each destination row whole lanes inside one register, which
is what lets the engine write it through the lane enables it already has.

Coordinates come from **consecutive scalar registers** starting at `Sb`, one per
descriptor rank (§12.11). The block shape comes from the descriptor.

`t.ld.async` and `t.st.async` with `Sb = S63` raise `ILLEGAL_INSTRUCTION`. The
next index would be `S0`, which reads as zero and discards writes (§3), so the
second coordinate could not be written at all and the transfer would run at
coordinate 0 on that axis while the program believed it had set one. That is a
wrong address, and a wrong address is a wrong answer rather than a slow one, so
the encoding is refused instead.

A transfer's lane mask is **all or nothing**. A tile transfer writes whole
8-byte units across many rows, so a 16-lane mask has no per-element meaning
here; the engine performs the whole transfer if any lane is active and none of
it if none is. The case §2.2 fixes — a fully-masked-off instruction performing
no access and still writing its completion event — is therefore exact, and it
is the case a predicated pipeline stage relies on.

`t.prefetch` is deliberately exempt. It promises nothing (§9E.3) and may be
dropped entirely, so a coordinate it reads wrongly costs a prefetch and never a
result — which is precisely the line §1.4 draws between semantics and hints.

This is the kind of implicit wide register read a compiler cannot see in the
explicit operand fields, and it is why the read set is stated here rather than
left to be inferred.

### 9E.1 Order of operations (§12.11)

1. read the descriptor **snapshot**;
2. take the coordinates from the scalar registers;
3. compute the addresses;
4. check the bounds against the descriptor's extents;
5. expand into memory transactions;
6. write returns into the window according to the transfer configuration;
7. fill padded elements;
8. signal **one** completion against the token.

Step 8 happens once per command, whatever step 5 expanded into — that is what
makes the operation counts in `mb.arm` a static, checkable property.

### 9E.2 Geometry the engine refuses

`DESCRIPTOR_FAULT` rather than an approximation, for each of:

| Condition | Why |
|---|---|
| rank above `TENSOR_MAX_RANK` | more coordinate arithmetic than the engine has |
| `block_shape[1] * sizeof(dtype)` not a multiple of 8 | a returned 8-byte unit would straddle two window registers |
| `base` not 8-byte aligned | the first unit of every row would land shifted |
| `byte_stride[0]` not a multiple of 8 | rows after the first would land shifted, even with an aligned base |
| `byte_stride[1]` not `sizeof(dtype)` | the engine copies runs, it does not gather per element |
| the tile larger than the window | ISA §9D.2: the window is the bounds check |

The two alignment conditions are the same requirement stated at both ends: a
transfer places whole 8-byte units into window registers through the existing
lane write enables, so every row must both *be* a whole number of units and
*start* on one. A row stride of, say, 26 bytes satisfies neither, and an engine
that proceeded anyway would return data shifted by two bytes per row — a wrong
answer wearing the shape of a right one.

### 9E.3 `t.prefetch` promises nothing

Completion of a prefetch does not guarantee a later hit, does not create a
snapshot, and does not confer ownership (§17.2). It is a hint in the sense of
§1.4: it may be dropped entirely and the program must still be correct.

---

## 10. Traps

| Code | Name | Cause |
|---:|---|---|
| `0x00` | `NONE` | |
| `0x01` | `ILLEGAL_INSTRUCTION` | unknown encoding, reserved bit set, illegal field |
| `0x02` | `MISALIGNED_PC` | branch target not 8-byte aligned |
| `0x03` | `ACCESS_FAULT` | translation or permission failure |
| `0x04` | `DESCRIPTOR_FAULT` | rank above `TENSOR_MAX_RANK`, transfer larger than its window, or an out-of-bounds access under padding mode 2 |
| `0x05` | `BARRIER_FAULT` | arrival or operation count underflow, or a waiter woken by `mb.inval` |
| `0x06` | `RESOURCE_EXHAUSTED` | descriptor/window/barrier table full, or `td.release` with outstanding references |
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
| 3 | `ATOMICS_DEVICE` | device-scope atomics, executed at the home slice |
| 4 | `SYSTEM_ATOMICS` | system-scope atomics |
| 5 | `MATRIX_F16` | native `m.mma` with FP16 inputs |
| 6 | `MATRIX_BF16` | native `m.mma` with BF16 inputs |
| 7 | `TENSOR_DMA` | native `t.ld.async` / `t.st.async` |
| 8 | `MBARRIER` | native mbarrier |
| 9 | `PACKED_F16` | Stage 2 |
| 10 | `CACHE_HINTS` | Stage 4 |
| 11 | `MATRIX_FP8` | native `m.mma` with OCP FP8 E5M2 **and** E4M3 inputs |

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
7. A `t.ld.async` / `t.st.async` whose coordinate base is `S63` is illegal,
   because its second coordinate register would not exist (§9E).
