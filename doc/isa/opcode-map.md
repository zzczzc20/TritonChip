# TNCA ISA v0.1 -- opcode map

<!-- TNCA ISA v0.1 -- GENERATED FILE, DO NOT EDIT. Source: isa/tnca-isa-v0.1.json (itself generated from tools/asm/tnca_isa.py). Regenerate with `make isa`. Normative text: doc/isa/TNCA-ISA-v0.1.md. -->

155 instructions across 16 allocated majors. Normative semantics live in `doc/isa/TNCA-ISA-v0.1.md`; this file only records which encodings are taken.

## `SYS` (major `0x00`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `nop` | -- | -- | no |
| `0x01` | `s.rdspecial` | `sd`, `spec` | -- | no |
| `0x02` | `s.wrspecial` | `spec`, `sa` | -- | no |

## `SALU` (major `0x01`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `s.mov` | `sd`, `sa` | -- | no |
| `0x01` | `s.add` | `sd`, `sa`, `sb` | -- | no |
| `0x02` | `s.sub` | `sd`, `sa`, `sb` | -- | no |
| `0x03` | `s.mul` | `sd`, `sa`, `sb` | -- | no |
| `0x04` | `s.mulh.s` | `sd`, `sa`, `sb` | -- | no |
| `0x05` | `s.mulh.u` | `sd`, `sa`, `sb` | -- | no |
| `0x08` | `s.and` | `sd`, `sa`, `sb` | -- | no |
| `0x09` | `s.or` | `sd`, `sa`, `sb` | -- | no |
| `0x0a` | `s.xor` | `sd`, `sa`, `sb` | -- | no |
| `0x0b` | `s.andn` | `sd`, `sa`, `sb` | -- | no |
| `0x0c` | `s.shl` | `sd`, `sa`, `sb` | -- | no |
| `0x0d` | `s.lshr` | `sd`, `sa`, `sb` | -- | no |
| `0x0e` | `s.ashr` | `sd`, `sa`, `sb` | -- | no |
| `0x10` | `s.min.s` | `sd`, `sa`, `sb` | -- | no |
| `0x11` | `s.max.s` | `sd`, `sa`, `sb` | -- | no |
| `0x12` | `s.min.u` | `sd`, `sa`, `sb` | -- | no |
| `0x13` | `s.max.u` | `sd`, `sa`, `sb` | -- | no |
| `0x14` | `s.clz` | `sd`, `sa` | -- | no |
| `0x15` | `s.popc` | `sd`, `sa` | -- | no |
| `0x16` | `s.sext.w` | `sd`, `sa` | -- | no |
| `0x17` | `s.zext.w` | `sd`, `sa` | -- | no |
| `0x18` | `s.cmp` | `sd`, `sa`, `sb` | `cc_int` | no |
| `0x19` | `s.sel` | `sd`, `sa`, `sb`, `sc` | -- | no |
| `0x20` | `s.addi` | `sd`, `sa`, `i16s` | -- | no |
| `0x21` | `s.subi` | `sd`, `sa`, `i16s` | -- | no |
| `0x22` | `s.muli` | `sd`, `sa`, `i16s` | -- | no |
| `0x28` | `s.andi` | `sd`, `sa`, `i16z` | -- | no |
| `0x29` | `s.ori` | `sd`, `sa`, `i16z` | -- | no |
| `0x2a` | `s.xori` | `sd`, `sa`, `i16z` | -- | no |
| `0x2c` | `s.shli` | `sd`, `sa`, `i16z` | -- | no |
| `0x2d` | `s.lshri` | `sd`, `sa`, `i16z` | -- | no |
| `0x2e` | `s.ashri` | `sd`, `sa`, `i16z` | -- | no |
| `0x38` | `s.cmpi` | `sd`, `sa`, `i16s` | `cc_int` | no |
| `0x40` | `s.movz` | `sd`, `i16z` | `lsl` | no |
| `0x41` | `s.movk` | `sd`, `i16z` | `lsl` | no |
| `0x42` | `s.movn` | `sd`, `i16z` | `lsl` | no |

## `CTRL` (major `0x02`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `br` | `lbl24` | -- | no |
| `0x01` | `br.nz` | `sa`, `lbl16` | -- | no |
| `0x02` | `br.z` | `sa`, `lbl16` | -- | no |
| `0x04` | `br.r` | `sa` | -- | no |
| `0x08` | `call` | `sd`, `lbl24` | -- | no |
| `0x09` | `ret` | `sa` | -- | no |
| `0x10` | `yield` | -- | -- | no |
| `0x18` | `trap` | `i24` | -- | no |
| `0x1f` | `exit` | -- | -- | no |

## `VINT` (major `0x10`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `v.mov` | `vd`, `va` | -- | yes |
| `0x01` | `v.add` | `vd`, `va`, `vb` | -- | yes |
| `0x02` | `v.sub` | `vd`, `va`, `vb` | -- | yes |
| `0x03` | `v.mul` | `vd`, `va`, `vb` | -- | yes |
| `0x04` | `v.mulh.s` | `vd`, `va`, `vb` | -- | yes |
| `0x05` | `v.mulh.u` | `vd`, `va`, `vb` | -- | yes |
| `0x08` | `v.and` | `vd`, `va`, `vb` | -- | yes |
| `0x09` | `v.or` | `vd`, `va`, `vb` | -- | yes |
| `0x0a` | `v.xor` | `vd`, `va`, `vb` | -- | yes |
| `0x0b` | `v.andn` | `vd`, `va`, `vb` | -- | yes |
| `0x0c` | `v.shl` | `vd`, `va`, `vb` | -- | yes |
| `0x0d` | `v.lshr` | `vd`, `va`, `vb` | -- | yes |
| `0x0e` | `v.ashr` | `vd`, `va`, `vb` | -- | yes |
| `0x10` | `v.min.s` | `vd`, `va`, `vb` | -- | yes |
| `0x11` | `v.max.s` | `vd`, `va`, `vb` | -- | yes |
| `0x12` | `v.min.u` | `vd`, `va`, `vb` | -- | yes |
| `0x13` | `v.max.u` | `vd`, `va`, `vb` | -- | yes |
| `0x14` | `v.clz` | `vd`, `va` | -- | yes |
| `0x15` | `v.popc` | `vd`, `va` | -- | yes |
| `0x18` | `v.cmp` | `pd`, `va`, `vb` | `cc_int` | yes |
| `0x19` | `v.sel` | `vd`, `va`, `vb` | -- | yes |
| `0x1a` | `v.splat` | `vd`, `sa` | -- | yes |
| `0x1b` | `v.iota` | `vd` | -- | yes |
| `0x20` | `v.addi` | `vd`, `va`, `i16s` | -- | yes |
| `0x21` | `v.subi` | `vd`, `va`, `i16s` | -- | yes |
| `0x22` | `v.muli` | `vd`, `va`, `i16s` | -- | yes |
| `0x28` | `v.andi` | `vd`, `va`, `i16z` | -- | yes |
| `0x29` | `v.ori` | `vd`, `va`, `i16z` | -- | yes |
| `0x2a` | `v.xori` | `vd`, `va`, `i16z` | -- | yes |
| `0x2c` | `v.shli` | `vd`, `va`, `i16z` | -- | yes |
| `0x2d` | `v.lshri` | `vd`, `va`, `i16z` | -- | yes |
| `0x2e` | `v.ashri` | `vd`, `va`, `i16z` | -- | yes |
| `0x38` | `v.cmpi` | `pd`, `va`, `i16s` | `cc_int` | yes |
| `0x40` | `v.movi` | `vd`, `i16s` | -- | yes |

## `VFP` (major `0x11`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x01` | `vf.add` | `vd`, `va`, `vb` | `ftz` | yes |
| `0x02` | `vf.sub` | `vd`, `va`, `vb` | `ftz` | yes |
| `0x03` | `vf.mul` | `vd`, `va`, `vb` | `ftz` | yes |
| `0x06` | `vf.fma` | `vd`, `va`, `vb`, `vc` | `ftz` | yes |
| `0x07` | `vf.fms` | `vd`, `va`, `vb`, `vc` | `ftz` | yes |
| `0x10` | `vf.min` | `vd`, `va`, `vb` | `fmm` | yes |
| `0x11` | `vf.max` | `vd`, `va`, `vb` | `fmm` | yes |
| `0x14` | `vf.neg` | `vd`, `va` | -- | yes |
| `0x15` | `vf.abs` | `vd`, `va` | -- | yes |
| `0x18` | `vf.cmp` | `pd`, `va`, `vb` | `cc_fp` | yes |

## `CVT` (major `0x12`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `vf.cvt.s32.f32` | `vd`, `va` | -- | yes |
| `0x01` | `vf.cvt.u32.f32` | `vd`, `va` | -- | yes |
| `0x02` | `vf.cvt.f32.s32` | `vd`, `va` | `rnd` | yes |
| `0x03` | `vf.cvt.f32.u32` | `vd`, `va` | `rnd` | yes |

## `PRED` (major `0x13`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `p.mov` | `pd`, `pa` | -- | no |
| `0x01` | `p.and` | `pd`, `pa`, `pb` | -- | no |
| `0x02` | `p.or` | `pd`, `pa`, `pb` | -- | no |
| `0x03` | `p.xor` | `pd`, `pa`, `pb` | -- | no |
| `0x04` | `p.not` | `pd`, `pa` | -- | no |
| `0x05` | `p.andn` | `pd`, `pa`, `pb` | -- | no |
| `0x10` | `p.any` | `sd`, `pa` | -- | no |
| `0x11` | `p.all` | `sd`, `pa` | -- | no |
| `0x12` | `p.ballot` | `sd`, `pa` | -- | no |
| `0x13` | `p.frombits` | `pd`, `sa` | -- | no |
| `0x14` | `p.popc` | `sd`, `pa` | -- | no |

## `SHUF` (major `0x14`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `v.shuffle.xor` | `vd`, `va`, `sb` | -- | yes |
| `0x01` | `v.shuffle.up` | `vd`, `va`, `sb` | -- | yes |
| `0x02` | `v.shuffle.down` | `vd`, `va`, `sb` | -- | yes |
| `0x03` | `v.permute` | `vd`, `va`, `vb` | -- | yes |
| `0x04` | `v.rotate` | `vd`, `va`, `sb` | -- | yes |
| `0x08` | `v.extract` | `sd`, `va`, `lane` | -- | yes |
| `0x09` | `v.insert` | `vd`, `sa`, `lane` | -- | yes |

## `RED` (major `0x15`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `v.reduce.add` | `rdst`, `va` | `redb` | yes |
| `0x01` | `v.reduce.max.s` | `rdst`, `va` | `redb` | yes |
| `0x02` | `v.reduce.min.s` | `rdst`, `va` | `redb` | yes |
| `0x03` | `v.reduce.and` | `rdst`, `va` | `redb` | yes |
| `0x04` | `v.reduce.or` | `rdst`, `va` | `redb` | yes |
| `0x05` | `v.reduce.xor` | `rdst`, `va` | `redb` | yes |
| `0x06` | `v.reduce.max.u` | `rdst`, `va` | `redb` | yes |
| `0x07` | `v.reduce.min.u` | `rdst`, `va` | `redb` | yes |
| `0x08` | `vf.reduce.add` | `rdst`, `va` | `redb` | yes |
| `0x09` | `vf.reduce.max` | `rdst`, `va` | `redb` | yes |
| `0x0a` | `vf.reduce.min` | `rdst`, `va` | `redb` | yes |

## `SFU` (major `0x16`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `v.exp2.approx` | `vd`, `va` | -- | yes |
| `0x01` | `v.rcp.approx` | `vd`, `va` | -- | yes |
| `0x02` | `v.rsqrt.approx` | `vd`, `va` | -- | yes |
| `0x03` | `v.log2.approx` | `vd`, `va` | -- | yes |

## `LSU` (major `0x20`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `s.ld` | `sd`, `mem_si` | `ls` | yes |
| `0x08` | `s.st` | `mem_si`, `sd` | `ls` | yes |
| `0x10` | `v.ld` | `vd`, `mem_si` | `ls` | yes |
| `0x18` | `v.st` | `mem_si`, `vd` | `ls` | yes |
| `0x20` | `v.ld.gather` | `vd`, `mem_sv` | `ls` | yes |
| `0x28` | `v.st.scatter` | `mem_sv`, `vd` | `ls` | yes |

## `ATOM` (major `0x21`)

Allocated, no instructions defined yet.

## `TDESC` (major `0x30`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `td.load` | `dd`, `mem_s` | -- | no |
| `0x02` | `td.release` | `dd` | -- | no |
| `0x10` | `w.def` | `wd`, `winspec` | -- | no |
| `0x11` | `w.release` | `wd` | -- | no |

## `TXFER` (major `0x31`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `t.ld.async` | `wd`, `da`, `sb`, `sc` | `txf` | yes |
| `0x08` | `t.st.async` | `wd`, `da`, `sb`, `sc` | `txf` | yes |
| `0x10` | `t.prefetch` | `da`, `sb` | `txf` | yes |

## `THINT` (major `0x32`)

Allocated, no instructions defined yet.

## `MATRIX` (major `0x40`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `m.zero` | `ad` | -- | no |
| `0x01` | `m.mma.async` | `ad`, `va`, `vb`, `sc` | `mma` | no |

## `ACC` (major `0x41`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `a.vread` | `vd`, `aa`, `sb` | -- | yes |
| `0x01` | `a.vwrite` | `ad`, `sa`, `vb` | -- | yes |
| `0x02` | `a.mul.row` | `ad`, `sa`, `sb` | -- | yes |
| `0x03` | `a.fma.row` | `ad`, `sa`, `vb`, `sc` | -- | yes |
| `0x08` | `a.zero` | `ad` | -- | yes |

## `MBAR` (major `0x50`)

| func | mnemonic | operands | suffixes | maskable |
|---|---|---|---|---|
| `0x00` | `mb.init` | `bd` | -- | no |
| `0x01` | `mb.arm` | `sd`, `ba`, `cnt2` | -- | no |
| `0x02` | `mb.arrive` | `ba` | -- | no |
| `0x03` | `mb.test` | `sd`, `sa` | -- | no |
| `0x04` | `mb.wait` | `sa` | -- | no |
| `0x05` | `mb.inval` | `ba` | -- | no |

## `ROLE` (major `0x51`)

Allocated, no instructions defined yet.

## `DEBUG` (major `0x60`)

Allocated, no instructions defined yet.
