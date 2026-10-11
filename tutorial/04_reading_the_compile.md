# 04 · Reading what the compiler did

**Run it:** `PYTHONPATH=python:compiler/python python tutorial/examples/04_inspect_the_compile.py`

Triton dumps its stages when you ask; so does this compiler. Use it when
a kernel is slower than you expected, or when you want to know whether
an option did anything.

## Asking for the dump

Either in code:

```python
aot.compile_aot(kernel, ..., options={'opt_level': 2, 'dump_dir': '/tmp/d'})
```

or from the environment, without touching your program:

```sh
TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=/tmp/dump python your_program.py
```

A dump never changes the code. That is checked, not promised: the
compile with `dump_dir` and the compile without produce identical
machine words.

## The stages

```text
stages the compiler kept:
  default     ['asm', 'hex', 'tnir', 'ttir', 'ttir_inlined']
  at O2       ['asm', 'hex', 'tnir', 'tnir_opt', 'ttir', 'ttir_inlined', 'ttir_opt']

  files under scale_and_offset/: ['scale_and_offset.hex',
    'scale_and_offset.inlined.ttir', 'scale_and_offset.json',
    'scale_and_offset.opt.tnir', 'scale_and_offset.opt.ttir',
    'scale_and_offset.s', 'scale_and_offset.tnir', 'scale_and_offset.ttir']
```

| file | what it is |
|---|---|
| `.ttir` | the Triton IR the frontend produced. Absent for a TNCA IR source, which never had one. |
| `.inlined.ttir` | the same IR with Triton's library calls (`tl.exp`, `tl.sum`, ...) inlined. That happens at every level: no backend can lower a `tt.call`, so it is legalisation, not optimisation |
| `.opt.ttir` | after the pinned upstream canonicalizer and CSE |
| `.tnir` | the TNCA IR the importer built |
| `.opt.tnir` | after the TNCA-level passes |
| `.s` | the assembly |
| `.hex` | the machine words |
| `.json` | the kernel, the profile and the cache key |

**The `.opt.*` files appear only when something optimised.** A file
written regardless would report an optimisation that did not happen,
so their absence is information.

## The assembly

```text
    s.rdspecial  s1, arg_base
    s.ld.d       s2, [s1, #0]      // x_ptr: device_ptr64 at arg_base+0
    s.ld.d       s3, [s1, #8]      // o_ptr: device_ptr64 at arg_base+8
    s.ld.w       s4, [s1, #16]     // alpha: f32 at arg_base+16
    s.ld.sw      s5, [s1, #24]     // n: i32 at arg_base+24
    s.rdspecial  s1, pid.x
    s.shli       s6, s1, #4        // x16 as a shift
    v.iota       v0                // arange(0, VLEN)
    ...
    .li          s2, #1065353216
    v.splat      v0, s2            // a 32-bit constant does not fit v.movi (ISA 0x40)
```

Two things to read there.

`s.shli s6, s1, #4` is `pid * BLOCK` with `BLOCK=16`. At the default
level that is a materialised constant plus a multiply; the shift is a
TNCA lowering, not something the upstream passes do — measured on this
exact module, upstream canonicalisation leaves the multiply alone.

The `.li` plus `v.splat` is the literal `1.0` in the kernel. A float
literal is a 32-bit constant, `v.movi` takes sixteen bits, so it is
materialised in a scalar register and broadcast. That comment is in the
emitted assembly so you do not have to work it out.

## What each pass did

```text
opt_dead_pure_removed        1
opt_immediate_compares       0
opt_lane0_stores             0
opt_ops_after                18
opt_ops_before               18
opt_pipeline                 canonicalize,cse
opt_pipeline_version         1
opt_pure_scalar_merged       0
opt_strength_reduced         1
opt_unobserved_load_other    1
```

This lives in the artifact's manifest under `optional_metadata` →
`provenance`, so it travels with the compiled kernel:

```python
prov = art.manifest['optional_metadata']['provenance']
```

A pass reporting `0` is not an error — it means that pass had nothing to
do on this kernel, which is worth knowing when you expected otherwise.

The script also prints what the passes bought in machine words:

```text
    32 machine word(s) at the default, 27 at O2
```

## And it still computes the right thing

```text
    default  SUCCEEDED, 112 cycle(s), correct=True
    O2       SUCCEEDED, 99 cycle(s), correct=True
```

---

Next: [05 · Matrix multiply](05_matrix_multiply.md)
