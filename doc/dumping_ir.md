# Inspecting the Compilation Process: IR and Assembly Dumps

As with upstream Triton, every compilation stage can be dumped to disk or accessed directly from `CompileResult.asm`. The naming follows Triton conventions so that developers already accustomed to tuning Triton kernels can use familiar switches.

## Switches

| | TNCA | Triton Counterpart |
|---|---|---|
| Enable dump | `TNCA_KERNEL_DUMP=1` | `TRITON_KERNEL_DUMP=1` |
| Dump destination | `TNCA_DUMP_DIR=<path>` (default: `~/.tnca/dump`) | `TRITON_DUMP_DIR` |
| Bypass cache | `TNCA_ALWAYS_COMPILE=1` | `TRITON_ALWAYS_COMPILE=1` |
| Per-call option | `compile_aot(..., options={'dump_dir': '...'})` | — |
| Command line | `tnca-aot compile --dump-dir DIR` | — |

## Stages

In chronological order of generation:

| File | Contents |
|---|---|
| `<kernel>.ttir` | TTIR produced by the Triton frontend. **Only present for Triton sources**: `tnca-ir` programs never have TTIR, and writing an empty file would falsely suggest one exists. |
| `<kernel>.tnir` | Imported TNCA IR, prior to register allocation. |
| `<kernel>.s` | Assembly text, using the standard assembly file extension. |
| `<kernel>.hex` | Machine code image, one 64-bit word per line—consumed directly by R1 tools and device testbenches. |
| `<kernel>.json` | Metadata describing each of the above files and how they were compiled. |

Each kernel receives its own directory (`<dump-dir>/<kernel>/`), ensuring two kernels do not overwrite each other, while recompiling the same kernel overwrites previous dumps.

## Command Line

```
$ tnca-aot compile kern.py:axpy -S \
      --signature x_ptr=*fp32 ... --constexpr BLOCK=16 -o axpy.tnca
    s.rdspecial  s1, arg_base
    s.ld.d       s2, [s1, #0]      // x_ptr: device_ptr64 at arg_base+0
    ...
```

`-S` and `--emit` follow conventions from standard compilers (`gcc -S` for assembly, `clang -emit-llvm` for IR):

* `-S` — Equivalent to `--emit asm`
* `--emit ttir|tnir|asm|hex` — Emits a specific stage to stdout **without writing files** (inspecting vs. saving are separate concerns).
* `--dump-dir DIR` — Writes all stages into `DIR/<kernel>/`.

## Python

```python
res = aot.compile_aot(kernel, signature=..., constexprs=...)
print(res.asm['tnir'])        # Same purpose as Triton's kernel.asm['ttgir']
print(res.asm.keys())         # ['ttir', 'tnir', 'asm', 'hex']
res.dumped                    # stage -> written file path; empty when dumping is not enabled
```

`res.asm` is **always** populated, regardless of whether dumps are written to disk: these text representations already reside in memory at the end of each compilation stage, so retaining a reference costs virtually nothing.

## Two Key Boundaries

**Dumping does not modify the artifact.** Neither `dump_dir` nor `debug` enters the cache key or the manifest's `identity.options`. Otherwise, enabling dumps would produce a different kernel than the one you intended to inspect—the very reason for inspecting it. Triton excludes `TRITON_KERNEL_DUMP` from the cache key for the same reason.

This also prevents packaging user-specific local filesystem paths (e.g., home directory paths passed via `dump_dir`) into distributable artifacts.

**Requesting a dump always performs a real compilation.** A cache hit means "skip the compiler", whereas dumping means "observe the compiler run". When both coincide, compilation takes place—otherwise, writing files on the first run in a session and silently skipping subsequent runs would create confusing behavior. `TNCA_ALWAYS_COMPILE=1` is the counterpart to Triton's switch of the same name.

## Testing

`make r2-dump-tests` (DUMP-01...06). These are **not contract test cases**: [`doc/phase2_tasks_and_acceptance.md`](phase2_tasks_and_acceptance.md) does not require dumping functionality, so they do not count toward R2's 84 mandatory acceptance cases, as explicitly noted in the test file header.
