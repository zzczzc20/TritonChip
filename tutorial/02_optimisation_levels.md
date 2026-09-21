# 02 · Optimisation levels

**Run it:** `PYTHONPATH=python:compiler/python python tutorial/examples/02_opt_levels.py`

```python
options={'opt_level': 2}
```

That is the whole interface. What follows is what each level does and
how to tell whether it helped.

## The levels

| | what it does |
|---|---|
| default | what the compiler has always emitted. The baseline you compare against. |
| `opt_level=0` | the compatibility path. **Byte-identical to the default.** |
| `opt_level=1` | upstream MLIR canonicalisation and CSE, and cheaper lowerings for constants, compares and masked loads. |
| `opt_level=2` | everything in O1, plus LinearLayout-driven layout decisions. |

O0 existing as a separate name that emits the *same bytes* is the point:
if it differed, a program compiled at O0 would no longer be the baseline,
and every comparison against it would be measuring two changes at once.

## What it prints

```text
mode            words  cycles   vs B0  code == B0  correct
B0 (default)       30     226      +0  yes         yes
O0                 30     226      +0  yes         yes
O1                 24     204     +22  no          yes
O2                 24     204     +22  no          yes
```

The kernel is a masked AXPY with a real tail — `n=17` against `BLOCK=16`
over a grid of two, so the second program has one active lane.

## The thing worth internalising

**Six fewer instructions bought twenty-two fewer cycles here. On the
GEMM in [05](05_matrix_multiply.md), seventy-three fewer instructions
buy about twenty-four fewer cycles.**

Those are not contradictory; they are the same fact seen twice. An
instruction removed from a cycle where the core was *already stalled* —
waiting on memory, waiting on the matrix unit — removes nothing. The
static count and the cycle count are different measurements and they
come apart exactly where the machine is waiting rather than issuing.

So when you are judging whether an option helped, read the cycles. The
word count tells you what the compiler did; the cycle count tells you
whether it mattered.

## What O1 actually did to this kernel

From [04](04_reading_the_compile.md)'s dump of a similar kernel:

```text
opt_strength_reduced         1      pid * 16 became a shift
opt_unobserved_load_other    1      a masked load's `other` nobody could observe
opt_dead_pure_removed        1      a constant with no remaining user
opt_ops_before / after      22 / 18 what upstream canonicalisation and CSE removed
```

The first is TNCA's own lowering. The last is the pinned upstream MLIR
passes running on the real Triton IR — the same canonicalizer and CSE
Triton itself uses.

## Correctness does not change

All four modes produce identical numbers, checked against an oracle
computed in Python. That is the minimum bar for an optimisation and it
is checked on every run, not assumed.

---

Next: [03 · Layout views](03_layout_views.md)
