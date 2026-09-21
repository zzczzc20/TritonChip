# 03 · Layout views, and what the algebra is worth

**Run it:** `PYTHONPATH=python:compiler/python python tutorial/examples/03_layout_views.py`

This is the one case where O2 does something O1 cannot, and the example
is built so you can switch that something off and measure the
difference.

## The program

Written against the TNCA IR builder rather than Triton, because a layout
view is a TNCA concept:

```python
v = b.load(p_in, mask=keep, other=0, dtype='f32')

t = b.view(v, TRANSPOSE, ['row', 'col'], [ROWS, COLS])   # read as row/col
u = b.view(t, BACK, ['reg'], [N])                        # and back again

b.store(p_out, b.vbin('add', u, u, dtype='f32'), mask=keep)
```

Read a 4×4 tile as rows and columns, read it back the way it came,
double it, store it. The two views cancel.

A layout is given the way upstream `LinearLayout` takes it — for each
input dimension, one basis row per bit of its index, saying where that
bit sends the value:

```python
TRANSPOSE = [('reg', [[1, 0], [2, 0], [0, 1], [0, 2]])]
BACK      = [('row', [[1], [2]]), ('col', [[4], [8]])]
```

## What it prints

```text
mode              words  cycles  correct
B0 (default)         63     215  yes
O2                    9      61  yes
O2 with no_ll        58     208  yes
```

## Reading that

At **O2** the compiler hands both layouts to the real upstream
`LinearLayout`, composes them, and compares the composition against
upstream's identity over all sixteen positions. It is the identity, so
no value has to move and nothing is emitted for either view. Sixty-three
words become nine; 215 cycles become 61.

**`no_ll`** switches off the layout algebra and keeps everything else.
Each view is then materialised the only way the ISA allows: an index
vector built from the lane index with shifts and XORs, then a
`v.permute`. That costs 58 words and 208 cycles.

**The 147-cycle difference between `no_ll` and O2 is what the layout
reasoning is worth on this program** — measured by turning it off, not
argued from the design.

## Why this comparison is trustworthy

All three modes produce the same numbers. The views cancel
mathematically whether or not anything notices, so the *only* thing that
differs between the runs is what the compiler decided. That is what
makes the cycle difference attributable to the layout algebra and
nothing else.

## What a view is not

A layout maps a *virtual* position to tensor indices. It says nothing
about which memory bank a value lands in — a bank is a property of the
physical register that allocation chooses later. Reading bank behaviour
off a layout would be wrong, and the compiler's cost model deliberately
has no access to layouts for that reason.

If a view cannot be realised on this ISA, you get a diagnostic with a
position rather than a silently invented instruction.

---

Next: [04 · Reading the compile](04_reading_the_compile.md)
