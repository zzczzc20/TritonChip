# 01 · Your first kernel

**Run it:** `PYTHONPATH=python:compiler/python python tutorial/examples/01_hello_vector_add.py`

A Triton kernel, compiled for the TNCA core, run on the simulated
hardware, and checked against an answer computed in Python.

## The kernel

```python
@triton.jit
def vector_add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x + y, mask=mask)
```

Ordinary Triton. It goes through the real Triton frontend — not a
reimplementation of it — and comes out as TNCA machine code.

## The four steps

```python
device.prepare(PROFILE, verbose=False)          # 1. build the device, once
art = aot.compile_aot(vector_add, ...)          # 2. compile
with Session(profile=PROFILE) as s:             # 3. run
    ...
want = [xs[i] + ys[i] for i in range(n)]        # 4. check
```

`device.prepare` builds the simulator if it is not already built.
`Session` deliberately will **not** build one behind your back: a launch
that quietly compiled Verilog would hide minutes of work inside what
looks like a function call.

## What it prints

```text
compiled  16 machine word(s), cache key a73b552fd72d
launched  SUCCEEDED, 101 device cycle(s)
result    [100.0, 100.0, 100.0, 100.0] ...
expected  [100.0, 100.0, 100.0, 100.0] ...
correct   every element matches
```

Sixteen machine words for the whole kernel, and 101 device cycles to
run it. Those are real hardware cycles from the RTL, not a host timing.

## The signature, and why it is separate from the arguments

```python
signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
           'n': 'i32', 'BLOCK': 'constexpr'},
constexprs={'BLOCK': n},
```

The signature says what each parameter **is**; the values come later, at
launch. One artifact runs with any arguments of that shape — you will
see that used in [07](07_backends_and_cache.md), where the same compiled
kernel is launched repeatedly with different values and the compiler is
never invoked again.

A `constexpr` is different: it is baked in at compile time, so changing
`BLOCK` gives you a different artifact.

## Buffers

```python
x = s.allocate(n * 4, name='x')
s.upload(x, f32(xs))
...
got = unf32(s.download(o, n * 4))
```

Device memory is allocated from the session, and the names are for you —
they appear in diagnostics when something goes wrong with that buffer.
Host and device memory are separate: `upload` and `download` are real
transfers, and there is no coherence between them. The runtime enforces
that access is serial rather than trusting you to remember.

---

Next: [02 · Optimisation levels](02_optimisation_levels.md)
