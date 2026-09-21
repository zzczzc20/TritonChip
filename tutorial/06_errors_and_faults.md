# 06 — What is refused, what faults, and what you are told

[`examples/06_errors_and_faults.py`](examples/06_errors_and_faults.py)

A compiler that accepts nonsense and a runtime that returns plausible
rubbish are both worse than ones that stop. This page is the set of
mistakes people actually make, and exactly what each one says.

Run it:

```sh
PYTHONPATH=python:compiler/python ~/tritonenv/bin/python tutorial/examples/06_errors_and_faults.py
```

## Every failure has a code

`tnca.errors.TncaError` carries a `.code`. You branch on the code, not
on the text — the text is for you, the code is for your program.

```python
from tnca.errors import TncaError

try:
    art = aot.compile_aot(kernel, signature=SIG, profile=PROFILE, ...)
except TncaError as e:
    print(e.code)      # 'UNSUPPORTED', 'PROFILE_MISMATCH', ...
```

## Compile time

```
at compile time:
  an option this driver does not know          [COMPILE_FAILED]
                                               [COMPILE_FAILED] unknown compile option(s): optimise_level
  a profile that does not exist                [PROFILE_MISMATCH]
                                               [PROFILE_MISMATCH] no profile 'core-imaginary'
  a block wider than one element per lane      [UNSUPPORTED]
                                               [UNSUPPORTED] a block of 256 spans 16 vector registers
```

`optimise_level` is a typo for `opt_level`. A driver that ignored
unknown options would compile at O0 and hand you an artifact you
believed was optimised; every measurement after that would be of the
wrong thing. So an unknown option is an error, not a shrug.

`BLOCK=256` on a machine with 16 lanes needs a tensor spread over 16
vector registers with its index mapping intact. That is not built, so it
is refused rather than half-done.

## Load and launch time

```
at load and launch time:
  an artifact built for another profile        [PROFILE_MISMATCH]
                                               [PROFILE_MISMATCH] artifact is for profile 'core-sim', this session is
                                               'core-vrf-sync-v1'
                                               and 0 device request(s) were made
  an argument the kernel does not take         [ARGUMENT_INVALID]
                                               [ARGUMENT_INVALID] argument mismatch: missing [], unexpected
                                               ['nonsense']
  a missing argument                           [ARGUMENT_INVALID]
                                               [ARGUMENT_INVALID] argument mismatch: missing ['o_ptr'], unexpected []
```

Read the middle line again: **0 device requests were made**. The
mismatch is caught from the artifact's own header before anything is
written to the device. That matters — a check performed after a partial
upload leaves the device in a state nobody described.

The example asserts this, rather than describing it:

```python
before = s.dev.seq
show("an artifact built for another profile", lambda: s.load(ref.path))
ok &= (s.dev.seq - before) == 0
```

Arguments are matched by name against the artifact's recorded ABI, in
both directions: an extra one is as much an error as a missing one.
Passing `n=16` when the kernel wanted `n` and `o_ptr` is a bug, and
silently defaulting the missing pointer to zero would turn it into a
memory fault a hundred lines later.

## A fault on the device

Not everything is caught before launch. This is a kernel that really
traps:

```python
def trapping(b, target):
    v = b.const(1)
    b.assert_eq(v, 0xDEAD, 7)          # 1 is not 0xDEAD
    b.exit()
```

```
a kernel that really faults on the device:
  the event ended FAILED, ok=False
  and says why: [KERNEL_FAULT] [KERNEL_FAULT] kernel trapped: code 0x40 at pc 0x50
  the session afterwards: still usable
```

Three things to take from that line:

- the **event** records the failure (`ev.state`, `ev.ok`, `ev.error`),
  so a launch you did not `wait()` on still knows it failed;
- the fault carries a **code and a pc**, which is the address in the
  artifact's disassembly — `04`'s `asm` dump is how you find the
  instruction;
- the **session survives**. A drainable fault leaves the device usable.
  One that does not says so, rather than letting you keep submitting to
  a device in an undefined state.

A note on what does *not* fault: passing a huge `n` to the masked kernel
of [01](01_first_kernel.md) is not an error. It only widens a mask, and
all sixteen lanes were inside the buffer already. A fault has to come
from the program doing something the machine refuses, not from an
argument you consider unreasonable.

## The shape of error handling you want

```python
ev = s.submit(mod, args)
try:
    s.wait(ev)
except TncaError:
    pass                      # the event already knows
if not ev.ok:
    log(ev.error.code, ev.error)
```

Next: [07 — backends, profiles and the cache](07_backends_and_cache.md).
