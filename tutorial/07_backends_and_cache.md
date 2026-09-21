# 07 — Backends, profiles, and the compilation cache

[`examples/07_backends_and_cache.py`](examples/07_backends_and_cache.py)

Two ideas that are easy to confuse, and one piece of plumbing that only
matters when it is wrong.

> A **profile** says what machine you are compiling *for* — the ISA, the
> capabilities, the register file. It is part of the artifact's
> identity.
>
> A **backend** says how this host *reaches* a device. It is not part of
> the artifact at all.

The consequence you care about: changing the profile means a different
artifact; changing the backend means the same artifact, no recompile.

## What is available

```python
from tnca import backend
backend.available()
```

```
backends this installation knows:
  fpga-prototype   interface-defined    a future board. The interface is defined and
                                        nothing implements it; opening it is an error,
                                        not a redirection to the simulator.
  sim-file         reference-available  the simulated device over a file rendezvous,
                                        which is what this repository has always used
  sim-socket       reference-available  the same RTL, reached over a TCP socket through
                                        a relay process: a different connection medium
                                        for the same implementation
```

Those states are not decoration. `interface-defined` means the contract
exists and nothing implements it, and asking for one is an error:

```
  asking for one that is not implemented: [BACKEND_NOT_IMPLEMENTED]
    [BACKEND_NOT_IMPLEMENTED] the backend 'fpga-prototype' is declared
    and not implemented
```

A registry that quietly fell back to the simulator when the board was
missing would let a platform that was never there appear to work. The
registry refuses to register anything claiming to be platform-verified,
and it never falls back.

## One artifact, two connections

```python
with Session(profile=PROFILE, backend='sim-socket') as s:
    mod = s.load(art.path)       # the same file as for sim-file
```

```
one artifact, key f6b102750788:
  sim-file     via a shared directory     81 cycle(s), correct=True
  sim-socket   via tcp                    81 cycle(s), correct=True
```

The cycle counts are equal because both media reach the *same* RTL. That
equality is the evidence: a socket transport that fabricated a plausible
answer would agree on the values and not on the cycles.

## The cache

```python
from tnca.cache import Cache
cache = Cache('/some/dir')
art = aot.compile_aot(kernel, ..., cache=cache)
art.from_cache        # True if this compile was served, not performed
art.cache_key         # what it was served under
```

```
the cache, which is about not compiling twice:
  the same request twice      from_cache=True, key f6b102750788
  a different opt level       from_cache=False, key 89f2e6fece51
```

What is in the key is the whole question. The **optimisation level is**,
because a kernel compiled at O0 served for an O2 request would make
every measurement after it a measurement of the wrong artifact — which
is exactly the mistake [02](02_optimisation_levels.md) is built to
avoid. Also in the key: the source, the signature, the constexprs, the
profile, and the toolchain lock hash, so a Triton upgrade does not serve
you yesterday's code.

A **diagnostic option is not**:

```
  a diagnostic option is not in the key, because it changes nothing:
    with dump_dir: key f6b102750788 (same: True)
```

`dump_dir` and `debug` only change what is written *beside* the
artifact. If they were in the key, turning on a dump to investigate a
bug would hand you a different kernel than the one you were
investigating. Triton keeps `TRITON_KERNEL_DUMP` out of its own key for
the same reason.

## Choosing a profile

```python
from tnca import device
device.prepare('core-vrf-sync-v1')     # builds the simulated device once
```

The profile decides what the compiler may emit. `core-vrf-sync-v1` is
the single core with the banked, finite-port register file — the one
every page here uses. Compile the same source for a different profile
and you get a different artifact with a different key, and
[06](06_errors_and_faults.md) shows what happens if you try to load it
into the wrong session.

Next: [08 — a GEMM written as a Triton kernel](08_triton_gemm.md).
