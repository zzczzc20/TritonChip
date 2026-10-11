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
  fpga-u280        reference-available  the TNCA device on an Alveo U280: the U50 kernel
                                        and register port (rtl/fpga/tnca_fpga_top.v)
                                        built for the U280 shell, reached through the
                                        same U50Backend
  fpga-u50         reference-available  the TNCA device on an Alveo U50:
                                        rtl/fpga/tnca_fpga_top.v reached over its
                                        register port. The host side is proven against
                                        the RTL; the board link is not, and opening this
                                        with no board raises rather than substituting
                                        the stand-in
  fpga-u50-sim     reference-available  the same device and the same register protocol,
                                        on real RTL, driven by a simulator standing in
                                        for the board
  sim-file         reference-available  the simulated device over a file rendezvous,
                                        which is what this repository has always used
  sim-socket       reference-available  the same RTL, reached over a TCP socket through
                                        a relay process: a different connection medium
                                        for the same implementation
```

The two board backends are real: `fpga-u280` is what
[11](11_running_on_the_u280.md) runs on. `fpga-u50-sim` is the same
register protocol driven against the RTL in a simulator, which is how
the board's host side is tested without a board — and how a board run
is checked cycle for cycle.

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
one artifact, key 058db3dcd3a1:
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
  the same request twice      from_cache=True, key 058db3dcd3a1
  a different opt level       from_cache=False, key b8e49aae5408
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
    with dump_dir: key 058db3dcd3a1 (same: True)
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
