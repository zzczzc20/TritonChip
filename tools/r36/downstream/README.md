# A downstream branch, rehearsed

This directory is what a platform branch looks like from the outside. It
is the rehearsal R36-HAND-01 asks for, and it is kept honest by three
rules it is checked against:

1. **It reads only delivered things.** The hardware export package and
   the installed wheel. Nothing in this directory refers to a path in
   the main repository, and `tools/r36/handoff_tests.py` greps for that
   rather than trusting it.

2. **It patches nothing.** No change to the common compiler, the runtime
   state machine or the compute core. It adds a file and registers a
   backend through the published extension point. The checker compares
   the installed package's files before and after.

3. **It runs on real RTL.** The small program really executes, its
   answers are checked against an oracle computed here, and the shared
   conformance suite runs against the backend it registered.

## What it contains

| file | what |
|---|---|
| `platform_backend.py` | a backend registered from outside the package, built from the EXPORT's RTL |
| `run_rehearsal.py` | the rehearsal: register, open, upload, launch, download, check |

## The extension point

```python
from tnca import backend

backend.register('my-board', factory,
                 state=backend.REFERENCE_AVAILABLE,
                 what='what this binding is')
```

That is the whole surface. A backend implements `tnca.backend.Backend`,
which is eight methods, and the runtime's task and buffer state machines
are not among them: section 3.1 keeps device lifetime separate from the
compile target and from the runtime's bookkeeping, and a binding that
needed to reach into either would be an interface gap to fix on the
mainline rather than something to work around here.
