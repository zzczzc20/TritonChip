#!/usr/bin/env python3
"""The downstream rehearsal itself -- R36-HAND-01.

Run from a directory that is not the repository, against an installed
package and a hardware export package:

    python run_rehearsal.py --package /path/to/export

What it does, in the order a platform branch would do it:

    1. register a backend through the published extension point
    2. build a device from the EXPORT's RTL, not the repository's
    3. open it, upload a program, launch, download, check the answer
       against an oracle computed here
    4. run the shared conformance suite against it

Nothing here compiles a kernel. The contract asks for "compile or load",
and loading is the harder claim to make honestly: a runtime-only install
has no Triton and no compiler, and this has to work in one.
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The minimal correctness program, its address and its expected value
# all come from the shared conformance suite rather than being copied
# here. The first version of this file kept its own copy of the words,
# which had been taken from a DIFFERENT program -- the export's smoke
# test, which writes to another address -- so the rehearsal launched
# successfully and then compared the wrong location and reported WRONG.
# Two copies of a constant is one too many.


def _load_suite(path):
    """The shared conformance suite, imported so it sees our registry.

    Found beside this file, or named with --suite. It used to be looked
    for one directory UP, which inside the repository is tools/r36 and
    outside it is wherever the rehearsal happened to be unpacked -- so
    on one run it imported a stale copy left next to the directory.
    """
    conf = path or os.path.join(HERE, 'conformance.py')
    if not os.path.exists(conf):
        print(f"the shared conformance suite is not at {conf}. It is part "
              f"of the handoff package; pass --suite if it is elsewhere.")
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location('_conf', conf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', required=True,
                    help='the hardware export package')
    ap.add_argument('--program', help='a JSON list of machine words; '
                                      'the built-in one is used otherwise')
    ap.add_argument('--suite',
                    help='the shared conformance suite; by default the '
                         'copy beside this file')
    ap.add_argument('--json')
    args = ap.parse_args()

    import platform_backend
    from tnca import backend

    suite = _load_suite(args.suite)
    if suite is None:
        return 2

    print("the downstream rehearsal")
    print(f"  package {args.package}")
    print(f"  tnca from {os.path.dirname(backend.__file__)}")

    name, device = platform_backend.install(args.package)
    print(f"  registered {name!r} through backend.register, device built "
          f"from the package's own RTL")
    print(f"    {os.path.relpath(device, args.package)}")

    info = backend.available()[name]
    print(f"  the registry now lists it as {info['state']!r}")

    words = (json.load(open(args.program)) if args.program
             else suite.MINIMAL_PROGRAM)
    dev = backend.open_backend(name, deadline_cycles=200_000)
    dev.start()
    ok = True
    try:
        caps = dev.capabilities()
        print(f"  capabilities: medium {caps.get('medium')!r}, built from "
              f"{os.path.basename(caps.get('built_from', '?'))}")
        dev.upload(0, b''.join(struct.pack('<Q', w) for w in words))
        dev.upload(suite.MARK_ADDR, struct.pack('<Q', 0))
        r = dev.launch(0, 0)
        print(f"  launch: status {r.status}, {r.r0} cycle(s), {r.r1} "
              f"instruction(s) retired")
        ok &= r.ok
        _q, blob = dev.download(suite.MARK_ADDR, 8)
        got = struct.unpack('<Q', blob)[0]
        print(f"  the program wrote {got:#x}; the oracle says "
              f"{suite.MARK:#x}: "
              f"{'correct' if got == suite.MARK else 'WRONG'}")
        ok &= got == suite.MARK
    finally:
        dev.stop()

    print()
    print("  and the shared conformance suite, against the backend this "
          "rehearsal registered:")
    log = []
    good, record = suite.run_backend(name, backend.available()[name], log)
    for line in log:
        print(f"  {line.strip()[:100]}")
    ok &= good

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'backend': name, 'device': device,
                       'package': args.package, 'conformance': record,
                       'ok': bool(ok)}, fh, indent=1)
    print()
    print(f"  rehearsal {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
