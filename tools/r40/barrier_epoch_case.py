#!/usr/bin/env python3
"""Cross the 16-bit barrier generation boundary through AOT and Session."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', type=int, default=65537)
    args = parser.parse_args()
    if args.arms <= 65536:
        raise ValueError('this acceptance case requires more than 65,536 arms')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='tnca-r41-barrier-epoch/1', status='running',
                  requested_arms=args.arms, board_acceptance='not-measured')
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        import triton  # noqa: F401
        from archbench.provenance import software_tree
        from tnca import aot, backend as backend_registry, device
        from tnca.runtime import Session
        from tnca.triton_backend.driver import backend_choice
        profile = os.environ.get('TNCA_PROFILE', 'core-vrf-sync-v5')
        source = software_tree(ROOT, refresh=True)
        def program(b, target):
            base = b.special('arg_base')
            destination = b.sload(base, 0)
            limit = b.sbini('add', b.sload(base, 8), 1)
            barrier = b.barrier()
            with b.loop(b.const(1), limit):
                token = b.barrier_arm(barrier)
                b.barrier_wait(token)
            b.sstore(destination, token, size='d')
            b.exit()
        artifact = out/'barrier_epoch.tnca'
        aot.compile_aot(program, kernel_name='barrier_epoch', source_kind='tnca-ir',
                        profile=profile, signature={'out':'*i64', 'arms':'i64'},
                        output=str(artifact))
        backend, kwargs = backend_choice()
        kwargs.setdefault('deadline_cycles', int(os.environ.get('TNCA_DEADLINE', '5000000')))
        kwargs.setdefault('host_timeout', float(os.environ.get('TNCA_HOST_TIMEOUT', '600')))
        if backend_registry.needs_elaborator(backend):
            device.prepare(profile)
        with Session(profile=profile, backend=backend, **kwargs) as session:
            module = session.load(artifact)
            result = session.allocate(8)
            event = session.wait(session.submit(module, {'out':result, 'arms':args.arms}))
            report.update(backend=backend, software_tree=source,
                          artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                          event=dict(state=event.state, ok=event.ok,
                                     device_cycles=event.device_cycles, error=str(event.error)),
                          memory_stats=session.memory_stats)
            assert event.ok, event.error
            observed = struct.unpack('<Q', session.download(result))[0]
            report['observed'] = dict(token=observed, generation=observed >> 16,
                                      barrier=observed & 0xffff)
            assert observed == args.arms << 16, report['observed']
            assert software_tree(ROOT, refresh=True) == source
        report['status'] = 'pass'
    except BaseException:
        report.update(status='failed-preserved', error=traceback.format_exc())
        save()
        raise
    save()
    print(json.dumps(report))


if __name__ == '__main__':
    main()
