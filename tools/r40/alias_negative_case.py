#!/usr/bin/env python3
"""Compound bounds, illegal-alias and storage-ownership runtime refusals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='tnca-r41-alias-negative/1', status='running', checks=[],
                  board_acceptance='not-measured')
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        import triton  # noqa: F401
        from archbench.provenance import software_tree
        from tnca import aot, backend as backend_registry, device
        from tnca.errors import TncaError
        from tnca.locality import plan_commands
        from tnca.runtime import Buffer, Session
        from tnca.triton_backend.driver import backend_choice
        profile = os.environ.get('TNCA_PROFILE', 'core-vrf-sync-v5')
        source = software_tree(ROOT, refresh=True)
        artifact = out/'two_inputs.tnca'
        aot.compile_aot(lambda b, target: b.exit(), kernel_name='two_inputs',
                        source_kind='tnca-ir', profile=profile,
                        signature={'a':'*i8', 'b':'*i8'}, output=str(artifact))
        backend, kwargs = backend_choice()
        kwargs.setdefault('deadline_cycles', int(os.environ.get('TNCA_DEADLINE', '5000000')))
        kwargs.setdefault('host_timeout', float(os.environ.get('TNCA_HOST_TIMEOUT', '600')))
        if backend_registry.needs_elaborator(backend):
            device.prepare(profile)
        with Session(profile=profile, backend=backend, **kwargs) as session:
            module = session.load(artifact)
            storage = session.allocate(128)
            session.upload(storage, bytes(storage.size))
            before = session.latest_event
            def refused(name, call):
                try:
                    call()
                except TncaError as error:
                    assert session.latest_event is before, name+' reached admission'
                    report['checks'].append(dict(name=name, status='refused-before-admission',
                                                 error_type=type(error).__name__, error=str(error)))
                else:
                    raise AssertionError(name+' was accepted')
            refused('undeclared-overlapping-storage-arguments',
                    lambda: session.submit(module, {'a':storage, 'b':storage}))
            refused('out-of-bounds-view',
                    lambda: session.submit(module, {'a':storage, 'b':(storage, storage.size)}))
            forged = Buffer(storage.id+1000, 1 << 63, storage.size,
                            session.id, session.epoch, 'forged-high-bit')
            forged.valid = True
            command = (session, module, {'a':storage, 'b':forged}, (1,1,1), (), (), session.epoch, {})
            refused('forged-high-bit-storage-owner', lambda: plan_commands([command]))
            overlap = Buffer(storage.id+1001, storage.addr, storage.size,
                             session.id, session.epoch, 'injected-overlap')
            overlap.valid = True
            session._buffers[overlap.id] = overlap
            try:
                command = (session, module, {'a':storage, 'b':overlap}, (1,1,1), (), (), session.epoch, {})
                refused('distinct-storage-handles-overlap', lambda: plan_commands([command]))
            finally:
                del session._buffers[overlap.id]
            report.update(backend=backend, software_tree=source,
                          artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
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
