#!/usr/bin/env python3
"""Bounded large-grid admission, queue overflow refusal and exact completion."""
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
    parser.add_argument('--programs', type=int, default=65537)
    args = parser.parse_args()
    if args.programs <= 4096:
        raise ValueError('the case requires a grid above the legacy admission limit')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='tnca-r41-grid-admission/1', status='running',
                  requested_programs=args.programs, checks=[], events=[],
                  board_acceptance='not-measured')
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        import triton  # noqa: F401
        from archbench.provenance import software_tree
        from tnca import aot, backend as backend_registry, device
        from tnca.errors import AdmissionError, QueueFull
        from tnca.runtime import MAX_UNFINISHED, Session
        from tnca.triton_backend.driver import backend_choice
        profile = os.environ.get('TNCA_PROFILE', 'core-vrf-sync-v5')
        source = software_tree(ROOT, refresh=True)
        def streamed(b, target):
            base = b.special('arg_base')
            result, counter = b.sload(base, 0), b.sload(base, 8)
            x, y, z = [b.program_id(i) for i in range(3)]
            nx, ny = b.num_programs(0), b.num_programs(1)
            linear = b.sbin('add', x, b.sbin('mul', nx,
                b.sbin('add', y, b.sbin('mul', ny, z))))
            b.sstore(b.sbin('add', result, b.sbini('mul', linear, 8)), linear, size='d')
            b.atomic('add', counter, b.const(1), size='d')
            b.fence()
            b.exit()
        artifact = out/'grid_admission.tnca'
        aot.compile_aot(streamed, kernel_name='grid_admission', source_kind='tnca-ir',
                        profile=profile, signature={'out':'*i64', 'counter':'*i64'},
                        options={'rctx_spmd':1}, output=str(artifact))
        backend, kwargs = backend_choice()
        kwargs.setdefault('deadline_cycles', int(os.environ.get('TNCA_DEADLINE', '5000000')))
        kwargs.setdefault('host_timeout', float(os.environ.get('TNCA_HOST_TIMEOUT', '600')))
        if backend_registry.needs_elaborator(backend):
            device.prepare(profile)
        with Session(profile=profile, backend=backend, large_grid=False, **kwargs) as session:
            module = session.load(artifact)
            result = session.allocate(args.programs*8)
            counter = session.allocate(8)
            session.upload(result, bytes([255])*result.size)
            session.upload(counter, bytes(8))
            base_limit = (session.dev.physical_grid_capability()[1]
                          if session.physical_grid else 4096)
            extension_limit = session.dev.large_grid_limit()
            limit = extension_limit or base_limit
            assert base_limit < 0xFFFFFFFF and args.programs <= limit
            refused_programs = max(args.programs, base_limit + 1)
            before = session.latest_event
            try:
                session.submit(module, {'out':result, 'counter':counter}, grid=(refused_programs,1,1))
                raise AssertionError('above-capability grid was admitted without opt-in')
            except AdmissionError as error:
                assert session.latest_event is before
                report['checks'].append(dict(name='opt-in-required', status='pass',
                    base_limit=base_limit, requested=refused_programs, error=str(error)))
                save()
                print('grid admission: opt-in refusal passed', flush=True)
            session.large_grid = True
            long_event = session.submit(module, {'out':result, 'counter':counter},
                                        grid=(args.programs,1,1))
            dependent = []
            overflow = None
            for index in range(MAX_UNFINISHED):
                try:
                    dependent.append(session.submit(module, {'out':result, 'counter':counter},
                        grid=(1,1,1), depends_on=[long_event]))
                except QueueFull as error:
                    overflow = error
                    break
            assert overflow is not None and len(dependent) == MAX_UNFINISHED-1
            assert len(session.unfinished) == MAX_UNFINISHED
            report['checks'].append(dict(name='bounded-queue-overflow', status='pass',
                queue_limit=MAX_UNFINISHED, admitted_dependent=len(dependent), error=str(overflow)))
            save()
            print('grid admission: bounded queue passed; draining', flush=True)
            events = [long_event] + dependent
            for event in events:
                session.wait(event)
                assert event.ok, event.error
                report['events'].append(dict(job_id=event.job_id, state=event.state,
                                             device_cycles=event.device_cycles))
                save()
            raw = session.download(result)
            values = struct.unpack('<%dQ' % args.programs, raw)
            completions = struct.unpack('<Q', session.download(counter))[0]
            assert values == tuple(range(args.programs))
            assert completions == args.programs + len(dependent)
            report['checks'].append(dict(name='exact-grid-and-drain', status='pass',
                unique_program_ids=len(set(values)), completion_atomics=completions,
                output_sha256=hashlib.sha256(raw).hexdigest()))
            save()
            print('grid admission: exact output and atomic count passed', flush=True)
            before = session.latest_event
            try:
                session.submit(module, {'out':result, 'counter':counter}, grid=(limit+1,1,1))
                raise AssertionError('grid above advertised capacity was admitted')
            except AdmissionError as error:
                assert session.latest_event is before
                report['checks'].append(dict(name='advertised-capacity-refusal', status='pass',
                    advertised_limit=limit, requested=limit+1, error=str(error)))
            report.update(backend=backend, software_tree=source,
                          artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                          memory_stats=session.memory_stats)
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
