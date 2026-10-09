#!/usr/bin/env python3
"""Original TileBench through compile_aot and Session, at each optimisation level."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
import execute as E
import upstream_cases
from tnca import aot, device
from tnca.runtime import Session, TensorDescriptor


def run(name, case, level, profile, directory, *, compile_only=False, session_kwargs=None,
        session_profile=None):
    fn, signature, constexprs = case.kernel_spec
    artifact = aot.compile_aot(fn, signature=signature, constexprs=constexprs,
                              profile=device.base_profile(profile), options={'opt_level': level},
                              output=str(directory / f'{name.replace("/", "_")}-O{level}.tnca'))
    if compile_only:
        return {}, dict(detail='compiled through AOT', artifact=artifact.path,
                        scratch_bytes=artifact.manifest['resources']['scratch_bytes'])
    outputs = [(case.out, case.out_offset, case.expect, case.rtol, case.atol)]
    outputs += [(b, o, v, 0., 0.) for b, o, v in case.extra_outputs]
    grid = tuple(p+1 for p in case.program_id)
    # The public runtime launches complete grids. Validate both batches when
    # the direct RTL case selected only batch 1; the oracle uses input data.
    if case.kernel == 'batched_matmul':
        a, bt = case.buffers['a'][1], case.buffers['bt'][1]
        expected = [sum(a[z*256+m*16+k]*bt[z*256+n*16+k] for k in range(16))
                    for z in range(2) for m in range(16) for n in range(16)]
        outputs = [(case.out, 0, expected, case.rtol, case.atol)]
    # session_profile: run on a different machine than the one compiled for
    # (a profile that executes another's artifacts).
    with Session(profile=session_profile or profile, deadline_cycles=500_000,
                 **(session_kwargs or {})) as session:
        module = session.load(artifact.path)
        handles, initial = {}, {}
        for name, (dtype, values) in case.buffers.items():
            size = E.DTYPES[dtype][0]
            data = bytearray(b''.join(w.to_bytes(8, 'little') for w in E.pack(values, dtype)))[:len(values)*size]
            data += b'\xa5'*8
            for buf, offset, expect, _, _ in outputs:
                if name == buf and not case.preserve_output:
                    data[offset*size:(offset+len(expect))*size] = b'\xcd'*(len(expect)*size)
            handle = session.allocate(len(data), name=name)
            session.upload(handle, data)
            handles[name], initial[name] = handle, bytes(data)
        args = dict(case.scalars)
        for name, handle in handles.items():
            if name in signature:
                offset = case.offsets.get(name, 0)*E.DTYPES[case.buffers[name][0]][0]
                args[name] = (handle, offset) if offset else handle
        for name, desc in case.descriptors.items():
            args[name] = TensorDescriptor(handles[desc['buffer']], desc['shape'], desc['strides'],
                                          desc['block'], case.buffers[desc['buffer']][0])
        event = session.submit(module, args, grid=grid)
        session.wait(event)
        if event.state != 'SUCCEEDED':
            raise RuntimeError(str(event.error))
        captured = {}
        count = 0
        for name, handle in handles.items():
            data = session.download(handle)
            if len(data) != len(initial[name]):
                raise AssertionError(f'{name}: truncated download')
            dtype, values = case.buffers[name]
            size, _, decode = E.DTYPES[dtype]
            unchanged = bytearray(b'\1'*len(data))
            for buf, offset, expect, rtol, atol in outputs:
                if name != buf:
                    continue
                start, end = offset*size, (offset+len(expect))*size
                got = [decode(int.from_bytes(data[i:i+size], 'little')) for i in range(start, end, size)]
                ok, detail = E._compare_values(got, expect, rtol, atol)
                if not ok:
                    raise AssertionError(f'{buf}: {detail}')
                unchanged[start:end] = bytes(end-start)
                count += len(expect)
            if any(check and a != b for check, a, b in zip(unchanged, data, initial[name])):
                raise AssertionError(f'{name}: input, output tail or guard overwritten')
            captured[name] = (dtype, [decode(int.from_bytes(data[i:i+size], 'little'))
                                      for i in range(0, len(values)*size, size)])
    return captured, dict(detail=f'{count} elements correct; inputs, tails and guards intact',
                          grid=grid, cycles=event.device_cycles, program_cycles=event.program_cycles,
                          counters=event.counters,
                          artifact=artifact.path, scratch_bytes=artifact.manifest['resources']['scratch_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'Tilebench')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--levels', nargs='+', type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument('--profile', default='core-sim')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--json', type=Path, default=ROOT/'sim/build/r37_aot_run.json')
    args = parser.parse_args()
    directory = args.json.parent / (args.json.stem+'_artifacts')
    directory.mkdir(parents=True, exist_ok=True)
    cases = upstream_cases.build(E.Case, args.source.resolve())
    known = {c.kernel for c in cases.values()}
    if args.only and set(args.only)-known:
        parser.error('unknown operators: '+str(set(args.only)-known))
    inventory = {r['id'] for r in json.loads((ROOT/'spec/r37_tilebench.json').read_text())['operators']}
    if inventory != known:
        parser.error('case inventory differs from the 45-operator specification')
    if not args.compile_only:
        device.prepare(args.profile, verbose=False)
    rows = []
    for level in args.levels:
        captured = {}
        for name, original in cases.items():
            if args.only and original.kernel not in args.only:
                continue
            case = copy.copy(original)
            case.buffers = copy.deepcopy(original.buffers)
            row = dict(case=name, operator=case.kernel, level=level, ok=False)
            start = time.monotonic()
            try:
                for buf, feed in (() if args.compile_only else case.feeds.items()):
                    dtype, data = captured[feed[0]][feed[1]]
                    if len(feed) == 3:
                        nr, nc = feed[2]
                        data = [data[r*nc+c] for c in range(nc) for r in range(nr)]
                    case.buffers[buf] = (dtype, data)
                captured[name], detail = run(name, case, level, args.profile, directory, compile_only=args.compile_only)
                row.update(ok=True, **detail)
            except Exception as error:
                row['detail'] = f'{type(error).__name__}: {error}'
            source = Path(case.kernel_spec[0].fn.__code__.co_filename)
            row.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       seconds=round(time.monotonic()-start, 2))
            rows.append(row)
            args.json.write_text(json.dumps(dict(profile=args.profile, levels=args.levels,
                                                 kind='aot-compile' if args.compile_only else 'aot-session', cases=rows,
                                                 passed=sum(r['ok'] for r in rows), total=len(rows)), indent=2)+'\n')
            print(f'O{level} {"PASS" if row["ok"] else "FAIL"} {name}: {row["detail"]}', flush=True)
    operators = []
    for level in args.levels:
        for operator in sorted(known if not args.only else set(args.only)):
            selected = [r for r in rows if r['level'] == level and r['operator'] == operator]
            expected = sum(c.kernel == operator for c in cases.values())
            operators.append(dict(level=level, operator=operator, passed=sum(r['ok'] for r in selected),
                                  total=expected, ok=len(selected) == expected and all(r['ok'] for r in selected)))
    report = json.loads(args.json.read_text())
    report.update(completed=True, operators=operators, operators_passed=sum(r['ok'] for r in operators),
                  operators_total=len(operators))
    args.json.write_text(json.dumps(report, indent=2)+'\n')
    label = 'AOT compilation' if args.compile_only else 'AOT + Session'
    print(f'{label}: {sum(r["ok"] for r in rows)}/{len(rows)} cases; '
          f'{report["operators_passed"]}/{len(operators)} operator/level pairs pass', flush=True)
    return 0 if rows and all(r['ok'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
