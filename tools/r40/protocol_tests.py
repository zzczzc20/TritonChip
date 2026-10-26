#!/usr/bin/env python3
"""R40-00 acceptance 1: the protocol refuses every way of dressing up a number.

    python3 tools/r40/protocol_tests.py [--evidence] [--json cases.json]

Synthetic ArchBench records (protocol 2), so this needs neither torch nor a
board. One clean pair is built -- two configurations of the same frozen
subset that differ only in the knob under test, with full provenance and
timed samples -- and checked to be accepted. Then each manipulation
doc/r40_tasks_and_acceptance.md R40-00 lists is applied to a copy of it and
must be refused, for the right reason:

  a slow point deleted, an unmeasured point zero-filled, two images mixed on
  one side, a tolerance changed, a weight changed, a provenance field
  missing, per-core cycles summed into a makespan, only the clock changed,
  autotuning or replay warm-up inside the timed samples, and the two sides
  measuring different things, a platform difference nobody declared, a
  timing kind the experiment did not freeze, and a provenance bundle that
  was edited or lacks an artifact that ran.

Identical inputs must give S_geo = 1 exactly with an interval containing 1
and the verdict no-significant-gain. tools/r40/acceptance.py is checked the
same way: a gate with no command, a make target that does not exist,
missing evidence, evidence that does not name the gate, a pass without
checks and a capability gate claiming perf-pass are none of them PASS.
"""
import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablate as AB          # noqa: E402
import acceptance as AC      # noqa: E402

FAMILIES = {'add': 3, 'matmul': 3, 'softmax': 2, 'llama_decode': 2}


def subset():
    pts = []
    for fam, n in FAMILIES.items():
        for i in range(n):
            pts.append(dict(id=f'op.{fam}/hbm/fp16-case{i}', sha256=f'{fam}{i:02d}' * 4,
                            family=fam))
    return pts


def record(pt, value, *, knob='0', image='xclbin:aaaa', clock=100e6, kind='graph-makespan'):
    samples = [value * (1 + 0.002 * ((k * 7) % 5 - 2)) for k in range(30)]
    return dict(schema='archbench-torch-record/2', point_id=pt['id'], point_sha256=pt['sha256'],
                family=pt['family'], tier='hbm', status='pass',
                platform=dict(kind='tnca', profile='core-r40', backend='fpga-u280', opt_level=2,
                              inductor={'max_block': 16}, device_id=image, image=image,
                              clock_policy='100MHz', software_tree='tree:1234',
                              suite='hbm:abcd', weights='balanced',
                              knobs={'TNCA_MECH': knob}),
                provenance=dict(software_tree='tree:1234', artifacts=['sha:1'],
                                profile='core-r40', capabilities=['tensor_dma'], image=image,
                                clock_hz=clock, launch_map={'cores': [0]}),
                time=dict(kind=kind, unit='s', samples=samples, device=value,
                          device_cycles=round(value * clock)))


def write(recs, d, name):
    p = Path(d) / name
    p.write_text(''.join(json.dumps(r) + '\n' for r in recs))
    return p


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', type=Path, help='also write the cases here')
    ap.add_argument('--evidence', action='store_true',
                    help='record gate R40-00.protocol in sim/build/r40/R40-00.json')
    args = ap.parse_args(argv)
    pts = subset()
    sub = {p['id']: p['sha256'] for p in pts}
    base_a = [record(p, 1e-3 * (1 + i)) for i, p in enumerate(pts)]
    base_b = [record(p, 0.8e-3 * (1 + i), knob='1') for i, p in enumerate(pts)]
    results, ok_all = [], True

    def expect(name, a, b, kind, **kw):
        nonlocal ok_all
        with tempfile.TemporaryDirectory() as d:
            pa, pb = write(a, d, 'a.jsonl'), write(b, d, 'b.jsonl')
            try:
                res = AB.compare(pa, pb, kw.pop('subset', sub), kw.pop('vary', ['TNCA_MECH']), **kw)
                got = ('accepted', res)
            except AB.Refused as e:
                got = ('refused', e.kind)
        if kind == 'accepted':
            ok = got[0] == 'accepted'
        elif kind == 'identity':
            ok = (got[0] == 'accepted' and got[1]['s_geo'] == 1.0
                  and got[1]['ci95'][0] <= 1.0 <= got[1]['ci95'][1]
                  and got[1]['verdict'] == 'no-significant-gain')
        else:
            ok = got == ('refused', kind)
        ok_all &= ok
        shown = got[1] if got[0] == 'refused' else f"S_geo {got[1]['s_geo']:.3f} {got[1]['verdict']}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44s} expect {kind:<12s} got {got[0]} {shown}")
        results.append(dict(name=name, expect=kind, ok=ok,
                            got=got[0] if got[0] == 'accepted' else f'refused:{got[1]}'))

    # The clean pair, and identity.
    expect('a clean comparison', base_a, base_b, 'accepted')
    expect('identical inputs', base_a, copy.deepcopy(base_a), 'identity')

    # The frozen plan calls it graph-makespan; the physical producer names
    # its device clock. Its raw endpoint proof is mandatory for this alias.
    def physical_records(records):
        records = copy.deepcopy(records)
        for r in records:
            cycles = round(r['time']['device'] * r['provenance']['clock_hz'])
            edges = []
            for index in range(30):
                start = 100 + index * cycles * 2
                edges.append([dict(clock='mmio-device-clock/1',epoch=1,session_id=1,
                                   start=start+a,end=start+b)
                              for a,b in ((0,cycles//4),(cycles//3,cycles//2),
                                          (3*cycles//4,cycles))])
            active = sum(e['end']-e['start'] for e in edges[0])
            r['time'].update(kind='device-cycles-serial-grid',unit='cycles',launches=3,
                             samples=[active]*30,device=active,device_cycles=active)
            r['timings'] = {'device-graph-makespan':dict(unit='cycles',samples=[cycles]*30,
                warmup=5,deterministic=False,endpoint_schema='tnca-launch-endpoints/1',
                endpoint_records=edges)}
        return records
    physical_a, physical_b = physical_records(base_a), physical_records(base_b)
    expect('frozen graph metric from raw device clock',physical_a,physical_b,'accepted',timing='graph-makespan')
    expect('explicit device graph metric',physical_a,physical_b,'accepted',timing='device-graph-makespan')
    for field in ('endpoint_schema','endpoint_records'):
        b = copy.deepcopy(physical_b)
        del b[0]['timings']['device-graph-makespan'][field]
        expect('device graph without '+field,physical_a,b,'timing',timing='graph-makespan')
    b = copy.deepcopy(physical_b)
    del b[0]['timings']['device-graph-makespan']['endpoint_records'][0][1]
    expect('interior device launch removed',physical_a,b,'timing',timing='graph-makespan')
    b = copy.deepcopy(physical_b)
    b[0]['timings']['device-graph-makespan']['endpoint_records'][0][-1]['epoch'] += 1
    expect('device graph crosses reset epoch',physical_a,b,'timing',timing='graph-makespan')
    b = copy.deepcopy(physical_b)
    b[0]['timings']['device-graph-makespan']['unit'] = 's'
    expect('device graph counter units changed',physical_a,b,'timing',timing='graph-makespan')
    b = copy.deepcopy(physical_b)
    b[0]['timings']['device-graph-makespan']['derived_from'] = 'sum-of-kernels'
    expect('device graph declares an active sum',physical_a,b,'timing',timing='graph-makespan')

    # 1. a slow point deleted from B
    b = [r for r in base_b if r['point_id'] != pts[-1]['id']]
    expect('slow point deleted', base_a, b, 'missing')
    # 2. an unmeasured point zero-filled
    b = copy.deepcopy(base_b)
    b[2]['time']['samples'] = [0.0] * 30
    b[2]['time']['device'] = 0
    expect('unmeasured point zero-filled', base_a, b, 'zero-fill')
    b = copy.deepcopy(base_b)
    b[3]['status'] = 'not-measured'
    expect('unmeasured point left in as not-measured', base_a, b, 'missing')
    # 3. two images on one side
    b = copy.deepcopy(base_b)
    for r in b[:4]:
        r['platform']['image'] = r['platform']['device_id'] = r['provenance']['image'] = 'xclbin:bbbb'
    expect('mixed images on one side', base_a, b, 'mixed')
    # 4. tolerance changed (a different point definition)
    b = copy.deepcopy(base_b)
    b[1]['point_sha256'] = 'f' * 64
    expect('tolerance changed', base_a, b, 'config')
    # 5. weight changed
    b = copy.deepcopy(base_b)
    for r in b:
        r['platform']['weights'] = 'reference-time:9999'
    expect('weights changed', base_a, b, 'config')
    # 6. provenance missing
    b = copy.deepcopy(base_b)
    del b[4]['provenance']['software_tree']
    expect('provenance missing', base_a, b, 'provenance')
    b = copy.deepcopy(base_b)
    b[0]['provenance']['launch_map'] = {}
    expect('empty launch map', base_a, b, 'provenance')
    # 7. per-core cycles summed into a makespan
    b = copy.deepcopy(base_b)
    for r in b:
        r['time']['per_core_active'] = [r['time']['device'] / 3] * 3
    expect('core cycles summed as makespan', base_a, b, 'timing')
    b = copy.deepcopy(base_b)
    b[5]['time']['derived_from'] = 'sum-of-kernels'
    expect('kernel times summed as makespan', base_a, b, 'timing')
    # 8. only the clock changed (same cycles, more MHz)
    b = copy.deepcopy(base_a)
    for r in b:
        cyc = r['time']['device_cycles']
        r['provenance']['clock_hz'] = 125e6
        r['platform']['clock_policy'] = '125MHz'
        r['time']['samples'] = [v * 0.8 for v in r['time']['samples']]
        r['time']['device'] *= 0.8
        r['time']['device_cycles'] = cyc
    expect('only the clock changed', base_a, b, 'clock-only')
    # 9. autotuning / replay warm-up inside the timed samples
    b = copy.deepcopy(base_b)
    b[6]['time']['samples'] = [dict(value=v, autotune=(k == 0)) for k, v in
                               enumerate(b[6]['time']['samples'])]
    expect('autotune inside the timed samples', base_a, b, 'warmup')
    b = copy.deepcopy(base_b)
    b[7]['time']['includes_replay_warmup'] = True
    expect('replay warm-up inside the timed region', base_a, b, 'warmup')
    # 10. different timing kinds on the two sides
    b = copy.deepcopy(base_b)
    for r in b:
        r['time']['kind'] = 'resident-call-latency'
    expect('different timing kinds', base_a, b, 'timing')
    # 11. a knob changed that is not the one under test
    b = copy.deepcopy(base_b)
    for r in b:
        r['platform']['knobs']['TNCA_OTHER'] = '1'
    expect('an undeclared knob changed', base_a, b, 'config')
    # 12. a protocol-1 record outside the B39 baseline
    b = copy.deepcopy(base_b)
    b[0]['schema'] = 'archbench-torch-record/1'
    expect('protocol-1 record on the B side', base_a, b, 'provenance')

    # 13. a platform difference is accepted only when declared
    b = copy.deepcopy(base_b)
    for r in b:
        r['platform']['image'] = r['platform']['device_id'] = r['provenance']['image'] = 'xclbin:cccc'
    expect('another image, not declared', base_a, b, 'config')
    expect('another image, declared (platform.image, platform.device_id)', base_a, b, 'accepted',
           vary=['TNCA_MECH', 'platform.image', 'platform.device_id'])
    # 14. several timing kinds in one record: the experiment's is compared
    def with_latency(recs, scale):
        out = copy.deepcopy(recs)
        for r in out:
            r['timings'] = {'graph-makespan': dict(r['time']),
                            'request-latency': dict(unit='s', samples=[v * scale for v in r['time']['samples']])}
        return out
    expect('request latency chosen from timings', with_latency(base_a, 1.1), with_latency(base_b, 1.1),
           'accepted', timing='request-latency')
    expect('a timing kind neither side measured', with_latency(base_a, 1.1), with_latency(base_b, 1.1),
           'timing', timing='kernel-makespan')
    # 15. the provenance bundle is audited file by file
    sys.path.insert(0, str(HERE.parents[1]))
    from archbench.provenance import write_bundle
    with tempfile.TemporaryDirectory() as bd:
        def bundled(recs, tag):
            out = copy.deepcopy(recs)
            for r in out:
                d = Path(bd) / tag / r['point_id'].replace('/', '__')
                (d / 'kernels').mkdir(parents=True, exist_ok=True)
                (d / 'kernels' / 'k.ttir').write_text(f"module {r['point_id']}\n")
                r['provenance']['launch_map'] = {'cores': [0], 'launches': []}
                r['provenance']['bundle'] = write_bundle(d, point_id=r['point_id'], graph_text=['g'],
                                                         wrapper_code=['w'], launches=[], dump_dir=None)
            return out
        ba, bb = bundled(base_a, 'a'), bundled(base_b, 'b')
        expect('bundles intact', ba, bb, 'accepted', audit_bundles=True)
        Path(bb[3]['provenance']['bundle']['dir'], 'kernels', 'k.ttir').write_text('edited\n')
        expect('a bundle file edited after the run', ba, bb, 'provenance', audit_bundles=True)
        bb2 = copy.deepcopy(bb)
        bb2[0]['provenance']['launch_map']['launches'] = [dict(kernel='k', artifact='ab' * 32)]
        Path(bb[3]['provenance']['bundle']['dir'], 'kernels', 'k.ttir').write_text(
            f"module {bb[3]['point_id']}\n")
        expect('an artifact that ran is missing from the bundle', ba, bb2, 'provenance',
               audit_bundles=True)

    # acceptance.py: nothing missing can pass.
    with tempfile.TemporaryDirectory() as d:
        ev = Path(d) / 'ev.json'
        ev.write_text(json.dumps(dict(gates={
            'g-ok': dict(status='capability-pass', checks=[dict(name='x', ok=True)]),
            'g-nochecks': dict(status='capability-pass'),
            'g-badcheck': dict(status='capability-pass', checks=[dict(name='x', ok=False)]),
            'g-perfcap': dict(status='perf-pass', checks=[dict(name='x', ok=True)]),
        })))
        rel = str(ev)
        cases = [
            ('a gate with evidence and checks', dict(id='g-ok', kind='capability',
                                                      command='make help-nonexistent-ok', evidence=rel),
             None, 'fail'),        # the make target does not exist
            ('no command', dict(id='g-ok', kind='capability', evidence=rel), None, 'fail'),
            ('evidence missing', dict(id='g-ok', kind='capability', command='true',
                                      evidence='sim/build/r40/does-not-exist.json'), None,
             'not-measured'),
            ('evidence does not name the gate', dict(id='g-other', kind='capability',
                                                     command='true', evidence=rel), None, 'fail'),
            ('a pass without checks', dict(id='g-nochecks', kind='capability', command='true',
                                           evidence=rel), None, 'fail'),
            ('a pass with a failing check', dict(id='g-badcheck', kind='capability',
                                                 command='true', evidence=rel), None, 'fail'),
            ('a capability gate claiming perf-pass', dict(id='g-perfcap', kind='capability',
                                                          command='true', evidence=rel), None, 'fail'),
            ('the command failed', dict(id='g-ok', kind='capability', command='true',
                                        evidence=rel), 1, 'fail'),
            ('all present', dict(id='g-ok', kind='capability', command='true', evidence=rel),
             0, 'capability-pass'),
        ]
        for name, gate, ran, want in cases:
            got, why = AC.judge(gate, ran=ran)
            ok = got == want
            ok_all &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  acceptance: {name:<34s} expect {want:<16s} got {got} ({why})")
            results.append(dict(name=f'acceptance: {name}', expect=want, ok=ok, got=got))
    print(f"r40-protocol  {'PASS' if ok_all else 'FAIL'}  ({sum(r['ok'] for r in results)}/{len(results)})")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(cases=results), indent=1) + '\n')
    if args.evidence:
        import evidence
        evidence.record('R40-00.protocol', 'capability-pass' if ok_all else 'fail',
                        [dict(name=r['name'], ok=r['ok'], expect=r['expect'], got=r['got'])
                         for r in results],
                        detail=f"{sum(r['ok'] for r in results)}/{len(results)} cases")
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(run())
