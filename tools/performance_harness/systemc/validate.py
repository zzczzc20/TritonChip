#!/usr/bin/env python3
"""GB10 functional/resource validation, RTL differential and trace perturbation.

These experiments validate mechanisms. They do not qualify R40 performance.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import time

from run import HERE, run, sha
from capture import capture

MASK = (1 << 64)-1


def config(path, **changes):
    values = dict(line.split('#')[0].strip().split('=', 1)
                  for line in (HERE/'configs/r40-three-core.cfg').read_text().splitlines()
                  if '=' in line.split('#')[0])
    values.update({k: str(v) for k, v in changes.items()})
    path.write_text(''.join(f'{k}={v}\n' for k, v in values.items()))
    return path


class Traffic:
    def __init__(self, path):
        self.file = path.open('x')
        self.words = {}
        self.id = 0

    def row(self, address, *, op=0, value=0, be=255, compare=0, core=0,
            kind='T', cycle=0, expected=None, check=None):
        old = self.words.get(address, 0)
        if kind == 'T':
            new = value
            if op >= 2:
                width = 64 if be == 255 else 32
                shift = 32 if be == 240 else 0
                mask = (1 << width)-1
                a, b = (old >> shift) & mask, (value >> shift) & mask
                sa = a-(1 << width) if a >> (width-1) else a
                sb = b-(1 << width) if b >> (width-1) else b
                aop = op-2
                if aop == 0: new = a+b
                elif aop == 1: new = a & b
                elif aop == 2: new = a | b
                elif aop == 3: new = a ^ b
                elif aop in (4, 11): new = b
                elif aop == 5: new = a if sa < sb else b
                elif aop == 6: new = b if sa < sb else a
                elif aop == 7: new = min(a, b)
                elif aop == 8: new = max(a, b)
                elif aop == 9: new = b if a == compare & mask else a
                elif aop == 10: new = a
                else: raise ValueError('atomic')
                new = (new & mask) << shift
            if op and op != 12:
                byte_mask = sum(255 << (8*b) for b in range(8) if be & (1 << b))
                self.words[address] = (old & ~byte_mask) | (new & byte_mask)
        elif kind == 'I':
            self.words[address] = value
        self.id += 1
        if check is None: check = op != 1 and kind == 'T'
        if expected is None: expected = old
        self.file.write(f'{self.id} {cycle} {core} {op} {address:#x} {value:#x} {be:#x} {compare:#x} {int(check)} {expected:#x} {kind}\n')

    def close(self): self.file.close()


def require_run(binary, cfg, trace, out, monitor="aggregate"):
    result = run(binary, cfg, trace, out, monitor=monitor)
    if result['status'] != 'completed-prediction':
        raise AssertionError(f'model failure; preserve {out}')
    return result


def validate(binary, out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = config(out/'base.cfg', trace_events=0)
    results = {}
    # All implemented AOPs, both 32-bit lanes, signed extrema, masked stores,
    # repeated same-line misses, merge order and return-of-old semantics.
    trace = out/'atomic.trace'
    t = Traffic(trace)
    t.row(0x1000, kind='I', value=0x8000000080000000)
    for repeat in range(8):
        for be in (255, 15, 240):
            for aop in range(12):
                t.row(0x1000, op=aop+2, be=be, value=MASK-repeat, compare=t.words[0x1000] & (MASK if be == 255 else 0xffffffff))
    for be in (1, 128, 85, 170, 0, 255):
        t.row(0x1000, op=1, be=be, value=0x123456789abcdef0)
        t.row(0x1000)
    t.row(0x1000, kind='F')
    t.row(0x1000, kind='H', expected=t.words[0x1000], check=True)
    t.close()
    results['atomics'] = require_run(binary, cfg, trace, out/'atomics')['summary']
    # Dirty thrashing with bounded resources. Core address regions are disjoint;
    # same-line operations retain an identical XY route and input order.
    trace = out/'random.trace';t = Traffic(trace);rng = random.Random(731)
    for i in range(12000):
        core = i % 3
        a = (core+1)*0x100000 + rng.randrange(512)*64 + rng.randrange(8)*8
        t.row(a, core=core, op=1 if rng.randrange(3) else 0,
              value=rng.getrandbits(64), be=rng.randrange(256))
    t.close()
    stress = config(out/'stress.cfg', sets=2, ways=2, mshrs=2, targets=4, hbm_latency_cycles=80, trace_events=1000)
    results['dirty_thrash'] = require_run(binary, stress, trace, out/'dirty_thrash')['summary']
    counters = results['dirty_thrash']['counters']
    assert sum(v for k, v in counters.items() if k.endswith('_writebacks')) > 0
    assert sum(v for k, v in counters.items() if k.endswith('_mshr_full_events')) > 0
    for core in range(3): assert counters[f'core_{core}_completed'] == 4000
    # Host reads are external memory probes. Stores remain dirty until a fence
    # drains all requesters and acknowledges every writeback.
    trace = out/'visibility.trace';t = Traffic(trace)
    t.row(0x1000, op=1, value=0xfedcba)
    t.row(0x1000, kind='H', check=True, expected=0)
    t.row(0x1000, kind='F')
    t.row(0x1000, kind='H', check=True, expected=0xfedcba)
    t.close()
    results['visibility'] = require_run(binary, cfg, trace, out/'visibility')['summary']
    # Same traffic, one/all outlets and independent physical PC partitions.
    trace = out/'bandwidth.trace';t = Traffic(trace)
    for i in range(3000):
        t.row((i % 3)*(1 << 28)+(i//3)*64, core=i % 3)
    t.close()
    for n in (1, 3):
        variant = config(out/f'ports{n}.cfg', active_outlets=n, outstanding=2)
        results[f'ports{n}'] = require_run(binary, variant, trace, out/f'ports{n}')['summary']
    assert results['ports1']['graph_cycles'] > results['ports3']['graph_cycles']
    assert all(results['ports3']['counters'].get(f'outlet_{i}_reads', 0) for i in range(3))
    assert all(results['ports3']['counters'].get(f'pc_{i}_bytes', 0) for i in range(3))
    # Mutation: a model that ignores expected data must fail this case.
    trace = out/'wrong.trace';trace.write_text('1 0 0 0 0x1000 0 255 0 1 1 T\n')
    bad = run(binary, cfg, trace, out/'mutation')
    assert bad['returncode'] == 1 and bad['summary']['mismatches'] == 1
    # Configuration and address refusals must not emit completed reports.
    for name, text in [('bounds', '1 0 0 0 0x40000000 0 255 0 0 0 T\n'),
                       ('mmio', '1 0 0 0 0x7fff0000 0 255 0 0 0 T\n'),
                       ('atomic_mask', '1 0 0 2 0x1000 0 1 0 0 0 T\n')]:
        trace = out/f'{name}.trace';trace.write_text(text)
        bad = run(binary, cfg, trace, out/name)
        assert bad['status'] == 'failed' and bad['returncode'] != 0
    # Trace host overhead is sampled in alternating order; target simulation
    # cycles and full responses must be identical for both modes.
    trace = out/'random.trace';pairs=[]
    for pair in range(3):
        values={}
        for mode in ((0, 1) if pair % 2 == 0 else (1, 0)):
            c = config(out/f'trace{pair}-{mode}.cfg', trace_events=100000 if mode else 0)
            r = require_run(binary, c, trace, out/f'trace{pair}-{mode}', monitor='trace')
            values[mode] = r
        assert values[0]['summary'] == values[1]['summary']
        assert sha(out/f'trace{pair}-0/responses.tsv') == sha(out/f'trace{pair}-1/responses.tsv')
        pairs.append(dict(baseline=values[0]['host_elapsed_seconds'], instrumented=values[1]['host_elapsed_seconds'],
                          relative=values[1]['host_elapsed_seconds']/values[0]['host_elapsed_seconds']-1))
    results['trace_overhead'] = dict(pairs=pairs, simulated_cycle_delta=0,
        observed_host_relative_range=[min(p['relative'] for p in pairs), max(p['relative'] for p in pairs)],
        median_relative=statistics.median(p['relative'] for p in pairs),
        scope='host serialization overhead on this workload; no physical instrumentation claim')
    # Independent randomized RTL traces. No parameter fitting on these seeds.
    for seed in (7, 29):
        capture_dir = out/f'rtl{seed}'
        report = capture(capture_dir, count=8000, seed=seed)
        c = config(out/f'rtl{seed}.cfg', nx=2, ny=2, sets=16, ways=4, full_line_tag=0,
                   mshrs=4, targets=8, outlets=1, active_outlets=1)
        r = require_run(binary, c, capture_dir/'requests.trace', out/f'rtl_replay{seed}')
        assert r['summary']['issued'] == report['requests']
        results[f'rtl{seed}'] = dict(capture=report, replay=r['summary'])
    result = dict(schema='tnca-systemc-validation/1', status='passed', results=results,
                  scope='memory subsystem mechanisms; whole-chip integration and production timing are not qualified',
                  performance_gate='not-measured', timing_error_bound_cycles=None)
    (out/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(validate(a.binary, a.out)['status'])
