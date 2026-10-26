#!/usr/bin/env python3
"""Compare two core traces (README.md section 2) and judge a point set.

    tracediff.py A.trace B.trace [--data] [--order-only] [--max-report N] [--json OUT]
    tracediff.py gate --a DIR_A --b DIR_B --out alignment.json [--data] [--order-only]

A is the reference (the RTL), B the candidate (the model). Launches are paired
in order; within a launch every stream is compared record by record, cycles
relative to the launch. The first divergence of each stream is reported with
both records. `gate` writes the verdict of README section 4.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

SCHEMA = 'tnca-core-alignment/1'
STREAMS = ('length', 'issue', 'request', 'response')


class TraceError(ValueError):
    pass


class Launch:
    __slots__ = ('start', 'end', 'pc', 'roles', 'issue', 'request', 'response')

    def __init__(self, start, pc, roles):
        self.start, self.end, self.pc, self.roles = start, None, pc, roles
        self.issue, self.request, self.response = [], [], []

    @property
    def cycles(self):
        return None if self.end is None else self.end - self.start


def _hex(s):
    """A hex field; None when the simulator printed x/z digits (an unknown value,
    e.g. the write data of a read, which the RTL never drives)."""
    return None if 'x' in s or 'z' in s or 'X' in s or 'Z' in s else int(s, 16)


def parse(path):
    """-> dict(core, launches=[Launch], cycles, stray=(issue, request, response))."""
    path = Path(path)
    launches, current = [], None
    stray = Launch(0, 0, 0)
    header = end = None
    pending = {}  # tag -> request index within the current launch
    with path.open() as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]
            try:
                if kind == 'TNCA_CORE_TRACE':
                    if header is not None:
                        raise TraceError('duplicate header')
                    if parts[1] != '1':
                        raise TraceError(f'unknown trace version {parts[1]}')
                    header = dict(kv.split('=', 1) for kv in parts[2:])
                    continue
                if header is None:
                    raise TraceError('record before header')
                if end is not None:
                    raise TraceError('record after END')
                if kind == 'L':
                    if current is not None and current.end is None:
                        raise TraceError('launch inside an open launch')
                    current = Launch(int(parts[1]), int(parts[2], 16), int(parts[3]))
                    launches.append(current)
                    pending = {}
                    continue
                target = current if current is not None and current.end is None else stray
                base = target.start
                if kind == 'I':
                    target.issue.append((int(parts[1]) - base, int(parts[2]), int(parts[3], 16)))
                elif kind == 'Q':
                    cyc, we, addr, be, tag, amo, wdata = (int(parts[1]) - base, int(parts[2]), int(parts[3], 16),
                                                          int(parts[4], 16), int(parts[5]), int(parts[6]),
                                                          _hex(parts[7]))
                    if not we:
                        wdata = be = None  # a read carries no data; its byte enables are not driven
                    pending[tag] = len(target.request)
                    target.request.append((cyc, we, addr, be, amo, wdata))
                    target.response.append(None)
                elif kind == 'P':
                    cyc, tag, rdata = int(parts[1]) - base, int(parts[2]), _hex(parts[3])
                    idx = pending.pop(tag, None)
                    if idx is None:
                        raise TraceError(f'response for unknown tag {tag}')
                    req_cycle = target.request[idx][0]
                    target.response[idx] = (cyc - req_cycle, rdata)
                elif kind == 'E':
                    if current is None or current.end is not None:
                        raise TraceError('E without an open launch')
                    current.end = int(parts[1])
                elif kind == 'END':
                    end = (int(parts[1]), int(parts[2]))
                else:
                    raise TraceError(f'unknown record {kind}')
            except (IndexError, ValueError) as e:
                raise TraceError(f'{path}:{n}: {e}') from None
    if header is None:
        raise TraceError(f'{path}: no header')
    if end is None:
        raise TraceError(f'{path}: no END (the simulation did not finish)')
    if end[0] != len(launches):
        raise TraceError(f'{path}: END counts {end[0]} launches, {len(launches)} recorded')
    return dict(core=int(header.get('core', 0)), launches=launches, cycles=end[1], stray=stray)


def _view(record, stream, data):
    """The compared projection of a record."""
    if record is None:
        return None
    if stream == 'request':
        cyc, we, addr, be, amo, wdata = record
        return (cyc, we, addr, be, amo) + ((wdata,) if data and we else ())
    if stream == 'response':
        lat, rdata = record
        return (lat,) + ((rdata,) if data else ())
    return record


def _same(ra, rb):
    """Projections equal; a None field (unknown data) matches anything."""
    if ra is None or rb is None:
        return ra is rb
    return len(ra) == len(rb) and all(x == y or x is None or y is None for x, y in zip(ra, rb))


def _strip_cycle(v):
    return None if v is None else v[1:]


def compare_launch(a, b, *, data=False, order_only=False):
    """-> {stream: 'equal' | divergence dict}."""
    out = {}
    if a.cycles == b.cycles or order_only:
        out['length'] = 'equal'
    else:
        out['length'] = dict(a=a.cycles, b=b.cycles)
    for stream in ('issue', 'request', 'response'):
        xa, xb = getattr(a, stream), getattr(b, stream)
        n = max(len(xa), len(xb))
        found = 'equal'
        for i in range(n):
            ra = _view(xa[i], stream, data) if i < len(xa) else None
            rb = _view(xb[i], stream, data) if i < len(xb) else None
            if order_only and stream != 'response':
                ra, rb = _strip_cycle(ra), _strip_cycle(rb)
            if order_only and stream == 'response':
                ra = None if ra is None else ra[1:]
                rb = None if rb is None else rb[1:]
                if not data:
                    continue
            if not _same(ra, rb):
                found = dict(index=i, a=_fmt(xa[i] if i < len(xa) else None, stream),
                             b=_fmt(xb[i] if i < len(xb) else None, stream),
                             count_a=len(xa), count_b=len(xb))
                break
        out[stream] = found
    return out


def _fmt(record, stream):
    if record is None:
        return None
    if stream == 'issue':
        cyc, rctx, pc = record
        return dict(cycle=cyc, rctx=rctx, pc=f'{pc:016x}')
    if stream == 'request':
        cyc, we, addr, be, amo, wdata = record
        return dict(cycle=cyc, we=we, addr=f'{addr:016x}', be=None if be is None else f'{be:02x}', amo=amo,
                    wdata=None if wdata is None else f'{wdata:016x}')
    lat, rdata = record
    return dict(latency=lat, rdata=None if rdata is None else f'{rdata:016x}')


def compare(ta, tb, *, data=False, order_only=False, max_report=5):
    """Compare two parsed traces -> point result dict."""
    la, lb = ta['launches'], tb['launches']
    result = dict(launches_a=len(la), launches_b=len(lb), launches=min(len(la), len(lb)),
                  aligned_launches=0, cycles_a=sum(x.cycles or 0 for x in la),
                  cycles_b=sum(x.cycles or 0 for x in lb), divergences=[])
    streams = {s: 'equal' for s in STREAMS}
    for k in range(min(len(la), len(lb))):
        c = compare_launch(la[k], lb[k], data=data, order_only=order_only)
        if all(v == 'equal' for v in c.values()):
            result['aligned_launches'] += 1
            continue
        for s in STREAMS:
            if c[s] != 'equal' and streams[s] == 'equal':
                streams[s] = dict(launch=k, **c[s]) if isinstance(c[s], dict) else c[s]
        if len(result['divergences']) < max_report:
            result['divergences'].append(dict(launch=k, **{s: c[s] for s in STREAMS if c[s] != 'equal'}))
    if len(la) != len(lb):
        streams['length'] = dict(launch_count_a=len(la), launch_count_b=len(lb)) if streams['length'] == 'equal' else streams['length']
    result['streams'] = streams
    result['equal'] = all(v == 'equal' for v in streams.values()) and len(la) == len(lb)
    if result['cycles_a']:
        result['cycle_error'] = (result['cycles_b'] - result['cycles_a']) / result['cycles_a']
    return result


def describe(result, name='point'):
    lines = [f"{name}: launches {result['launches_a']} vs {result['launches_b']}, "
             f"aligned {result['aligned_launches']}, cycles {result['cycles_a']} vs {result['cycles_b']}"
             + (f" ({result['cycle_error']:+.2%})" if 'cycle_error' in result else '')]
    for s in STREAMS:
        v = result['streams'][s]
        if v == 'equal':
            continue
        lines.append(f"  first {s} divergence: {json.dumps(v)}")
    return '\n'.join(lines)


def gate(dir_a, dir_b, *, data=False, order_only=False, max_report=5):
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    names = sorted({p.stem for p in dir_a.glob('*.trace')} | {p.stem for p in dir_b.glob('*.trace')})
    points, missing, aligned, diverged, errors = {}, [], 0, 0, []
    for name in names:
        pa, pb = dir_a / f'{name}.trace', dir_b / f'{name}.trace'
        if not pa.exists() or not pb.exists():
            missing.append(name)
            points[name] = dict(status='missing', a=pa.exists(), b=pb.exists())
            continue
        try:
            r = compare(parse(pa), parse(pb), data=data, order_only=order_only, max_report=max_report)
        except TraceError as e:
            missing.append(name)
            points[name] = dict(status='malformed', error=str(e))
            continue
        r['status'] = 'aligned' if r['equal'] else 'diverged'
        points[name] = r
        aligned += r['equal']
        diverged += not r['equal']
        if 'cycle_error' in r:
            errors.append(abs(r['cycle_error']))
    if not names:
        verdict, reason = 'no-verdict', 'no traces on either side'
    elif missing:
        verdict, reason = 'no-verdict', f'{len(missing)} point(s) missing or malformed: ' + ', '.join(missing[:5])
    elif diverged:
        verdict, reason = 'diverged', f'{diverged} of {len(names)} point(s) diverge'
    else:
        verdict, reason = 'aligned', f'all {len(names)} point(s) equal in every compared stream'
    errors.sort()
    agg = dict(points=len(names), aligned=aligned, diverged=diverged, missing=len(missing),
               median_abs_cycle_error=statistics.median(errors) if errors else None,
               p95_abs_cycle_error=errors[min(len(errors) - 1, int(0.95 * len(errors)))] if errors else None)
    return dict(schema=SCHEMA, verdict=verdict, reason=reason, options=dict(data=data, order_only=order_only),
                aggregates=agg, points=points)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data', action='store_true', help='also compare write data and read data')
    common.add_argument('--order-only', action='store_true', help='compare sequences without cycles')
    common.add_argument('--max-report', type=int, default=5)
    common.add_argument('--json', help='write the result here')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[common])
    sub = ap.add_subparsers(dest='cmd')
    g = sub.add_parser('gate', help='judge two directories of per-point traces', parents=[common])
    g.add_argument('--a', required=True, help='reference traces (the RTL)')
    g.add_argument('--b', required=True, help='candidate traces (the model)')
    g.add_argument('--out', required=True, help='alignment.json')
    p = sub.add_parser('pair', help='compare two trace files', parents=[common])
    p.add_argument('a')
    p.add_argument('b')
    if argv is None:
        argv = sys.argv[1:]
    if 'gate' not in argv and 'pair' not in argv and argv and argv != ['-h'] and argv != ['--help']:
        argv = ['pair'] + argv
    args = ap.parse_args(argv)
    opts = dict(data=args.data, order_only=args.order_only, max_report=args.max_report)
    if args.cmd == 'gate':
        r = gate(args.a, args.b, **opts)
        Path(args.out).write_text(json.dumps(r, indent=1, sort_keys=True) + '\n')
        print(f"{r['verdict']}: {r['reason']}")
        a = r['aggregates']
        if a['median_abs_cycle_error'] is not None:
            print(f"cycle error median {a['median_abs_cycle_error']:.2%}, p95 {a['p95_abs_cycle_error']:.2%}")
        return {'aligned': 0, 'diverged': 1, 'no-verdict': 2}[r['verdict']]
    if args.cmd == 'pair':
        try:
            r = compare(parse(args.a), parse(args.b), **opts)
        except TraceError as e:
            print(f'no-verdict: {e}')
            return 2
        print(describe(r, Path(args.a).stem))
        for d in r['divergences']:
            print('  launch', d['launch'], json.dumps({k: v for k, v in d.items() if k != 'launch'}))
        if args.json:
            Path(args.json).write_text(json.dumps(r, indent=1, sort_keys=True) + '\n')
        return 0 if r['equal'] else 1
    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
