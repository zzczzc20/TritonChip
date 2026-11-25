#!/usr/bin/env python3
"""Extract a replay input from an RTL recording: the one boundary stream the
model replays, nothing else of the trace.

    replay_extract.py --level core   RUN_RTL/traces/<point>.trace        OUT.replay
    replay_extract.py --level uncore RUN_RTL/traces/<point>.uncore.trace OUT.replay

The model (src/platform.inc) takes only this file as a replay input and
refuses a trace. Cycles are relative to each launch.

    TNCA_REPLAY 1 level=core
    L                                  one launch
    B <cycle> <we> <addr:016x> <be:02x> <latency|->   a beat of the core's port, in port order;
                                       `-` when the RTL never answered it inside the launch
    E

    TNCA_REPLAY 1 level=uncore
    L
    X <cycle> <we> <addr:016x> <src> <tag>   a request the device's memory port accepted
    Y <cycle> <src> <tag>                    a response the port delivered
    E

A core-level replay carries no instruction record (`I`) and an uncore-level
one no slice stream (`S`, `R`, `M`, `D`): a model handed this file learns the
boundary and nothing the gate compares above it.
"""
import argparse
import sys
from pathlib import Path


def extract_core(lines):
    out = ['TNCA_REPLAY 1 level=core']
    beats, pending, start, in_launch = [], {}, 0, False

    def flush():
        for b in beats:
            lat = str(b['latency']) if b['answered'] else '-'
            out.append(f"B {b['cycle']} {b['we']} {b['addr']:016x} {b['be']:02x} {lat}")
        out.append('E')
        beats.clear()
        pending.clear()

    for line in lines:
        f = line.split()
        if not f:
            continue
        k = f[0]
        if k == 'L':
            if in_launch:
                flush()
            start = int(f[1])
            in_launch = True
            out.append('L')
        elif k == 'Q' and in_launch:
            c, we, addr, be, tag = int(f[1]), int(f[2]), int(f[3], 16), int(f[4], 16), f[5]
            pending[tag] = len(beats)
            beats.append(dict(cycle=c - start, we=1 if we else 0, addr=addr, be=be, latency=0, answered=False))
        elif k == 'P' and in_launch:
            c, tag = int(f[1]), f[2]
            i = pending.pop(tag, None)
            if i is not None:
                beats[i]['latency'] = (c - start) - beats[i]['cycle']
                beats[i]['answered'] = True
        elif k == 'E' and in_launch:
            flush()
            in_launch = False
    if in_launch:
        flush()
    return out


def extract_uncore(lines):
    out = ['TNCA_REPLAY 1 level=uncore']
    start, launches = 0, 0
    for line in lines:
        f = line.split()
        if not f:
            continue
        k = f[0]
        if k == 'L':
            if launches:
                out.append('E')
            start = int(f[1])
            launches += 1
            out.append('L')
        elif k == 'X' and launches:
            c, we, addr, src, tag = int(f[1]), int(f[2]), int(f[3], 16), int(f[4]), int(f[5])
            out.append(f'X {c - start} {1 if we else 0} {addr:016x} {src} {tag}')
        elif k == 'Y' and launches:
            c, src, tag = int(f[1]), int(f[2]), int(f[3])
            out.append(f'Y {c - start} {src} {tag}')
    if launches:
        out.append('E')
    return out


def extract(level, src, dst):
    """Write the replay input for one trace file; returns the number of launches."""
    text = Path(src).read_text().splitlines()
    out = extract_core(text) if level == 'core' else extract_uncore(text)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text('\n'.join(out) + '\n')
    return sum(1 for line in out if line == 'L')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--level', choices=['core', 'uncore'], required=True)
    ap.add_argument('trace')
    ap.add_argument('out')
    args = ap.parse_args(argv)
    n = extract(args.level, args.trace, args.out)
    print(f'{args.out}: {n} launches', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
