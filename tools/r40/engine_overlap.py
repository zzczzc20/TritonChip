"""Actual single-core engine service intervals from +engtrace/+trace.

Only transfer commands and op=1 matrix commands count. Descriptor config,
accumulator zero/row operations, and gaps between commands do not count as
DMA/MMA overlap. Mailbox sequence numbers bind each interval to its launch.
"""
import re

ENGINE = re.compile(r'^ENG (\d+) (mtx-issue|mtx-done|txf-issue|txf-done)(?: (.*))?$')
REPLY = re.compile(r'^\[dev\] seq=(\d+) op=(\d+).* -> status=(\d+)')


def parse(text):
    launches = {}
    pending = {'mtx': None, 'txf': None}
    intervals = {'mma': [], 'dma': []}
    last_cycle = -1
    for line in text.splitlines():
        event = ENGINE.match(line)
        if event:
            cycle, action, detail = event.groups()
            cycle = int(cycle)
            if cycle < last_cycle:
                raise ValueError('engine clocks went backwards within a launch')
            last_cycle = cycle
            engine = action[:3]
            if action.endswith('issue'):
                # Non-transfer configuration writes have no DMA service.
                if engine == 'txf' and not (detail or '').endswith('xfr'):
                    continue
                if pending[engine] is not None:
                    raise ValueError(f'{engine} issued before its previous completion')
                kind = 'mma' if engine == 'mtx' and (detail or '').endswith('op1') else 'dma' if engine == 'txf' else None
                pending[engine] = (cycle, kind)
            elif pending[engine] is not None:
                start, kind = pending[engine]
                if cycle <= start:
                    raise ValueError('engine completion has no positive service interval')
                if kind:
                    intervals[kind].append((start, cycle))
                pending[engine] = None
            continue
        reply = REPLY.match(line)
        if reply and int(reply[2]) in (3, 9):
            if int(reply[3]) != 0 or any(value is not None for value in pending.values()):
                raise ValueError('faulted launch or unfinished engine trace')
            seq = int(reply[1])
            if seq in launches:
                raise ValueError('duplicate launch sequence')
            overlap = sum(max(0, min(m1, d1) - max(m0, d0))
                          for m0, m1 in intervals['mma'] for d0, d1 in intervals['dma'])
            launches[seq] = dict(mma_intervals=intervals['mma'], dma_intervals=intervals['dma'],
                                 mma_cycles=sum(b-a for a,b in intervals['mma']),
                                 dma_cycles=sum(b-a for a,b in intervals['dma']),
                                 overlap_cycles=overlap)
            intervals = {'mma': [], 'dma': []}
            last_cycle = -1
    if any(value is not None for value in pending.values()) or any(intervals.values()):
        raise ValueError('trace ended before its launch completion')
    if not launches:
        raise ValueError('no traced launch completions')
    return launches


def selected(text, sequences):
    launches = parse(text)
    missing = set(sequences) - set(launches)
    if missing:
        raise ValueError(f'measured launch sequences missing: {sorted(missing)}')
    return {seq: launches[seq] for seq in sequences}
