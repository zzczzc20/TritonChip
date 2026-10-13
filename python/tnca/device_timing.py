"""Common-clock spans, distinct from summed active kernel cycles."""
MASK = (1 << 64) - 1
HALF = 1 << 63


def clock_span(records):
    """Combine endpoint records within one session/epoch, or unavailable.

    Signed offsets from an anchor handle one counter wrap and out-of-order
    completion. An interval of half the counter range is ambiguous/refused.
    Missing endpoints never become a partial graph measurement.
    """
    if not records or any(r is None for r in records):
        return None
    first = records[0]
    domain = (first['clock'], first['epoch'], first['session_id'])
    anchor = first['start']
    starts, ends = [], []
    for record in records:
        if (record['clock'], record['epoch'], record['session_id']) != domain:
            raise ValueError('device timestamps cross a clock domain or reset epoch')
        duration = (record['end'] - record['start']) & MASK
        if not 0 < duration < HALF:
            raise ValueError('invalid or ambiguous device timestamp interval')
        offset = ((record['start'] - anchor + HALF) & MASK) - HALF
        starts.append(offset)
        ends.append(offset + duration)
    start, end = min(starts), max(ends)
    if not 0 < end - start < HALF:
        raise ValueError('ambiguous device graph clock span')
    return dict(clock=domain[0], epoch=domain[1], session_id=domain[2],
                start=(anchor + start) & MASK, end=(anchor + end) & MASK,
                cycles=end - start)
