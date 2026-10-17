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


def audit_graph_samples(samples, endpoint_records, *, launch_counts):
    """Recompute saved graph samples from every raw launch's uint64 endpoints.

    This runs after measurement. Missing launches, mixed domains/epochs and
    samples assembled from active-cycle sums cannot become valid makespans.
    Each graph call is checked independently; independent workers may use
    different sessions, but launches within one call must share a domain.
    """
    if (not isinstance(samples, list) or not samples
            or not isinstance(endpoint_records, list)
            or not isinstance(launch_counts, list)
            or len(samples) != len(endpoint_records) or len(samples) != len(launch_counts)):
        raise ValueError('one complete launch-endpoint list is required per graph sample')
    spans = []
    for sample, records, count in zip(samples, endpoint_records, launch_counts):
        if type(sample) is not int or sample <= 0 or not isinstance(records, list) or not records:
            raise ValueError('positive integer cycles and nonempty launch endpoints required')
        if type(count) is not int or count <= 0 or len(records) != count:
            raise ValueError('raw endpoint count differs from the recorded graph launch count')
        for record in records:
            if (not isinstance(record, dict)
                    or not {'clock', 'epoch', 'session_id', 'start', 'end'} <= record.keys()
                    or record['clock'] != 'mmio-device-clock/1'
                    or any(type(record[k]) is not int or not 0 <= record[k] <= MASK
                           for k in ('start', 'end'))):
                raise ValueError('invalid or missing common-clock launch endpoints')
        span = clock_span(records)
        if span['cycles'] != sample:
            raise ValueError('graph sample differs from its raw launch-endpoint span')
        spans.append(span)
    return spans
