"""Numerical execution and timing-prediction eligibility are separate."""

VALID_STATUSES = {'uncalibrated', 'uncalibrated-fixed-point-checked'}


def timing_status(counters, *, slow=False):
    commands = counters.get('dram_slow_commands', 0)
    if not slow and not commands:
        return 'uncalibrated'
    if (not commands or counters.get('dram_slow_fixed_point_checked', 0) != commands
            or any(counters.get(k, 0) for k in
                   ('dram_slow_unconverged', 'dram_slow_unmatched', 'dram_child_failures'))):
        return 'inconclusive-fixed-point'
    return 'uncalibrated-fixed-point-checked'


def prediction_status(row, *, configured_slow=False):
    explicit = row.get('model_timing_status')
    if explicit is not None:
        return explicit
    # Historical fast-model data remain available. A slow result predating
    # request/latency validation cannot inherit its numerical PASS as timing.
    return timing_status(row.get('dram') or {}, slow=configured_slow or
                         (row.get('knobs') or {}).get('TNCA_DRAM_PATH') == 'slow')
