"""Decode TNCA Proton captures into Chrome Trace, without a GPU dependency.

This is the TNCA record backend/exporter, not a replacement for upstream
Proton's CUDA/CUPTI collector. No host timestamp stands in for device time.
"""
import math
import statistics
import struct

IDENTITY_FIELDS = ('software_sha256', 'config_sha256', 'image_sha256',
                   'contract_sha256', 'clock_hz', 'backend')


def check_identity(identity):
    for key in IDENTITY_FIELDS:
        if key not in identity:
            raise ValueError(f'missing capture identity: {key}')
        value = identity[key]
        if key.endswith('_sha256'):
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError(f'{key} must be a full lowercase SHA256')
    if (type(identity['clock_hz']) is not int or identity['clock_hz'] <= 0
            or not isinstance(identity['backend'], str) or not identity['backend']):
        raise ValueError('capture needs a positive device clock and explicit backend')


def chrome_trace(data, layout, *, identity):
    """Decode a completed, ownership-transferred fresh capture. Incomplete = error.

    Each core is a separate clock domain. Trace origins are independently
    rebased for display; cross-core visual alignment does NOT imply ordering.
    """
    check_identity(identity)
    if layout.get('schema') != 'tnca-proton-record/1':
        raise ValueError('unknown Proton capture schema')
    required = layout['required_bytes']
    if len(data) != required:
        raise ValueError(f'capture length {len(data)} != {required}')
    roles, stride = layout['roles'], layout['stride_bytes']
    programs = math.prod(layout['grid'])
    captures, origins = [], {}
    for program in range(programs):
        for slot, role in enumerate(roles):
            offset = (program * len(roles) + slot) * stride
            valid, core = struct.unpack_from('<QQ', data, offset)
            if valid != 1:
                raise ValueError(f'incomplete capture: program {program}, role {role["role"]}')
            n = len(role['endpoints'])
            times = struct.unpack_from(f'<{n}Q', data, offset + 16)
            if not n or any(b < a for a, b in zip(times, times[1:])):
                raise ValueError('nonmonotonic clock or unsupported uint64 wrap')
            origins[core] = min(origins.get(core, times[0]), times[0])
            captures.append((program, role, core, times))
    events = []
    for core in sorted(origins):
        events.append(dict(ph='M', name='process_name', pid=core, tid=0,
                           args=dict(name=f'core {core} (independent clock origin)')))
    for program, role, core, times in captures:
        stack, seen = [], set()
        tid = program * len(roles) + roles.index(role)
        for endpoint, cycle in zip(role['endpoints'], times):
            name = endpoint['name']
            if endpoint['start']:
                if name in seen:
                    raise ValueError('reused Proton scope name')
                seen.add(name)
                stack.append((name, cycle))
            else:
                if not stack or stack[-1][0] != name:
                    raise ValueError('unbalanced capture layout')
                _, begin = stack.pop()
                delta, duration = begin - origins[core], cycle - begin
                if max(delta, duration) >= 1 << 53:
                    raise ValueError('capture exceeds exact Chrome timestamp range')
                events.append(dict(ph='X', cat='tnca.proton', name=name, pid=core, tid=tid,
                    ts=delta * 1e6 / identity['clock_hz'],
                    dur=duration * 1e6 / identity['clock_hz'],
                    args=dict(program=program, role=role['role'], begin_cycle=begin,
                              end_cycle=cycle, cycles=duration)))
        if stack:
            raise ValueError('unclosed capture scope')
    return dict(traceEvents=events, displayTimeUnit='ns', metadata=dict(
        identity=identity, capture_layout=layout, core_clock_origins=origins,
        clock_alignment='independent core origins; cross-core ordering unavailable',
        performance_status='diagnostic-only', overhead_status='not-measured'))


def paired_overhead(pairs, *, identity, max_relative_overhead):
    """Raw AB/BA paired device makespans, no constant subtraction correction.

    Threshold is a declared measurement policy, not an acceptance-contract
    change. Even a passing diagnostic run never replaces uninstrumented data.
    """
    check_identity(identity)
    if (not math.isfinite(max_relative_overhead) or max_relative_overhead < 0
            or len(pairs) < 5):
        raise ValueError('need >=5 AB/BA pairs and a nonnegative overhead limit')
    relative = []
    code_pair = None
    for index, pair in enumerate(pairs):
        expected = 'AB' if index % 2 == 0 else 'BA'
        if pair.get('order') != expected or pair.get('identity') != identity:
            raise ValueError('paired runs must alternate AB/BA and have identical identities')
        a, b = pair['baseline_cycles'], pair['instrumented_cycles']
        if any(type(v) is not int or v <= 0 for v in (a, b)):
            raise ValueError('overhead requires positive raw device cycles')
        if pair.get('baseline_code_sha256') == pair.get('instrumented_code_sha256'):
            raise ValueError('instrumentation needs distinct bound code identities')
        for key in ('baseline_code_sha256', 'instrumented_code_sha256'):
            check_identity(dict(identity, software_sha256=pair.get(key)))
        current = (pair['baseline_code_sha256'], pair['instrumented_code_sha256'])
        if code_pair is not None and current != code_pair:
            raise ValueError('paired code identities changed during calibration')
        code_pair = current
        relative.append((b - a) / a)
    return dict(identity=identity, pairs=pairs, relative_deltas=relative,
                median_relative_overhead=statistics.median(relative),
                observed_relative_range=[min(relative), max(relative)],
                pair_count=len(pairs),
                range_interpretation='observed calibration range, not a probabilistic coverage guarantee',
                region_timing_error_bound_cycles=None,
                region_error_interpretation='whole-program perturbation does not bound each region error',
                worst_absolute_relative_change=max(map(abs, relative)),
                max_relative_overhead=max_relative_overhead,
                status=('within-diagnostic-budget' if max(map(abs, relative)) <= max_relative_overhead
                        else 'perturbed-use-coarser-scopes'),
                performance_status='diagnostic-only; no corrected performance claim')
