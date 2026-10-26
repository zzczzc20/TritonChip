#!/usr/bin/env python3
"""An image's resources and timing, from Vivado's routed reports, as JSON.

    python3 tools/r40/resources.py --util full_util_routed.rpt \
        --timing <..._timing_summary_routed.rpt> [--clock ap_clk] \
        [--image <xclbin or sha>] [--clock-mhz 100.0] --out resources.json

doc/r40_tasks_and_acceptance.md sec 3.2 item 4 and sec 3.4: every
comparison reports what the mechanism cost -- LUT, FF, BRAM, URAM and DSP
for the whole chip and for each SLR, the SLR crossings, and post-route setup
and hold slack of the kernel clock and of the design. tools/r40/ablate.py
--resources puts two of these beside a speedup. Reads the text reports the
v++ link leaves in its impl_1 run directory; nothing here runs Vivado.

Use --diagnose-placement --util <placed report> --build-manifest <JSON>
[--implementation-log <log or .gz>] --out <new JSON> to inspect existing
CLB packing and local SLL demand without requiring final timing evidence.
This mode produces diagnostic evidence only and never qualifies an image.
"""
import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

TOTAL_ROWS = {'lut': 'CLB LUTs', 'ff': 'CLB Registers', 'bram': 'Block RAM Tile',
              'uram': 'URAM', 'dsp': 'DSPs'}
OPTIONAL_ROWS = {'clb': 'CLB', 'control_sets': 'Unique Control Sets'}


def _num(x):
    x = x.strip().lstrip('<')
    try:
        return int(x)
    except ValueError:
        return float(x)


def _util_tables(text):
    """Identify columns explicitly; a per-SLR column is never global Used."""
    header = None
    for line in text.splitlines():
        if not line.startswith('|'):
            if not line.startswith('+'):
                header = None
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if cells[0] == 'Site Type':
            header = cells
        elif header is not None:
            if len(cells) != len(header):
                raise ValueError('resource row does not match its table header')
            yield header, cells


def parse_util(text):
    """Explicit global Used counts and independently named per-SLR counts.

    A report containing only per-SLR tables has no global total. Do not
    silently treat its first SLR as the whole device or infer missing values.
    """
    total, slr = {}, {}
    labels = dict(TOTAL_ROWS, **OPTIONAL_ROWS)
    for header, cells in _util_tables(text):
        for key, label in labels.items():
            if cells[0].rstrip('*').strip() != label:
                continue
            columns = [(name, i) for i, name in enumerate(header) if re.fullmatch(r'SLR\d+', name)]
            if 'Used' in header and not columns:
                number = _num(cells[header.index('Used')])
                if key in total and total[key] != number:
                    raise ValueError(f'conflicting global resource count for {key}')
                total[key] = number
            for name, i in columns:
                number = _num(cells[i])
                previous = slr.setdefault(name, {})
                if key in previous and previous[key] != number:
                    raise ValueError(f'conflicting {name} resource count for {key}')
                previous[key] = number
    crossings = {}
    for m in re.finditer(r'^\|\s*(SLR\d) <-> (SLR\d)\s*\|\s*(\d+)\s*\|\s*\|\s*(\d+)', text, re.M):
        crossings[f'{m.group(1)}<->{m.group(2)}'] = dict(used=int(m.group(3)), available=int(m.group(4)))
    return total, slr, crossings


def placement_diagnosis(text, implementation_log='', *, pressure_threshold=.9):
    """Read existing physical evidence without predicting candidate timing.

    Density is an adjustable diagnostic hint, not an acceptance criterion.
    Reported per-column SLL demand is an estimate, separate from placed SLL
    connectivity. Neither is a resource floor for modified RTL.
    """
    if (type(pressure_threshold) not in (int, float) or not math.isfinite(pressure_threshold)
            or not 0 < pressure_threshold <= 1):
        raise ValueError('pressure threshold must be in (0, 1]')
    total, slr, crossings = parse_util(text)
    percentages = {}
    for header, cells in _util_tables(text):
        if cells[0] not in ('CLB', 'CLB LUTs', 'Unique Control Sets'):
            continue
        for i, name in enumerate(header):
            match = re.fullmatch(r'(SLR\d+)\s*%', name)
            if match:
                # Preserve inequalities instead of turning <0.01 into exact.
                if cells[i] and (not math.isfinite(_num(cells[i])) or _num(cells[i]) < 0):
                    raise ValueError('invalid per-SLR utilization percentage')
                percentages.setdefault(match[1], {})[cells[0]] = cells[i]
    dense = []
    for name, values in percentages.items():
        value = values.get('CLB')
        if value and not value.startswith('<') and float(value) >= pressure_threshold * 100:
            dense.append(name)
    columns = {}
    capacity = None
    for line in implementation_log.splitlines():
        match = re.search(r'Number SLLs per Column:\s*(\d+)', line)
        if match:
            capacity = int(match[1])
            # A later partial snapshot must not inherit a cut from an older
            # snapshot and appear to be a complete simultaneous observation.
            columns = {}
        match = re.match(r'\s*SLR\s+\[(\d+)-(\d+)\]\s+(.*?)\s+Demand:\s*(\d+)\s+Available:\s*(\d+)\s+Utilization\(%\):\s*([0-9.]+)', line)
        if match:
            demands = [int(n) for n in re.findall(r'(\d+)\s+\(\s*\d+%\)', match[3])]
            if not demands or sum(demands) != int(match[4]):
                raise ValueError('SLL column demands do not match the reported sum')
            if capacity is None or capacity <= 0:
                raise ValueError('SLL column capacity is missing')
            if len(demands) * capacity != int(match[5]):
                raise ValueError('SLL column capacities do not match the reported total')
            columns[f'SLR{match[1]}<->SLR{match[2]}'] = dict(
                estimated_demand=demands, available_per_column=capacity,
                overloaded_columns=[i for i, n in enumerate(demands) if n > capacity],
                peak_demand_ratio=max(demands) / capacity,
                aggregate_demand_ratio=int(match[4]) / int(match[5]))
    state = re.search(r'^\|\s*Design State\s*:\s*(.*?)\s*$', text, re.M)
    return dict(schema='tnca-placement-diagnosis/1', status='diagnostic-only',
                design_state=state[1] if state else None, total=total, slr=slr,
                global_total_available=bool(total), reported_slr_percentages=percentages,
                summed_reported_slrs={key: sum(v[key] for v in slr.values())
                    # Unique control sets can occur in multiple SLRs; their
                    # per-region cardinalities are not an additive global count.
                    for key in dict(TOTAL_ROWS, clb='CLB')
                    if slr and all(key in v for v in slr.values())},
                density_threshold=pressure_threshold, dense_slrs=sorted(dense),
                placed_slr_connectivity=crossings, estimated_sll_by_column=columns,
                congestion_warning_present='[Route 35-447]' in implementation_log,
                interpretation='observed packing and estimated crossing pressure; not a candidate resource floor',
                gate_status='not-measured', routed_timing='not-measured',
                predicted_speedup=None, candidate_area=None)


def parse_timing(text, clock):
    """Design WNS/WHS and the named clock's own, from the summary's head."""
    head = '\n'.join(text.splitlines()[:600])
    out = {}
    m = re.search(r'Design Timing Summary.*?\n\s*-+.*?\n\s*([-0-9.]+)\s+([-0-9.]+)\s+\d+\s+\d+\s+([-0-9.]+)',
                  head, re.S)
    if m:
        out['design_wns_ns'], out['design_whs_ns'] = float(m.group(1)), float(m.group(3))
    intra = head.split('Intra Clock Table', 1)
    if len(intra) == 2:
        m = re.search(r'^\s*' + re.escape(clock) + r'\s+([-0-9.]+)\s+([-0-9.]+)\s+\d+\s+\d+\s+([-0-9.]+)',
                      intra[1], re.M)
        if m:
            out['wns_ns'], out['whs_ns'] = float(m.group(1)), float(m.group(3))
    m = re.search(r'^\s*' + re.escape(clock) + r'\s+\{[0-9. ]+\}\s+([0-9.]+)\s+([0-9.]+)', head, re.M)
    if m:
        out['period_ns'], out['target_mhz'] = float(m.group(1)), float(m.group(2))
    return out


def _read_report(path):
    return (gzip.open(path, 'rt', errors='replace').read() if path.suffix == '.gz'
            else path.read_text(errors='replace'))


def placement_bundle(util, log, build_manifest, *, pressure_threshold=.9):
    """Bind diagnostics to existing build evidence and the analysis source."""
    from contract_v2 import LOCK_SHA256
    identity = json.loads(build_manifest.read_text())
    if (not isinstance(identity, dict) or identity.get('contract_sha256') != LOCK_SHA256
            or not identity.get('hardware_source') or not identity.get('implementation_root')
            or type(identity.get('clock_target_hz')) is not int or identity['clock_target_hz'] <= 0):
        raise ValueError('build manifest needs hardware source, build path, positive target clock and frozen contract')
    result = placement_diagnosis(_read_report(util), _read_report(log) if log else '',
                                 pressure_threshold=pressure_threshold)
    if not result['slr']:
        raise ValueError('placement diagnosis requires actual per-SLR resource evidence')
    if result['design_state'] not in ('Placed', 'Fully Placed', 'Routed', 'Fully Routed'):
        raise ValueError('placement diagnosis requires a placed/routed report; synthesis assignments are insufficient')
    result['build_identity'] = identity
    result['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (util, log, build_manifest) if p is not None}
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['measured_clock_hz'] = None
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--util', type=Path, required=True)
    ap.add_argument('--timing', type=Path)
    ap.add_argument('--diagnose-placement', action='store_true',
                    help='diagnose existing packing/SLL evidence; never certify timing or routing')
    ap.add_argument('--implementation-log', type=Path, help='existing implementation log, optionally .gz')
    ap.add_argument('--build-manifest', type=Path, help='source/build/target-clock/contract identity for diagnostics')
    ap.add_argument('--pressure-threshold', type=float, default=.9,
                    help='adjustable CLB density hint for diagnostics; not an acceptance gate')
    ap.add_argument('--clock', default='ap_clk')
    ap.add_argument('--image', help='the xclbin (hashed) or its sha256')
    ap.add_argument('--clock-mhz', type=float, help='the measured device clock')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    if args.diagnose_placement:
        if args.build_manifest is None:
            ap.error('--diagnose-placement requires --build-manifest')
        if args.timing or args.image or args.clock_mhz is not None:
            ap.error('placement diagnostics do not accept image/timing/measured-clock qualification arguments')
        result = placement_bundle(args.util, args.implementation_log, args.build_manifest,
                                  pressure_threshold=args.pressure_threshold)
        rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.out:
            with args.out.open('x') as stream:
                stream.write(rendered)
        else:
            sys.stdout.write(rendered)
        return 0
    if args.timing is None:
        ap.error('--timing is required unless --diagnose-placement is selected')
    if args.implementation_log or args.build_manifest or args.pressure_threshold != .9:
        ap.error('placement diagnostic options require --diagnose-placement')
    total, slr, crossings = parse_util(args.util.read_text())
    timing = parse_timing(args.timing.read_text(errors='replace'), args.clock)
    if args.clock_mhz:
        timing['clock_mhz'] = args.clock_mhz
    image = args.image
    if image and Path(image).is_file():
        image = hashlib.sha256(Path(image).read_bytes()).hexdigest()
    missing = [k for k in TOTAL_ROWS if k not in total] + \
              [k for k in ('wns_ns', 'whs_ns') if k not in timing]
    doc = dict(schema='tnca-r40-resources/1', image=image, clock=args.clock, total=total, slr=slr,
               slr_crossings=crossings, timing=timing,
               sources={k: dict(name=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                        for k, f in (('util', args.util), ('timing', args.timing))})
    text = json.dumps(doc, indent=1) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    if missing:
        print(f'resources: not found in the reports: {missing}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
