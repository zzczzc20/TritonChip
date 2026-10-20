#!/usr/bin/env python3
"""Group failing setup paths of a routed TNCA image into source/destination cones.

Input is either the TSV written by tools/r40/census_failing_endpoints.tcl (one
row per failing endpoint) or a Vivado text report (report_timing /
report_timing_summary) whose path headers are parsed. Text reports only
contain the paths Vivado printed, so their coverage is reported against the
design TNS/endpoint totals given on the command line.

Offline analysis only: no Vivado, no board, no checkpoint access.

  group_failing_paths.py --tsv failing_endpoints.tsv --out groups.json
  group_failing_paths.py --report timing_summary_routed.rpt --tns -6679.026 \
      --endpoints 17332 --out groups.json
"""
import argparse
import collections
import json
import re
import sys

PREFIX = 'level0_i/ulp/tnca_u50_1/inst/device/'


def family(pin):
    """Normalise a pin to its storage family (bits, replicas and RAM slices)."""
    name = pin[len(PREFIX):] if pin.startswith(PREFIX) else pin
    parts = name.split('/')[:-1]  # drop the pin
    if parts and re.fullmatch(r'RAM[A-Z0-9_]+', parts[-1]):
        parts = parts[:-1]
    out = []
    for p in parts:
        p = re.sub(r'\[\d+\]', '[*]', p)
        p = re.sub(r'_reg_\d+_\d+_\d+_\d+$', '_reg', p)
        p = re.sub(r'_replica(_\d+)?$', '_replica', p)
        p = re.sub(r'_rep(__\d+)?$', '_rep', p)
        out.append(p)
    return '/'.join(out)


def unit(fam):
    """Module-instance level of a family (drop the register leaf)."""
    return fam.rsplit('/', 1)[0] if '/' in fam else fam


def tile_of(pin):
    m = re.search(r'g_tile\[(\d+)\]', pin)
    return int(m.group(1)) if m else None


def parse_tsv(path):
    rows = []
    with open(path) as f:
        head = f.readline().rstrip('\n').split('\t')
        for line in f:
            v = dict(zip(head, line.rstrip('\n').split('\t')))
            rows.append(dict(slack=float(v['slack_ns']), source=v['startpoint'],
                             destination=v['endpoint'], levels=int(v['logic_levels']),
                             delay=float(v['datapath_ns']), logic=float(v['logic_ns']),
                             net=float(v['net_ns']), start_slr=v.get('start_slr'),
                             end_slr=v.get('end_slr'), slr_crossings=None))
    return rows


def parse_report(path, group=None):
    """Setup paths of a Vivado text report (one row per printed path)."""
    rows, row, section_group, in_data = [], None, None, False
    with open(path, errors='replace') as f:
        for line in f:
            # Only the launch-to-capture data section carries data-path SLR
            # crossings and nets; clock-tree lines before/after are excluded.
            if row is not None and 'C_Q)' in line and 'Prop_' in line:
                in_data = True
            elif in_data and re.match(r'^\s+\(clock ', line):
                in_data = False
            if line.startswith('Max Delay Paths'):
                section_group = 'max'
            elif line.startswith('Min Delay Paths'):
                section_group = 'min'
            m = re.match(r'^Slack \((VIOLATED|MET)\)\s*:\s*(-?[0-9.]+)ns', line)
            if m:
                if row:
                    rows.append(row)
                row = dict(slack=float(m.group(2)), kind=section_group, crossings=[])
                in_data = False
                continue
            if row is None:
                continue
            m = re.match(r'^\s+Source:\s+(\S+)', line)
            if m:
                row['source'] = m.group(1)
            m = re.match(r'^\s+Destination:\s+(\S+)', line)
            if m:
                row['destination'] = m.group(1)
            m = re.match(r'^\s+Path Group:\s+(\S+)', line)
            if m:
                row['group'] = m.group(1)
            m = re.match(r'^\s+Path Type:\s+(Setup|Hold)', line)
            if m:
                row['type'] = m.group(1)
            m = re.match(r'^\s+Data Path Delay:\s+([0-9.]+)ns\s+\(logic\s+([0-9.]+)ns.*route\s+([0-9.]+)ns', line)
            if m:
                row['delay'], row['logic'], row['net'] = map(float, m.groups())
            m = re.match(r'^\s+Logic Levels:\s+(\d+)', line)
            if m:
                row['levels'] = int(m.group(1))
            m = re.search(r'SLR Crossing\[(\d)->(\d)\]', line)
            if m and in_data:
                row['crossings'].append(m.group(1) + '->' + m.group(2))
            m = re.search(r'net \(fo=(\d+), routed\)\s+([0-9.]+)\s+[0-9.]+\s+reconfigurable\s+(\S+)', line)
            if m and in_data:
                row.setdefault('nets', []).append((int(m.group(1)), float(m.group(2)), m.group(3)))
    if row:
        rows.append(row)
    out = []
    for r in rows:
        if r.get('type') != 'Setup' or 'source' not in r:
            continue
        if group and r.get('group') != group:
            continue
        data_x = list(r['crossings'])
        fanouts = sorted((n for n in r.get('nets', []) if n[0] >= 64), reverse=True)
        out.append(dict(slack=r['slack'], source=r['source'], destination=r['destination'],
                        levels=r.get('levels'), delay=r.get('delay'), logic=r.get('logic'),
                        net=r.get('net'), start_slr=None, end_slr=None,
                        slr_crossings=data_x,
                        high_fanout=[dict(fanout=a, net_ns=b, net=c[len(PREFIX):] if c.startswith(PREFIX) else c)
                                     for a, b, c in fanouts]))
    return out


def group_rows(rows):
    fams = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: r['slack']):
        key = (family(r['source']), family(r['destination']))
        g = fams.setdefault(key, dict(source_family=key[0], destination_family=key[1],
                                      endpoints=0, tns_ns=0.0, worst=r, tiles=collections.Counter(),
                                      levels=[], logic_fraction=[], slr_paths=collections.Counter()))
        g['endpoints'] += 1
        g['tns_ns'] += r['slack']
        g['tiles'][str(tile_of(r['destination']))] += 1
        if r.get('levels') is not None:
            g['levels'].append(r['levels'])
        if r.get('delay'):
            g['logic_fraction'].append(r['logic'] / r['delay'])
        if r.get('slr_crossings') is not None:
            g['slr_paths']['->'.join(r['slr_crossings']) or 'none'] += 1
        elif r.get('start_slr'):
            g['slr_paths'][f"{r['start_slr']}->{r['end_slr']}"] += 1
    out = []
    for g in fams.values():
        w = g['worst']
        out.append(dict(
            source_family=g['source_family'], destination_family=g['destination_family'],
            source_unit=unit(g['source_family']), destination_unit=unit(g['destination_family']),
            endpoints=g['endpoints'], tns_ns=round(g['tns_ns'], 3), worst_slack_ns=w['slack'],
            worst_levels=w.get('levels'), worst_delay_ns=w.get('delay'), worst_logic_ns=w.get('logic'),
            worst_net_ns=w.get('net'), worst_source=w['source'], worst_destination=w['destination'],
            worst_slr_crossings=w.get('slr_crossings'), worst_high_fanout=w.get('high_fanout'),
            levels_range=[min(g['levels']), max(g['levels'])] if g['levels'] else None,
            mean_logic_fraction=round(sum(g['logic_fraction']) / len(g['logic_fraction']), 3)
            if g['logic_fraction'] else None,
            destination_tiles=dict(g['tiles']), slr_paths=dict(g['slr_paths'])))
    units = collections.OrderedDict()
    for g in out:
        k = (g['source_unit'], g['destination_unit'])
        u = units.setdefault(k, dict(source_unit=k[0], destination_unit=k[1], endpoints=0,
                                     tns_ns=0.0, worst_slack_ns=g['worst_slack_ns'], families=0))
        u['endpoints'] += g['endpoints']
        u['tns_ns'] = round(u['tns_ns'] + g['tns_ns'], 3)
        u['worst_slack_ns'] = min(u['worst_slack_ns'], g['worst_slack_ns'])
        u['families'] += 1
    return out, sorted(units.values(), key=lambda u: u['tns_ns'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--tsv')
    src.add_argument('--report')
    ap.add_argument('--group', default='clk_kernel_00_unbuffered_net')
    ap.add_argument('--tns', type=float, help='design TNS of the path group (ns), for coverage')
    ap.add_argument('--endpoints', type=int, help='failing endpoints of the path group, for coverage')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    rows = parse_tsv(a.tsv) if a.tsv else parse_report(a.report, a.group)
    failing = [r for r in rows if r['slack'] < 0]
    fams, units = group_rows(failing)
    covered_tns = round(sum(r['slack'] for r in failing), 3)
    report = dict(
        schema='tnca-r40-failing-path-groups/1',
        input=a.tsv or a.report,
        input_kind='census-tsv' if a.tsv else 'printed-report-paths',
        path_group=a.group,
        failing_paths_parsed=len(failing),
        design_failing_endpoints=a.endpoints,
        design_tns_ns=a.tns,
        parsed_tns_ns=covered_tns,
        endpoint_coverage=(len(failing) / a.endpoints) if a.endpoints else None,
        tns_coverage=(covered_tns / a.tns) if a.tns else None,
        complete=bool(a.endpoints and len(failing) >= a.endpoints),
        unit_groups=units,
        family_groups=fams)
    with open(a.out, 'w') as f:
        json.dump(report, f, indent=1)
        f.write('\n')
    print(f"parsed {len(failing)} failing paths in {len(fams)} families / {len(units)} unit pairs;"
          f" TNS covered {covered_tns} of {a.tns}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
