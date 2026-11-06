#!/usr/bin/env python3
"""Stage ceilings of the full link, computed from the recorded runs only.

    python3 tools/r41/link_budget_report.py [--write]

Reads the raw evidence under evidence/r41/hbm_bw/link_budget/ (and the
Ramulator2 stream probe of evidence/r41/hbm_bw/model/), computes every number
of the LINK_BUDGET.md tables, prints them as Markdown and, with --write,
stores them in link_budget/stage_table.json. No measurement is a literal in
this file; the only constants are the target's own definition (32 GB/s, three
cores, 100 MHz, BANDWIDTH_PLAN.md section 2) and structural facts that are
cited to their RTL source where they are used.

Domains: `board` (run-1 image, 100 MHz), `rtl` (RTL simulation on GB10),
`model` (SystemC/Ramulator2, uncalibrated). A ceiling is per core in bytes
per cycle; chip GB/s is that times three cores times 100 MHz.
"""
import argparse
import glob
import json
from pathlib import Path
import re
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[2]
E = ROOT/'evidence/r41/hbm_bw/link_budget'
TARGET_BYTES_PER_SECOND = 32e9          # BANDWIDTH_PLAN.md section 1
CORES = 3
CLOCK_HZ = 100e6
REQUIRED = TARGET_BYTES_PER_SECOND / CORES / CLOCK_HZ
WORD = 8          # bytes per core-port request: tnca_tile CORE_DW = 64 when HYBRID_LSU = 0
LINE = 64         # bytes per NUCA line: tnca_tile LINE_BYTES


# the memory-bound families of the original suite (the run-1 board records)
STREAMING = ('add', 'copy', 'silu', 'gelu_tanh', 'swiglu', 'softmax', 'rmsnorm', 'layernorm', 'transpose')


def gbps(per_core):
    return per_core * CORES * CLOCK_HZ / 1e9


# ---------------------------------------------------------------------------- board
def board():
    rows = []
    with tarfile.open(E/'board_run1/run1_streaming.tgz') as tar:
        for member in tar.getmembers():
            if not re.match(r'runs/op\.[^/]+-q\.jsonl$', member.name):
                continue
            r = json.loads(tar.extractfile(member).read().decode().splitlines()[0])
            if r['status'] != 'pass' or r['family'] not in STREAMING:
                continue
            launches = r['provenance']['launch_map']['launches']
            programs = sum(l['grid'][0] * l['grid'][1] * l['grid'][2] for l in launches)
            cycles = r['timings']['device-cycles-serial-grid']['median']
            d = r['demand']
            moved = d['read_bytes'] + d['write_bytes']
            rows.append(dict(point=r['point_id'], family=r['family'], tier=r['tier'], cycles=cycles,
                             programs=programs, bytes=moved, chip_bytes_per_cycle=moved / cycles,
                             per_core=moved / cycles / CORES, gbps=moved / cycles * CLOCK_HZ / 1e9,
                             cycles_per_program=cycles / programs, bytes_per_program=moved / programs,
                             clock_hz=r['timings']['device-cycles-serial-grid']['clock_hz'],
                             image=r['platform']['image_sha256']))
    return sorted(rows, key=lambda x: x['point'])


# ---------------------------------------------------------------------------- rtl
def rtl():
    return rtl_rows(sorted(glob.glob(str(E/'rtl/batch*/*.jsonl'))))


def rtl_rows(paths):
    """Stage metrics of every RTL record in `paths` (also used by g4_rtl_report.py)."""
    out = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r['status'] != 'pass' or not r['timed_launches']:
                out.append(dict(point=r['point'], status=r['status'], problems=r.get('problems'), source=path))
                continue
            cycles = sum(l['cycles'] for l in r['timed_launches'])
            tiles = [dict() for _ in r['timed_launches'][0]['tiles']]
            ports = [dict() for _ in r['timed_launches'][0]['ports']]
            for l in r['timed_launches']:
                for i, row in enumerate(l['tiles'].values()):
                    for k, v in row.items():
                        tiles[i][k] = tiles[i].get(k, 0) + v
                for i, row in enumerate(l['ports'].values()):
                    for k, v in row.items():
                        ports[i][k] = ports[i].get(k, 0) + v
            per = r['time']['per_launch']
            programs = sum(p['grid'][0] * p['grid'][1] * p['grid'][2] for p in per)
            d = r['demand']
            moved = d['read_bytes'] + d['write_bytes']
            s = lambda key: sum(t.get(key, 0) for t in tiles)       # noqa: E731
            p = lambda key: sum(q.get(key, 0) for q in ports)       # noqa: E731
            core_cycles = cycles * CORES
            row = dict(point=r['point'], status='pass', source=Path(path).name, cycles=cycles,
                       device_cycles=r['device_cycles_samples'], programs=programs, bytes=moved,
                       per_core=moved / core_cycles, cycles_per_program=cycles / programs,
                       issued_per_program=s('issued') / programs,
                       ipc=s('issued') / core_cycles,
                       issue_ceiling=moved / s('issued'),
                       stall_lsu=s('x_lsu') / core_cycles, stall_hazard=s('x_hazard') / core_cycles,
                       stall_fpu=s('x_fpu') / core_cycles, stall_vrf=(s('x_vrf') + s('x_vrfwb')) / core_cycles,
                       stall_frontend=s('x_frontend') / core_cycles,
                       port_words=s('creq'), port_write_words=s('creq_we'),
                       port_words_per_cycle=s('creq') / core_cycles,
                       port_bytes_per_program=s('creq') * WORD / programs,
                       ni_backpressure=s('inj_stall') / core_cycles,
                       home_requests_per_cycle=s('hreq') / core_cycles,
                       home_backpressure=s('hreq_stall') / core_cycles,
                       home_engine_busy=1 - s('e_sel') / core_cycles,
                       home_engine_cycles_per_request=(core_cycles - s('e_sel')) / s('hreq'),
                       home_hits=s('hit'), home_misses=s('miss'), home_merges=s('merge'), home_writebacks=s('wb'),
                       line_reads=s('mreq_rd'), line_writes=s('mreq_wr'),
                       line_read_bytes_per_demand_read=s('mreq_rd') * LINE / d['read_bytes'],
                       outlet_busy=p('busy') / core_cycles,
                       outlet_bytes_per_cycle=(p('r') + p('w')) * 32 / core_cycles)
            if s('c_occ_sum'):
                row.update(port_mean_outstanding=s('c_occ_sum') / core_cycles,
                           port_mean_latency=s('c_occ_sum') / s('creq'),
                           home_mean_latency=s('h_occ_sum') / s('hreq'))
            if p('rd_occ_sum'):
                row.update(outlet_read_latency=p('rd_occ_sum') / max(1, p('ar')),
                           outlet_mean_outstanding_reads=p('rd_occ_sum') / core_cycles)
            row['tiles'], row['ports'] = tiles, ports
            out.append(row)
    return out


# ---------------------------------------------------------------------------- fixtures
def home():
    d = json.loads((E/'home/results.json').read_text())
    return [dict(label=r['label'], mode=r['mode'], dram_latency=r['dram_latency'], stall=r['dram_random_stall'],
                 cycles_per_word=r['cycles_per_word'], bytes_per_cycle=r['bytes_per_cycle'],
                 misses=r['miss_total'], merges=r['merge'], writebacks=r['wb']) for r in d['rows']]


def noc():
    rows = []
    for path in sorted((E/'noc').glob('*.log')):
        m = re.search(r'data=(\d+) payload=(\d+) links=(\d+) stall=(\d+) requests=(\d+) injection_span=(\d+)',
                      path.read_text())
        data, payload, links, stall, requests, span = map(int, m.groups())
        rows.append(dict(log=path.name, data_bits=data, links=links, random_stall=stall, requests=requests,
                         span=span, bytes_per_cycle=requests * data / 8 / span))
    return rows


def outlet():
    rows = []
    for path in sorted((E/'outlet').glob('*.log')):
        m = re.search(r'LINE STREAM MEASURE pipeline=(\d+) write=(\d+) slots=(\d+) delay=(\d+) lines=(\d+) '
                      r'issue_span=(\d+)', path.read_text())
        if not m:
            continue
        pipe, write, slots, delay, lines, span = map(int, m.groups())
        rows.append(dict(log=path.name, pipelined_issue=pipe, write=write, slots=slots, reply_delay=delay,
                         lines=lines, cycles_per_line=(span + 1) / lines,
                         bytes_per_cycle=lines * LINE / (span + 1)))
    return rows


def pc_model():
    d = json.loads((ROOT/'evidence/r41/hbm_bw/model/r41-hbmbw-stream-probe.json').read_text())
    return [dict(mode=c['mode'], window=c['window_lines'], lines=c['lines'], gbps=c['per_pc_GBps'],
                 bytes_per_cycle=c['per_pc_GBps'] * 1e9 / CLOCK_HZ)
            for c in d['cases'] if c['lines'] == 16384 and c['issue_kernel_cycles'] == 1]


def census():
    path = E/'census/census.json'
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------- stage table
def stages(b, r, h, n, o, pc):
    def pick(name):
        return next((x for x in r if x['point'].startswith(name) and x.get('status') == 'pass'), None)
    add = pick('op.add/membound/fp16-small-contig')
    big = [x for x in b if x['family'] == 'add' and x['tier'] == 'hbm']
    table = []

    def row(stage, ceiling, domain, basis, observed=None):
        table.append(dict(stage=stage, ceiling_bytes_per_cycle_per_core=ceiling,
                          ceiling_chip_gbps=gbps(ceiling) if ceiling else None,
                          required_bytes_per_cycle_per_core=REQUIRED,
                          shortfall=REQUIRED / ceiling if ceiling else None,
                          domain=domain, basis=basis, observed_in_fp16_add=observed))
    if add:
        row('instruction issue (one instruction per cycle, fp16 add as compiled)', add['issue_ceiling'], 'rtl',
            f"{add['bytes']/add['programs']:.0f} B and {add['issued_per_program']:.1f} issued instructions per "
            f"program (RTL counters); observed IPC {add['ipc']:.3f}",
            observed=add['per_core'])
    ni = max(n, key=lambda x: x['bytes_per_cycle'])
    row('core port, LSU and tensor transfer engine (64-bit word per request)', ni['bytes_per_cycle'], 'rtl',
        f"one {WORD}-byte request accepted per cycle by the core NI ({ni['requests']} in {ni['span']} cycles, "
        f"{ni['log']}); the LSU and the transfer engine share this port",
        observed=add['port_words_per_cycle'] * WORD if add else None)
    if add and 'port_mean_latency' in add:
        # LSU_INFLIGHT vector commands of VLEN fp16 lanes, each waiting one measured round trip
        inflight = json.loads((ROOT/'evidence/r41/stage2/integration/image.json').read_text())['parameters']['LSU_INFLIGHT']
        vector = 16 * 2
        row('LSU memory-level parallelism (commands in flight / round-trip latency)',
            inflight * vector / add['port_mean_latency'], 'rtl',
            f"LSU_INFLIGHT={inflight} commands of {vector} B (image parameter) over a mean round trip of "
            f"{add['port_mean_latency']:.1f} cycles per word request (Little's law on RTL counters; mean "
            f"{add['port_mean_outstanding']:.2f} words outstanding per core)", observed=add['per_core'])
    row('NI and mesh NoC (one 216-bit flit with 64-bit data per cycle and virtual network)',
        ni['bytes_per_cycle'], 'rtl', f"{ni['log']}: {ni['requests']} packets in {ni['span']} cycles",
        observed=None)
    hot = next(x for x in h if x['mode'] == 1 and x['dram_latency'] == 40 and x['stall'])
    rd = next(x for x in h if x['mode'] == 2 and x['dram_latency'] == 40 and not x['stall'])
    wr = next(x for x in h if x['mode'] == 3 and x['dram_latency'] == 40 and not x['stall'])
    stream = 1 / ((2 / 3) / rd['bytes_per_cycle'] + (1 / 3) / wr['bytes_per_cycle'])
    row('NUCA home slice, hits', hot['bytes_per_cycle'], 'rtl',
        f"{hot['cycles_per_word']:.2f} cycles per 8-byte word ({hot['label']})")
    row('NUCA home slice, streaming (2 reads : 1 write, as fp16 add)', stream, 'rtl',
        f"reads {rd['cycles_per_word']:.2f} and writes {wr['cycles_per_word']:.2f} cycles per word with a "
        f"40-cycle synthetic DRAM ({rd['label']}, {wr['label']})",
        observed=add['home_requests_per_cycle'] * WORD if add else None)
    o8 = next(x for x in o if x['pipelined_issue'] == 0 and x['slots'] == 8 and x['write'] == 0)
    o32 = next(x for x in o if x['pipelined_issue'] == 0 and x['slots'] == 32 and x['write'] == 0)
    row('AXI outlet (tnca_line_axi, default issue, 8 lines outstanding)', o8['bytes_per_cycle'], 'rtl',
        f"{o8['cycles_per_line']:.2f} cycles per line with a {o8['reply_delay']}-cycle responder; "
        f"{o32['cycles_per_line']:.2f} with 32 slots (issue limit)",
        observed=add['outlet_bytes_per_cycle'] if add else None)
    p8 = next(x for x in pc if x['window'] == 8 and x['mode'] == 'read')
    p64 = next(x for x in pc if x['window'] == 64 and x['mode'] == 'read')
    row('HBM pseudo-channel behind one outlet (8-line window)', p8['bytes_per_cycle'], 'model',
        f"Ramulator2 HBM2 PC, {p8['gbps']} GB/s read with 8 lines in flight, {p64['gbps']} GB/s with 64 "
        f"(issue-limited at one line per cycle)")
    if big:
        row('end to end, fp16 add, 64 Mi elements (measured)', big[0]['per_core'], 'board',
            f"{big[0]['cycles']} cycles, {big[0]['cycles_per_program']:.1f} cycles per 16-element program")
    return table


def markdown(b, r, h, n, o, pc, table):
    lines = ['| Stage | Ceiling B/cycle/core | Chip GB/s at 100 MHz | Shortfall vs 106.7 | Domain | Basis |',
             '|---|---:|---:|---:|---|---|']
    for t in table:
        c = t['ceiling_bytes_per_cycle_per_core']
        lines.append(f"| {t['stage']} | {c:.2f} | {gbps(c):.2f} | {REQUIRED / c:.0f}x | {t['domain']} | "
                     f"{t['basis']} |")
    lines += ['', '| Point (RTL) | Cycles | Programs | B/cycle/core | Issued/program | IPC | LSU stall | '
              'Port words/cycle | Port latency | Home busy | Home cycles/request | Line reads per demand read | '
              'Outlet busy |', '|---' * 13 + '|']
    for x in r:
        if x.get('status') != 'pass':
            lines.append(f"| {x['point']} | {x['status']} | | | | | | | | | | | |")
            continue
        lat = f"{x['port_mean_latency']:.0f}" if 'port_mean_latency' in x else ''
        lines.append(f"| {x['point']} | {x['cycles']} | {x['programs']} | {x['per_core']:.3f} | "
                     f"{x['issued_per_program']:.1f} | {x['ipc']:.3f} | {x['stall_lsu']:.2f} | "
                     f"{x['port_words_per_cycle']:.3f} | {lat} | {x['home_engine_busy']:.2f} | "
                     f"{x['home_engine_cycles_per_request']:.2f} | {x['line_read_bytes_per_demand_read']:.2f} | "
                     f"{x['outlet_busy']:.3f} |")
    lines += ['', '| Point (board, run 1) | Cycles | Programs | Cycles/program | B/program | B/cycle/core | GB/s |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for x in b:
        lines.append(f"| {x['point']} | {x['cycles']} | {x['programs']} | {x['cycles_per_program']:.1f} | "
                     f"{x['bytes_per_program']:.0f} | {x['per_core']:.3f} | {x['gbps']:.3f} |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    b, r, h, n, o, pc = board(), rtl(), home(), noc(), outlet(), pc_model()
    table = stages(b, r, h, n, o, pc)
    print(markdown(b, r, h, n, o, pc, table))
    if a.write:
        (E/'stage_table.json').write_text(json.dumps(dict(
            schema='tnca-link-budget/1', required_bytes_per_cycle_per_core=REQUIRED,
            definition=dict(target_gbps=TARGET_BYTES_PER_SECOND / 1e9, cores=CORES, clock_hz=CLOCK_HZ),
            stages=table, board=b, rtl=r, home=h, noc=n, outlet=o, pc_model=pc, census=census()),
            indent=1) + '\n')


if __name__ == '__main__':
    main()
