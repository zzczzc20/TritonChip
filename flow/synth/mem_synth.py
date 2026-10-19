#!/usr/bin/env python3
"""
Memory-preserving structural synthesis -- doc/phase3_tasks_and_acceptance.md
R3-10.

The difference from flow/synth/synth.py is one pass. That one maps
everything to gates, which turns a register file into a few hundred
thousand flops and answers a question nobody asked. This one stops before
`memory_map`, so an array that a real flow would give to an SRAM stays a
memory object and can be counted as one.

    proc; opt; memory -nomap; opt

`memory -nomap` is the whole memory pass except the mapping: memory_dff
folds an output register into the read port, memory_share merges the
sixteen per-lane writes into one port with a wide enable, memory_collect
builds the object. Stopping at `memory_collect` alone -- which is what the
first version of this did -- reports sixteen write ports and an
asynchronous read for RTL that has neither.

What this does NOT do is claim an area or a frequency. There is no PDK
here and no FPGA mapping, so SYN-04 allows "memory-inferred" and nothing
more; SYN-05's Fmax, power and physical area are NOT MEASURED, and a gate
depth is not converted into a GHz.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RTL = os.path.join(ROOT, 'rtl')
INC = os.path.join(RTL, 'common')

# The structural run has a hard cap. The contract sets it: "结构综合单任务
# 上限1200秒；超时就是未完成，不能将缺失模块记录为0 gate."
TIMEOUT = 1200

# top module -> the files it needs. Kept explicit rather than globbed: a
# module that silently stopped being included would report zero cells, and
# zero is the number this contract most wants not to be invented.
# `flatten` for a top that instantiates others: without it the memories sit
# inside the sub-module's definition and the top reports none. The first
# version of this said tnca_vrf_banked had zero memory bits, which is a
# strange thing to believe about a register file.
TOPS = {
    'tnca_rf_bank': (['core/tnca_rf_bank.v'], False),
    'tnca_vrf_banked': (['core/tnca_rf_bank.v', 'core/tnca_vrf_banked.v'],
                        True),
    'tnca_operand_collector': (['core/tnca_operand_collector.v'], False),
    'tnca_vector_rf': (['core/tnca_vector_rf.v'], False),
    'tnca_acc_rf': (['core/tnca_acc_rf.v'], False),
}

# Below this, an inferred "memory" is a handful of control bits that any
# real flow maps to flops -- the collector's three-bit need/have masks come
# out as memory objects with eight read ports, which no SRAM has and none
# was asked for. They are counted separately rather than added to the data
# capacity or hidden. SYN-01: "remaining valid/bypass/collector bits 独立
# 列账".
SMALL_ARRAY_BITS = 1024


def control_array(top, memory):
    # The accumulator's per-row/lane valid bitmap is 4096 bits, larger than
    # the generic small-array cutoff. It is reset and has 257 write ports;
    # counting it as accumulator data overstates the declared 16 KiB capacity.
    return memory['bits'] < SMALL_ARRAY_BITS or (top == 'tnca_acc_rf' and memory['name'] == 'valid')

FF_CELLS = ('$dff', '$dffe', '$sdff', '$sdffe', '$sdffce', '$adff',
            '$adffe', '$dffsr', '$dffsre', '$aldff', '$aldffe')
LATCH_CELLS = ('$dlatch', '$dlatchsr', '$adlatch')


def bits(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and set(value) <= set('01xz'):
        return int(value.replace('x', '0').replace('z', '0'), 2)
    return 0


def run_top(top, files, flatten=False):
    out = {'top': top, 'ok': False}
    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'out.json')
        srcs = ' '.join(os.path.join(RTL, f) for f in files)
        script = (f"read_verilog -DSYNTHESIS -I {INC} {srcs}; "
                  f"hierarchy -top {top}; "
                  + ("flatten; " if flatten else "")
                  + f"proc; opt -fast; memory -nomap; opt -fast; "
                  f"check; stat; write_json {js}")
        try:
            r = subprocess.run(['yosys', '-p', script], capture_output=True,
                               text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            out['error'] = f"over the {TIMEOUT}s structural cap: not done"
            return out
        out['log'] = r.stdout[-4000:]
        if r.returncode != 0 or not os.path.exists(js):
            out['error'] = (r.stderr or r.stdout)[-400:]
            return out

        net = json.load(open(js))
        mod = net['modules'].get(top) or {}
        mems, ff_bits, latches, cells = [], 0, 0, {}
        for _cname, cell in (mod.get('cells') or {}).items():
            t = cell['type']
            cells[t] = cells.get(t, 0) + 1
            p = cell.get('parameters', {})
            if t.startswith('$mem'):
                mems.append({
                    'name': _cname,
                    'width': bits(p.get('WIDTH')),
                    'depth': bits(p.get('SIZE')),
                    'bits': bits(p.get('WIDTH')) * bits(p.get('SIZE')),
                    'rd_ports': bits(p.get('RD_PORTS')),
                    'wr_ports': bits(p.get('WR_PORTS')),
                    'rd_sync': bool(bits(p.get('RD_CLK_ENABLE'))),
                    'transparent': bool(bits(p.get('RD_TRANSPARENCY_MASK'))),
                })
            elif t in FF_CELLS:
                ff_bits += bits(p.get('WIDTH')) or 1
            elif t in LATCH_CELLS:
                latches += 1

        # yosys says what it could not resolve; the contract wants those
        # named rather than absorbed into a pass.
        # Every warning, split into the ones with a known cause and the
        # ones without. "Replacing memory X with list of registers" is the
        # expected outcome for a small control array -- it is what should
        # happen to a three-bit mask with eight readers -- so it is
        # explained and listed as such. The summary line ("Warnings: N
        # unique messages") is not a warning; counting it as one hid the
        # real ones behind a number.
        raw = [l.strip() for l in r.stdout.splitlines()
               if l.strip().startswith('Warning:')]
        explained = [l for l in raw
                     if 'with list of registers' in l]
        unexplained = [l for l in raw if l not in explained]
        warnings = unexplained
        # `check` names what it could not resolve. SYN-03 wants latches,
        # combinational loops, multiple drivers and unconnected required
        # ports named rather than absorbed into a pass, so its findings are
        # carried out of the log rather than left in it.
        problems = [l.strip() for l in r.stdout.splitlines()
                    if re.search(r'(found|Found) \d+ problems|'
                                 r'combinational loop|multiple drivers|'
                                 r'wire .* is used but has no driver',
                                 l)]
        data = [m for m in mems if not control_array(top, m)]
        small = [m for m in mems if control_array(top, m)]
        out.update(problems=problems, explained_warnings=explained)
        out.update(ok=True, memories=mems, data_memories=data,
                   small_arrays=small, ff_bits=ff_bits,
                   latches=latches, cells=cells,
                   memory_bits=sum(m['bits'] for m in data),
                   small_bits=sum(m['bits'] for m in small),
                   logic_cells=sum(n for t, n in cells.items()
                                   if not t.startswith('$mem')
                                   and t not in FF_CELLS),
                   warnings=warnings[:10])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tops', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    wanted = args.tops or sorted(TOPS)
    # The tops are independent yosys processes, so they run together. The
    # wall time is then the slowest one rather than their sum, which
    # matters because tnca_acc_rf -- 256 rows behind 257 write ports -- is
    # most of it, and CI pays this on every push.
    results = {}
    if len(wanted) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(wanted), 5)) as pool:
            futures = {}
            for top in wanted:
                if top not in TOPS:
                    continue
                files, flat = TOPS[top]
                futures[top] = pool.submit(run_top, top, files, flat)
            for top, fut in futures.items():
                results[top] = fut.result()
    report = {'tops': {}, 'cap_seconds': TIMEOUT,
              'flow': 'proc; opt; memory -nomap; opt',
              'mapping': 'memory-inferred; no PDK, no FPGA RAM primitive',
              'fmax': 'NOT MEASURED', 'area': 'NOT MEASURED',
              'power': 'NOT MEASURED'}
    print("R3-10 memory-preserving structural synthesis")
    print(f"  flow: {report['flow']}   cap: {TIMEOUT}s per top")
    for top in wanted:
        if top not in TOPS:
            print(f"  {top}: not a declared top")
            continue
        files, flat = TOPS[top]
        res = results.get(top) or run_top(top, files, flatten=flat)
        report['tops'][top] = res
        if not res['ok']:
            print(f"  {top:<24s} FAILED  {res.get('error', '')[:60]}")
            continue
        mb = res['memory_bits']
        print(f"  {top:<24s} {len(res['data_memories'])} data memory "
              f"object(s), {mb} bit(s) = {mb / 8192:.1f} KiB; "
              f"{res['ff_bits']} flop bit(s) + {res['small_bits']} bit(s) "
              f"in {len(res['small_arrays'])} small array(s); "
              f"{res['logic_cells']} logic cell(s), "
              f"{res['latches']} latch(es)")
        for m in res['data_memories']:
            print(f"      {m['name']}: {m['width']}b x {m['depth']} = "
                  f"{m['bits']} bits, {m['rd_ports']} read port(s), "
                  f"{m['wr_ports']} write port(s), "
                  f"{'synchronous' if m['rd_sync'] else 'ASYNCHRONOUS'} read,"
                  f" {'transparent' if m['transparent'] else 'read-first'}")
        for m in res['small_arrays']:
            print(f"      (control) {m['name']}: {m['bits']} bits, "
                  f"{m['rd_ports']} read port(s) -- flops in any real flow, "
                  f"not counted as data capacity")
    print(f"  Fmax / area / power: NOT MEASURED (no library)")
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
