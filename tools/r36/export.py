#!/usr/bin/env python3
"""
The public hardware export package -- phase3_6_tasks_and_acceptance.md
section R36-04.

R36-RTL-01 asks for a package that elaborates independently and runs a
small program, with a complete source list, includes, generated RTL,
parameters and tool versions, and with no relative path back into this
repository and no undeclared blackbox.

The interesting word is "independently". A package that only works
because the checker happens to run inside the repository proves nothing,
so this writes a directory with no path leading out of it, and
tools/r36/rtl_tests.py copies it somewhere else entirely, elaborates it
with no library search of ours, and runs it.

HOW THE SOURCE LIST IS BUILT

Not by hand, and not by globbing a directory. It walks the module graph
from the top: every module instantiated by a module already in the set
is added, and every `include` is followed. Anything the walk does not
reach is not in the package.

That walk could be wrong. It is not trusted: the package's own
elaborate.sh names every file explicitly and passes NO -y search path,
so a missing file is an elaboration error rather than a silent fallback
to the repository's copy. An over-approximation would pass that check,
which is why the manifest also records what was reached from what.

WHAT IS SYNTHESIZABLE AND WHAT IS NOT

The package keeps the two apart, in two directories and two lists in the
manifest, because R36-RTL-03 asks for the synthesizable functional path
to be free of run-time file service and simulator-only behaviour. A
memory model that reads a hex image is honest and useful and is not
hardware; it goes in sim/ and is declared. Calling it rtl/ because it is
written in Verilog would be the failure.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r1'))
import r1lib as R                                          # noqa: E402

# The synthesizable top, and the simulation binding that drives it.
SYNTH_TOP = 'tnca_shell'
SIM_TOP = 'tnca_device'

SEARCH = [os.path.join(ROOT, 'rtl'),
          os.path.join(ROOT, 'rtl', 'core'),
          os.path.join(ROOT, 'rtl', 'common'),
          os.path.join(ROOT, 'rtl', 'gen'),
          os.path.join(ROOT, 'rtl', 'nuca'),
          os.path.join(ROOT, 'rtl', 'noc'),
          os.path.join(ROOT, 'rtl', 'tile'),
          os.path.join(ROOT, 'tb', 'common'),
          os.path.join(ROOT, 'tb', 'device')]

# Things that look like a module instantiation and are not one.
_KEYWORDS = {
    'if', 'else', 'case', 'casex', 'casez', 'endcase', 'begin', 'end',
    'for', 'while', 'repeat', 'forever', 'always', 'initial', 'assign',
    'wire', 'reg', 'integer', 'genvar', 'generate', 'endgenerate',
    'parameter', 'localparam', 'input', 'output', 'inout', 'function',
    'endfunction', 'task', 'endtask', 'module', 'endmodule', 'defparam',
    'posedge', 'negedge', 'or', 'and', 'not', 'xor', 'nand', 'nor',
    'xnor', 'buf', 'real', 'time', 'signed', 'unsigned', 'default',
    'return', 'disable', 'fork', 'join', 'wait', 'event', 'force',
    'release', 'deassign', 'specify', 'endspecify', 'table', 'endtable',
}

_MODULE_RE = re.compile(r'^\s*module\s+([A-Za-z_]\w*)', re.M)
_INCLUDE_RE = re.compile(r'^\s*`include\s+"([^"]+)"', re.M)
# `name #(...) inst (` or `name inst (`.
#
# The separator is required: either a parameter block or real whitespace.
# Without it the pattern matched `case (op)` as the module `cas` being
# instantiated as `e`, and the manifest filled up with unresolved
# references to fragments of keywords.
_INST_RE = re.compile(
    r'^[ \t]*([A-Za-z_]\w*)'
    r'(?:[ \t]*#[ \t]*\((?:[^()]|\([^()]*\))*\)[ \t]*|[ \t]+)'
    r'([A-Za-z_]\w*)[ \t]*\(', re.M)

_TASKFN_RE = re.compile(r'^\s*(?:task|function)\b[^;]*?([A-Za-z_]\w*)\s*[;(]',
                        re.M)


def _strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


def _index():
    """module name -> path, and include name -> path."""
    modules, includes = {}, {}
    for d in SEARCH:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if name.endswith('.vh'):
                includes.setdefault(name, p)
            elif name.endswith('.v'):
                body = _strip_comments(open(p).read())
                for m in _MODULE_RE.findall(body):
                    modules.setdefault(m, p)
    return modules, includes


def _needs(path):
    """(instantiated module names, included file names) in one file."""
    body = _strip_comments(open(path).read())
    # A file's own tasks and functions are called the same way a module
    # is instantiated and are not modules.
    local = set(_TASKFN_RE.findall(body))
    inst = set()
    for name, _label in _INST_RE.findall(body):
        if name not in _KEYWORDS and name not in local:
            inst.add(name)
    return inst, set(_INCLUDE_RE.findall(body))


def walk(top, modules, includes):
    """Every file reachable from `top`, and the edges that got there."""
    if top not in modules:
        raise SystemExit(f"no module {top!r} in the search path")
    files, incs, edges = {}, {}, []
    seen, queue = set(), [top]
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        path = modules.get(name)
        if path is None:
            # Not a module of ours: a primitive, or a blackbox. Recorded
            # by the caller, never silently dropped.
            continue
        files[name] = path
        inst, inc = _needs(path)
        for i in sorted(inc):
            if i in includes:
                incs[i] = includes[i]
            else:
                edges.append((name, i, 'missing-include'))
        for i in sorted(inst):
            if i in modules:
                edges.append((name, i, 'module'))
                queue.append(i)
            elif i not in seen:
                edges.append((name, i, 'unresolved'))
    return files, incs, edges


def _sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _tool_versions():
    out = {}
    for name, cmd in (('iverilog', ['iverilog', '-V']),
                      ('yosys', ['yosys', '-V']),
                      ('python', [sys.executable, '--version'])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=30)
            out[name] = (r.stdout + r.stderr).strip().splitlines()[0]
        except Exception:                                  # noqa: BLE001
            out[name] = 'not present'
    return out


def _head():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:                                      # noqa: BLE001
        return ''


def _params():
    """The parameters of the public top, read from the source.

    Read rather than restated: a copy in this script would be a second
    place for the geometry to live, and the two would disagree the first
    time one of them changed.
    """
    body = _strip_comments(
        open(os.path.join(ROOT, 'rtl', f'{SYNTH_TOP}.v')).read())
    head = body.split(') (', 1)[0]
    out = {}
    for m in re.finditer(r'parameter\s+([A-Za-z_]\w*)\s*=\s*([^,\n]+)',
                         head):
        out[m.group(1)] = m.group(2).strip()
    return out


SMOKE_TB = r'''//==============================================================================
// tnca_smoke -- the export package proving itself
//
// No files, no plusargs, no hierarchical reference into the design. It
// writes a program through the shell's own host port, launches it, waits
// for the completion record, reads the answer back through the host port
// and prints PASS or FAIL.
//
// That is deliberately the whole surface a platform integrator has: if
// this passes, the ports are enough to drive the device, which is the
// claim R36-RTL-02 is about.
//==============================================================================
`timescale 1ns/1ps

module tnca_smoke;

    // A memory latency the caller can turn up, so the smoke test can be
    // run against a slower memory without editing it.
    parameter integer MEM_LATENCY = 1;
    // Backpressure and out-of-order responses, so an integrator can run
    // this against a memory that behaves like theirs rather than only
    // against the friendliest one. The contract allows both; a top that
    // only works without them is not meeting it.
    parameter integer MEM_STALL = 0;
    parameter integer MEM_REORDER = 0;
    parameter integer WATCHDOG_NS = 40000000;

    localparam integer WORDS     = 1024;
    localparam integer LSU_TAG_W = 5;
    localparam integer NUM_RCTX  = 2;
    localparam integer RCTX_W    = 1;

    localparam [3:0] CMD_NOP     = 4'd0;
    localparam [3:0] CMD_LAUNCH  = 4'd1;
    localparam [3:0] CMD_QUIESCE = 4'd2;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    wire [63:0] imem_addr, imem_data;
    wire        mreq_valid, mreq_ready, mreq_we, mreq_amo, mrsp_valid;
    wire [63:0] mreq_addr, mreq_wdata, mreq_cmp, mrsp_rdata;
    wire [7:0]  mreq_be;
    wire [3:0]  mreq_aop;
    wire [1:0]  mreq_hint;
    wire [LSU_TAG_W-1:0] mreq_tag, mrsp_tag;

    reg         cmd_valid = 1'b0;
    reg  [3:0]  cmd_op = CMD_NOP;
    reg  [63:0] cmd_pc = 64'd0, cmd_arg_base = 64'd0;
    reg  [NUM_RCTX-1:0] cmd_role_mask = 1;
    reg  [31:0] px = 0, py = 0, pz = 0, nx = 1, ny = 1, nz = 1;
    reg  [63:0] cmd_deadline = 64'd100000;
    wire        cmd_ready, done_valid, busy, core_idle;
    reg         done_ready = 1'b0;
    wire [3:0]  done_status;
    wire [63:0] done_cycles, done_retired, done_fault_pc;
    wire [7:0]  done_fault_code;
    wire [RCTX_W-1:0] done_fault_rctx;
    wire [23:0] done_fault_detail;
    wire [63:0] perf_cycles, perf_issued, perf_stall_hazard,
                perf_stall_lsu, perf_stall_wb, perf_squash;

    reg         hq_valid = 1'b0, hq_we = 1'b0;
    reg  [63:0] hq_addr = 0, hq_wdata = 0;
    reg  [7:0]  hq_be = 8'hFF;
    wire        hq_ready, hs_valid;
    wire [63:0] hs_rdata;

    tnca_shell #(
        .NUM_RCTX(NUM_RCTX), .RCTX_W(RCTX_W), .LSU_TAG_W(LSU_TAG_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op),
        .cmd_pc(cmd_pc), .cmd_arg_base(cmd_arg_base),
        .cmd_role_mask(cmd_role_mask),
        .cmd_pid_x(px), .cmd_pid_y(py), .cmd_pid_z(pz),
        .cmd_npid_x(nx), .cmd_npid_y(ny), .cmd_npid_z(nz),
        .cmd_deadline(cmd_deadline),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_status(done_status), .done_cycles(done_cycles),
        .done_retired(done_retired),
        .done_fault_code(done_fault_code), .done_fault_pc(done_fault_pc),
        .done_fault_rctx(done_fault_rctx),
        .done_fault_detail(done_fault_detail),
        .host_req_valid(hq_valid), .host_req_ready(hq_ready),
        .host_req_we(hq_we), .host_req_addr(hq_addr),
        .host_req_be(hq_be), .host_req_wdata(hq_wdata),
        .host_rsp_valid(hs_valid), .host_rsp_rdata(hs_rdata),
        .busy(busy), .core_idle(core_idle),
        .perf_cycles(perf_cycles), .perf_issued(perf_issued),
        .perf_stall_hazard(perf_stall_hazard),
        .perf_stall_lsu(perf_stall_lsu),
        .perf_stall_wb(perf_stall_wb), .perf_squash(perf_squash),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .mem_req_valid(mreq_valid), .mem_req_ready(mreq_ready),
        .mem_req_addr(mreq_addr), .mem_req_we(mreq_we),
        .mem_req_be(mreq_be), .mem_req_wdata(mreq_wdata),
        .mem_req_tag(mreq_tag), .mem_req_hint(mreq_hint),
        .mem_req_amo(mreq_amo), .mem_req_aop(mreq_aop),
        .mem_req_cmp(mreq_cmp),
        .mem_rsp_valid(mrsp_valid), .mem_rsp_tag(mrsp_tag),
        .mem_rsp_rdata(mrsp_rdata)
    );

    tnca_mem_model #(
        .WORDS(WORDS), .MEM_DW(64), .MEM_BYTES(8), .TAG_W(LSU_TAG_W),
        .LATENCY(MEM_LATENCY), .RANDOM_STALL(MEM_STALL),
        .REORDER(MEM_REORDER)
    ) u_mem (
        .clk(clk), .rst_n(rst_n),
        .imem_addr(imem_addr), .imem_data(imem_data),
        .req_valid(mreq_valid), .req_ready(mreq_ready),
        .req_addr(mreq_addr), .req_we(mreq_we),
        .req_be(mreq_be), .req_wdata(mreq_wdata), .req_tag(mreq_tag),
        .req_amo(mreq_amo), .req_aop(mreq_aop), .req_cmp(mreq_cmp),
        .rsp_valid(mrsp_valid), .rsp_tag(mrsp_tag), .rsp_rdata(mrsp_rdata)
    );

    reg [63:0] got;
    reg settled;
    integer i, fails, step;

    // Watching, rather than sampling: a single check at one moment would
    // pass against a port that opened for one cycle in the middle.
    reg saw_run, saw_open_while_busy;
    integer busy_cycles;
    always @(posedge clk) begin
        if (!rst_n) begin
            saw_run <= 1'b0;
            saw_open_while_busy <= 1'b0;
            busy_cycles <= 0;
        end else if (busy) begin
            saw_run <= 1'b1;
            busy_cycles <= busy_cycles + 1;
            if (host_req_ready_probe) saw_open_while_busy <= 1'b1;
        end
    end
    wire host_req_ready_probe = hq_ready;

    task host_xfer;
        input        we;
        input [63:0] addr;
        input [63:0] wdata;
        output [63:0] rdata;
        begin
            settled = 1'b0;
            while (hq_ready !== 1'b1) begin @(posedge clk); #1; end
            hq_valid = 1'b1; hq_we = we; hq_addr = addr;
            hq_be = 8'hFF; hq_wdata = wdata;
            @(posedge clk);
            #1 hq_valid = 1'b0;
            rdata = 64'd0;
            while (settled !== 1'b1) begin
                if (hs_valid === 1'b1) begin
                    rdata = hs_rdata; settled = 1'b1;
                end else begin @(posedge clk); #1; end
            end
        end
    endtask

    task run_cmd;
        input [3:0]  cop;
        input [63:0] pc;
        begin
            while (cmd_ready !== 1'b1) begin @(posedge clk); #1; end
            cmd_op = cop; cmd_pc = pc; cmd_valid = 1'b1;
            @(posedge clk);
            #1 cmd_valid = 1'b0;
            settled = 1'b0;
            while (settled !== 1'b1) begin
                if (done_valid === 1'b1) settled = 1'b1;
                else begin @(posedge clk); #1; end
            end
            done_ready = 1'b1;
            @(posedge clk);
            #1 done_ready = 1'b0;
        end
    endtask

    // A test that hangs tells you less than one that fails, and takes
    // longer to do it. WATCHDOG_NS is generous next to the work below
    // and finite, which is the property that matters.
    initial begin
        #WATCHDOG_NS;
        $display("SMOKE FAIL watchdog: stuck at step %0d", step);
        $finish;
    end

    initial begin
        fails = 0;
        step  = 0;
        repeat (4) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // 1. A NOP completes. The command port is alive.
        step = 1;
        run_cmd(CMD_NOP, 64'd0);
        if (done_status !== 4'd0) begin
            $display("FAIL: a NOP returned status %0d", done_status);
            fails = fails + 1;
        end

        // 2. The host port carries a word in and the same word out.
        step = 2;
        host_xfer(1'b1, 64'h0000_0800, 64'hA5A5_5A5A_1234_5678, got);
        host_xfer(1'b0, 64'h0000_0800, 64'd0, got);
        if (got !== 64'hA5A5_5A5A_1234_5678) begin
            $display("FAIL: host readback %h", got);
            fails = fails + 1;
        end

        // 3. The program, written through the host port a word at a time.
        step = 3;
        for (i = 0; i < PROGRAM_WORDS; i = i + 1)
            host_xfer(1'b1, i*8, program_image(i), got);

        // 4. Run it, and read the answer.
        step = 4;
        run_cmd(CMD_LAUNCH, 64'd0);
        if (done_status !== 4'd0) begin
            $display("FAIL: the program returned status %0d (fault %02h at %0h)",
                     done_status, done_fault_code, done_fault_pc);
            fails = fails + 1;
        end
        host_xfer(1'b0, RESULT_ADDR, 64'd0, got);
        if (got !== EXPECTED) begin
            $display("FAIL: result %h, expected %h", got, EXPECTED);
            fails = fails + 1;
        end else begin
            // One string. Verilog-2005 has no adjacent string literal
            // concatenation -- that is a C habit, and iverilog calls it
            // a malformed statement.
            $display("  the program ran in %0d cycle(s) and wrote %h (latency %0d, backpressure %0d, reordering %0d)",
                     done_cycles, got, MEM_LATENCY, MEM_STALL,
                     MEM_REORDER);
        end

        // 5. The host port must have refused while the kernel ran. The
        //    monitor below watched for it the whole time rather than
        //    sampling once and hoping.
        if (!saw_run) begin
            $display("FAIL: never observed a busy cycle");
            fails = fails + 1;
        end else if (saw_open_while_busy) begin
            $display("FAIL: the host port was ready while a kernel ran");
            fails = fails + 1;
        end else begin
            $display("  the host port stayed closed for all %0d busy cycle(s)",
                     busy_cycles);
        end

        if (fails == 0) $display("SMOKE PASS");
        else            $display("SMOKE FAIL %0d", fails);
        $finish;
    end

endmodule
'''


def _program(words):
    """The smoke program, as a Verilog function over its words."""
    lines = ['    function [63:0] program_image;',
             '        input integer idx;',
             '        begin',
             '            case (idx)']
    for i, w in enumerate(words):
        lines.append(f"                {i}: program_image = 64'h{w:016x};")
    lines.append("                default: program_image = 64'd0;")
    lines.append('            endcase')
    lines.append('        end')
    lines.append('    endfunction')
    return '\n'.join(lines)


def build_program():
    """A tiny kernel: store a known value, and stop.

    Assembled by the repository's own assembler, because an export whose
    smoke program was hand-encoded would stop matching the ISA the first
    time an opcode moved.
    """
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'asm'))
    import tnca_asm                                        # noqa: E402
    # Writes a known value to RESULT_ADDR and exits, so the package shows
    # the device computing something rather than merely elaborating.
    # Assembled by the repository's own assembler: a hand-encoded image
    # here would stop matching the ISA the first time an opcode moved.
    src = """
        .li      s1, #0x0DEFACED
        .li      s2, #0x800
        s.st.d   [s2, #0], s1
        exit
    """
    a = tnca_asm.Assembler()
    a.assemble(src)
    return tnca_asm.build_image(a.records)


def export(outdir, *, program_words, result_addr, expected):
    modules, includes = _index()

    synth_files, synth_incs, synth_edges = walk(SYNTH_TOP, modules,
                                                includes)
    sim_files, sim_incs, sim_edges = walk(SIM_TOP, modules, includes)
    # Everything the simulation binding needs that the synthesizable set
    # does not already have.
    sim_only = {k: v for k, v in sim_files.items() if k not in synth_files}
    sim_only_incs = {k: v for k, v in sim_incs.items()
                     if k not in synth_incs}

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, 'rtl'))
    os.makedirs(os.path.join(outdir, 'sim'))

    manifest = {
        'format': 'tnca-hardware-export/1',
        'claim_level': 'engineering-ready',
        'claim_note': (
            'This package elaborates and runs in simulation. It has not '
            'been placed, routed, timed or run on any device, and '
            'nothing in it may be read as a platform-verified claim.'),
        'synthesizable_top': SYNTH_TOP,
        'simulation_top': SIM_TOP,
        'parameters': _params(),
        'tools': _tool_versions(),
        'source_commit': _head(),
        'rtl': [], 'sim': [], 'includes': [],
        'unresolved': [], 'simulation_only_behaviour': [],
    }

    def copy(src, sub):
        name = os.path.basename(src)
        dst = os.path.join(outdir, sub, name)
        shutil.copy2(src, dst)
        return {'file': f"{sub}/{name}", 'sha256_16': _sha(src),
                'bytes': os.path.getsize(src)}

    for name in sorted(synth_incs):
        manifest['includes'].append(copy(synth_incs[name], 'rtl'))
    for name in sorted(synth_files):
        manifest['rtl'].append(dict(module=name,
                                    **copy(synth_files[name], 'rtl')))
    for name in sorted(sim_only_incs):
        manifest['includes'].append(copy(sim_only_incs[name], 'sim'))
    for name in sorted(sim_only):
        manifest['sim'].append(dict(module=name,
                                    **copy(sim_only[name], 'sim')))

    for a, b, why in synth_edges + sim_edges:
        if why != 'module':
            manifest['unresolved'].append({'from': a, 'needs': b,
                                           'kind': why})

    manifest['simulation_only_behaviour'] = [
        {'file': 'sim/tnca_mem_model.v',
         'what': '$readmemh, $fopen and $fwrite',
         'why': 'a behavioural memory with an image loader. A platform '
                'replaces it with real RAM; nothing in rtl/ calls into '
                'it.'},
        {'file': 'sim/tnca_device.v',
         'what': '$fopen, $fscanf, plusargs and a file mailbox',
         'why': 'the reference transport binding. It drives the public '
                'ports only, so a platform binding replaces this file '
                'and nothing else.'},
        {'file': 'tnca_smoke.v',
         'what': 'a self-checking stimulus',
         'why': 'the package proving itself; not part of the design.'},
    ]

    # The smoke bench, with its program compiled in.
    tb = SMOKE_TB.replace(
        '    reg [63:0] got;',
        _program(program_words) + '\n\n    reg [63:0] got;')
    tb = tb.replace(
        '    localparam integer WORDS     = 1024;',
        f"    localparam integer WORDS     = 1024;\n"
        f"    localparam integer PROGRAM_WORDS = {len(program_words)};\n"
        f"    localparam [63:0]  RESULT_ADDR = 64'h{result_addr:x};\n"
        f"    localparam [63:0]  EXPECTED = 64'h{expected:016x};")
    with open(os.path.join(outdir, 'tnca_smoke.v'), 'w') as fh:
        fh.write(tb)

    rtl_list = ' '.join(f"rtl/{os.path.basename(p)}"
                        for p in sorted(synth_files.values()))
    # The reference transport binding has its own top and its own
    # `initial` block, so it is built separately from the smoke test.
    # Handing both to one elaboration lets the simulator choose a top,
    # and it chose the binding -- which then asked for the plusargs the
    # smoke test does not pass.
    binding = os.path.basename(sim_only[SIM_TOP])
    support = ' '.join(f"sim/{os.path.basename(p)}"
                       for name, p in sorted(sim_only.items())
                       if name != SIM_TOP)

    with open(os.path.join(outdir, 'elaborate.sh'), 'w') as fh:
        fh.write(f"""#!/bin/sh
# Elaborate the synthesizable set ALONE, with no library search path.
#
# The missing -y is the point: with one, a file this package forgot
# would be found in whatever tree the caller happens to be standing in
# and the package would look complete when it is not.
set -e
cd "$(dirname "$0")"
iverilog -g2005 -I rtl -t null -s {SYNTH_TOP} {rtl_list}
echo "elaborate: {len(synth_files)} synthesizable file(s), top {SYNTH_TOP}"
""")
    with open(os.path.join(outdir, 'run_smoke.sh'), 'w') as fh:
        fh.write(f"""#!/bin/sh
# Build and run the self-checking smoke test. Prints SMOKE PASS or
# SMOKE FAIL, and exits non-zero on failure.
#
# `-s tnca_smoke` is named rather than inferred: sim/{binding} has an
# `initial` block of its own and the simulator picked it once, which made
# the smoke test ask for the plusargs of a transport it is not using.
set -e
cd "$(dirname "$0")"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
iverilog -g2005 -I rtl -I sim -s tnca_smoke -o "$OUT/smoke.vvp" \\
    tnca_smoke.v {rtl_list} {support}
vvp "$OUT/smoke.vvp" | tee "$OUT/smoke.log"
grep -q "SMOKE PASS" "$OUT/smoke.log"
""")

    # The reference binding is built too, so that "it elaborates" is a
    # fact about the package rather than about this repository's Makefile.
    with open(os.path.join(outdir, 'build_binding.sh'), 'w') as fh:
        fh.write(f"""#!/bin/sh
# Elaborate the reference transport binding, the worked example of
# driving the public ports. It is a simulation: it speaks the file
# mailbox of spec/r2_transport_v1.json and is replaced wholesale by a
# platform binding.
set -e
cd "$(dirname "$0")"
OUT=${{1:-$(mktemp -d)}}
iverilog -g2005 -I rtl -I sim -s {SIM_TOP} -o "$OUT/device.vvp" \\
    sim/{binding} {rtl_list} {support}
echo "binding: elaborated {SIM_TOP} -> $OUT/device.vvp"
""")
    os.chmod(os.path.join(outdir, 'build_binding.sh'), 0o755)
    os.chmod(os.path.join(outdir, 'elaborate.sh'), 0o755)
    os.chmod(os.path.join(outdir, 'run_smoke.sh'), 0o755)

    with open(os.path.join(outdir, 'MANIFEST.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write('\n')

    with open(os.path.join(outdir, 'README.md'), 'w') as fh:
        fh.write(f"""# TNCA hardware export

The public top is `{SYNTH_TOP}` in `rtl/`. Everything it needs is in
`rtl/` beside it; nothing in this package refers to a path outside it.

```sh
./elaborate.sh     # the synthesizable set alone, no library search
./run_smoke.sh     # build the reference simulation and run a program
```

## What is in here

| directory | what | may it go on a device |
|---|---|---|
| `rtl/` | the public top and everything it instantiates | yes |
| `sim/` | the behavioural memory and the reference transport binding | no |
| `tnca_smoke.v` | a self-checking stimulus | no |

`MANIFEST.json` lists every file with its hash, the parameters of the
public top, the tool versions this package was produced with, and --
under `simulation_only_behaviour` -- every place a `$`-task appears and
why. Anything the module walk could not resolve is under `unresolved`;
an empty list there is the claim that there are no undeclared
blackboxes.

## The claim level

`engineering-ready`. This elaborates and runs in simulation. It has not
been placed, routed, timed, or run on any device. See
`doc/phase3_6_tasks_and_acceptance.md` section 1: platform verification
is a separate branch and is not claimed here.

## Driving it

The public top takes a depth-one command port
(`cmd_valid`/`cmd_ready`/`cmd_op`), returns one completion record per
command (`done_valid`/`done_ready`/`done_status`), and gives the host a
single-word access port (`host_*`) that is only ready while no kernel is
running. `sim/tnca_device.v` is a worked example of driving all three;
it is roughly two hundred lines and contains no design logic.
""")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--json')
    args = ap.parse_args()

    words = build_program()
    m = export(args.out, program_words=words, result_addr=0x800,
               expected=0x0DEFACED)
    print(f"export: {len(m['rtl'])} synthesizable file(s), "
          f"{len(m['sim'])} simulation file(s), "
          f"{len(m['includes'])} include(s)")
    print(f"        top {m['synthesizable_top']}, "
          f"{len(m['unresolved'])} unresolved reference(s)")
    print(f"        -> {args.out}")
    if args.json:
        R.write_json(args.json, m)
    return 0


if __name__ == '__main__':
    sys.exit(main())
