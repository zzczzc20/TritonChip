#!/usr/bin/env python3
"""R42 lineelig: unit tests of tnca_tensor LINE_HALF (half-line rows), RTL simulation (Icarus Verilog).

    python3 tools/r42/lineelig_tests.py miter   [--seeds N]
    python3 tools/r42/lineelig_tests.py lock    --ref tnca_tensor_ref.v [--seeds N]
    python3 tools/r42/lineelig_tests.py mutants [--seeds N]
    python3 tools/r42/lineelig_tests.py directed
    python3 tools/r42/lineelig_tests.py core

miter   the line engine with LINE_HALF 1 against the word engine (tb/core/tb_tensor_line.v, HBIAS 1: boxes of
        32-byte pieces added to the random program): memories, register files and completion counts identical.
        Also the same program with LINE_HALF 0 (the delivered line engine) as a control.
lock    the engine as it was (--ref: the base commit's rtl/core/tnca_tensor.v with the module renamed
        tnca_tensor_ref) beside the current engine with LINE_HALF 0, both line engines, same environment: every
        output compared on every cycle (the default parameter keeps the engine cycle-identical).
directed a fixed list of boxes (A/B/C tiles of 32-byte rows, strides of 96 and 224... bytes, 16-byte-aligned bases, partial rows,
        negative coordinates, rank 1, rows of 64 bytes at a 32-byte offset, an inactive command, stores, a transposed load), each with
        the number of 32-byte pieces the half-line path must move for it; the engines run in step and memory and registers are compared
        after every command.
core    tensor programs of the core's own regression on tb_core_hybrid (the core with the line-wide port, the run-2 image's tag
        width), without the line engine, with it, and with it and LINE_HALF: every program must pass in all three; the cycles are printed.
mutants copies of tnca_tensor.v with one change each in the half-line logic; the miter must fail on every one.
"""
import argparse, re, subprocess, sys, tempfile, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CFG = ((20, 20, 40), (0, 0, 8), (40, 40, 120))
TB = 'tb/core/tb_tensor_line.v'


def build(tmp, name, params, rtl, extra_src=()):
    out = str(Path(tmp) / name)
    cmd = ['iverilog', '-g2012', '-Irtl/common', '-s', 'tb_tensor_line'] + \
          [f'-Ptb_tensor_line.{k}={v}' for k, v in params.items()] + ['-o', out, TB, str(rtl)] + [str(x) for x in extra_src]
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=300)
    return out


def run(vvp):
    r = subprocess.run(['vvp', vvp], cwd=ROOT, capture_output=True, text=True, timeout=3000)
    return r.stdout


def one(tmp, params, rtl, extra_src=()):
    b = build(tmp, 'tl.vvp', params, rtl, extra_src)
    out = run(b)
    m = re.search(r'TENSOR LINE MITER PASS.*', out)
    ok = m is not None and 'FATAL' not in out and 'FAIL' not in out
    return ok, (m.group(0) if m else out.strip().splitlines()[-1] if out.strip() else 'no output')


def miter(a):
    bad = n = 0
    with tempfile.TemporaryDirectory(prefix='lineelig-') as t:
        for lhalf, lsw in ((1, 5), (1, 6), (0, 5)):       # LSW 6: the run-2 image's 64 line slots
            for seed in range(1, a.seeds + 1):
                for vw, rq, lat in CFG:
                    p = dict(SEED=seed, VW_STALL=vw, REQ_STALL=rq, MAXLAT=lat, LHALF=lhalf, HBIAS=1, LSW=lsw)
                    ok, msg = one(t, p, ROOT / 'rtl/core/tnca_tensor.v')
                    n += 1; bad += not ok
                    print(f"{'PASS' if ok else 'FAIL'} LINE_HALF={lhalf} LSW={lsw} seed={seed} vw={vw} rq={rq} lat={lat} {msg}", flush=True)
    print(f'MITER {n - bad} of {n} configurations pass', flush=True)
    return 1 if bad else 0


def directed(a):
    bad = n = 0
    with tempfile.TemporaryDirectory(prefix='lineelig-') as t:
        for lhalf in (1, 0):
            for lsw in (5, 6):
                for vw, rq, lat in CFG:
                    p = dict(SEED=1, VW_STALL=vw, REQ_STALL=rq, MAXLAT=lat, LHALF=lhalf, DIRECTED=1, LSW=lsw)
                    ok, msg = one(t, p, ROOT / 'rtl/core/tnca_tensor.v')
                    n += 1; bad += not ok
                    print(f"{'PASS' if ok else 'FAIL'} DIRECTED LINE_HALF={lhalf} LSW={lsw} vw={vw} rq={rq} lat={lat} {msg}", flush=True)
    print(f'DIRECTED {n - bad} of {n} configurations pass', flush=True)
    return 1 if bad else 0


CORE_PROGS = (('t12_tensor', ''), ('t30_xpose', ''), ('t33_xpose8', ''), ('t33_tdesc_load', ''), ('t34_tdesc_store', ''),
              ('t17_descfault', '+expectfault=04'), ('t35_tdesc_fpad2', '+expectfault=04'), ('t39_freeze_drop', ''))


def core(a):
    sys.path.insert(0, str(ROOT / 'tools/asm'))
    from tnca_asm import Assembler, build_image
    bad = n = 0
    with tempfile.TemporaryDirectory(prefix='lineelig-core-') as t:
        vvps = {}
        for line, half in ((0, 0), (1, 0), (1, 1)):
            vvps[(line, half)] = str(Path(t) / f'core_{line}{half}.vvp')
            subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-y', 'rtl/common', '-y', 'rtl/core', '-y', 'rtl/gen', '-y', 'tb/common',
                            '-s', 'tb_core_hybrid', '-Ptb_core_hybrid.MEM_LATENCY=6', '-Ptb_core_hybrid.RANDOM_STALL=1',
                            '-Ptb_core_hybrid.MEM_REORDER=1', '-Ptb_core_hybrid.LSU_TAG_W=8', '-Ptb_core_hybrid.HAS_TENSOR_XPOSE=2',
                            f'-Ptb_core_hybrid.TENSOR_LINE={line}', f'-Ptb_core_hybrid.TENSOR_LINE_HALF={half}',
                            '-o', vvps[(line, half)], 'tb/core/tb_core_hybrid.v'], cwd=ROOT, check=True, timeout=300)
        for prog, args in CORE_PROGS:
            hexf = Path(t) / f'{prog}.hex'
            src = (ROOT / f'tb/core/progs/{prog}.s').read_text()
            hexf.write_text(''.join(f'{w:016x}\n' for w in build_image(Assembler().assemble(src), size_words=65536)))
            row = []
            for key, vvp in vvps.items():
                r = subprocess.run(['vvp', vvp, f'+hex={hexf}', '+maxcycles=1000000'] + args.split(), cwd=ROOT,
                                   capture_output=True, text=True, timeout=1800).stdout
                ok = re.search(r'^PASS', r, re.M) is not None and 'FAIL' not in r and 'FATAL' not in r
                m = re.search(r'finish called at (\d+)', r)
                row.append((key, ok, int(m.group(1)) // 10000 if m else -1))
                n += 1; bad += not ok
            print(f"{'PASS' if all(x[1] for x in row) else 'FAIL'} {prog:18s} " +
                  '  '.join(f"line={k[0]} half={k[1]}: {c} cycles" for k, ok, c in row), flush=True)
    print(f'CORE {n - bad} of {n} runs pass', flush=True)
    return 1 if bad else 0


def lock(a):
    bad = n = 0
    with tempfile.TemporaryDirectory(prefix='lineelig-') as t:
        for hb in (0, 1):
            for seed in range(1, a.seeds + 1):
                for vw, rq, lat in CFG:
                    p = dict(SEED=seed, VW_STALL=vw, REQ_STALL=rq, MAXLAT=lat, LHALF=0, HBIAS=hb, LOCK=1)
                    ok, msg = one(t, p, ROOT / 'rtl/core/tnca_tensor.v', [a.ref])
                    n += 1; bad += not ok
                    print(f"{'PASS' if ok else 'FAIL'} LOCK HBIAS={hb} seed={seed} vw={vw} rq={rq} lat={lat} {msg}", flush=True)
    print(f'LOCK {n - bad} of {n} configurations cycle-identical', flush=True)
    return 1 if bad else 0


# (name, old text, new text): each must change the half-line logic and be caught
MUTANTS = [
    ('load: source half inverted', "vw_data    <= ls_hs[lrdy_slot] ?", "vw_data    <= !ls_hs[lrdy_slot] ?"),
    ('load: destination half inverted', "vw_lane_en <= ls_hd[lrdy_slot] ?", "vw_lane_en <= !ls_hd[lrdy_slot] ?"),
    ('load: lanes all enabled', "vw_lane_en <= ls_hd[lrdy_slot] ? {{(VLEN/2){1'b1}}, {(VLEN/2){1'b0}}}", "vw_lane_en <= ls_hd[lrdy_slot] ? {VLEN{1'b1}}"),
    ('load: source half recorded from the window offset', "ls_hs[lfree] <= l_addr[5];", "ls_hs[lfree] <= t_flat[5];"),
    ('load: destination half recorded from the address', "ls_hd[lfree] <= t_flat[5];", "ls_hd[lfree] <= l_addr[5];"),
    ('load: piece flag dropped', "ls_h[lfree]  <= hm;", "ls_h[lfree]  <= 1'b0;"),
    ('store: byte enables of the wrong half', "<< (t_ps[5] ? PORT_BYTES/2 : 0);", "<< (t_ps[5] ? 0 : PORT_BYTES/2);"),
    ('store: byte enables all ones', "assign mem_req_be    = (l_req && hm && (st == S_LSRUN)) ? hl_be", "assign mem_req_be    = (l_req && hm && (st == S_LSRUN)) ? {PORT_BYTES{1'b1}}"),
    ('store: register half inverted', "wire [VLEN*ELEN/2-1:0] hl_half = t_flat[5] ?", "wire [VLEN*ELEN/2-1:0] hl_half = !t_flat[5] ?"),
    ('store: data not replicated', "assign mem_req_wdata = (l_req && hm) ? {2{hl_half}}", "assign mem_req_wdata = (l_req && hm) ? {hl_half, {(VLEN*ELEN/2){1'b0}}}"),
    ('walk: piece step 16 bytes', "t_ps <= t_ps + 64'd32;", "t_ps <= t_ps + 64'd16;"),
    ('walk: window offset step 64 bytes', "t_flat <= t_flat + 32'd32;", "t_flat <= t_flat + 32'd64;"),
    ('walk: row ends one piece early', "if ({24'd0, t_pn} + 32'd1 == t_row_pcs) begin\n                        t_pn <= 8'd0;", "if ({24'd0, t_pn} + 32'd2 == t_row_pcs) begin\n                        t_pn <= 8'd0;"),
    ('eligibility: row alignment not checked', "((t_acc[4:0] + t_CB[4:0]) == 5'd0) &&\n                   (((t_s0 | t_s1 | t_s2 | t_s3) & 64'h1f)", "1'b1 &&\n                   (((t_s0 | t_s1 | t_s2 | t_s3) & 64'h1f)"),
    ('eligibility: strides not checked', "(((t_s0 | t_s1 | t_s2 | t_s3) & 64'h1f) == 64'd0) &&\n                   (t_row_bytes[4:0]", "1'b1 &&\n                   (t_row_bytes[4:0]"),
    ('eligibility: containment not checked', "(t_row_bytes[4:0] == 5'd0) && g_inside;\n", "(t_row_bytes[4:0] == 5'd0);\n"),
    ('eligibility: row length not checked', "(t_row_bytes[4:0] == 5'd0) && g_inside;\n", "g_inside;\n"),
    ('store: next register read after the first piece', "(!hm || t_flat[5]);", "1'b1;"),
    ('store: register kept too long', "if (st == S_LSRUN && l_more && !hm_first) begin", "if (st == S_LSRUN && l_more && !hm_first && !hm) begin"),
]


def mutants(a):
    src = (ROOT / 'rtl/core/tnca_tensor.v').read_text()
    miss = 0
    with tempfile.TemporaryDirectory(prefix='lineelig-mut-') as t:
        for name, old, new in MUTANTS:
            if src.count(old) != 1:
                print(f'ERROR mutant "{name}": pattern found {src.count(old)} times', flush=True); miss += 1; continue
            f = Path(t) / 'mut.v'; f.write_text(src.replace(old, new))
            caught = False
            for seed in range(1, a.seeds + 1):
                for vw, rq, lat in CFG[:2]:
                    p = dict(SEED=seed, VW_STALL=vw, REQ_STALL=rq, MAXLAT=lat, LHALF=1, HBIAS=1)
                    try:
                        ok, msg = one(t, p, f)
                    except subprocess.CalledProcessError:
                        ok, msg = False, 'does not elaborate'
                    if not ok:
                        caught = True; break
                if caught: break
            print(f"{'CAUGHT' if caught else 'MISSED'} {name}", flush=True)
            miss += not caught
    print(f'MUTANTS {len(MUTANTS) - miss} of {len(MUTANTS)} caught', flush=True)
    return 1 if miss else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('miter', 'lock', 'mutants', 'directed', 'core'))
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--ref', default=None)
    a = ap.parse_args()
    if a.mode == 'lock' and not a.ref:
        ap.error('lock needs --ref')
    sys.exit({'miter': miter, 'lock': lock, 'mutants': mutants, 'directed': directed, 'core': core}[a.mode](a))
