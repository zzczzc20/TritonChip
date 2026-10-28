#!/usr/bin/env python3
"""The board-top RTL at the exact parameters of an implementation image.

The other board-top benches elaborate tnca_multicore_hbm_xo with a small
cache (4 sets x 2 ways) and the convergence options off. This one takes every
parameter from the image's image.tcl (TNCA_IMAGE_PARAMS: e.g. 32 sets x 8
ways, registered victim select/rank, cache data pipeline, local TMU replicas,
registered TMU writes, two-stage TLW pipe, two-stage links) and runs the
actual MMIO v2 register protocol through it:

  * code upload/readback through the coherent host window;
  * two physical grids (455 programs each) with every result slot checked;
  * runtime memory controls 1/1, 1/8, 3/1, 3/8 with a grid under each;
  * coherent host word writes, dirty-line download through the cache
    ownership handoff, clean invalidation before upload, and a partial-line
    handoff that keeps a dirty neighbour;
  * a deadline abort that drains and a final quiesce.

Every image parameter must either be a bench parameter or equal the RTL
default of tnca_multicore_hbm_xo; otherwise the run refuses to start.
Functional RTL only: no timing, resource or board claim.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.backends.multicore_mmio import MulticoreMMIOBackend, OP_CACHE_HANDOFF  # noqa: E402
from tnca.backends.multicore_mmio_sim import MulticoreMMIOLink                   # noqa: E402
from tnca.backends.u50_sim import _repo_root, _tools                             # noqa: E402
from tnca.errors import DependencyFailed                                         # noqa: E402

# image.tcl parameter -> board-bench parameter
BENCH = dict(SETS='SETS', SET_W='SET_W', WAYS='WAYS', WAY_W='WAY_W',
             DCO_REGISTERED_RANGES='DCO_REGISTERED_RANGES',
             NUCA_REGISTER_VICTIM_SELECT='NUCA_REGISTER_VICTIM_SELECT',
             NUCA_CACHE_DATA_PIPELINE='NUCA_CACHE_DATA_PIPELINE',
             NUCA_REGISTER_VICTIM_RANK='NUCA_REGISTER_VICTIM_RANK',
             DCO_TLW_PIPE_STAGES='DCO_TLW_PIPE_STAGES', DCO_REGISTER_WRITES='DCO_REGISTER_WRITES',
             DCO_LOCAL_REPLICAS='DCO_LOCAL_REPLICAS', NUCA_SHARED_DATA_READ='NUCA_SHARED_DATA_READ',
             AXI_LINE_LUTRAM='AXI_LINE_LUTRAM',
             # the image's tensor-transpose build (tnca_multicore_board_regs.HAS_TENSOR_XPOSE)
             HAS_TENSOR_XPOSE='HAS_TENSOR_XPOSE',
             # R41 G2 resource sizing
             **{k: k for k in ('NUM_RCTX', 'RCTX_W', 'PHYS_VREGS', 'PHYS_W', 'ACC_N', 'ACC_W',
                               'BAR_N', 'BAR_W', 'BAR_EPOCH_W', 'PHYS_ADDR_W')},
             # R41 G4 one-lane path (tnca_multicore_board_regs forwards them)
             **{k: k for k in ('HYBRID_LSU', 'HOME_BACKGROUND_READS', 'HOME_LINE_WRITE_NOFILL', 'HOME_LINE_PIPE',
                               'HOME_MSHR_N', 'LINE_SLOTS', 'AXI_OUT_N', 'AXI_PIPELINED_ISSUE')})
# Parameters the bench derives from others; the image must hold the derived value.
DERIVED = dict(AXI_ID_W=lambda image: max(1, (image.get('AXI_OUT_N', 8) - 1).bit_length()))
# Parameters the link's own constructor sets.
LINK = dict(MEM_PORTS='memory_ports', LINK_STAGES='link_stages', NOC_LINK_STAGES='noc_link_stages',
            DISPATCH_STAGES='dispatch_stages', MEM_HOME_STAGES='home_stages',
            CODE_LINK_STAGES='code_stages', CACHE_MAINTENANCE='maintenance',
            MEMORY_CONTROLS='memory_controls')


def image_parameters(path):
    text = Path(path).read_text()
    m = re.search(r'set TNCA_IMAGE_PARAMS \[dict create ([^\]]*)\]', text)
    if not m:
        raise SystemExit(f'{path}: no TNCA_IMAGE_PARAMS')
    words = m.group(1).split()
    return {words[i]: int(words[i + 1]) for i in range(0, len(words), 2)}


def rtl_defaults():
    src = (ROOT / 'rtl/board/tnca_multicore_hbm_xo.v').read_text()
    start = src.index('module tnca_multicore_hbm_xo')
    head = src[start:src.index('\n) (', start)]
    out = {k: int(v) for k, v in re.findall(r'\b([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*[,)\n]', head)}
    # AXI_ID_W defaults to clog2(AXI_OUT_N).
    out['AXI_ID_W'] = max(1, (out['AXI_OUT_N'] - 1).bit_length())
    return out


class ImageLink(MulticoreMMIOLink):
    def __init__(self, *, bench_parameters, **kwargs):
        self.bench_parameters = dict(bench_parameters)
        super().__init__(**kwargs)

    def _build(self):
        root = Path(_repo_root())
        iverilog, self._vvp = _tools()
        output = str(Path(self.dir) / 'image-geometry.vvp')
        top = 'tnca_multicore_board_regs'
        command = [iverilog, '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc', '-s', top, '-o', output]
        params = dict(N=self.core_count, PORTS=self.memory_ports, MEMORY_BYTES=self.memory_bytes,
                      STAGES=self.link_stages, NOC_PIPE=self.noc_link_stages,
                      DISPATCH_PIPE=self.dispatch_stages, HOME_PIPE=self.home_stages,
                      CODE_PIPE=self.code_stages, MAINTENANCE=self.maintenance,
                      MEMORY_CONTROLS=int(self.memory_controls))
        params.update(self.bench_parameters)
        for name, value in params.items():
            command += [f'-P{top}.{name}={value}']
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            command += ['-y', 'rtl/' + library]
        command += ['-y', 'rtl', '-y', 'rtl/fpga', '-y', 'rtl/board']
        command += [str(p.relative_to(root)) for p in sorted((root / 'rtl/common').glob('*.v'))]
        command += ['tb/common/tnca_axi_line_model.v', f'tb/fpga/{top}.v']
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise DependencyFailed('image-geometry harness did not elaborate',
                                   detail={'diagnostic': result.stderr[-2000:]})
        return output


def grid(dev, label):
    # Clear every result slot first, so each grid has to write all of them.
    assert dev.upload(0x30000, bytes(455 * 8)).ok
    reply = dev.launch_grid(0, 0x40000, (13, 7, 5), arg_stride=256)
    assert reply.ok, (label, reply.status)
    metrics = dev.physical_grid_metrics()
    assert sum(row['issued'] for row in metrics['cores']) == 455, (label, metrics)
    assert all(row['issued'] == row['completed'] for row in metrics['cores']), (label, metrics)
    got, data = dev.download(0x30000, 455 * 8)
    assert got.ok
    values = [int.from_bytes(data[k:k + 8], 'little') for k in range(0, len(data), 8)]
    wrong = [(k, hex(v)) for k, v in enumerate(values) if v != 4096 + k]
    assert not wrong, (label, f'{len(wrong)} wrong result slots', wrong[:16])
    return reply


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--image-tcl', default=str(ROOT / 'evidence/r40/v2/convergence/attempt03/image.tcl'))
    ap.add_argument('--image-profile', help='the profile the image implements (default: the backend\'s); '
                    'core-vrf-sync-v5-x8-g2 for an R41 G2 image with two contexts')
    ap.add_argument('--rtl-param', action='append', default=[], metavar='NAME=VALUE',
                    help='override a board-bench RTL parameter (repeatable)')
    args = ap.parse_args()
    image = image_parameters(args.image_tcl)
    defaults = rtl_defaults()
    bench, link = {}, {}
    for name, value in sorted(image.items()):
        if name in BENCH:
            bench[BENCH[name]] = value
        elif name in LINK:
            link[LINK[name]] = value
        elif name in DERIVED:
            if DERIVED[name](image) != value:
                raise SystemExit(f'{name}={value} differs from the value the bench derives, {DERIVED[name](image)}')
        elif name in defaults:
            if defaults[name] != value:
                raise SystemExit(f'{name}={value} differs from the RTL default {defaults[name]} '
                                 'and has no bench parameter')
        else:
            raise SystemExit(f'image parameter {name} is unknown to the bench and the RTL top')
    for item in args.rtl_param:
        name, separator, value = item.partition('=')
        if not separator or not re.fullmatch(r'[A-Z][A-Z0-9_]*', name) or not re.fullmatch(r'\d+', value):
            ap.error(f'--rtl-param expects NAME=VALUE with a non-negative integer, got {item!r}')
        bench[name] = int(value)
    stages = {link[k] for k in ('link_stages', 'noc_link_stages', 'dispatch_stages', 'home_stages', 'code_stages')}
    print(f'IMAGE GEOMETRY parameters={image}', flush=True)
    lk = ImageLink(bench_parameters=bench, core_count=3, memory_ports=link['memory_ports'],
                   link_stages=link['link_stages'], noc_link_stages=link['noc_link_stages'],
                   dispatch_stages=link['dispatch_stages'], home_stages=link['home_stages'],
                   code_stages=link['code_stages'], board_top=True,
                   maintenance=link['maintenance'], memory_controls=bool(link['memory_controls']),
                   host_timeout=3600)
    dev = MulticoreMMIOBackend(lk, core_count=3, memory_ports=link['memory_ports'],
                               hbm_dma=True, host_timeout=3600,
                               **({'image_profile': args.image_profile} if args.image_profile else {}))
    program = ROOT / 'tb/core/progs/r40_grid_mmio_memory.s'
    hexfile = Path(lk.dir) / 'grid.hex'
    subprocess.run(['python3', 'tools/asm/tnca_asm.py', str(program), '-o', str(hexfile), '--size', '256'],
                   cwd=ROOT, check=True, timeout=60)
    code = b''.join(int(line, 16).to_bytes(8, 'little') for line in hexfile.read_text().splitlines())
    try:
        dev.start()
        caps = dev.capabilities()
        assert caps['hbm_dma'] and caps['cache_ownership_handoff'], caps
        assert dev.upload(0, code).ok
        reply, data = dev.download(0, len(code))
        assert reply.ok and data == code
        print('IMAGE GEOMETRY code upload/readback PASS', flush=True)
        first = grid(dev, 'grid-1')
        second = grid(dev, 'grid-2')
        assert second.device_timing['start'] > first.device_timing['end']
        counters = dev.memory_port_metrics()
        assert all(row['read_lines'] > 0 and row['write_lines'] > 0 for row in counters), counters
        print(f'IMAGE GEOMETRY two grids PASS programs=455x2 counters={counters}', flush=True)
        maximum = dev.memory_configuration()['maximum_outstanding']
        for ports, limit in ((1, 1), (1, maximum), (link['memory_ports'], 1), (link['memory_ports'], maximum)):
            got = dev.configure_memory(active_ports=ports, outstanding=limit)
            assert (got['active_ports'], got['outstanding']) == (ports, limit)
            grid(dev, f'controls-{ports}/{limit}')
            print(f'IMAGE GEOMETRY memory controls {ports}/{limit} PASS', flush=True)
        base = 0x30000
        expected = bytearray((i * 17 + 3) & 255 for i in range(256))
        assert dev.upload(base, expected).ok
        for offset in range(0, 256, 8):
            assert dev._window(base + offset, None, write=False) == int.from_bytes(expected[offset:offset + 8], 'little')
        for offset in range(8, 256, 64):
            value = 0x1122334455667788 + offset
            dev._window(base + offset, value, write=True)
            expected[offset:offset + 8] = value.to_bytes(8, 'little')
        start = len(dev.history)
        reply, got = dev.download(base, 256)
        assert reply.ok and got == expected, 'dirty cache bytes never reached the backing memory'
        assert any(row['op'] == OP_CACHE_HANDOFF for row in dev.history[start:])
        for offset in range(0, 256, 8):
            dev._window(base + offset, None, write=False)
        replacement = bytes((255 - i) & 255 for i in range(256))
        assert dev.upload(base, replacement).ok
        for offset in range(0, 256, 8):
            assert dev._window(base + offset, None, write=False) == int.from_bytes(replacement[offset:offset + 8], 'little')
        expected = bytearray(replacement)
        dev._window(base + 16, 0xfedcba9876543210, write=True)
        expected[16:24] = (0xfedcba9876543210).to_bytes(8, 'little')
        assert dev.request(OP_CACHE_HANDOFF, base + 9, 1).ok
        dev.link.hbm_write(base + 9 - dev.local_limit, b'\xaa')
        expected[9] = 0xaa
        assert dev.download(base, 256)[1] == expected, 'partial handoff lost a dirty neighbour'
        print('IMAGE GEOMETRY coherent host window and ownership handoff PASS', flush=True)
        old = dev.deadline_cycles
        dev.deadline_cycles = 20
        timeout_reply = dev.launch_grid(0, 0x40000, (13, 7, 5), arg_stride=256)
        dev.deadline_cycles = old
        assert timeout_reply.status == 2
        drained = dev.physical_grid_metrics()
        assert all(row['issued'] == row['completed'] for row in drained['cores'])
        assert dev.quiesce().ok
        print(f'IMAGE GEOMETRY PASS sets={image.get("SETS")} ways={image.get("WAYS")} stages={sorted(stages)} '
              f'options={bench}', flush=True)
    finally:
        dev.stop()


if __name__ == '__main__':
    main()
