"""Sparse 8 GiB-window RTL stand-in for the three-core U280 board binding.

`fpga-multicore-sparse-sim` is `fpga-multicore-sim` with the board top
(rtl/board/tnca_multicore_hbm_xo.v), maintenance (DMA ownership handoff),
runtime memory controls and three outlets, and two differences:

  * the stand-in BO can start at any 64-byte aligned address (`hbm_base`;
    0 is what XRT gave the HBM[0:31] group on the board), not only at
    0x1_0000_0000;
  * the AXI memory behind each outlet backs only the window ranges listed in
    a sparse map (tb/common/tnca_sparse_map.vh). An 8 GiB window is then
    simulated with a few MiB of model memory, and an access to an unbacked
    line -- what a dropped or extra high address bit produces -- stops the
    simulator with the address instead of reading zero.

The RTL is elaborated from `rtl_root` (default: this checkout), so the exact
sources of an image (e.g. git archive 54249f7 for the B40 image) can be
simulated with this checkout's runtime, as the board runs them. Image
geometry parameters (SETS/WAYS/registered TMU/victim options) are passed
through `bench_parameters` exactly as tools/r40/image_geometry_tests.py does.

This is a test binding: it lives under tools/, registers itself only when a
harness asks, and changes nothing in python/tnca.
"""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'python',):
    if str(p) not in __import__('sys').path:
        __import__('sys').path.insert(0, str(p))

from tnca.backends.multicore_mmio import MulticoreMMIOBackend          # noqa: E402
from tnca.backends.multicore_mmio_sim import MulticoreMMIOLink         # noqa: E402
from tnca.backends.u50_sim import _tools                               # noqa: E402
from tnca.errors import ArgumentError, DependencyFailed                # noqa: E402

NAME = 'fpga-multicore-sparse-sim'
PC_BYTES = 256 << 20
LINE = 64

# B40 image parameters (evidence/r40/v2/convergence/attempt03/image.tcl) that
# are bench parameters of tb/fpga/tnca_multicore_board_regs.v.
IMAGE_BENCH = dict(SETS=32, SET_W=5, WAYS=8, WAY_W=3, DCO_REGISTERED_RANGES=1,
                   NUCA_REGISTER_VICTIM_SELECT=1, NUCA_CACHE_DATA_PIPELINE=1,
                   NUCA_REGISTER_VICTIM_RANK=1, DCO_TLW_PIPE_STAGES=2,
                   DCO_REGISTER_WRITES=1, DCO_LOCAL_REPLICAS=1)
# ... and the ones the link constructor carries (all two-stage on B40).
IMAGE_LINK = dict(link_stages=2, noc_link_stages=2, dispatch_stages=2, home_stages=2,
                  code_stages=2, maintenance=1, memory_controls=True)


def merge_ranges(ranges, window_bytes):
    """Sorted, line-aligned, disjoint (offset, bytes) ranges inside the window."""
    spans = []
    for off, size in ranges:
        lo = max(0, int(off) // LINE * LINE)
        hi = min(int(window_bytes), -(-(int(off) + int(size)) // LINE) * LINE)
        if hi > lo:
            spans.append((lo, hi))
    spans.sort()
    out = []
    for lo, hi in spans:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return [(lo, hi - lo) for lo, hi in out]


def write_map(path, ranges):
    path = Path(path)
    path.write_text(''.join(f'{off:x} {size:x}\n' for off, size in ranges))
    return sum(size for _, size in ranges) // LINE


def synthetic_allocation(base, size):
    """The board's physical PC map for a BO at `base`, for a stand-in.

    On the board this comes from the xclbin MEM_TOPOLOGY (HBM[n] at n*256 MiB,
    evidence/r41/hbm8/xclbin_hbm_map.json). The stand-in has no xclbin, so it
    carries the same layout relative to its own BO and says so.
    """
    banks = []
    cursor = base
    while cursor < base + size:
        pc = (cursor - base) // PC_BYTES
        lo, hi = cursor, min(base + size, base + (pc + 1) * PC_BYTES)
        banks.append(dict(pc=pc, memory_index=pc, tag=f'HBM[{pc}]', base=base + pc * PC_BYTES,
                          size=PC_BYTES, used=True, allocated_base=lo, allocated_size=hi - lo))
        cursor = hi
    return dict(physical_banks=banks, topology_sha256='synthetic-stand-in-no-xclbin',
                topology_source='synthetic: HBM[n] = n*256 MiB from the stand-in BO base')


class SparseBoardLink(MulticoreMMIOLink):
    def __init__(self, *, sparse_map, window_bytes, hbm_base=0, rtl_root=None,
                 bench_parameters=None, **kwargs):
        if type(window_bytes) is not int or window_bytes <= 0 or window_bytes % LINE:
            raise ArgumentError('window_bytes must be a positive whole number of lines')
        if type(hbm_base) is not int or hbm_base < 0 or hbm_base % LINE or hbm_base + window_bytes > 1 << 64:
            raise ArgumentError('hbm_base must be a line-aligned 64-bit BO address')
        self.sparse_map = str(Path(sparse_map).resolve())
        ranges = [tuple(int(v, 16) for v in line.split()) for line in
                  Path(self.sparse_map).read_text().split('\n') if line.strip()]
        if merge_ranges(ranges, window_bytes) != ranges:
            raise ArgumentError('sparse map must be sorted, disjoint, line aligned and inside the window')
        self.sparse_lines = sum(size for _, size in ranges) // LINE
        self.hbm_base = hbm_base
        self.rtl_root = Path(rtl_root).resolve() if rtl_root else ROOT
        self.bench_parameters = dict(bench_parameters or {})
        super().__init__(memory_bytes=window_bytes, board_top=True, **kwargs)
        # The stand-in carries the board's physical PC layout relative to its BO.
        self.hbm_allocation = None

    def describe(self):
        return (f'sparse RTL MMIO v2 stand-in: {self.core_count} cores, {self.memory_ports} outlets, '
                f'{self.memory_bytes} B window at BO {self.hbm_base:#x}, {self.sparse_lines} backed lines, '
                f'RTL {self.rtl_root}')

    def hbm_open(self, nbytes):
        size = min(int(nbytes), self.memory_bytes)
        if size <= 0 or size % LINE:
            raise ArgumentError('simulated AXI allocation must contain complete lines')
        self.hbm_allocation = dict(requested_bytes=int(nbytes), exposed_bytes=size,
                                   connected_memory=dict(index=None, used=1, size=self.memory_bytes,
                                                         base=self.hbm_base, tag='stand-in BO'),
                                   **synthetic_allocation(self.hbm_base, size))
        return self.hbm_base, size

    def _build(self):
        iverilog, self._vvp = _tools()
        top = 'tnca_multicore_board_regs'
        output = str(Path(self.dir) / 'sparse-board.vvp')
        rtl, tb = self.rtl_root / 'rtl', ROOT
        command = [iverilog, '-g2012', '-I', str(rtl / 'common'), '-I', str(rtl / 'noc'),
                   '-I', str(tb), '-s', top, '-o', output]
        params = dict(N=self.core_count, PORTS=self.memory_ports, MEMORY_BYTES=self.memory_bytes,
                      STAGES=self.link_stages, NOC_PIPE=self.noc_link_stages,
                      DISPATCH_PIPE=self.dispatch_stages, HOME_PIPE=self.home_stages,
                      CODE_PIPE=self.code_stages, MAINTENANCE=self.maintenance,
                      MEMORY_CONTROLS=int(self.memory_controls),
                      WIN_BASE=self.hbm_base, SPARSE=1, SPARSE_LINES=self.sparse_lines)
        params.update(self.bench_parameters)
        for name, value in params.items():
            command += [f'-P{top}.{name}={value}']
        command += [f'-P{top}.SPARSE_MAP="{self.sparse_map}"']
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            command += ['-y', str(rtl / library)]
        command += ['-y', str(rtl), '-y', str(rtl / 'fpga'), '-y', str(rtl / 'board')]
        command += [str(p) for p in sorted((rtl / 'common').glob('*.v'))]
        command += [str(tb / 'tb/common/tnca_axi_line_model.v'), str(tb / f'tb/fpga/{top}.v')]
        # Elaboration takes minutes; reuse an identical build (command and every
        # RTL/bench source byte) across case processes.
        import hashlib
        h = hashlib.sha256(json.dumps(command[1:]).replace(output, 'OUT').encode())
        for path in sorted(list(rtl.rglob('*.v')) + list(rtl.rglob('*.vh')) +
                           list((tb / 'tb/common').glob('*.v*')) + list((tb / 'tb/fpga').glob('*.v*'))):
            h.update(str(path).encode() + hashlib.sha256(path.read_bytes()).digest())
        cache = Path(os.environ.get('TNCA_HBM8_SIM_CACHE', ROOT / 'sim/build/hbm8-sparse'))
        cache.mkdir(parents=True, exist_ok=True)
        cached = cache / f'{h.hexdigest()[:32]}.vvp'
        if cached.exists():
            return str(cached)
        result = subprocess.run(command, cwd=tb, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise DependencyFailed('sparse board harness did not elaborate',
                                   detail={'diagnostic': result.stderr[-3000:]})
        tmp = cached.with_suffix(f'.{os.getpid()}.tmp')
        Path(output).replace(tmp)
        os.replace(tmp, cached)
        return str(cached)


def stand_in(**kwargs):
    kwargs.pop('bench', None)
    image = bool(kwargs.pop('image_geometry', False))
    link_kwargs = dict(IMAGE_LINK) if image else dict(maintenance=1, memory_controls=True)
    for key in ('link_stages', 'noc_link_stages', 'dispatch_stages', 'home_stages', 'code_stages'):
        if key in kwargs:
            link_kwargs[key] = kwargs.pop(key)
    link = SparseBoardLink(sparse_map=kwargs.pop('sparse_map'),
                           window_bytes=kwargs.pop('window_bytes'),
                           hbm_base=kwargs.pop('hbm_base', 0),
                           rtl_root=kwargs.pop('rtl_root', None),
                           bench_parameters=IMAGE_BENCH if image else None,
                           core_count=3, memory_ports=3,
                           host_timeout=kwargs.get('host_timeout', 30.0), **link_kwargs)
    kwargs.pop('core_count', None)
    kwargs.pop('memory_ports', None)
    return MulticoreMMIOBackend(link, core_count=3, memory_ports=3, **kwargs)


def register():
    """Make NAME openable by the runtime registry (test binding, this process only)."""
    from tnca import backend
    if NAME not in backend.available():
        backend.register(NAME, stand_in, state=backend.REFERENCE_AVAILABLE,
                         needs_elaborator=False, physical_grid=True,
                         what='sparse-window RTL stand-in for fpga-u280-multicore (tools/performance_harness/hbm8_sim.py)')
    return NAME


def backend_args(*, sparse_map, window_bytes, hbm_base=0, rtl_root=None, image_geometry=True):
    return dict(sparse_map=str(sparse_map), window_bytes=int(window_bytes), hbm_base=int(hbm_base),
                rtl_root=str(rtl_root) if rtl_root else None, image_geometry=bool(image_geometry),
                hbm_bytes=int(window_bytes), hbm_dma=True, device_clock=True,
                image_profile='core-vrf-sync-v5', core_count=3, memory_ports=3)
