"""Explicit RTL stand-in for MMIO v2; shares the board host binding."""
from pathlib import Path
import operator
import subprocess

from ..errors import ArgumentError, DependencyFailed
from .multicore_mmio import MulticoreMMIOBackend
from .u50_sim import SimLink, _repo_root, _tools


class MulticoreMMIOLink(SimLink):
    def __init__(self, *, core_count=3, memory_ports=3,
                 memory_bytes=1 << 20, link_stages=2, **kwargs):
        self.core_count = operator.index(core_count)
        self.memory_ports = operator.index(memory_ports)
        self.memory_bytes = operator.index(memory_bytes)
        self.link_stages = operator.index(link_stages)
        if not 1 <= self.memory_ports <= self.core_count <= 16:
            raise ArgumentError('invalid multicore MMIO geometry')
        if self.core_count > 4 and self.core_count % 4:
            raise ArgumentError('the rectangular stand-in needs a multiple of four above four cores')
        if self.memory_bytes < 4096 or self.memory_bytes % 64 or self.link_stages < 0:
            raise ArgumentError('memory must contain whole lines; link stages must be nonnegative')
        super().__init__(**kwargs)

    def describe(self):
        return f'RTL MMIO v2: {self.core_count} cores, {self.memory_ports} AXI outlets'

    def hbm_open(self, nbytes):
        size = min(operator.index(nbytes), self.memory_bytes)
        if size <= 0 or size % 64:
            raise ArgumentError('simulated AXI allocation must contain complete lines')
        return 0x100000000, size

    def _build(self):
        root = Path(_repo_root())
        iverilog, self._vvp = _tools()
        output = str(Path(self.dir) / 'multicore-mmio.vvp')
        command = [iverilog, '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc',
                   '-s', 'tnca_multicore_regs', '-o', output]
        for name, value in dict(N=self.core_count, PORTS=self.memory_ports,
                                MEMORY_BYTES=self.memory_bytes, STAGES=self.link_stages).items():
            command += [f'-Ptnca_multicore_regs.{name}={value}']
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            command += ['-y', 'rtl/' + library]
        command += ['-y', 'rtl', '-y', 'rtl/fpga']
        command += [str(p.relative_to(root)) for p in sorted((root / 'rtl/common').glob('*.v'))]
        command += ['tb/common/tnca_axi_line_model.v', 'tb/fpga/tnca_multicore_regs.v']
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise DependencyFailed('multicore MMIO harness did not elaborate',
                                   detail={'diagnostic': result.stderr[-2000:]})
        return output


def stand_in(**kwargs):
    kwargs.pop('bench', None)
    core_count = kwargs.pop('core_count', 3)
    memory_ports = kwargs.pop('memory_ports', min(3, core_count))
    link = MulticoreMMIOLink(core_count=core_count, memory_ports=memory_ports,
                            memory_bytes=kwargs.pop('memory_bytes', 1 << 20),
                            link_stages=kwargs.pop('link_stages', 2),
                            host_timeout=kwargs.get('host_timeout', 30.0))
    return MulticoreMMIOBackend(link, core_count=core_count,
                               memory_ports=memory_ports, **kwargs)
