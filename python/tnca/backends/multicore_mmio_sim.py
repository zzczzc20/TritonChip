"""Explicit RTL stand-in for MMIO v2; shares the board host binding."""
from pathlib import Path
import operator
import subprocess

from ..errors import ArgumentError, DependencyFailed
from .multicore_mmio import MulticoreMMIOBackend
from .u50_sim import SimLink, _repo_root, _tools


class MulticoreMMIOLink(SimLink):
    def __init__(self, *, core_count=3, memory_ports=3,
                 memory_bytes=1 << 20, link_stages=2, noc_link_stages=0, dispatch_stages=0, home_stages=0, code_stages=0, board_top=False, maintenance=False, memory_controls=False, journal=False, hybrid_lsu=False, sfu_lanes=1, home_background_reads=False, cache_sets=32, **kwargs):
        if type(journal) is not bool or (journal and not board_top):
            raise ArgumentError('journal requires a boolean and the board-top stand-in')
        if type(hybrid_lsu) is not bool or type(home_background_reads) is not bool or sfu_lanes not in (1, 2, 3, 4, 8):
            raise ArgumentError('invalid hybrid LSU/SFU geometry')
        if cache_sets not in (16, 32, 64):
            raise ArgumentError('cache_sets must be 16, 32 or 64')
        self.cache_sets = cache_sets
        self.home_background_reads = home_background_reads
        self.hybrid_lsu = hybrid_lsu
        self.sfu_lanes = operator.index(sfu_lanes)
        self.assert_fatal = True
        self.board_top = bool(board_top)
        self.memory_controls = bool(memory_controls)
        self.maintenance = operator.index(maintenance)
        if self.maintenance not in (0,1):
            raise ArgumentError('maintenance must be 0 or 1')
        self.core_count = operator.index(core_count)
        self.memory_ports = operator.index(memory_ports)
        self.memory_bytes = operator.index(memory_bytes)
        self.link_stages = operator.index(link_stages)
        self.noc_link_stages = operator.index(noc_link_stages)
        self.dispatch_stages = operator.index(dispatch_stages)
        self.home_stages = operator.index(home_stages)
        self.code_stages = operator.index(code_stages)
        if self.board_top and self.core_count != 3:
            raise ArgumentError('the U280 platform binding has three physical cores')
        if not (1 <= self.memory_ports <= min(3, self.core_count) and self.core_count <= 16):
            raise ArgumentError('invalid multicore MMIO geometry')
        if self.core_count > 4 and self.core_count % 4:
            raise ArgumentError('the rectangular stand-in needs a multiple of four above four cores')
        if self.memory_bytes < 4096 or self.memory_bytes % 64 or min(self.link_stages, self.noc_link_stages, self.dispatch_stages, self.home_stages, self.code_stages) < 0:
            raise ArgumentError('memory must contain whole lines; link stages must be nonnegative')
        super().__init__(**kwargs)
        self.journal_dir = str(Path(self.dir)/'journal') if journal else None

    def describe(self):
        return f'RTL MMIO v2: {self.core_count} cores, {self.memory_ports} AXI outlets'

    def hbm_open(self, nbytes):
        size = min(operator.index(nbytes), self.memory_bytes)
        if size <= 0 or size % 64:
            raise ArgumentError('simulated AXI allocation must contain complete lines')
        return 0x100000000, size

    def _dma_commands(self, offset, count, data=None):
        offset,count=operator.index(offset),operator.index(count)
        if not self.maintenance or offset<0 or count<0 or offset>self.memory_bytes-count:
            raise ArgumentError('DMA model range/capability is invalid')
        out=bytearray()
        width=8 if offset%8==0 and count%8==0 else 1
        chunk=4096*width
        for start in range(0,count,chunk):
            indices=range(start,min(count,start+chunk),width)
            read_op,write_op=('D','B') if width==8 else ('d','b')
            commands=[f'{read_op} {offset+i}' if data is None else
                      f'{write_op} {offset+i} {int.from_bytes(data[i:i+width],"little")}' for i in indices]
            values=self._round_trip(self._pending+commands)
            self._pending=[]
            if data is None:
                if len(values)!=len(commands) or any(not 0<=v<1<<(width*8) for v in values):
                    raise DependencyFailed('DMA model returned an invalid byte stream')
                for value in values:out.extend(value.to_bytes(width,'little'))
            elif values:
                raise DependencyFailed('DMA model returned values for a write')
        return bytes(out)

    def hbm_write(self, offset, data):
        data=bytes(data)
        self._dma_commands(offset,len(data),data)

    def hbm_read(self, offset, nbytes):
        return self._dma_commands(offset,nbytes)

    def _build(self):
        root = Path(_repo_root())
        iverilog, self._vvp = _tools()
        output = str(Path(self.dir) / 'multicore-mmio.vvp')
        top = 'tnca_multicore_board_regs' if self.board_top else 'tnca_multicore_regs'
        command = [iverilog, '-g2012', '-I', 'rtl/common', '-I', 'rtl/noc',
                   '-s', top, '-o', output]
        for name, value in dict(N=self.core_count, PORTS=self.memory_ports,
                                MEMORY_BYTES=self.memory_bytes, STAGES=self.link_stages,
                                HYBRID_LSU=int(self.hybrid_lsu), SFU_LANES=self.sfu_lanes,
                                HOME_BACKGROUND_READS=int(self.home_background_reads),
                                SETS=self.cache_sets, SET_W=self.cache_sets.bit_length()-1,
                                NOC_PIPE=self.noc_link_stages, DISPATCH_PIPE=self.dispatch_stages, HOME_PIPE=self.home_stages, CODE_PIPE=self.code_stages, MAINTENANCE=self.maintenance, MEMORY_CONTROLS=int(self.memory_controls)).items():
            command += [f'-P{top}.{name}={value}']
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            command += ['-y', 'rtl/' + library]
        command += ['-y', 'rtl', '-y', 'rtl/fpga', '-y', 'rtl/board']
        command += [str(p.relative_to(root)) for p in sorted((root / 'rtl/common').glob('*.v'))]
        command += ['tb/common/tnca_axi_line_model.v', f'tb/fpga/{top}.v']
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
                            noc_link_stages=kwargs.pop('noc_link_stages', 0),
                            dispatch_stages=kwargs.pop('dispatch_stages', 0),
                            home_stages=kwargs.pop('home_stages', 0),
                            code_stages=kwargs.pop('code_stages', 0),
                            board_top=kwargs.pop('board_top', False),
                            maintenance=kwargs.pop('maintenance', False),
                            memory_controls=kwargs.pop('memory_controls', False),
                            journal=kwargs.pop('journal', False),
                            hybrid_lsu=kwargs.pop('hybrid_lsu', False),
                            home_background_reads=kwargs.pop('home_background_reads', False),
                            sfu_lanes=kwargs.pop('sfu_lanes', 1),
                            cache_sets=kwargs.pop('cache_sets', 32),
                            trace=kwargs.pop('trace', False),
                            host_timeout=kwargs.get('host_timeout', 30.0))
    return MulticoreMMIOBackend(link, core_count=core_count,
                               memory_ports=memory_ports, **kwargs)
