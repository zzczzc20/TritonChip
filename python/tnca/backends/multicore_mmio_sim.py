"""Explicit RTL stand-in for MMIO v2; shares the board host binding."""
from pathlib import Path
import argparse
import operator
import os
import subprocess
import sys
import time

from .. import devices
from ..errors import ArgumentError, DependencyFailed
from .multicore_mmio import MulticoreMMIOBackend
from .u50_sim import SimLink, _repo_root, _tools


# A mailbox is two directories, one per direction: the host writes its
# commands and its goodbye into the inbox, which the device only reads; the
# device writes its responses and its exit status into the outbox, which the
# host only reads. The device writes nothing where the host can write -- a
# link the host left there would otherwise aim the device's own writes (the
# simulator opens its response file by name) at the observer's log.
EXIT_FILE = 'device.exit'   # in the outbox, when the simulator stops
DONE_FILE = 'host.done'     # in the inbox: the host's goodbye


class _ExternalDevice:
    """The process handle of a device this host did not start: its exit
    status is the file the device leaves when its simulator stops."""

    pid = None

    def __init__(self, path):
        self.path = path

    @property
    def returncode(self):
        try:
            with open(self.path) as stream:
                text = stream.read().strip()
        except OSError:
            return None
        try:
            return int(text) if text else None
        except ValueError:
            return None

    def poll(self):
        return self.returncode


class MulticoreMMIOLink(SimLink):
    def __init__(self, *, core_count=3, memory_ports=3,
                 memory_bytes=1 << 20, link_stages=2, noc_link_stages=0, dispatch_stages=0, home_stages=0, code_stages=0, board_top=False, maintenance=False, memory_controls=False, journal=False, hybrid_lsu=False, sfu_lanes=1, home_background_reads=False, cache_sets=None, rtl_params=None, stage_probes=False, device=None, inbox=None, outbox=None, **kwargs):
        if type(journal) is not bool or (journal and not board_top):
            raise ArgumentError('journal requires a boolean and the board-top stand-in')
        if type(hybrid_lsu) is not bool or type(home_background_reads) is not bool or sfu_lanes not in (1, 2, 3, 4, 8):
            raise ArgumentError('invalid hybrid LSU/SFU geometry')
        # None keeps the register bench's own default geometry (SETS=4), as
        # before the uncommitted 32-set default was checkpointed (AUDIT.md F14).
        if cache_sets is not None and cache_sets not in (4, 16, 32, 64):
            raise ArgumentError('cache_sets must be None, 4, 16, 32 or 64')
        self.cache_sets = cache_sets
        # Extra top-level bench parameters (for example an image's cache
        # geometry, tools/r40/image_geometry_tests.py); names are checked by
        # the elaborator, values must be integers. None keeps every default.
        rtl_params = dict(rtl_params or {})
        if any(not (isinstance(k, str) and k.isidentifier() and k.isupper()) or type(v) is not int
               for k, v in rtl_params.items()):
            raise ArgumentError('rtl_params must map upper-case parameter names to integers')
        # Observation-only stage counters (tb/fpga/tnca_stage_probes.vh).
        if type(stage_probes) is not bool:
            raise ArgumentError('stage_probes must be a boolean')
        self.plusargs = ['+stageprobe'] if stage_probes else []
        self.home_background_reads = home_background_reads
        self.hybrid_lsu = hybrid_lsu
        self.sfu_lanes = operator.index(sfu_lanes)
        explicit = kwargs.pop('_explicit', ())
        # A device description (spec/devices) is the machine: the wrapper is
        # elaborated with every parameter of its image, and the stand-in's own
        # geometry follows it.
        self.device = None
        if device is not None:
            if not board_top:
                raise ArgumentError('a device description is simulated in the board-top stand-in')
            try:
                self.device = device if isinstance(device, dict) else devices.load(device)
                devices.validate(self.device)
            except devices.DeviceError as e:
                raise ArgumentError(f'device description: {e}') from None
            u, plat = self.device['uncore'], self.device['platform']
            given = dict(core_count=core_count, memory_ports=memory_ports, link_stages=link_stages,
                         noc_link_stages=noc_link_stages, dispatch_stages=dispatch_stages,
                         home_stages=home_stages, code_stages=code_stages, maintenance=int(maintenance),
                         memory_controls=int(memory_controls))
            core_count, memory_ports = plat['core_count'], u['MEM_PORTS']
            link_stages, noc_link_stages = u['LINK_STAGES'], u['NOC_LINK_STAGES']
            dispatch_stages, home_stages, code_stages = u['DISPATCH_STAGES'], u['MEM_HOME_STAGES'], u['CODE_LINK_STAGES']
            maintenance, memory_controls = plat['CACHE_MAINTENANCE'], bool(plat['MEMORY_CONTROLS'])
            described = dict(core_count=core_count, memory_ports=memory_ports, link_stages=link_stages,
                             noc_link_stages=noc_link_stages, dispatch_stages=dispatch_stages,
                             home_stages=home_stages, code_stages=code_stages, maintenance=int(maintenance),
                             memory_controls=int(memory_controls))
            clash = {k: (v, described[k]) for k, v in given.items() if k in explicit and v != described[k]}
            if clash:
                raise ArgumentError('the device description decides the geometry; given against described: '
                                    + ', '.join(f'{k}={a} (described {b})' for k, (a, b) in sorted(clash.items())))
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
        # A mailbox is a device run elsewhere -- by the agent harness's
        # evaluator, in a container of its own (`serve` below) -- that this
        # link reaches only through its inbox and outbox. It neither builds
        # nor starts the simulator, and the device's process is not its child.
        self.mailbox = None
        if (inbox is None) != (outbox is None):
            raise ArgumentError('a mailbox is an inbox and an outbox')
        if inbox is not None:
            if journal:
                raise ArgumentError('a mailbox device keeps no journal')
            self.mailbox = (str(inbox), str(outbox))
            kwargs['workdir'] = self.mailbox[0]
        super().__init__(rtl_params=rtl_params, **kwargs)
        if self.mailbox is not None:
            self.cmd = os.path.join(self.mailbox[0], 'regs.cmd')
            self.rsp = os.path.join(self.mailbox[1], 'regs.rsp')
        self.journal_dir = str(Path(self.dir)/'journal') if journal else None

    def describe(self):
        base = f'RTL MMIO v2: {self.core_count} cores, {self.memory_ports} AXI outlets'
        return base + (f", device {self.device['name']}" if self.device else '')

    def open(self):
        if self.mailbox is None:
            return super().open()
        if self.proc is not None:
            from ..errors import TncaError
            raise TncaError('this link already has a device', code='INTERNAL')
        self.proc = _ExternalDevice(os.path.join(self.mailbox[1], EXIT_FILE))
        self._await_up()
        return self

    def close(self):
        if self.mailbox is None:
            return super().close()
        if self._pending and self.alive():
            try:
                self._round_trip([])
            except Exception:  # noqa: BLE001 -- closing; the device's own log has the cause
                pass
        with open(os.path.join(self.mailbox[0], DONE_FILE), 'w') as stream:
            stream.write('done\n')
        self.proc = None

    def image_header(self):
        """`define TNCA_IMAGE_PARAMS: every parameter of the description's
        image, in the wrapper's parameter-override syntax."""
        params = devices.image_params(self.device)
        return ('// GENERATED from device description ' + self.device['name']
                + ' by tnca.backends.multicore_mmio_sim. Do not edit.\n'
                + '`define TNCA_IMAGE_PARAMS ' + ','.join(f'.{k}({v})' for k, v in sorted(params.items())) + '\n')

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
        geometry = {} if self.cache_sets is None else dict(SETS=self.cache_sets, SET_W=self.cache_sets.bit_length()-1)
        bench = dict(HYBRID_LSU=int(self.hybrid_lsu), SFU_LANES=self.sfu_lanes, HOME_BACKGROUND_READS=int(self.home_background_reads), **geometry, N=self.core_count, PORTS=self.memory_ports,
                     MEMORY_BYTES=self.memory_bytes, STAGES=self.link_stages,
                     NOC_PIPE=self.noc_link_stages, DISPATCH_PIPE=self.dispatch_stages, HOME_PIPE=self.home_stages, CODE_PIPE=self.code_stages, MAINTENANCE=self.maintenance, MEMORY_CONTROLS=int(self.memory_controls))
        header = None
        if self.device is not None:
            bench['AXI_ID_W'] = devices.derived(self.device['uncore'])['AXI_ID_W']
            bench['AXI_OUT_N'] = self.device['uncore']['AXI_OUT_N']
            bench['AXI_DATA_W'] = self.device['uncore'].get('AXI_DATA_W', 256)
            header = Path(self.dir) / 'tnca_image_params.vh'
            header.write_text(self.image_header(), encoding='utf-8')
        for name, value in bench.items():
            command += [f'-P{top}.{name}={value}']
        for name, value in sorted(self.rtl_params.items()):
            command += [f'-P{top}.{name}={value}']
        for library in ('common', 'core', 'gen', 'nuca', 'noc', 'tile'):
            command += ['-y', 'rtl/' + library]
        command += ['-y', 'rtl', '-y', 'rtl/fpga', '-y', 'rtl/board']
        command += [str(p.relative_to(root)) for p in sorted((root / 'rtl/common').glob('*.v'))]
        command += ['tb/common/tnca_axi_line_model.v']
        if header is not None:
            command += [str(header)]
        command += [f'tb/fpga/{top}.v']
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise DependencyFailed('multicore MMIO harness did not elaborate',
                                   detail={'diagnostic': result.stderr[-2000:]})
        return output


def stand_in(**kwargs):
    kwargs.pop('bench', None)
    device = kwargs.pop('device', None)
    # Geometry arguments the caller actually gave: with a description they
    # must agree with it, and the ones not given are the description's.
    explicit = {k for k in ('core_count', 'memory_ports', 'link_stages', 'noc_link_stages', 'dispatch_stages',
                            'home_stages', 'code_stages', 'maintenance', 'memory_controls') if k in kwargs}
    if device is not None:
        desc = device if isinstance(device, dict) else devices.load(device)
        kwargs.setdefault('image_profile', desc['per_core_profile'])
        kwargs.setdefault('core_count', desc['platform']['core_count'])
        kwargs.setdefault('memory_ports', desc['uncore']['MEM_PORTS'])
        device = desc
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
                            cache_sets=kwargs.pop('cache_sets', None),
                            rtl_params=kwargs.pop('rtl_params', None),
                            stage_probes=kwargs.pop('stage_probes', False),
                            trace=kwargs.pop('trace', False),
                            device=device, _explicit=explicit, inbox=kwargs.pop('inbox', None),
                            outbox=kwargs.pop('outbox', None),
                            host_timeout=kwargs.get('host_timeout', 30.0))
    return MulticoreMMIOBackend(link, core_count=core_count,
                               memory_ports=memory_ports, **kwargs)


def serve(argv=None):
    """The device side of a mailbox: build the board-top stand-in for a
    device description, run it against the mailbox's files with the
    platform's observer log, and leave its exit status in the outbox.

        python -m tnca.backends.multicore_mmio_sim --device D --inbox IN --outbox OUT
               [--build DIR] [--observe FILE] [--memory-bytes N] [--watchdog CYCLES]

    --build is where the simulator is elaborated; the agent harness keeps it,
    and the observer log, where the host cannot write. The device only reads
    the inbox and only writes the outbox (and --build, and the observer).

    A mailbox serves one device, once. The host may write its first command
    before the device is up, so nothing found in a fresh inbox is removed; a
    mailbox a device has answered from (or left) is refused instead. The
    simulator is built at once but started only when the host's first command
    arrives: an idle simulator still runs its clock, and a device waiting for
    its host's turn (the evaluator runs a few points at a time) would take a
    CPU from the ones that are running.
    """
    ap = argparse.ArgumentParser(prog='tnca.backends.multicore_mmio_sim', description=serve.__doc__)
    ap.add_argument('--device', required=True, help='a description name in spec/devices, or its path')
    ap.add_argument('--inbox', required=True, help="the host's commands and goodbye, read here")
    ap.add_argument('--outbox', required=True, help='the responses and the exit status, written here')
    ap.add_argument('--build')
    ap.add_argument('--observe')
    ap.add_argument('--memory-bytes', type=int, default=1 << 20)
    ap.add_argument('--watchdog', type=int, default=200_000_000)
    a = ap.parse_args(argv)
    os.makedirs(a.outbox, exist_ok=True)
    used = [os.path.join(d, n) for d, n in ((a.outbox, 'regs.rsp'), (a.outbox, EXIT_FILE), (a.inbox, DONE_FILE))
            if os.path.lexists(os.path.join(d, n))]
    if used:
        print(f'the mailbox has been used ({", ".join(used)}); a device takes a fresh one',
              file=sys.stderr, flush=True)
        return 2
    exit_file = os.path.join(a.outbox, EXIT_FILE)
    code = 2
    try:
        try:
            build = a.build or os.path.join(a.outbox, 'build')
            os.makedirs(build, exist_ok=True)
            link = MulticoreMMIOLink(board_top=True, device=a.device, memory_bytes=a.memory_bytes, workdir=build)
            bench = link._build()
        except Exception as e:  # noqa: BLE001 -- a device that never came up; the host is told by the exit file
            print(f'the device did not build: {e}', file=sys.stderr, flush=True)
            return code
        command = os.path.join(a.inbox, 'regs.cmd')
        argv = [link._vvp, bench, f'+cmd={command}', f'+rsp={os.path.join(a.outbox, "regs.rsp")}',
                f'+pollcycles={link.pollcycles}', f'+intercycle={link.intercycle}', f'+watchdog={a.watchdog}',
                '+assert_fatal']
        if a.observe:
            argv.append(f'+observe={a.observe}')
        done = os.path.join(a.inbox, DONE_FILE)
        while not os.path.exists(command):
            if os.path.exists(done):
                # The host left without a command: nothing to serve.
                code = 0
                return code
            time.sleep(0.2)
        print('serving', link.describe(), flush=True)
        # The simulator runs until its watchdog unless the host says goodbye
        # (the link's close() leaves DONE_FILE), which ends it at once.
        proc = subprocess.Popen(argv)
        while proc.poll() is None:
            if os.path.exists(done):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(0.2)
        code = proc.returncode
        return code
    finally:
        # Whatever happened, the host is told: without the exit file it would
        # wait out its whole timeout for a device that is gone.
        with open(exit_file, 'w') as stream:
            stream.write(f'{code}\n')


if __name__ == '__main__':
    sys.exit(serve())
