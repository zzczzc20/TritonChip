"""Opt-in v2 register binding; no legacy image or simulator substitution."""
import operator
import os

from ..errors import ArgumentError, DeviceLost, DependencyFailed
from ..hbm_layout import (StripedHBMLayout, STRIPED_CAP, STRIPED_REGISTER,
                          STRIPED_SIGNATURE, STRIPED_BYTES)
from .u50 import U50Backend, XDMALink, IP_NAME, DEFAULT_WORDS, _xclbin_bank

OP_CACHE_HANDOFF = 18
CACHE_HANDOFF_CAP = 16
MEMORY_CONTROLS_CAP = 32


class MulticoreMMIOBackend(U50Backend):
    identity_magic = 0x0002_0000_544E_4341
    opcode_mask = 0xFF
    name = 'fpga-u280-multicore'

    def __init__(self, link, *, core_count=3, memory_ports=3,
                 hbm_dma=False, active_memory_ports=None, outstanding=None, **kwargs):
        self.core_count = operator.index(core_count)
        self.memory_ports = operator.index(memory_ports)
        if not (1 <= self.memory_ports <= min(3, self.core_count) and self.core_count <= 16):
            raise ArgumentError('MMIO v2 needs 1..3 outlets and 1..16 cores, with ports <= cores')
        self._requested_ports = (active_memory_ports if active_memory_ports is not None
                                 else os.environ.get('TNCA_RTL_MEM_PORTS'))
        self._requested_outstanding = (outstanding if outstanding is not None
                                      else os.environ.get('TNCA_RTL_OUTSTANDING'))
        self._memory_config = None
        kwargs['image_profile'] = kwargs.get('image_profile') or 'core-vrf-sync-v5'
        kwargs.setdefault('device_clock', True)
        super().__init__(link, hbm_dma=hbm_dma, **kwargs)

    def _open_window(self, stat):
        striped = bool(stat & STRIPED_CAP)
        if striped and self.link.read_reg(STRIPED_REGISTER) != STRIPED_SIGNATURE:
            raise DeviceLost('striped HBM image has an invalid layout signature')
        self._ownership_cap = bool(stat & CACHE_HANDOFF_CAP)
        if self.hbm_dma and not self._ownership_cap:
            raise ArgumentError('multicore DMA requires the image ownership-handoff capability')
        result = super()._open_window(stat)
        if striped:
            try:
                StripedHBMLayout(self.axi_window['base'], self.axi_window['size'])
            except ValueError as error:
                raise DependencyFailed(str(error)) from error
            if self.hbm_dma and not isinstance(getattr(self.link, 'hbm_layout', None), StripedHBMLayout):
                raise DependencyFailed('striped HBM DMA requires a layout-aware link')
            self.axi_window['layout'] = StripedHBMLayout.name
        return result

    def _handoff(self, addr, nbytes):
        if not nbytes:
            return
        reply = self.request(OP_CACHE_HANDOFF, addr, nbytes)
        stat = self.axi_status()
        if not reply.ok or (stat and (stat['error'] or stat['stray'])):
            raise DeviceLost('DMA ownership handoff failed; no host transfer was performed',
                             detail={'status': reply.status, 'axi': stat})

    def _put_payload(self, addr, words):
        self._check_window(addr, len(words), 'upload')
        register_words = self._dma_split(addr, len(words))
        if register_words < len(words):
            self._handoff(addr + register_words * 8, (len(words)-register_words)*8)
        return super()._put_payload(addr, words)

    def _put_payload_bytes(self, addr, data):
        nwords = len(data) // 8
        self._check_window(addr, nwords, 'upload')
        register_words = self._dma_split(addr, nwords)
        if register_words < nwords:
            self._handoff(addr + register_words * 8, (nwords - register_words) * 8)
        return super()._put_payload_bytes(addr, data)

    def _get_payload(self, addr, nwords):
        self._check_window(addr, nwords, 'download')
        register_words = self._dma_split(addr, nwords)
        if register_words < nwords:
            self._handoff(addr + register_words * 8, (nwords-register_words)*8)
        return super()._get_payload(addr, nwords)

    def start(self):
        self._memory_config = None
        try:
            super().start()
            if self.hbm_dma and not getattr(self, '_ownership_cap', False):
                raise ArgumentError('multicore DMA requires an ownership-capable HBM image')
            capability = self.physical_grid_capability()
            if capability is None or capability[0] != self.core_count:
                raise DeviceLost('v2 register image has a different physical core count')
            if self.physical_chunk_capability() != 4:
                raise DeviceLost('v2 register image has no four-lane physical chunk contract')
            if self.memory_port_capability() != self.memory_ports:
                raise DeviceLost('v2 register image has a different memory outlet contract')
            config = self.memory_configuration()
            if config is None:
                if self._requested_ports is not None or self._requested_outstanding is not None:
                    raise ArgumentError('this image has no runtime memory controls')
            else:
                self.configure_memory(
                    active_ports=config['physical_ports'] if self._requested_ports is None else self._requested_ports,
                    outstanding=config['maximum_outstanding'] if self._requested_outstanding is None else self._requested_outstanding)
            return self
        except BaseException:
            self.stop()
            raise

    def capabilities(self):
        out = super().capabilities()
        out.update(physical_grid=True, tmu_locality=True, tmu_ranges=8, memory_ports=self.memory_ports,
                   memory_bytes=self.memory_bytes, memory_configuration=(dict(self._memory_config) if self._memory_config is not None else None),
                   completion_counters='global-makespan-u64', fault_detail=False,
                   reaches='rtl/fpga/tnca_multicore_fpga.v',
                   protocol='r2_transport_v1 + r40 physical-grid extension, MMIO v2',
                   coherent_host_window=True, hbm_dma=self.hbm_dma and getattr(self, '_ownership_cap', False),
                   cache_ownership_handoff=getattr(self, '_ownership_cap', False))
        return out

    def memory_configuration(self):
        """Read actual image capability and quiescent-control state; legacy=None."""
        with self._io_lock:
            # Probe the legacy status flag first: old eight-bit register
            # maps may alias extended addresses onto command arguments.
            if not self.link.read_reg(0xb8) & MEMORY_CONTROLS_CAP:
                return None
            cap = self.link.read_reg(0x118)
            if cap >> 32 != 0x5234304D:
                raise DeviceLost('advertised memory-control capability has an invalid signature')
            ports, maximum = (cap >> 8) & 0xff, cap & 0xff
            if cap & 0xffff0000 != 0x00010000 or ports != self.memory_ports or not maximum:
                raise DeviceLost('invalid runtime memory-control capability')
            value = self.link.read_reg(0x110)
            active, limit = (value >> 8) & 0xff, value & 0xff
            if value >> 16 or active not in (1, ports) or not 1 <= limit <= maximum:
                raise DeviceLost('invalid runtime memory-control readback')
            return dict(version=1, physical_ports=ports, maximum_outstanding=maximum,
                        active_ports=active, outstanding=limit)

    def configure_memory(self, *, active_ports, outstanding):
        """Change routing/admission only after hardware confirms complete drain."""
        with self._io_lock:
            config = self.memory_configuration()
            if config is None:
                raise ArgumentError('this image has no runtime memory controls')
            def integer(value):
                try:
                    if isinstance(value, str):
                        return int(value, 10)
                    if isinstance(value, bool):
                        raise TypeError('boolean is not a memory-control count')
                    return operator.index(value)
                except (TypeError, ValueError) as exc:
                    raise ArgumentError('memory-control counts must be integers') from exc
            ports, limit = integer(active_ports), integer(outstanding)
            if ports not in (1, config['physical_ports']) or not 1 <= limit <= config['maximum_outstanding']:
                raise ArgumentError('unsupported runtime memory-control counts')
            if self.link.read_reg(0x48):
                raise ArgumentError('memory controls require an idle, drained device')
            self.link.write_reg(0x110, (ports << 8) | limit)
            actual = self.memory_configuration()
            if actual is None or (actual['active_ports'], actual['outstanding']) != (ports, limit):
                raise DeviceLost('hardware refused the memory-control change')
            self._memory_config = actual
            return dict(actual)

    def axi_counters(self):
        """Legacy C0..D8 registers do not exist in the v2 binding.

        Use memory_port_metrics() for actual per-outlet line counts. Reading
        the legacy addresses would return the register mux's default zeros.
        Read latency and AXI beat counts are unavailable through this ABI.
        """
        return None

    def launch_grid(self, *args, **kwargs):
        with self._io_lock:
            reply = super().launch_grid(*args, **kwargs)
            stat = self.axi_status()
            if stat and (stat['error'] or stat['stray']):
                raise DeviceLost('physical AXI fault requires platform recovery', detail=stat)
            if reply.ok and self.device_clock:
                reply.device_timing = self.launch_timestamps(reply.seq)
            return reply


class MulticoreXDMALink(XDMALink):
    """The platform's three argument aliases require one reachable BO group."""
    def hbm_open(self, nbytes):
        if self._accessor is None or not hasattr(self._accessor, 'bo_open'):
            return None
        self.hbm_layout = None
        banks = [_xclbin_bank(self.xclbin, self.ip_name, arg_index=i) for i in range(3)]
        if any(not bank['used'] for bank in banks) or len({bank['index'] for bank in banks}) != 1:
            raise DependencyFailed('multicore AXI masters must share the same XRT allocation group',
                                   detail={'banks': banks})
        striped = bool(self.read_reg(0xb8) & STRIPED_CAP)
        self._striped_required = striped
        if striped:
            if self.read_reg(STRIPED_REGISTER) != STRIPED_SIGNATURE:
                raise DeviceLost('invalid striped HBM layout signature')
            if operator.index(nbytes) != STRIPED_BYTES:
                raise ArgumentError('striped HBM requires an explicit 8 GiB allocation')
        allocation = super().hbm_open(nbytes)
        if striped:
            banks = self.hbm_allocation.get('physical_banks', [])
            expected = [(pc, pc << 28, 1 << 28) for pc in range(32)]
            actual = [(b.get('pc'), b.get('allocated_base'), b.get('allocated_size')) for b in banks]
            if (self.hbm_allocation.get('mapping_error') or actual != expected or
                    any(not b.get('used') or b.get('base') != b.get('allocated_base') or
                        b.get('size') != 1 << 28 for b in banks)):
                raise DependencyFailed('striped HBM requires verified PC0..31 physical topology')
            try:
                self.hbm_layout = StripedHBMLayout(*allocation)
            except ValueError as error:
                raise DependencyFailed(str(error)) from error
            self.hbm_allocation['layout'] = self.hbm_layout.name
        return allocation

    def hbm_write(self, offset, data):
        layout = getattr(self, 'hbm_layout', None)
        if getattr(self, '_striped_required', False) and layout is None:
            raise DependencyFailed('striped HBM allocation has not been verified')
        if layout is not None:
            return layout.write(self._accessor, offset, data)
        return super().hbm_write(offset, data)

    def hbm_read(self, offset, nbytes):
        layout = getattr(self, 'hbm_layout', None)
        if getattr(self, '_striped_required', False) and layout is None:
            raise DependencyFailed('striped HBM allocation has not been verified')
        if layout is not None:
            return layout.read(self._accessor, offset, nbytes)
        return super().hbm_read(offset, nbytes)


def board(**kwargs):
    kwargs.pop('bench', None)
    link = kwargs.pop('link', None)
    if link is None:
        link = MulticoreXDMALink(xclbin=kwargs.pop('xclbin', None),
                        accessor=kwargs.pop('accessor', None),
                        index=kwargs.pop('index', 0),
                        ip_name=kwargs.pop('ip_name', IP_NAME),
                        words=kwargs.pop('words', DEFAULT_WORDS))
    return MulticoreMMIOBackend(link, **kwargs)
