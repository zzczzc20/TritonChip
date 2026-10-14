"""Opt-in v2 register binding; no legacy image or simulator substitution."""
import operator

from ..errors import ArgumentError, DeviceLost, DependencyFailed
from .u50 import U50Backend, XDMALink, IP_NAME, DEFAULT_WORDS, _xclbin_bank

OP_CACHE_HANDOFF = 18
CACHE_HANDOFF_CAP = 16


class MulticoreMMIOBackend(U50Backend):
    identity_magic = 0x0002_0000_544E_4341
    opcode_mask = 0xFF
    name = 'fpga-u280-multicore'

    def __init__(self, link, *, core_count=3, memory_ports=3,
                 hbm_dma=False, **kwargs):
        self.core_count = operator.index(core_count)
        self.memory_ports = operator.index(memory_ports)
        if not (1 <= self.memory_ports <= min(3, self.core_count) and self.core_count <= 16):
            raise ArgumentError('MMIO v2 needs 1..3 outlets and 1..16 cores, with ports <= cores')
        kwargs['image_profile'] = kwargs.get('image_profile') or 'core-vrf-sync-v5'
        kwargs.setdefault('device_clock', True)
        super().__init__(link, hbm_dma=hbm_dma, **kwargs)

    def _open_window(self, stat):
        self._ownership_cap = bool(stat & CACHE_HANDOFF_CAP)
        if self.hbm_dma and not self._ownership_cap:
            raise ArgumentError('multicore DMA requires the image ownership-handoff capability')
        return super()._open_window(stat)

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

    def _get_payload(self, addr, nwords):
        self._check_window(addr, nwords, 'download')
        register_words = self._dma_split(addr, nwords)
        if register_words < nwords:
            self._handoff(addr + register_words * 8, (nwords-register_words)*8)
        return super()._get_payload(addr, nwords)

    def start(self):
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
            return self
        except BaseException:
            self.stop()
            raise

    def capabilities(self):
        out = super().capabilities()
        out.update(physical_grid=True, memory_ports=self.memory_ports,
                   memory_bytes=self.memory_bytes,
                   completion_counters='global-makespan-u64', fault_detail=False,
                   reaches='rtl/fpga/tnca_multicore_fpga.v',
                   protocol='r2_transport_v1 + r40 physical-grid extension, MMIO v2',
                   coherent_host_window=True, hbm_dma=self.hbm_dma and getattr(self, '_ownership_cap', False),
                   cache_ownership_handoff=getattr(self, '_ownership_cap', False))
        return out

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
        banks = [_xclbin_bank(self.xclbin, self.ip_name, arg_index=i) for i in range(3)]
        if any(not bank['used'] for bank in banks) or len({bank['index'] for bank in banks}) != 1:
            raise DependencyFailed('multicore AXI masters must share the same XRT allocation group',
                                   detail={'banks': banks})
        return super().hbm_open(nbytes)


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
