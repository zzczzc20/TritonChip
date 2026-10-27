"""Validated command driver for the 32-PC stream CU; no image loading."""
import math
import operator
import time

from ..errors import ArgumentError, DeviceLost
from ..hbm_layout import StripedHBMLayout, STRIPED_BYTES

MAGIC = 0x52343153424E4B01
CAPABILITY = 0x0020020000100080
PAGE_GROUP = 1 << 17
# Native-copy setup is substantial MMIO work. Explicit stream APIs can use
# smaller ranges; this initial native threshold requires board tuning.
NATIVE_COPY_MIN_BYTES = 32 << 20


def integer(value):
    if isinstance(value, bool):
        raise ArgumentError('stream counts and offsets must be integers, not booleans')
    try:
        return operator.index(value)
    except TypeError as error:
        raise ArgumentError('stream counts and offsets must be integers') from error


def descriptors(mode, source, destination, nbytes):
    mode, source, destination, nbytes = map(integer, (mode, source, destination, nbytes))
    if mode not in (0, 1, 2) or nbytes <= 0 or nbytes % 64:
        raise ArgumentError('stream operations need a supported mode and positive complete lines')
    for offset in ((source, destination) if mode == 0 else (destination,) if mode == 1 else (source,)):
        if offset < 0 or offset % 64 or offset > STRIPED_BYTES or nbytes > STRIPED_BYTES-offset:
            raise ArgumentError('stream range is unaligned or outside the 8 GiB allocation')
    if mode == 0 and ((source-destination) % PAGE_GROUP or max(source,destination)<min(source+nbytes,destination+nbytes)):
        raise ArgumentError('stream copy needs disjoint ranges with matching 128 KiB phase')
    def groups(offset):
        return {pc: (physical & ((1 << 28)-1), size//64)
                for pc,physical,size in StripedHBMLayout.pc_ranges(offset,nbytes)}
    reads = {} if mode == 1 else groups(source)
    writes = {} if mode == 2 else groups(destination)
    result = []
    for pc in range(32):
        src, nr = reads.get(pc,(0,0));dst, nw = writes.get(pc,(0,0))
        if mode == 0 and nr != nw:
            raise ArgumentError('source/destination PC traffic differs')
        result.append((src,dst,max(nr,nw),nr,nw))
    return result


class StreamController:
    def __init__(self, registers, *, timeout=30.0, poll_interval=0.001):
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(poll_interval) or poll_interval < 0:
            raise ArgumentError('stream timeout must be positive and polling interval nonnegative')
        self.registers = registers
        self.timeout, self.poll_interval = timeout, poll_interval
        self.failed = False
        self.last_result = None
        if registers.read(0) != MAGIC or registers.read(8) != CAPABILITY:
            raise DeviceLost('stream CU identity or capability differs')
        self.sequence = integer(registers.read(0x28))
        self._idle()

    def _idle(self):
        flags = self.registers.read(0x20)
        if self.failed or flags & 5:
            raise DeviceLost('stream CU is busy or faulted; no command was admitted', detail={'flags': flags})

    def _put(self, address, value):
        self.registers.write(address,value)
        if self.registers.read(address) != value:
            raise DeviceLost('stream CU refused configuration readback', detail={'register': address})

    def execute(self, mode, source, destination, nbytes, *, pattern=0, handoff):
        mode, source, destination, nbytes = map(integer,(mode,source,destination,nbytes))
        plan = descriptors(mode,source,destination,nbytes)
        pattern = integer(pattern)
        if not 0 <= pattern < 1 << 32:
            raise ArgumentError('stream fill pattern must fit uint32')
        self._idle()
        if self.sequence >= (1 << 64)-1:
            raise DeviceLost('stream sequence exhausted; a new device epoch is required')
        # Pure validation precedes core ownership changes and all CU writes.
        if mode != 1: handoff(source,nbytes)
        if mode != 2: handoff(destination,nbytes)
        started = False
        try:
            self._put(0x88,STRIPED_BYTES);self._put(0x90,0)
            self._put(0x40,(mode << 32)|pattern)
            mask = 0
            for pc,(src,dst,lines,_,_) in enumerate(plan):
                self._put(0x50,pc)
                self._put(0x58,src);self._put(0x60,dst);self._put(0x68,lines)
                if lines:mask |= 1 << pc
            self._put(0x48,mask)
            self.sequence += 1
            self._put(0x10,self.sequence)
            started = True
            self.registers.write(0x18,1)
            deadline = time.monotonic()+self.timeout
            while True:
                flags = self.registers.read(0x20)
                if flags & 4:
                    raise DeviceLost('stream AXI fault requires coordinated platform recovery',
                                     detail={'lanes': self.registers.read(0xa0)})
                done = self.registers.read(0x28)
                if done == self.sequence and not flags & 1:break
                if done > self.sequence:
                    raise DeviceLost('stream completion sequence advanced unexpectedly')
                if time.monotonic() >= deadline:
                    raise DeviceLost('stream command observation timed out; no completion was fabricated')
                time.sleep(self.poll_interval)
            status = self.registers.read(0x30)
            if status:
                raise DeviceLost('validated stream command was refused',detail={'status': status})
            cycles = self.registers.read(0x38)
            total_reads,total_writes = self.registers.read(0x78),self.registers.read(0x80)
            if cycles <= 0 or (total_reads,total_writes) != (sum(p[3] for p in plan),sum(p[4] for p in plan)):
                raise DeviceLost('stream completion has incorrect cycle/traffic counters')
            lane_records = []
            for pc,p in enumerate(plan):
                if not p[2]:continue
                self._put(0x50,pc)
                nr,nw = self.registers.read(0xb0),self.registers.read(0xb8)
                if (nr,nw) != p[3:5]:
                    raise DeviceLost('stream per-PC counters differ from the command')
                lane_cycles = self.registers.read(0xa8)
                if not max(nr,nw) <= lane_cycles <= cycles:
                    raise DeviceLost('stream lane timing violates its interface bound')
                lane_records.append(dict(pc=pc,read_lines=nr,write_lines=nw,cycles=lane_cycles))
            if self.registers.read(0x28) != self.sequence or self.registers.read(0x20) & 5:
                raise DeviceLost('stream completion changed during its snapshot')
            result = dict(sequence=self.sequence,mode=mode,cycles=cycles,
                          read_bytes=total_reads*64,write_bytes=total_writes*64,
                          result_sum=self.registers.read(0x70),lanes=lane_records,
                          clock_contract_hz=100000000)
            self.last_result = result
            return result
        except BaseException:
            if started:self.failed = True
            raise
