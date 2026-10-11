"""The U50 platform binding -- a board, from the host.

doc/phase3_6_tasks_and_acceptance.md R36-HAND-02 and section 3.1.

WHERE THIS SITS

Section 3.1 separates four things that used to be one: a compile
target, an execution backend, a hardware implementation, and a platform
binding. This file is the second over the fourth. The profile is
core-vrf-sync-v1 -- one core, banked register file -- and the hardware
is rtl/fpga/tnca_fpga_top.v, which wraps the same rtl/tnca_shell.v that
the export package publishes. R36-RTL-03 asks the simulation binding and
the platform shell to share one core implementation; they do, and this
is what reaches the second one.

WHAT IS REUSED, AND WHY THAT IS THE POINT

`U50Backend` subclasses `tnca.transport.Transport` -- the `sim-file`
binding -- and overrides the medium and nothing else. That is the same
decision `backends/socket_relay.py` made, and `tools/r36/downstream/
platform_backend.py` says why in one line: "a platform branch inherits
the host side and supplies the device. What it overrides is where the
device comes from, which is the only thing that is actually different."

So the sequence number, the stale-reply rule, the grid packing in
`launch` and the `history` bookkeeping are NOT reimplemented here. They
are the protocol, and a second copy of a protocol is how two bindings
start answering differently -- which is exactly what the comment on
`_await` in transport.py is about. What differs between a shared
directory and a board is how a request is put somewhere and how an
answer is picked up: `_put_request`, `_read_reply`, `_put_payload`,
`_get_payload`, plus the four lifetime methods.

THE PROTOCOL IS NOT RENEGOTIATED

spec/r2_transport_v1.json, unchanged. R2-10: swapping the medium does
not change the protocol. rtl/fpga/tnca_mmio.v carries the same opcodes
and statuses on the other end, and `tb/fpga/tb_tnca_fpga.v` drives them
against the real RTL and checks the values against a hand-computed
oracle. That bench is the protocol reference this file is written to.

THE TWO LINKS

`U50Link` is the register-level medium -- read a register, write a
register, and nothing else. Below it are two implementations:

    XDMALink   the board. In this file, because it names no simulator.
    SimLink    the same registers driven on the real RTL by a simulator.
               In backends/u50_sim.py, which is why that file exists
               separately: a module that talks to an FPGA should not
               name a simulator, and tools/r36/backend_tests.py scans
               for exactly that.

They are separate names -- `fpga-u50` and `fpga-u50-sim` -- and neither
falls back to the other. Section 3.4 forbids the fallback in as many
words: a missing backend that quietly became the simulator would report
a platform working that was never there.

WHAT THIS FILE CANNOT TELL YOU

That the board works. Nothing here has run on one. The register
protocol above `U50Link` is proven against the RTL; `XDMALink` is not
proven at all, and its one unverifiable step is isolated and marked
rather than spread through the file. Section 3.2 keeps engineering-ready
and platform-verified apart, and this binding is the first of those.
"""
import abc
import ctypes
import os
import time

from .. import backend as _backend
from ..errors import ArgumentError, DependencyFailed, DeviceLost, TncaError, Timeout
from ..transport import (OP_DOWNLOAD, OP_LAUNCH, OP_PING, OP_QUIESCE,
                         OP_SHUTDOWN, OP_UPLOAD, Reply, Transport, WORD)

# ---------------------------------------------------------------------------
# The register map
#
# Byte offsets, 64 bits each, copied from rtl/fpga/tnca_mmio.v. Two copies of
# a map is one too many and the alternative is worse: read out of the RTL at
# import time and a binding that cannot be opened until the RTL is checked
# out. So it is copied, and it is copied with the file named, and the bench
# that drives both ends is the thing that keeps them equal.
#
# The high half of the mailbox is the transport protocol and the low half is
# the payload window, which is not part of the protocol at all -- a board
# with a DMA engine replaces the window and keeps the mailbox. See the
# ADAPTATION RESPONSIBILITIES table in rtl/fpga/tnca_fpga_top.v.
# ---------------------------------------------------------------------------
R_REQ_SEQ = 0x00
R_REQ_OP = 0x08
R_REQ_A0 = 0x10
R_REQ_A1 = 0x18
R_REQ_A2 = 0x20
R_REQ_A3 = 0x28
R_REQ_A4 = 0x30
R_REQ_A5 = 0x38
R_REQ_DEADLINE = 0x40
R_DOORBELL = 0x48

#: The opcode field is four bits wide, and that is the DEVICE's shape and
#: not this binding's to widen: rtl/fpga/tnca_mmio.v:416 decodes
#: `req_op[3:0]`. So an opcode that does not fit is not refused -- it
#: arrives as a DIFFERENT operation. 99 is 0x63, which reaches the device
#: as 3, which is OP_LAUNCH, and a caller who asked for an operation the
#: device has not got would have had a kernel run instead. Unknown
#: opcodes INSIDE the field stay the device's to refuse, and it does,
#: with RSP_REFUSED; this is only about what cannot be carried at all.
OP_MASK = 0xF

R_RSP_SEQ = 0x50
R_RSP_STATUS = 0x58
R_RSP_R0 = 0x60
R_RSP_R1 = 0x68

R_HOST_ADDR = 0x70
R_HOST_DATA = 0x78
R_HOST_BE = 0x80
R_HOST_CTRL = 0x88
R_HOST_STATUS = 0x90

R_MAGIC = 0x98
R_SERVED = 0xA0
# R40-04: the AXI window of an image with external memory (rtl/fpga/tnca_mmio.v,
# DATA_AXI). An image without one reads all three as zero.
R_AXI_BASE = 0xA8
R_AXI_SIZE = 0xB0
R_AXI_STAT = 0xB8
AXI_STAT_ERR = 0x1          # sticky: an AXI error response came back
AXI_STAT_PRESENT = 0x2      # the image has the window
AXI_STAT_STRAY = 0x4        # sticky: an AXI response matched nothing in flight
R_AXI_RD = 0xC0             # AXI reads issued, free-running
R_AXI_WR = 0xC8             # AXI writes issued
R_AXI_LAT = 0xD0            # sum over cycles of the reads in flight
R_AXI_RBEAT = 0xD8          # R beats taken (reads x burst length)

#: Version 1.0 and "TNCA" in ASCII. Read before anything else is written:
#: a host that has opened the wrong BAR, or a board with no bitstream
#: loaded, answers something else, and finding that out from a miscounted
#: sequence number is finding it out late.
MAGIC = 0x0001_0000_544E_4341

CTRL_WE = 0x1
CTRL_START = 0x2

ST_BUSY = 0x1
ST_VALID = 0x2
ST_REFUSED = 0x4

#: rtl/fpga/tnca_fpga_top.v's WORDS parameter. It is the range the payload
#: window will accept, and it is here so that the host can refuse an
#: out-of-range transfer before putting it on the bus rather than reading a
#: refusal back afterwards.
#:
#: 65536 and not something smaller, because that is the size of the device
#: the profile names: tb/device/tnca_device.v declares it for this profile,
#: and `Session`'s heap limit defaults to 0x80000, which is this many words
#: to the byte. Two independent statements of the same number.
#:
#: Built at 16384 instead, the board's memory ended exactly where the
#: profile's heap begins, so every compiled kernel was refused on its first
#: upload -- while a hand-written program keeping to low addresses passed
#: every check. That is why the mismatch survived a driver that exercised
#: the entire protocol, and why the end-to-end test is a compiled tutorial
#: kernel rather than a program written to fit.
DEFAULT_WORDS = 65536

#: The payload window's byte enables. All eight, so a write of a word writes
#: a word. The register resets to this in the RTL; it is set again on open
#: because a board that was left mid-transfer by a previous session is a
#: board whose window is in a state this session did not choose.
BE_ALL = 0xFF


class U50Link(abc.ABC):
    """The register-level medium, and the whole of what a board must supply.

    Deliberately four methods plus lifetime. Everything above this line --
    the protocol, the guards, the payload loop -- is shared by the board and
    its stand-in, so that what the simulation proves is the code the board
    runs and not a description of it.
    """

    #: How many 64-bit words the device's memory holds. Must equal the WORDS
    #: parameter the RTL was built with; a host that disagrees with the
    #: device about this is a host whose range guard refuses transfers the
    #: device would have accepted.
    words = DEFAULT_WORDS

    @abc.abstractmethod
    def open(self):
        """Take the device. Returns self, or raises with a reason."""

    @abc.abstractmethod
    def close(self):
        """Release it. Must be safe on a device that already died."""

    @abc.abstractmethod
    def alive(self):
        """True while the medium can still carry a register access."""

    @abc.abstractmethod
    def read_reg(self, offset):
        """One 64-bit register. Raises DeviceLost if the medium is gone."""

    @abc.abstractmethod
    def write_reg(self, offset, value):
        """One 64-bit register. Raises DeviceLost if the medium is gone."""

    def describe(self):
        """What this medium is, for capabilities()."""
        return self.__class__.__name__

    def log_tail(self, n=6):
        """The last few lines of whatever this medium keeps. May be empty.

        A board has no log and returning nothing is honest; a stand-in that
        keeps one is where "the device stopped answering" gets its reason.
        """
        return []

    # ---- R40-04: external memory (optional) -------------------------------
    def hbm_open(self, nbytes):
        """Allocate the AXI window's memory: (device address, bytes), or None
        when this medium has no way to. Only an image whose R_AXI_STAT says it
        has a window is asked."""
        return None

    def hbm_write(self, offset, data):
        """Bytes into the window's memory at `offset`, by DMA."""
        raise NotImplementedError

    def hbm_read(self, offset, nbytes):
        """Bytes out of the window's memory at `offset`, by DMA."""
        raise NotImplementedError


class U50Backend(Transport):
    """core-vrf-sync-v1, on rtl/fpga/tnca_fpga_top.v, over a register port.

    Named `fpga-u50` and not `fpga-prototype`. That is not a detail:
    tools/r36/backend_tests.py holds `fpga-prototype` to being
    interface-defined and asserts that opening it raises
    BACKEND_NOT_IMPLEMENTED, which is the existing contract for "a future
    board", and it holds at least one backend to staying interface-defined
    so that the state means something. A second name registers an
    implementation instead of editing the state machine -- section 3.4's
    "an external branch registers an implementation".
    """

    name = 'fpga-u50'
    state = _backend.REFERENCE_AVAILABLE
    launch_is_idempotent = False

    def __init__(self, link, *, epoch=0, host_timeout=30.0,
                 deadline_cycles=2_000_000, poll_interval=0.001,
                 trace=False, image_profile=None, hbm_bytes=None, hbm_dma=True,
                 **kw):
        # The profile the loaded image (or the simulated harness) was built
        # as. Nothing in the hardware says which (spec/r1_profiles.json: the
        # R39 switches change no capability register), so it is what the
        # build declares: rtl/fpga/tnca_fpga_top.v's defaults are
        # core-vrf-sync-v2 since R39, and an older image (r39b and before)
        # is core-vrf-sync-v1 and says so through this argument.
        self.image_profile = image_profile or BOARD_PROFILE
        # Transport's constructor sets up four filenames and a child
        # process. None of that is here, so it is not called.
        if link is None:
            raise ArgumentError(
                'a U50 backend needs a link',
                remedy='the medium is not optional: build an XDMALink for '
                       'a board, or a SimLink to drive the same registers '
                       'on the RTL. Nothing is chosen for you, because a '
                       'default that silently picked the stand-in is the '
                       'substitution section 3.4 forbids')
        self.link = link
        # There is no bench and no directory. Kept as attributes anyway
        # rather than left absent: `Transport` sets them in its constructor,
        # so anything that reaches for `.bench` expecting the binding it was
        # promised would get an AttributeError from a code path that has
        # nothing to do with a board. None says the same thing, in a way a
        # caller can test.
        self.bench = None
        self.dir = None
        # And this one for the same reason, which was learned the hard way.
        # `Transport._await` reaches for `self.log` when a wait runs out and
        # `request()` when the medium is gone; neither is a board path, both
        # are reachable from one, and an AttributeError raised there replaces
        # a WAIT_TIMEOUT the caller could act on with a traceback it cannot.
        # There is no log file on a board -- `log_tail` above already says
        # so -- and None says it in a way a caller can test.
        self.log = None
        self.epoch = epoch
        self.host_timeout = host_timeout
        self.deadline_cycles = deadline_cycles
        # A board register read is a bus transaction and a stand-in's is an
        # event step; neither wants to be polled as fast as the host can
        # loop, and both want to be polled often enough that the poll is not
        # the latency. One millisecond is far under both.
        self.poll_interval = poll_interval
        self.trace = trace
        self.seq = 0
        self.history = []
        self.opened = False
        # R40-04: an image with an AXI window (R_AXI_STAT) gets its memory
        # here at start(). How much: hbm_bytes, or TNCA_HBM_BYTES, or 1 GiB.
        # Transfers wholly inside the window go by DMA unless hbm_dma is off
        # (the register window still reaches every address).
        self.hbm_bytes = int(hbm_bytes or os.environ.get('TNCA_HBM_BYTES') or (1 << 30))
        self.hbm_dma = bool(int(hbm_dma)) if not isinstance(hbm_dma, bool) else hbm_dma
        self.local_limit = None
        self.memory_bytes = None

    # ---- lifetime ---------------------------------------------------------
    def start(self):
        if self.opened:
            raise TncaError('this backend already has a device',
                            code='INTERNAL')
        # `link.open()` is INSIDE the try, and that is not tidiness. It was
        # outside first: a link that started a device and then failed to
        # get an answer from it raised out of `open()`, the handler below
        # was never reached, `close()` was never called, and the device
        # stayed running. It was found by looking for it with the driver
        # long gone, which is a poor way to find one.
        try:
            self.link.open()
            self.opened = True
            magic = self.link.read_reg(R_MAGIC)
            if magic != MAGIC:
                raise DeviceLost(
                    f"the device answered {magic:#018x} from its identity "
                    f"register, not {MAGIC:#018x}",
                    detail={'epoch': self.epoch, 'magic': magic,
                            'medium': self.link.describe()},
                    remedy='this is the wrong BAR, or a board with no '
                           'bitstream loaded, or a bitstream that is not '
                           'this device. Check the identity register '
                           'before anything else: every later symptom of '
                           'this looks like a protocol bug')
            # The device's `served` is 64 bits wide and it resets when the
            # bitstream does, not when this process starts. The rule the
            # doorbell applies is `req_seq > served` (rtl/fpga/tnca_mmio.v's
            # MS_IDLE), so a session that began numbering at one again would
            # ring a bell the device is already past -- and a duplicate ring
            # is dropped in silence, by design: no answer, no completion, no
            # error. Adopting the watermark is what lets a second host
            # process talk to a board the first one has already used, and it
            # is why `served` is 64 bits and readable at all.
            self.seq = self.link.read_reg(R_SERVED)
            # The window's byte enables belong to this session, not to
            # whatever the last one left them as.
            self.link.write_reg(R_HOST_BE, BE_ALL)
            # The stand-in for sim-file's "await sequence 0", and a stronger
            # claim than it is: a device that reads its identity back and
            # then answers a ping has a working command path, where a
            # process that started has a working process.
            self.request(OP_PING)
            stat = self.link.read_reg(R_AXI_STAT)
            if stat & AXI_STAT_PRESENT:
                self._open_window(stat)
        except BaseException:
            self.opened = False
            self.link.close()
            raise
        return self

    def alive(self):
        return self.opened and self.link.alive()

    def _open_window(self, stat):
        """R40-04: give the image's AXI window its memory.

        The device reaches address A >= LOCAL_LIMIT at R_AXI_BASE + (A -
        LOCAL_LIMIT); the memory is a buffer XRT allocates in the bank the
        link connected m_axi_hbm to, and R_AXI_BASE is that buffer's device
        address. LOCAL_LIMIT comes from the device (R_AXI_STAT 63:32, in 4 KiB
        pages), not from a constant here.
        """
        local = (stat >> 32) << 12
        got = self.link.hbm_open(self.hbm_bytes)
        if got is None:
            raise DependencyFailed(
                'the image has an AXI window and this medium cannot allocate '
                'its memory',
                detail={'medium': self.link.describe(), 'axi_stat': stat},
                remedy='open the board through XDMALink (XRT), which allocates '
                       'it; without it every address from LOCAL_LIMIT up would '
                       'read zero')
        base, size = got
        self.link.write_reg(R_AXI_BASE, base)
        self.link.write_reg(R_AXI_SIZE, size)
        if (self.link.read_reg(R_AXI_BASE), self.link.read_reg(R_AXI_SIZE)) != (base, size):
            raise DeviceLost('the AXI window registers did not keep what was written',
                             detail={'base': base, 'size': size,
                                     'medium': self.link.describe()})
        self.local_limit = local
        self.memory_bytes = local + size
        self.link.words = self.memory_bytes // WORD
        self.axi_window = dict(base=base, size=size, local_limit=local, dma=self.hbm_dma)

    def axi_status(self):
        """R_AXI_STAT's sticky flags, or None on an image without a window."""
        if self.local_limit is None:
            return None
        stat = self.link.read_reg(R_AXI_STAT)
        return dict(error=bool(stat & AXI_STAT_ERR), stray=bool(stat & AXI_STAT_STRAY))

    def axi_counters(self):
        """The bridge's free-running counters, or None without a window:
        reads, writes, read-cycles in flight (mean read latency in cycles is
        read_cycles / reads, by Little's law) and R beats (read_beats / reads
        is the burst length)."""
        if self.local_limit is None:
            return None
        r = self.link.read_reg
        return dict(reads=r(R_AXI_RD), writes=r(R_AXI_WR), read_cycles=r(R_AXI_LAT),
                    read_beats=r(R_AXI_RBEAT))

    def launch(self, *a, **kw):
        reply = super().launch(*a, **kw)
        if self.local_limit is not None:
            st = self.axi_status()
            if st['error'] or st['stray']:
                # Sticky until the image is reset: every later result is
                # suspect too, so the device is reported lost, not faulted.
                raise DeviceLost(
                    'the AXI memory answered with an error during a launch'
                    if st['error'] else
                    'the AXI memory answered a request that was not in flight',
                    detail=dict(self._gone_detail(), axi=st),
                    remedy='reload the image; R_AXI_STAT stays set until reset')
        return reply

    def stop(self):
        if not self.opened:
            return
        try:
            if self.link.alive():
                self.request(OP_SHUTDOWN)
        except TncaError:
            pass
        self.opened = False
        try:
            self.link.close()
        except Exception:
            pass

    def _gone_detail(self):
        return {'medium': self.link.describe(), 'epoch': self.epoch}

    def _log_tail(self, n=6):
        return self.link.log_tail(n)

    # ---- the medium -------------------------------------------------------
    def _put_request(self, seq, op, a0, a1, a2, a3, a4, a5):
        """The descriptor, then the ring.

        Order matters and it is the order the doorbell needs: the sequence
        is written first because the device's rule is `req_seq > served`,
        and a ring that arrived while `req_seq` still held the previous
        request's number would either be dropped as a duplicate or, worse,
        served against a descriptor half of which is the previous one's.
        rtl/fpga/tnca_mmio.v's MS_IDLE clears the ring flag whether it
        serves it or drops it, so a dropped ring is not a stuck one.
        """
        # Refused here rather than written and truncated into a different
        # operation. This is the only place that knows the field is four
        # bits; the device cannot tell a truncated opcode from a real one,
        # so nobody downstream can catch it.
        if op < 0 or op > OP_MASK:
            raise TncaError(
                f"opcode {op} does not fit the device's "
                f"{OP_MASK + 1}-wide opcode field",
                code='ARGUMENT_INVALID',
                remedy='the board decodes four bits of the opcode register, '
                       'so this would arrive as a different operation '
                       'rather than as a refusal. Ask for one the device '
                       'has, or drive a binding whose opcode field is wide '
                       'enough to carry the request')

        w = self.link.write_reg
        w(R_REQ_SEQ, seq)
        w(R_REQ_OP, op)
        w(R_REQ_A0, a0)
        w(R_REQ_A1, a1)
        w(R_REQ_A2, a2)
        w(R_REQ_A3, a3)
        w(R_REQ_A4, a4)
        w(R_REQ_A5, a5)
        w(R_REQ_DEADLINE, self.deadline_cycles)
        w(R_DOORBELL, 1)

    def _read_reply(self):
        """The response registers, read as one consistent set or not at all.

        The four are written together on one clock edge and read one at a
        time over a bus, so a host that read them naively could take the
        sequence from one response and the status from the next -- and
        would then believe a status that belongs to a request it has not
        made yet. The repair is the ordinary one: read the sequence, read
        the fields, read the sequence again, and believe the set only if
        the number did not move.

        Sequence zero means nothing has answered yet. No request carries
        it -- `start()` adopts the device's `served` and `request()` steps
        before it sends, so the first sequence of any session is at least
        one -- so "nothing yet" and "the previous answer is still standing"
        are the same observation and `_await` waits on both.
        """
        for _ in range(8):
            before = self.link.read_reg(R_RSP_SEQ)
            if before == 0:
                return None
            status = self.link.read_reg(R_RSP_STATUS)
            r0 = self.link.read_reg(R_RSP_R0)
            r1 = self.link.read_reg(R_RSP_R1)
            if self.link.read_reg(R_RSP_SEQ) == before:
                return Reply(before, status & 0xF, r0, r1)
        raise DeviceLost(
            'the response registers would not hold still for one read',
            detail={'medium': self.link.describe(), 'epoch': self.epoch},
            remedy='the device is completing requests faster than this '
                   'medium can be read, so no response can be attributed '
                   'to a request. This is a host-side reading problem and '
                   'not a device fault')

    def _check_window(self, addr, nwords, what):
        """The range the payload window will accept, checked here first.

        rtl/fpga/tnca_mmio.v refuses an address at or past the end, on an
        unaligned address and a transfer of nothing, and answers ST_REFUSED.
        All three are also checked here, and that is not belt and braces:
        a refusal that reached the bus would come back as a status word,
        and a host reading a status word to find out whether its own
        argument was legal is a host that has moved an argument check onto
        the wire. What the device's own guard is for is the case where the
        two ends disagree about the size of memory.
        """
        if int(addr) % WORD:
            raise ArgumentError(
                f'{what} address {addr:#x} is not {WORD}-byte aligned',
                remedy=f'the window is addressed in {WORD}-byte words')
        if int(nwords) <= 0:
            raise ArgumentError(f'a {what} of {nwords} word(s)',
                                remedy='a transfer of nothing is a mistake, '
                                       'not a fast path')
        if int(addr) // WORD + int(nwords) > self.link.words:
            raise ArgumentError(
                f'{what} of {nwords} word(s) at {addr:#x} runs past the '
                f'end of {self.link.words} word(s) of memory',
                remedy='memory the host has not written is not zero on a '
                       'board, so a transfer past the end is not a way to '
                       'place a sentinel')

    def _window(self, addr, word, *, write):
        """One word through the payload window.

        The address is written every time rather than leaned on. The RTL
        advances `host_addr` by eight after each completed access, so a
        host that left it alone would walk forward on its own -- and a
        burst that walked forward on its own is a burst where one refused
        or dropped access silently shifts every word after it. Writing the
        address each time costs one register write per word and makes each
        access independently checkable, which is the trade this makes
        deliberately rather than by omission.
        """
        w = self.link.write_reg
        w(R_HOST_ADDR, addr)
        if write:
            w(R_HOST_DATA, word)
        w(R_HOST_CTRL, CTRL_START | (CTRL_WE if write else 0))

        deadline = time.time() + self.host_timeout
        while True:
            if not self.link.alive():
                raise DeviceLost(
                    'the medium stopped answering mid-transfer',
                    detail=dict(self._gone_detail(), tail=self._log_tail()))
            st = self.link.read_reg(R_HOST_STATUS)
            if not st & ST_BUSY:
                break
            if time.time() > deadline:
                raise Timeout(
                    f'the payload window stayed busy for '
                    f'{self.host_timeout}s at {addr:#x}',
                    detail=dict(self._gone_detail(), status=st),
                    remedy='this is a HOST wait. The device may still be '
                           'working and the window may still complete; '
                           'this is not a verdict on the transfer')
            time.sleep(self.poll_interval)

        if st & ST_REFUSED:
            raise DeviceLost(
                f'the device refused {addr:#x}, which this host had '
                f'checked against {self.link.words} word(s)',
                detail=dict(self._gone_detail(), addr=addr, status=st),
                remedy='the host and the device disagree about how much '
                       'memory there is. The device is right: '
                       'DEFAULT_WORDS must equal the WORDS parameter the '
                       'RTL was built with')
        return None if write else self.link.read_reg(R_HOST_DATA)

    def _dma_split(self, addr, nwords):
        """How many of the words from `addr` go through the register window:
        all of them, unless part of the range is in an AXI window that DMA
        reaches, in which case only those below LOCAL_LIMIT."""
        if self.local_limit is None or not self.hbm_dma:
            return nwords
        return max(0, min(nwords, (self.local_limit - addr) // WORD))

    def _put_payload(self, addr, words):
        self._check_window(addr, len(words), 'upload')
        k = self._dma_split(addr, len(words))
        for i, word in enumerate(words[:k]):
            self._window(addr + i * WORD, word, write=True)
        if k < len(words):
            data = b''.join(int(w).to_bytes(WORD, 'little') for w in words[k:])
            self.link.hbm_write(addr + k * WORD - self.local_limit, data)

    def _get_payload(self, addr, nwords):
        self._check_window(addr, nwords, 'download')
        k = self._dma_split(addr, nwords)
        out = bytearray()
        for i in range(k):
            got = self._window(addr + i * WORD, None, write=False)
            out += got.to_bytes(WORD, 'little')
        if k < nwords:
            out += self.link.hbm_read(addr + k * WORD - self.local_limit, (nwords - k) * WORD)
        return bytes(out)

    def capabilities(self):
        # `Backend.capabilities`, not `Transport.capabilities`. The one above
        # describes a shared directory and a bench file, and this is neither;
        # what is wanted here is the contract's own fields plus this
        # binding's. Named rather than `super()` so that a later edit to the
        # file medium cannot quietly add a field to a board's capabilities.
        out = _backend.Backend.capabilities(self)
        out.update({'medium': self.link.describe(),
                    'reaches': 'rtl/fpga/tnca_fpga_top.v: rtl/tnca_shell.v '
                               'wrapped for a board, over its register port',
                    'protocol': 'r2_transport_v1',
                    'profile': self.image_profile,
                    # The mailbox carries an opcode field and the device
                    # refuses one it does not have, so a caller can send
                    # one and see it refused.
                    'raw_request': True})
        return out


# ---------------------------------------------------------------------------
# The board's registers, through XRT
#
# Everything from here to XDMALink is board-dependent and none of it has run
# on a board. What it is built from is three things that are on the machine
# and are not the device: XRT's headers, `nm` on its library, and xclbinutil
# reading a shipped .xclbin. Where a fact came from one of those rather than
# from memory, the comment says which one.
# ---------------------------------------------------------------------------

#: The kernel's name as XRT knows it, which is NOT the name in fpga/kernel.xml.
#:
#: xclIPName2Index's header says "@ipName: IP name. usually `<kernel
#: name>:<instance name>`". Confirmed against a shipped .xclbin rather than
#: assumed: dumping that file's IP_LAYOUT section shows a kernel named `hello`
#: with a single instance as `hello:hello_1`. This kernel is `tnca_u50` in
#: fpga/kernel.xml with one instance, so it is this.
#:
#: XDMALink takes it as a parameter because it is the one thing in this
#: section that a different build could disagree about -- a second kernel in
#: the same xclbin, or an instance numbering that starts elsewhere.
IP_NAME = 'tnca_u50:tnca_u50_1'

# What rtl/fpga/tnca_fpga_top.v builds by default (R39: core-vrf-sync-v2).
BOARD_PROFILE = 'core-vrf-sync-v2'

#: XRT's verbosity, from `enum xclVerbosityLevel` in deprecated/xrt.h:
#: XCL_QUIET = 0, XCL_INFO = 1, XCL_WARN = 2, XCL_ERROR = 3.
#:
#: WARN and not INFO, which is talkative about enumerating devices on every
#: open, and not QUIET, which is also quiet about the reason a register access
#: was refused -- which is the one thing worth hearing from XRT here.
XCL_WARN = 2

#: The magic every .xclbin starts with.
XCLBIN_MAGIC = b'xclbin2'

#: How much of an .xclbin is searched for the loaded device's uuid.
#:
#: A search and not an offset, deliberately -- read `_loaded_xclbin_uuid`
#: below for what assuming an offset cost the first time a board was loaded.
#: The uuid sits inside the header, and the header is a few hundred bytes:
#: 0x150 in the platform-shipped .xclbin the old constant was measured on,
#: 0x1a0 in the one v++ built for this branch. Four kilobytes is a window
#: that does not need revisiting when it moves again, and sixteen exact bytes
#: do not turn up by accident, so a window wider than needed costs nothing.
XCLBIN_UUID_SEARCH = 4096

#: The drm render node that is XRT device index 0, and the file under it that
#: holds the uuid of the image the device is running.
_DRM_RENDER_BASE = 128

#: libxrt_core.so, loaded once. Module state and not a class attribute,
#: because it is one library for the process and a second copy would be a
#: second set of signatures to keep right.
_LIB = None


def _loaded_xclbin_uuid(index):
    """The sixteen bytes the device says it is running.

    Not read out of the .xclbin file, and that is a correction rather than a
    preference.

    The uuid used to be read from the file at a fixed offset, 336, which had
    been found by searching a platform-shipped .xclbin for the bytes
    xclbinutil reported. That number is not a property of the format. The
    header v++ writes puts m_uniqueId eighty bytes further along, so on the
    first .xclbin this branch ever loaded the read returned sixteen zero
    bytes; xclOpenContext rejected them with "Invalid argument", and XRT
    threw across the C boundary, aborting the process before any of `open`'s
    error handling could run. The failure looked like the board refusing and
    was a constant that happened to fit one file.

    The kernel has known the answer since the image was loaded:
    /sys/class/drm/renderD<N>/device/xclbinuuid is the uuid of the image
    running on that device. It needs no knowledge of any header layout, and
    it cannot disagree with the device, because it is the device's own
    answer -- which is the whole of what xclOpenContext is asking about.

    N is 128 + index, the drm convention XRT's index follows. On this machine
    renderD128 is the U50's user function 0000:01:00.1; renderD129 is an
    NVIDIA card, which is not an XRT device at all. The lookup fails closed,
    so a card whose render node is numbered differently stops here with a
    named reason instead of opening a context against the wrong uuid.
    """
    node = (f'/sys/class/drm/renderD{_DRM_RENDER_BASE + index}'
            '/device/xclbinuuid')
    try:
        with open(node) as fh:
            text = fh.read().strip()
    except OSError as exc:
        raise DependencyFailed(
            f'could not read the loaded image uuid from {node}',
            detail={'node': node, 'index': index, 'errno': exc.errno,
                    'xrt': _xrt_version()},
            remedy='xclOpenContext takes the uuid of the image running on '
                   'the device, and this is where the kernel publishes it. '
                   f'If the render node for this device is not renderD'
                   f'{_DRM_RENDER_BASE + index}, this is the wrong node and '
                   'the fix is the right index, not another way of reading '
                   'it') from exc

    raw = text.replace('-', '')
    if len(raw) != 32:
        raise DependencyFailed(
            f'{node} does not hold a uuid (found {text!r})',
            detail={'node': node, 'value': text, 'xrt': _xrt_version()},
            remedy='that device has no image loaded, so there is no uuid for '
                   'a context to be opened against. Load one first: '
                   'flow/fpga/program.sh, which is a deliberate step and '
                   'says so')
    return bytes.fromhex(raw)


def _file_carries_uuid(path, uuid):
    """Whether the named .xclbin claims the identity the device is running.

    A membership test rather than a lookup: the sixteen bytes have to appear
    in the header, and where in it they appear is not asked. That is the
    question worth asking here anyway -- pointing the link at a file other
    than the loaded image is the mistake the uuid was in this function to
    catch, and this catches it without betting on a layout.
    """
    with open(path, 'rb') as fh:
        head = fh.read(XCLBIN_UUID_SEARCH)
    if head[:7] != XCLBIN_MAGIC:
        raise DependencyFailed(
            f'{path} does not start with "xclbin2"',
            detail={'path': path, 'found': repr(head[:8]),
                    'xrt': _xrt_version()},
            remedy='it is not an .xclbin, so it is not the image the device '
                   'is running')
    return uuid in head


class _XclBOProperties(ctypes.Structure):
    """deprecated/xrt.h: struct xclBOProperties."""
    _fields_ = [('handle', ctypes.c_uint32), ('flags', ctypes.c_uint32),
                ('size', ctypes.c_uint64), ('paddr', ctypes.c_uint64),
                ('reserved', ctypes.c_int)]


NULLBO = 0xFFFF_FFFF
XCL_BO_SYNC_BO_TO_DEVICE = 0
XCL_BO_SYNC_BO_FROM_DEVICE = 1

# xclbin.h section kinds
_MEM_TOPOLOGY, _CONNECTIVITY, _IP_LAYOUT = 6, 7, 8
_ASK_GROUP_TOPOLOGY, _ASK_GROUP_CONNECTIVITY = 26, 27


def _xclbin_sections(path):
    """{kind: bytes} of an .xclbin (xclbin.h: struct axlf; the section
    headers start at 456 -- the 304-byte prefix, then axlf_header with
    m_numSections at 448 -- and are 40 bytes each)."""
    import struct
    with open(path, 'rb') as fh:
        blob = fh.read()
    if not blob.startswith(XCLBIN_MAGIC):
        raise DependencyFailed(f'{path} is not an .xclbin', detail={'path': path})
    (n,) = struct.unpack_from('<I', blob, 448)
    out = {}
    for i in range(n):
        kind, _name, off, size = struct.unpack_from('<I16s4xQQ', blob, 456 + 40 * i)
        out.setdefault(kind, blob[off:off + size])
    return out


def _xclbin_bank(path, ip_name, arg_index=0):
    """The memory index XRT allocates in for argument `arg_index` of
    `ip_name`, and that bank's tag and size in bytes.

    From the group sections when the link made any (a connection to a range
    such as HBM[0:31] is one group bank spanning them), else the plain ones;
    of the banks the argument is connected to, the largest, which is the
    group. xclbin.h: ip_layout {int32 count; ip_data[80 B] from 8},
    connectivity {int32 count; connection[12 B] from 4}, mem_topology {int32
    count; mem_data[40 B] from 8, m_size in KiB}.
    """
    import struct
    sec = _xclbin_sections(path)
    ips = sec.get(_IP_LAYOUT)
    if ips is None:
        raise DependencyFailed(f'{path} has no IP_LAYOUT', detail={'path': path})
    (n_ip,) = struct.unpack_from('<i', ips, 0)
    names = [struct.unpack_from('<IIQ64s', ips, 8 + 80 * i)[3].split(b'\0')[0].decode()
             for i in range(n_ip)]
    if ip_name not in names:
        raise DependencyFailed(f'{path} has no IP {ip_name!r}', detail={'ips': names})
    ip = names.index(ip_name)
    conn = sec.get(_ASK_GROUP_CONNECTIVITY) or sec.get(_CONNECTIVITY)
    topo = sec.get(_ASK_GROUP_TOPOLOGY) or sec.get(_MEM_TOPOLOGY)
    if conn is None or topo is None:
        raise DependencyFailed(f'{path} has no connectivity for {ip_name!r}',
                               detail={'sections': sorted(sec)})
    (n_c,) = struct.unpack_from('<i', conn, 0)
    (n_m,) = struct.unpack_from('<i', topo, 0)
    mems = []
    for i in range(n_m):
        mtype, used, size_kb, base, tag = struct.unpack_from('<BB6xQQ16s', topo, 8 + 40 * i)
        mems.append(dict(index=i, used=used, size=size_kb * 1024, base=base,
                         tag=tag.split(b'\0')[0].decode(errors='replace')))
    banks = []
    for i in range(n_c):
        arg, ipi, mi = struct.unpack_from('<iii', conn, 4 + 12 * i)
        if arg == arg_index and ipi == ip and 0 <= mi < n_m:
            banks.append(mems[mi])
    if not banks:
        raise DependencyFailed(f'argument {arg_index} of {ip_name!r} is connected to no memory',
                               detail={'path': path})
    return max(banks, key=lambda m: m['size'])


def _xrt_library():
    """libxrt_core.so, loaded once, with every signature declared.

    Declared here rather than at the call sites, and that is not style:
    ctypes without argtypes does not fail on a wrong argument. It passes an
    integer where a pointer was expected and returns whatever the callee made
    of it, which for a register read is a number that looks like a register.
    Each signature below is the one in the header named beside it.
    """
    global _LIB
    if _LIB is not None:
        return _LIB

    root = os.environ.get('XILINX_XRT', '/opt/xilinx/xrt')
    path = os.path.join(root, 'lib', 'libxrt_core.so')
    try:
        lib = ctypes.CDLL(path, use_errno=True)
    except OSError as e:
        raise DependencyFailed(
            'XRT\'s library is not loadable, so there is no way to reach '
            'the board\'s registers from here',
            detail={'path': path, 'error': str(e), 'xrt': _xrt_version()},
            remedy='this binding drives a board and will not stand in for '
                   'one. Install XRT, or point XILINX_XRT at it, or pass '
                   'accessor= an object with read(offset) and '
                   'write(offset, value), or use the fpga-u50-sim binding '
                   'to drive the same registers on the RTL') from e

    c = ctypes
    # xuid_t is `unsigned char[16]` (xclbin.h), and as a parameter that is a
    # pointer to its first byte.
    u8 = c.POINTER(c.c_ubyte)

    # deprecated/xrt.h:
    #   xclOpen(unsigned int, const char*, enum xclVerbosityLevel)
    lib.xclOpen.argtypes = [c.c_uint, c.c_char_p, c.c_int]
    lib.xclOpen.restype = c.c_void_p

    # deprecated/xrt.h: void xclClose(xclDeviceHandle)
    lib.xclClose.argtypes = [c.c_void_p]
    lib.xclClose.restype = None

    # deprecated/xrt.h: int xclIPName2Index(xclDeviceHandle, const char*)
    lib.xclIPName2Index.argtypes = [c.c_void_p, c.c_char_p]
    lib.xclIPName2Index.restype = c.c_int

    # deprecated/xrt.h:
    #   int xclOpenContext(xclDeviceHandle, const xuid_t, unsigned, bool)
    lib.xclOpenContext.argtypes = [c.c_void_p, u8, c.c_uint, c.c_bool]
    lib.xclOpenContext.restype = c.c_int

    # deprecated/xrt.h:
    #   int xclCloseContext(xclDeviceHandle, const xuid_t, unsigned)
    lib.xclCloseContext.argtypes = [c.c_void_p, u8, c.c_uint]
    lib.xclCloseContext.restype = c.c_int

    # experimental/xrt-next.h: the datum is 32 bits wide and `offset` is a byte
    # offset into the IP's register space -- which is why a 64-bit register
    # costs two of these. Both return 0 on success, and datap is only
    # meaningful when they did.
    lib.xclRegRead.argtypes = [c.c_void_p, c.c_uint, c.c_uint,
                               c.POINTER(c.c_uint32)]
    lib.xclRegRead.restype = c.c_int
    lib.xclRegWrite.argtypes = [c.c_void_p, c.c_uint, c.c_uint, c.c_uint32]
    lib.xclRegWrite.restype = c.c_int

    # R40-04, the AXI window's memory: deprecated/xrt.h (XRT 2.16), each
    # checked with `nm -D` on libxrt_core.so. xclBufferHandle is `unsigned
    # int` on Linux and NULLBO 0xffffffff; xclWriteBO/xclReadBO copy to and
    # from the BO's host backing store and xclSyncBO moves a range of it by
    # DMA (XCL_BO_SYNC_BO_TO_DEVICE = 0, ..._FROM_DEVICE = 1).
    lib.xclAllocBO.argtypes = [c.c_void_p, c.c_size_t, c.c_int, c.c_uint]
    lib.xclAllocBO.restype = c.c_uint
    lib.xclFreeBO.argtypes = [c.c_void_p, c.c_uint]
    lib.xclFreeBO.restype = None
    lib.xclGetBOProperties.argtypes = [c.c_void_p, c.c_uint, c.POINTER(_XclBOProperties)]
    lib.xclGetBOProperties.restype = c.c_int
    lib.xclWriteBO.argtypes = [c.c_void_p, c.c_uint, c.c_void_p, c.c_size_t, c.c_size_t]
    lib.xclWriteBO.restype = c.c_size_t
    lib.xclReadBO.argtypes = [c.c_void_p, c.c_uint, c.c_void_p, c.c_size_t, c.c_size_t]
    lib.xclReadBO.restype = c.c_size_t
    lib.xclSyncBO.argtypes = [c.c_void_p, c.c_uint, c.c_int, c.c_size_t, c.c_size_t]
    lib.xclSyncBO.restype = c.c_int

    _LIB = lib
    return lib


class _XRTRegisters:
    """The board's register space, reached through XRT's C API.

    WHY ctypes AND NOT pyxrt

    `XDMALink`'s docstring used to say there were two shapes and that this
    was the step needing a board. This is the shape, and it was chosen from
    the headers on the machine:

      * pyxrt is built for one Python, and this backend has to import under
        whichever one runs it. libxrt_core.so has no such opinion, and
        ctypes reaches it from any of them. (pyxrt on this machine is also
        packaged without an `ip` class at all, so the first shape was not
        available to choose.)
      * the functions below are the deprecated-but-still-exported C API.
        All seven were checked with `nm -D` on the library before this was
        written, because a header declaring a function says nothing about
        whether the library still has it.

    WHAT IT TAKES, AND IN WHAT ORDER

        xclOpen(index, NULL, XCL_WARN)     -> device handle
        xclIPName2Index(handle, ip_name)   -> ipIndex, negative if absent
        xclOpenContext(handle, uuid, ipIndex, shared=False)
        xclRegRead / xclRegWrite(handle, ipIndex, byte offset, ...)
        xclCloseContext, xclClose          on the way out

    The context is not optional. xclRegRead's header says so in as many words
    -- "Caller should own an exclusive context on the IP obtained via
    xclOpenContext()" -- and a register read without one is a read whose
    failure would have nothing to do with the device.

    The UUID that context needs comes from the device and not from the file
    -- `_loaded_xclbin_uuid` below has the correction and the board that
    forced it. `xclbin` is still not optional, for a different reason now: it
    is what lets this link check that the file it was pointed at is the image
    the device is running, instead of a design nobody measured.

    WHY THERE ARE TWO 32-BIT ACCESSES PER 64-BIT REGISTER

    xclRegRead and xclRegWrite are 32-bit; every register in
    rtl/fpga/tnca_mmio.v is 64. So a register is two accesses, at `offset`
    for the low half and `offset + 4` for the high, which is the aperture
    rtl/fpga/tnca_axi_lite.v decodes.

    The WRITE ORDER is low then high, and it is not a preference:
    tnca_axi_lite latches the low half into a shadow register and lets the
    HIGH-half write complete the register write, composing {high,
    shadow_low}. High first completes a write with the previous value's low
    half still in the shadow -- which arrives as a register holding another
    register's low bits, and reads as a protocol bug.

    The READ ORDER is low then high and has no such rule, but it does have a
    tear: the device can update the register between the two accesses, and
    the halves would then come from two different values. That is not
    repaired here because it is repaired above. `U50Backend._read_reply`
    reads the sequence, then the fields, then the sequence again, and
    believes the set only if the number held still -- which is exactly the
    repair this needs, in the place that knows what the registers mean.
    A medium does not, and one that tried would be guessing.

    WHAT IT CANNOT DO

    Nothing here has been executed against a board. XRT's headers, `nm` on
    its library and xclbinutil on a shipped .xclbin are three documents and
    none of them is a device. Section 3.2 keeps engineering-ready and
    platform-verified apart and this is still the first.
    """

    def __init__(self, index=0, xclbin=None, ip_name=IP_NAME, what=None):
        self.index = index
        self.xclbin = xclbin
        self.ip_name = ip_name
        self._what = what or f'U50 index {index}'
        self._lib = None
        self._handle = None
        self._ip = None
        # Kept as an attribute and not a local: ctypes does not own the
        # buffer, and the context holds the pointer for as long as it is open.
        self._uuid = None
        # R40-04: the AXI window's buffer object, if one was allocated.
        self._bo = None
        self._bo_size = 0

    # ---- lifetime ---------------------------------------------------------
    def open(self):
        # The uuid the context needs is the device's own answer, so this no
        # longer depends on knowing where it sits in a file. What the file is
        # still good for is the question below it.
        uuid = _loaded_xclbin_uuid(self.index)

        if not self.xclbin:
            raise DependencyFailed(
                'the board link was not told which .xclbin is loaded',
                detail={'medium': self._what, 'xclbin': self.xclbin},
                remedy='the uuid itself now comes from the device, so this is '
                       'not about reaching it -- it is about checking that '
                       'the file named here is the image the device is '
                       'running, which cannot be done without being named '
                       'one. XDMALink(xclbin=<path>), or open the backend '
                       'with xclbin= that path')

        if not _file_carries_uuid(self.xclbin, uuid):
            raise DependencyFailed(
                f'{self.xclbin} is not the image the device is running',
                detail={'xclbin': self.xclbin, 'medium': self._what,
                        'xrt': _xrt_version()},
                remedy='that file does not carry the uuid of the loaded '
                       'image, so driving the device through it would be '
                       'driving a design nobody measured. Either load it, or '
                       'name the one that is loaded -- '
                       '/sys/class/drm/renderD*/device/xclbinuuid is what the '
                       'device says it is running')

        lib = _xrt_library()

        handle = lib.xclOpen(self.index, None, XCL_WARN)
        if not handle:
            errno = ctypes.get_errno()
            raise DependencyFailed(
                f'XRT would not open device {self.index}',
                detail={'index': self.index, 'errno': errno,
                        'xclbin': self.xclbin, 'xrt': _xrt_version()},
                remedy='either there is no card at that index, or the driver '
                       'is not loaded, or another process holds it. XRT '
                       'prints its own reason at the verbosity this opens '
                       'with, so the message above this one is the detail')
        self._lib = lib
        self._handle = handle

        ip = lib.xclIPName2Index(handle, self.ip_name.encode())
        if ip < 0:
            self._close_everything()
            raise DependencyFailed(
                f'the loaded xclbin has no IP called {self.ip_name!r}',
                detail={'ip_name': self.ip_name, 'rc': ip,
                        'xclbin': self.xclbin, 'medium': self._what},
                remedy='the name xclIPName2Index takes is "<kernel '
                       'name>:<instance name>", which is not the name in '
                       'kernel.xml. Dump the xclbin\'s IP_LAYOUT section '
                       'with xclbinutil and pass the name it lists: '
                       'XDMALink(xclbin=..., ip_name=...)')

        self._uuid = (ctypes.c_ubyte * 16)(*uuid)
        rc = lib.xclOpenContext(handle, self._uuid, ip, False)
        if rc != 0:
            self._close_everything()
            raise DependencyFailed(
                f'XRT would not give an exclusive context on '
                f'{self.ip_name!r} (returned {rc})',
                detail={'ip_name': self.ip_name, 'ipIndex': ip, 'rc': rc,
                        'xclbin': self.xclbin, 'medium': self._what},
                remedy='a context is refused when another client already '
                       'holds an exclusive one on the same IP, or when the '
                       'xclbin named here is not the one loaded on the '
                       'device. Both are worth knowing before a register '
                       'read fails for a reason that looks like the device')
        self._ip = ip
        return self

    def close(self):
        """Safe on a device that already died, and safe to call twice."""
        self._close_everything()

    # ---- R40-04: the AXI window's memory ------------------------------------
    def bo_open(self, nbytes, bank):
        """Allocate `nbytes` in memory index `bank`; returns its device address."""
        if self._handle is None:
            raise DeviceLost('the board is not open', detail={'medium': self._what})
        bo = self._lib.xclAllocBO(self._handle, nbytes, 0, bank)
        if bo == NULLBO:
            raise DependencyFailed(
                f'XRT would not allocate {nbytes} bytes in memory index {bank}',
                detail={'medium': self._what, 'bank': bank, 'bytes': nbytes},
                remedy='ask for less (TNCA_HBM_BYTES), or check the bank the '
                       'link connected m_axi_hbm to (xclbinutil --info)')
        props = _XclBOProperties()
        rc = self._lib.xclGetBOProperties(self._handle, bo, ctypes.byref(props))
        if rc != 0:
            self._lib.xclFreeBO(self._handle, bo)
            raise DependencyFailed('XRT gave no properties for the buffer it allocated',
                                   detail={'rc': rc, 'medium': self._what})
        self._bo, self._bo_size = bo, nbytes
        return props.paddr

    def bo_write(self, offset, data):
        n = len(data)
        if offset < 0 or offset + n > self._bo_size:
            raise ArgumentError(f'{n} bytes at {offset:#x} runs past the AXI window')
        buf = ctypes.create_string_buffer(bytes(data), n)
        if self._lib.xclWriteBO(self._handle, self._bo, buf, n, offset) != 0:
            raise DeviceLost('XRT refused to stage a write into the AXI window',
                             detail={'offset': offset, 'bytes': n, 'medium': self._what})
        rc = self._lib.xclSyncBO(self._handle, self._bo, XCL_BO_SYNC_BO_TO_DEVICE, n, offset)
        if rc != 0:
            raise DeviceLost('the DMA into the AXI window failed',
                             detail={'rc': rc, 'offset': offset, 'bytes': n, 'medium': self._what})

    def bo_read(self, offset, nbytes):
        if offset < 0 or offset + nbytes > self._bo_size:
            raise ArgumentError(f'{nbytes} bytes at {offset:#x} runs past the AXI window')
        rc = self._lib.xclSyncBO(self._handle, self._bo, XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, offset)
        if rc != 0:
            raise DeviceLost('the DMA out of the AXI window failed',
                             detail={'rc': rc, 'offset': offset, 'bytes': nbytes,
                                     'medium': self._what})
        buf = ctypes.create_string_buffer(nbytes)
        if self._lib.xclReadBO(self._handle, self._bo, buf, nbytes, offset) != 0:
            raise DeviceLost('XRT refused to read back the AXI window',
                             detail={'offset': offset, 'bytes': nbytes, 'medium': self._what})
        return buf.raw

    def _close_everything(self):
        if self._bo is not None and self._lib is not None and self._handle is not None:
            try:
                self._lib.xclFreeBO(self._handle, self._bo)
            except Exception:
                pass
        self._bo = None
        lib, self._lib = self._lib, None
        handle, self._handle = self._handle, None
        ip, self._ip = self._ip, None
        uuid, self._uuid = self._uuid, None
        if lib is None or handle is None:
            return
        try:
            if ip is not None and uuid is not None:
                lib.xclCloseContext(handle, uuid, ip)
        except Exception:
            # A context that cannot be closed is not a reason to leak the
            # device handle below, and neither failure has a caller who
            # could do anything with it.
            pass
        try:
            lib.xclClose(handle)
        except Exception:
            pass

    # ---- the four methods a U50Link needs ---------------------------------
    def read(self, offset):
        lo = self._read32(offset)
        hi = self._read32(offset + 4)
        return lo | (hi << 32)

    def write(self, offset, value):
        value = int(value) & 0xFFFF_FFFF_FFFF_FFFF
        # Low half first. See the class docstring: the high-half write is
        # what completes the register write, so it has to come second.
        self._write32(offset, value & 0xFFFF_FFFF)
        self._write32(offset + 4, (value >> 32) & 0xFFFF_FFFF)

    def describe(self):
        return self._what

    # ---- one 32-bit access ------------------------------------------------
    def _read32(self, offset):
        if self._handle is None:
            raise DeviceLost('the board is not open',
                             detail={'medium': self._what})
        out = ctypes.c_uint32(0)
        rc = self._lib.xclRegRead(self._handle, self._ip, offset,
                                  ctypes.byref(out))
        if rc != 0:
            raise DeviceLost(
                f'XRT refused a register read at {offset:#x} '
                f'(returned {rc})',
                detail={'medium': self._what, 'offset': offset, 'rc': rc,
                        'ip_name': self.ip_name, 'ipIndex': self._ip},
                remedy='a read that XRT refuses is not a read that returned '
                       'zero: nothing was transferred. If the device is '
                       'still there, this is the IP index or the offset, '
                       'and the two are independent -- the index comes from '
                       'the xclbin and the offset from rtl/fpga/tnca_mmio.v')
        return out.value

    def _write32(self, offset, word):
        if self._handle is None:
            raise DeviceLost('the board is not open',
                             detail={'medium': self._what})
        rc = self._lib.xclRegWrite(self._handle, self._ip, offset, word)
        if rc != 0:
            raise DeviceLost(
                f'XRT refused a register write at {offset:#x} '
                f'(returned {rc})',
                detail={'medium': self._what, 'offset': offset, 'rc': rc,
                        'value': word, 'ip_name': self.ip_name,
                        'ipIndex': self._ip},
                remedy='nothing was written. Whether the device is gone or '
                       'the offset is wrong is told apart by whether reads '
                       'at other offsets still work')


class XDMALink(U50Link):
    """The board itself. NOT EXERCISED, and this docstring says so first.

    Everything above it in this file is proven against the real RTL by
    `fpga-u50-sim`; this class has never run on hardware. It is here
    because a binding with no board link is a binding that cannot be
    finished, and because writing it down is what makes the one step that
    needs a board visible as one step.

    THAT STEP

    `_open_access()` -- and nothing else -- depends on the platform's
    control path, and it was written by reading that control path rather
    than by trying one. On the U50 the register file sits behind the
    shell's AXI-Lite control, and the way in is XRT's C API through
    `_XRTRegisters` above: xclOpen, xclIPName2Index, xclOpenContext, then
    xclRegRead and xclRegWrite at a byte offset, twice per 64-bit register.

    pyxrt was the other candidate and was not chosen. On this machine it
    is packaged for one Python only and without an `ip` class at all, so
    it was not a choice that could be made here anyway.

    What is left that only a board can settle:

      * that `ip_name` is the name this kernel is given at link time,
      * that an exclusive context on it is granted,
      * that the register file answers at the offsets the map names.

    `accessor=` still exists and is still the way past all three without
    editing this file. Failing to build one is a DependencyFailed naming
    what is missing -- never a fallback to the stand-in, which section
    3.4 forbids in as many words.
    """

    def __init__(self, xclbin=None, accessor=None, index=0,
                 words=DEFAULT_WORDS, what=None, ip_name=IP_NAME):
        self.xclbin = xclbin
        self.index = index
        self.words = words
        self.ip_name = ip_name
        self._accessor = accessor
        self._what = what or f'U50 index {index}'
        self._device = None

    def describe(self):
        return self._what

    def open(self):
        if self._accessor is None:
            self._accessor = self._open_access()
        return self

    def close(self):
        acc, self._accessor = self._accessor, None
        if acc is not None and hasattr(acc, 'close'):
            try:
                acc.close()
            except Exception:
                pass

    def alive(self):
        # "The handle is open", not "the board answers". Asking a board that
        # question costs a bus transaction, and every caller in this file
        # reads a register immediately after asking -- so the read is the
        # test, and a board that is gone is found by it raising DeviceLost.
        return self._accessor is not None

    def read_reg(self, offset):
        if self._accessor is None:
            raise DeviceLost('the board is not open',
                             detail={'medium': self._what})
        return self._accessor.read(offset) & 0xFFFF_FFFF_FFFF_FFFF

    def write_reg(self, offset, value):
        if self._accessor is None:
            raise DeviceLost('the board is not open',
                             detail={'medium': self._what})
        self._accessor.write(offset, int(value) & 0xFFFF_FFFF_FFFF_FFFF)

    # ---- R40-04: the AXI window's memory, through XRT ----------------------
    def hbm_open(self, nbytes):
        if self._accessor is None or not hasattr(self._accessor, 'bo_open'):
            return None
        bank = _xclbin_bank(self.xclbin, self.ip_name, arg_index=0)
        nbytes = min(int(nbytes), bank['size'])
        self.hbm_bank = bank
        return self._accessor.bo_open(nbytes, bank['index']), nbytes

    def hbm_write(self, offset, data):
        self._accessor.bo_write(offset, data)

    def hbm_read(self, offset, nbytes):
        return self._accessor.bo_read(offset, nbytes)

    def _open_access(self):
        """Build the platform accessor. The board-dependent step.

        Every call this makes was checked against the header that declares
        it and against `nm` on the library that exports it; `_XRTRegisters`
        above carries those notes. What no amount of reading settles is the
        board, and the class docstring lists the three things that are.

        The accessor is closed on the way out of a failure rather than left
        to the caller. `U50Backend.start()` does close the link when
        `open()` raises, but by the time it does, this method has already
        thrown and a device handle that was opened in here is not something
        the layer above can see. That is the same shape as the leak that
        left devices running before `link.open()` was moved inside that
        try -- found by looking for one with the driver long gone.
        """
        acc = _XRTRegisters(index=self.index, xclbin=self.xclbin,
                            ip_name=self.ip_name, what=self._what)
        try:
            acc.open()
        except BaseException:
            acc.close()
            raise
        return acc


def _xrt_version():
    """XRT's version as a string, or None. Reported, not required."""
    for p in ('/opt/xilinx/xrt/version.json',
              os.path.join(os.environ.get('XILINX_XRT', '/opt/xilinx/xrt'),
                           'version.json')):
        try:
            import json
            with open(p) as fh:
                return json.load(fh).get('Version')
        except Exception:
            continue
    return None


def board(**kw):
    """The factory `fpga-u50` is registered with. See backend.py.

    Here rather than registered at the bottom of this file, so that the
    registry stays one list in one place. A module that registered itself
    on import would also make `import tnca.backends.u50` a side effect,
    which is how a binding becomes unremovable.
    """
    kw.pop('bench', None)
    link = kw.pop('link', None)
    if link is None:
        link = XDMALink(xclbin=kw.pop('xclbin', None),
                        accessor=kw.pop('accessor', None),
                        index=kw.pop('index', 0),
                        # Named here so that a build that named its kernel
                        # differently is a caller's argument and not an
                        # edit to this file. See IP_NAME.
                        ip_name=kw.pop('ip_name', IP_NAME),
                        words=kw.pop('words', DEFAULT_WORDS))
    return U50Backend(link, **kw)
