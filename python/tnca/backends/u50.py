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
                 trace=False, **kw):
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
            # The window's byte enables belong to this session, not to
            # whatever the last one left them as.
            self.link.write_reg(R_HOST_BE, BE_ALL)
            # The stand-in for sim-file's "await sequence 0", and a stronger
            # claim than it is: a device that reads its identity back and
            # then answers a ping has a working command path, where a
            # process that started has a working process.
            self.request(OP_PING)
        except BaseException:
            self.opened = False
            self.link.close()
            raise
        return self

    def alive(self):
        return self.opened and self.link.alive()

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
        it, because `self.seq` starts at one, so "nothing yet" and "the
        previous answer is still standing" are the same observation and
        `_await` waits on both.
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

    def _put_payload(self, addr, words):
        self._check_window(addr, len(words), 'upload')
        for i, word in enumerate(words):
            self._window(addr + i * WORD, word, write=True)

    def _get_payload(self, addr, nwords):
        self._check_window(addr, nwords, 'download')
        out = bytearray()
        for i in range(nwords):
            got = self._window(addr + i * WORD, None, write=False)
            out += got.to_bytes(WORD, 'little')
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
                    'profile': 'core-vrf-sync-v1',
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

#: XRT's verbosity, from `enum xclVerbosityLevel` in deprecated/xrt.h:
#: XCL_QUIET = 0, XCL_INFO = 1, XCL_WARN = 2, XCL_ERROR = 3.
#:
#: WARN and not INFO, which is talkative about enumerating devices on every
#: open, and not QUIET, which is also quiet about the reason a register access
#: was refused -- which is the one thing worth hearing from XRT here.
XCL_WARN = 2

#: Where the loaded xclbin's UUID is inside the xclbin file, and how that is
#: known.
#:
#: xclOpenContext wants "UUID of the xclbin image running on the device", and
#: no getter for it exists: xclDeviceInfo2 has no uuid field and there is no
#: XclbinId function (both checked in the headers rather than remembered).
#: XRT's own accessor is xclbin_uuid() in experimental/xclbin_util.h -- and it
#: is `static inline`, so it is not an exported symbol and ctypes cannot reach
#: it. Its body is four lines: check the magic, then copy the header's m_uuid.
#: This is those four lines.
#:
#: 336 is not read out of a struct definition. It is where the UUID that
#: xclbinutil --info reports for a shipped .xclbin was actually found, by
#: searching the file's first kilobyte for those sixteen bytes; they matched in
#: straight order at 336, which is axlf.m_header at 304 plus m_uuid at 32. The
#: magic check above it is the guard, so a file that is not an xclbin is
#: refused rather than parsed at an offset that would then be meaningless.
XCLBIN_MAGIC = b'xclbin2'
XCLBIN_UUID_OFFSET = 336

#: libxrt_core.so, loaded once. Module state and not a class attribute,
#: because it is one library for the process and a second copy would be a
#: second set of signatures to keep right.
_LIB = None


def _xclbin_uuid(path):
    """The sixteen bytes xclOpenContext wants, read out of the file."""
    with open(path, 'rb') as fh:
        head = fh.read(XCLBIN_UUID_OFFSET + 16)
    if head[:7] != XCLBIN_MAGIC:
        raise DependencyFailed(
            f'{path} does not start with "xclbin2"',
            detail={'path': path, 'found': repr(head[:8]),
                    'xrt': _xrt_version()},
            remedy='xclOpenContext takes the UUID of the xclbin loaded on '
                   'the device, and the only place that UUID can be read '
                   'from is the file itself. This is not one')
    return head[XCLBIN_UUID_OFFSET:XCLBIN_UUID_OFFSET + 16]


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

    The UUID that context needs is why `xclbin` is not optional either. Read
    the four lines in `_xclbin_uuid` above for why it can only come from the
    file, and note what that asserts: the file named here is the image loaded
    on the board. If it is not, xclOpenContext refuses -- which is the loud
    version of that mistake and not a quiet one.

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

    # ---- lifetime ---------------------------------------------------------
    def open(self):
        if not self.xclbin:
            raise DependencyFailed(
                'the board link was not told which .xclbin is loaded',
                detail={'medium': self._what, 'xclbin': self.xclbin},
                remedy='xclOpenContext takes the UUID of the image running '
                       'on the device, and XRT hands that UUID out nowhere '
                       '-- not from xclDeviceInfo2 and not from any getter. '
                       'It is read out of the file, so the file has to be '
                       'named: XDMALink(xclbin=<path>), or open the backend '
                       'with xclbin= that path')

        uuid = _xclbin_uuid(self.xclbin)
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

    def _close_everything(self):
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
