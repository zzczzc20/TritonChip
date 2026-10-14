"""Execution backends, and the boundary a platform branch extends at.

doc/phase3_6_tasks_and_acceptance.md sections 3.1 and 4.2.

Section 3.1 separates four things that had been one:

    a compile target/profile   ISA, ABI, capabilities, resources, costs
    an execution backend       device lifetime, requests, memory, faults
    a hardware implementation  the RTL, its ports, latencies, resources
    a platform binding         a simulator, a board shell, a build

This module owns the second. What it is NOT allowed to be is the first:
a backend does not compile anything, does not own the buffer or event
state machine, and never quietly edits a kernel. The runtime keeps all
of that; a backend moves bytes and starts work.

Section 3.4 also asks a registry to distinguish three states, because
they are not the same claim and a stage that blurs them is claiming
more than it did:

    interface-defined     the contract exists; nothing implements it
    reference-available   an implementation runs here, through this
                          interface, on real hardware or a protocol
                          loop to it
    platform-verified     a real platform ran it. RESERVED -- nothing
                          in this repository may claim it.

An unknown, missing or unimplemented backend is a stable diagnostic.
Section 3.4: it "不能自动转到仿真并返回成功" -- falling back to the
simulator and reporting success is the specific thing forbidden, because
then a platform that was never there appears to work.
"""
import abc

from .errors import TncaError

INTERFACE_DEFINED = 'interface-defined'
REFERENCE_AVAILABLE = 'reference-available'
PLATFORM_VERIFIED = 'platform-verified'

STATES = (INTERFACE_DEFINED, REFERENCE_AVAILABLE, PLATFORM_VERIFIED)


class BackendError(TncaError):
    """A backend could not be opened, or does not exist."""


#: The device's word. A transfer is addressed in bytes and moved in
#: words, so an address that is not a multiple of this one cannot be
#: expressed on the wire.
WORD = 8


def check_transfer(addr, nbytes, *, what='transfer'):
    """The argument rules every backend shares.

    Shared, rather than implemented once per binding. The file-mailbox
    transport checked alignment and the socket binding did not, so the
    same illegal call was ARGUMENT_INVALID on one backend and, on the
    other, went out on the wire, made the device raise, and took the
    connection down with it. One rule in one place is the fix; the
    conformance suite is what found it.
    """
    if int(addr) < 0:
        raise TncaError(f"{what} address {addr} is negative",
                        code='ARGUMENT_INVALID')
    if int(addr) % WORD:
        raise TncaError(
            f"{what} address {addr:#x} is not {WORD}-byte aligned",
            code='ARGUMENT_INVALID')
    if int(nbytes) <= 0:
        raise TncaError(f"a {what} of {nbytes} byte(s)",
                        code='ARGUMENT_INVALID',
                        remedy='a transfer of nothing is a mistake, not a '
                               'fast path')


class Backend(abc.ABC):
    """One device, one epoch, one connection.

    The lifecycle section 4.2 requires: discovery and capability query,
    connect and close, the visible range of an allocation, upload and
    download, load and launch, wait and status, drain, errors and
    explicit recovery.

    What it does not get to decide: whether a launch is retried. Section
    4.2 is explicit that a timeout must not resend a launch, because a
    kernel that runs twice is worse than one that did not finish.
    """

    #: What this implementation binds to. Not a profile: a profile is a
    #: compile target, and section 3.1 forbids treating a target as a
    #: particular executable or board.
    name = 'abstract'
    state = INTERFACE_DEFINED
    #: Does a launch survive being retried? Answering yes needs evidence.
    launch_is_idempotent = False

    @abc.abstractmethod
    def start(self):
        """Connect. Returns self, or raises with a reason."""

    @abc.abstractmethod
    def stop(self):
        """Close. Must be safe to call on a device that already died."""

    @abc.abstractmethod
    def alive(self):
        """True while the device can still answer."""

    @abc.abstractmethod
    def upload(self, addr, data):
        """Host to device. Returns a Reply."""

    @abc.abstractmethod
    def download(self, addr, nbytes):
        """Device to host. Returns (Reply, bytes)."""

    @abc.abstractmethod
    def launch(self, pc, arg_base, role_mask=1, pid=(0, 0, 0),
               npid=(1, 1, 1)):
        """Start one program. Returns a Reply carrying its cycles."""

    @abc.abstractmethod
    def quiesce(self):
        """Drain: every accepted request finished, nothing in flight."""

    @abc.abstractmethod
    def ping(self):
        """Is it there, and how much has it served? Returns a Reply.

        Part of the interface rather than one binding's convenience: the
        conformance suite needs a request with no side effect to bracket
        the ones that have them, and a binding that cannot answer "are
        you there" cannot be checked for having stayed there.
        """

    def perf_read(self, index):
        """One device counter (spec/r38/counters.json), or None.

        Not abstract: a binding with no counter path is a legal binding,
        and it must say "unavailable" -- None -- rather than a number.
        A zero here would be a measurement.
        """
        return None

    def capabilities(self):
        """What this binding can do, for a caller that needs to ask.

        Deliberately about the BINDING, not the hardware: the hardware's
        capabilities are the profile's business and are checked when an
        artifact is loaded.
        """
        return {'name': self.name, 'state': self.state,
                'launch_is_idempotent': self.launch_is_idempotent,
                'supports_cancel': False,
                'concurrent_requests': 1,
                # Whether a caller can put an arbitrary opcode on the
                # wire. The file mailbox can; a structured protocol
                # cannot, and says so rather than having the conformance
                # suite discover it as an AttributeError.
                'raw_request': False}


# ---------------------------------------------------------------------------
# the registry

_REGISTRY = {}


def register(name, factory, *, state=INTERFACE_DEFINED, what='',
             needs_elaborator=True, needs_hardware=False, prepare_bench=None, physical_grid=False):
    """Make a backend openable by name.

    `needs_elaborator` says whether opening this backend means reading the
    profile's own simulator build. True for everything that drives a
    compiled artifact of the profile; False for a device reached some
    other way. A board over PCIe has no simulator build to read, and
    asking for one on its behalf is what made
    `Session(backend='fpga-u50')` raise "no simulator for profile ..." on
    precisely the machine that had the board and not the simulator.

    `needs_hardware` says whether opening it means a physical device
    being present. It is not the same claim as `needs_elaborator` and
    neither implies the other: the simulator bindings need no hardware
    but do read a build, and a board needs the hardware and reads
    nothing. It is here because a reference-available binding is
    otherwise held to "runnable wherever this is installed", and the
    conformance suite enforces that by opening every one of them. A
    board cannot meet that, and the suite had no way to be told so.

    The defaults are the ones a backend had before these existed, so a
    binding that says nothing about either keeps the behaviour it had.
    """
    if state not in STATES:
        raise ValueError(f"{state!r} is not one of {STATES}")
    if state == PLATFORM_VERIFIED:
        raise ValueError(
            "platform-verified is reserved: nothing in this repository "
            "has run on a real platform, and section 3.2 keeps "
            "engineering-ready and platform-verified apart")
    _REGISTRY[name] = {'factory': factory, 'state': state, 'what': what,
                       'needs_elaborator': bool(needs_elaborator),
                       'needs_hardware': bool(needs_hardware),
                       'prepare_bench': prepare_bench, 'physical_grid': bool(physical_grid)}


def available():
    """Every registered backend and its state."""
    return {n: {'state': e['state'], 'what': e['what']}
            for n, e in sorted(_REGISTRY.items())}


def needs_elaborator(name):
    """Whether opening `name` means reading the profile's simulator.

    An unknown name answers True, and deliberately. The caller is about to
    be told the name is unknown, and it should hear that from
    `open_backend`, which says so plainly, rather than from
    `device.require`, which would say there is no simulator and send
    somebody looking for the wrong problem.
    """
    entry = _REGISTRY.get(name)
    return True if entry is None else entry['needs_elaborator']


def needs_hardware(name):
    """Whether opening `name` means a physical device being present.

    An unknown name answers False, the opposite of `needs_elaborator`
    above, and for the same kind of reason: a caller about to be told
    the name is unknown should not also be told its hardware is
    missing. False says no more than "this is not a reason to excuse
    it", which leaves `open_backend` to give the plain answer.
    """
    entry = _REGISTRY.get(name)
    return False if entry is None else entry['needs_hardware']


def physical_grid(name):
    entry = _REGISTRY.get(name)
    return bool(entry and entry['physical_grid'])


def prepare_bench(name, profile, **kwargs):
    entry = _REGISTRY.get(name)
    if entry and entry['prepare_bench']:
        return entry['prepare_bench'](profile=profile, **kwargs)
    from . import device
    return device.require(profile) if needs_elaborator(name) else None


def open_backend(name, **kwargs):
    """Open one by name, or say plainly why not.

    Never falls back. A caller who asked for a backend and got the
    simulator would be told a platform works when it was never there.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        raise BackendError(
            f"there is no backend named {name!r}",
            code='BACKEND_UNKNOWN',
            detail={'known': sorted(_REGISTRY)},
            remedy='name one of the backends listed, or register yours '
                   'before opening it. Nothing is substituted: a missing '
                   'backend that silently became the simulator would '
                   'report a platform working that was never there.')
    if entry['state'] == INTERFACE_DEFINED:
        raise BackendError(
            f"the backend {name!r} is declared and not implemented",
            code='BACKEND_NOT_IMPLEMENTED',
            detail={'state': entry['state'], 'what': entry['what']},
            remedy='an interface-defined backend is a contract with no '
                   'implementation behind it yet')
    return entry['factory'](**kwargs)


def _lazy(module, attr):
    """Resolve an implementation when it is opened, not when imported.

    An implementation imports this module for the interface, so
    importing it back here at load time is a cycle. It also means a
    backend whose dependencies are missing costs nothing until somebody
    asks for it, and then says what is missing.
    """
    def factory(**kwargs):
        import importlib
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            raise BackendError(
                f"the backend's implementation {module} will not import: "
                f"{e}",
                code='BACKEND_UNAVAILABLE',
                remedy='its dependencies are not here. Nothing is '
                       'substituted.') from e
        return getattr(mod, attr)(**kwargs)
    return factory


def _register_builtin():
    """The bindings this repository ships.

    Both drive the same RTL. They differ in how the host reaches it,
    which is the boundary section 4.1 asks to be replaceable: "同一语义
    和实现的不同 transport binding，应能使用同一 artifact".
    """
    register('sim-file', _lazy('tnca.transport', 'Transport'),
             state=REFERENCE_AVAILABLE,
             what='the simulated device over a file rendezvous, which is '
                  'what this repository has always used')
    register('sim-multicore', _lazy('tnca.backends.multicore_file', 'MulticoreFileBackend'),
             state=REFERENCE_AVAILABLE, what='physical-core coherent mesh over a persistent mailbox',
             prepare_bench=_lazy('tnca.backends.multicore_file', 'prepare'), physical_grid=True)
    register('sim-socket',
             _lazy('tnca.backends.socket_relay', 'SocketBackend'),
             state=REFERENCE_AVAILABLE,
             what='the same RTL, reached over a TCP socket through a '
                  'relay process: a different connection medium for the '
                  'same implementation')
    # The U50 platform binding, in two parts that share one backend.
    #
    # `fpga-u50` is the board: rtl/fpga/tnca_fpga_top.v over its register
    # port, through an XDMALink. `fpga-u50-sim` is the same backend, the
    # same protocol and the same RTL, with a simulator standing in for the
    # board -- section 3.4's "a protocol loop to it", which is what makes
    # the board binding reference-available rather than platform-verified.
    #
    # Both are lazy for the usual reason: u50_sim imports a simulator and
    # u50 imports XRT, and neither should cost anything until it is asked
    # for. `fpga-prototype` above is untouched and stays interface-defined
    # -- that is the name whose contract is "declared and not implemented",
    # and giving it an implementation is the thing the state is there to
    # prevent.
    register('fpga-u50', _lazy('tnca.backends.u50', 'board'),
             state=REFERENCE_AVAILABLE, needs_elaborator=False,
             needs_hardware=True,
             what='the TNCA device on an Alveo U50: rtl/fpga/tnca_fpga_top.v '
                  'reached over its register port. The host side is proven '
                  'against the RTL; the board link is not, and opening this '
                  'with no board raises rather than substituting the '
                  'stand-in')
    # The same kernel and register protocol on an Alveo U280 (flow/fpga/
    # boards.sh, TNCA_BOARD=u280): one backend, a second name, so a record
    # says which card produced it.
    register('fpga-u280', _lazy('tnca.backends.u50', 'board'),
             state=REFERENCE_AVAILABLE, needs_elaborator=False,
             needs_hardware=True,
             what='the TNCA device on an Alveo U280: the U50 kernel and '
                  'register port (rtl/fpga/tnca_fpga_top.v) built for the '
                  'U280 shell, reached through the same U50Backend')
    register('fpga-u50-sim', _lazy('tnca.backends.u50_sim', 'stand_in'),
             state=REFERENCE_AVAILABLE, needs_elaborator=False,
             what='the same device and the same register protocol, on real '
                  'RTL, driven by a simulator standing in for the board')
    register('fpga-prototype', None, state=INTERFACE_DEFINED,
             what='a future board. The interface is defined and nothing '
                  'implements it; opening it is an error, not a '
                  'redirection to the simulator.')


_register_builtin()
