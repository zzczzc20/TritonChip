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

    def capabilities(self):
        """What this binding can do, for a caller that needs to ask.

        Deliberately about the BINDING, not the hardware: the hardware's
        capabilities are the profile's business and are checked when an
        artifact is loaded.
        """
        return {'name': self.name, 'state': self.state,
                'launch_is_idempotent': self.launch_is_idempotent,
                'supports_cancel': False,
                'concurrent_requests': 1}


# ---------------------------------------------------------------------------
# the registry

_REGISTRY = {}


def register(name, factory, *, state=INTERFACE_DEFINED, what=''):
    """Make a backend openable by name."""
    if state not in STATES:
        raise ValueError(f"{state!r} is not one of {STATES}")
    if state == PLATFORM_VERIFIED:
        raise ValueError(
            "platform-verified is reserved: nothing in this repository "
            "has run on a real platform, and section 3.2 keeps "
            "engineering-ready and platform-verified apart")
    _REGISTRY[name] = {'factory': factory, 'state': state, 'what': what}


def available():
    """Every registered backend and its state."""
    return {n: {'state': e['state'], 'what': e['what']}
            for n, e in sorted(_REGISTRY.items())}


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
    register('sim-socket',
             _lazy('tnca.backends.socket_relay', 'SocketBackend'),
             state=REFERENCE_AVAILABLE,
             what='the same RTL, reached over a TCP socket through a '
                  'relay process: a different connection medium for the '
                  'same implementation')
    register('fpga-prototype', None, state=INTERFACE_DEFINED,
             what='a future board. The interface is defined and nothing '
                  'implements it; opening it is an error, not a '
                  'redirection to the simulator.')


_register_builtin()
