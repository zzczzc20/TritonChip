"""
The runtime session -- doc/phase2_tasks_and_acceptance.md R2-08.

    allocate -> upload -> load -> submit -> wait -> download -> free

One session owns one device, one address space, one queue and one event table.
The state machine is section 3.4's, and the states are not decoration: they
are the difference between "the task finished" and "the host stopped looking".

    SUBMITTED -> VALIDATED -> QUEUED -> RUNNING -> DRAINING -> SUCCEEDED
                        a fault -> FAILING -> DRAINING -> FAILED
                    undrainable -> DEVICE_LOST  (the session is POISONED)

Four rules live here because nowhere else can enforce them.

**Admission happens before anything starts.** A bad argument, a buffer that is
too small, an artifact for another profile -- each is refused with no RUNNING
event and no side effect on the device. A runtime that starts a kernel and
then notices is a runtime whose failures have already written somewhere.

**The argument block is a snapshot.** It is encoded at submit and never read
again, so a caller who mutates their dict afterwards -- or reuses a scratch
object for the next submit -- cannot change a task that has already been
accepted (ABI-04).

**A lease is counted, not guessed.** A buffer an unfinished task named may not
be freed, uploaded to, or downloaded from; it answers BUFFER_BUSY and does not
wait, because implicitly waiting turns a programming error into a hang.

**A host wait is not a verdict.** `wait(timeout=...)` expiring means the host
stopped looking. The task keeps running, it is not cancelled, and it is
certainly not successful. The device has its own, separate deadline, and that
one is a fault.
"""
import json
import os
import struct
import threading
import time
from collections import OrderedDict

from . import _data, abi, artifact, manifest as M
from . import transport as T
from .tensor import TensorDescriptor
from .errors import (AdmissionError, ArgumentError, ArtifactError,
                     BufferBusy, BufferError, CapabilityError, DependencyFailed,
                     DeviceLost, ProfileMismatch, QueueFull, SessionBusy,
                     SessionPoisoned, TncaError, Timeout, VersionError)

WORD = 8
MAX_UNFINISHED = 8          # the contract's floor; this is exactly it
DEFAULT_ALIGN = 64

# States
SUBMITTED, VALIDATED, QUEUED = 'SUBMITTED', 'VALIDATED', 'QUEUED'
RUNNING, DRAINING = 'RUNNING', 'DRAINING'
SUCCEEDED, FAILING, FAILED = 'SUCCEEDED', 'FAILING', 'FAILED'
TERMINAL = {SUCCEEDED, FAILED}

class Buffer:
    """A device allocation. The handle a pointer argument takes."""

    __slots__ = ('id', 'addr', 'size', 'reserved', 'session_id', 'epoch',
                 'alive', 'leases', 'valid', 'name')

    def __init__(self, bid, addr, size, session_id, epoch, name=None,
                 reserved=None):
        self.id = bid
        self.addr = addr
        self.size = size
        # What the allocator actually set aside, which is `size` rounded up
        # to the alignment. Kept so `free` returns the whole block rather
        # than leaking the padding on every allocation.
        self.reserved = reserved if reserved is not None else size
        self.session_id = session_id
        self.epoch = epoch
        self.alive = True
        self.leases = 0
        # Written by a task that FAILED. The contract does not promise a
        # rollback; it promises that such a buffer is not silently consumed
        # as if it held a result.
        self.valid = True
        self.name = name or f"buf{bid}"

    def __repr__(self):
        return (f"<Buffer {self.name} {self.size}B @{self.addr:#x}"
                f"{'' if self.alive else ' FREED'}"
                f"{'' if self.valid else ' INVALID'}>")


class Event:
    """One submitted task. Exactly one terminal state, ever."""

    def __init__(self, job_id, epoch, kernel):
        self.job_id = job_id
        self.epoch = epoch
        self.kernel = kernel
        self.state = SUBMITTED
        self.error = None
        self.history = [(SUBMITTED, time.time())]
        self.device_cycles = None
        # One entry per program, in linear order. `device_cycles` is their
        # sum because the runtime launches a grid one program at a time;
        # that sum is serial-grid time, not the makespan of a device that
        # runs programs concurrently.
        self.program_cycles = None
        # Session(counters=True): {'available', 'programs', 'total'}; None
        # when counters were not asked for. See _counter_summary.
        self.counters = None
        self.done = threading.Event()

    def _to(self, state, error=None):
        if self.state in TERMINAL:
            # The one invariant worth an assertion: a second terminal state
            # would mean two answers to "did it work?".
            raise TncaError(
                f"job {self.job_id} is already {self.state}; refusing to "
                f"move it to {state}", code='INTERNAL')
        self.state = state
        if error is not None:
            self.error = error
        self.history.append((state, time.time()))
        if state in TERMINAL:
            self.done.set()

    @property
    def terminal(self):
        return self.state in TERMINAL

    @property
    def ok(self):
        return self.state == SUCCEEDED

    def __repr__(self):
        return (f"<Event job={self.job_id} epoch={self.epoch} "
                f"{self.kernel!r} {self.state}"
                + (f" {self.error.code}" if self.error else '') + ">")


class Module:
    """A loaded artifact: its code is on the device and its manifest checked."""

    def __init__(self, path, manifest, code_addr, params, epoch):
        self.path = path
        self.manifest = manifest
        self.code_addr = code_addr
        self.params = params
        self.epoch = epoch
        self.name = manifest['identity']['kernel_name']

    def __repr__(self):
        return f"<Module {self.name!r} @{self.code_addr:#x}>"


class _Allocator:
    """A first-fit allocator over the device's address space.

    Deliberately simple and deliberately NOT reusing a freed block
    immediately: reuse-after-free is the bug this layer is supposed to make
    impossible to hit by accident, and an allocator that hands the same
    address straight back makes a stale handle look like a live one.
    """

    def __init__(self, base, limit):
        self.base = base
        self.limit = limit
        self.next = base
        self.free_list = []

    def alloc(self, size, align=DEFAULT_ALIGN):
        if size <= 0:
            raise BufferError(
                f"allocation of {size} bytes",
                remedy='a zero-byte buffer has no address that means '
                       'anything; allocate at least one byte')
        need = (size + align - 1) // align * align
        for i, (addr, blocksize) in enumerate(self.free_list):
            aligned = (addr + align - 1) // align * align
            prefix = aligned - addr
            if blocksize >= prefix + need:
                self.free_list.pop(i)
                if prefix:
                    self.free_list.append((addr, prefix))
                if blocksize > prefix + need:
                    self.free_list.append((aligned + need, blocksize - prefix - need))
                return aligned, need
        addr = (self.next + align - 1) // align * align
        if addr + need > self.limit:
            raise BufferError(
                f"out of device memory: {size} bytes needs {need} and "
                f"{self.limit - addr} remain",
                remedy='free a buffer, or use a profile with more memory')
        self.next = addr + need
        return addr, need

    def release(self, addr, size):
        self.free_list.append((addr, size))


def _open_device(name, bench, **kwargs):
    """The one place the runtime learns which device it is talking to.

    Phase 3.6 R36-BE-01: the general runtime holds no simulator or board
    startup logic. It names a backend and the registry resolves it, so
    an external branch registers an implementation rather than editing
    the state machine below.
    """
    from . import backend as _backend
    return _backend.open_backend(name, bench=bench, **kwargs)


def _counter_spec():
    with open(_data.path('spec', 'r38', 'counters.json')) as fh:
        spec = json.load(fh)
    return sorted(spec['counters'], key=lambda c: c['index'])


def _counter_summary(deltas):
    """Per-program counter differences and their sum over the grid.

    The sum of a counter over programs is a total of events. For the
    cycle-like counters it is serial-grid time, like device_cycles, and
    not the makespan of a device that runs programs concurrently.
    """
    total = {}
    for d in deltas:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return {'available': True, 'timing': 'serial-grid',
            'programs': deltas, 'total': total}


class Session:
    """One device, one address space, one queue."""

    _counter = 0

    def __init__(self, profile='core-sim', *, bench=None, workdir=None,
                 code_base=0, heap_base=0x20000, heap_limit=0x80000,
                 deadline_cycles=2_000_000, host_timeout=600.0, trace=False,
                 backend='sim-file', counters=False):
        Session._counter += 1
        self.id = Session._counter
        self.profile = profile
        # Phase 3.6 section 3.1: the profile is a COMPILE target and the
        # backend is how this host reaches a device. They were one thing
        # and are now two, because a target that meant "this simulator
        # executable" could never describe a board.
        self.backend_name = backend
        self.epoch = 0
        self.poisoned = False
        self.closed = False

        # Located, never built. R2-02 allows the simulator to be built at
        # install time or by an explicit `prepare`, and forbids building it
        # per launch; `device.locate` looks and `device.prepare` builds, so
        # there is no flag that could default the wrong way.
        from . import device as _device
        self.bench = bench or _device.require(profile)

        self._workdir = workdir
        self._deadline = deadline_cycles
        # Recorded rather than derived: a replay has to be told what the
        # original run was given, and a seed nobody wrote down is a seed
        # nobody can reuse.
        self.seeds = {}
        self._host_timeout = host_timeout
        self._trace = trace
        # Off by default. On, every program's launch is bracketed by a read
        # of each device counter and the event carries the differences.
        # The reads happen while the device is idle, outside the window
        # done_cycles measures, so they add host time and no device cycles.
        self._counters = _counter_spec() if counters else None
        self._lock = threading.RLock()
        self._buffers = OrderedDict()
        self._modules = []
        self._queue = []
        self._events = OrderedDict()
        self._next_buffer = 0
        self._next_job = 0
        self._code_next = code_base
        self._heap = _Allocator(heap_base, heap_limit)
        self._worker = None
        self._stop = threading.Event()

        self.dev = _open_device(self.backend_name, self.bench,
                                workdir=self._workdir,
                                deadline_cycles=deadline_cycles,
                                host_timeout=host_timeout, trace=trace,
                                epoch=self.epoch)
        self.dev.start()
        self._start_worker()

    # ---- housekeeping -----------------------------------------------------
    def _snapshot(self):
        """Every counter now, by name; None if the device cannot say."""
        out = {}
        for c in self._counters:
            v = self.dev.perf_read(c['index'])
            if v is None:
                return None
            out[c['name']] = v
        return out

    def _check_usable(self):
        if self.closed:
            raise TncaError('this session is closed', code='SESSION_BUSY')
        if self.poisoned:
            raise SessionPoisoned(
                'the device was lost and this session is poisoned',
                detail={'epoch': self.epoch},
                remedy='call recover() to start a new epoch; every handle '
                       'from this one is invalid')

    def _start_worker(self):
        self._stop.clear()
        self._worker = threading.Thread(target=self._run_queue, daemon=True,
                                        name=f"tnca-session-{self.id}")
        self._worker.start()

    @property
    def unfinished(self):
        with self._lock:
            return [e for e in self._events.values() if not e.terminal]

    # ---- buffers ----------------------------------------------------------
    def allocate(self, nbytes, *, name=None):
        self._check_usable()
        with self._lock:
            addr, size = self._heap.alloc(nbytes)
            self._next_buffer += 1
            buf = Buffer(self._next_buffer, addr, nbytes, self.id,
                         self.epoch, name, reserved=size)
            self._buffers[buf.id] = buf
            return buf

    def _validate_buffer(self, buf, *, why):
        if not isinstance(buf, Buffer):
            raise BufferError(f"{why}: not a buffer handle")
        if buf.session_id != self.id:
            raise BufferError(
                f"{why}: that handle belongs to session {buf.session_id}, "
                f"this is session {self.id}")
        if buf.epoch != self.epoch:
            raise BufferError(
                f"{why}: that handle is from epoch {buf.epoch} and this "
                f"session is on epoch {self.epoch}",
                remedy='every handle from before recover() is invalid')
        if not buf.alive:
            raise BufferError(f"{why}: {buf.name} has been freed")
        if buf.leases:
            raise BufferBusy(
                f"{why}: {buf.name} is named by {buf.leases} unfinished "
                f"task(s)",
                remedy='wait for them; this does not wait implicitly, '
                       'because doing so turns a mistake into a hang')

    def upload(self, buf, data, offset=0):
        self._check_usable()
        with self._lock:
            self._validate_buffer(buf, why='upload')
            if offset < 0 or offset + len(data) > buf.size:
                raise BufferError(
                    f"upload of {len(data)} bytes at offset {offset} runs "
                    f"past {buf.name} ({buf.size} bytes)")
            reply = self.dev.upload(buf.addr + offset, data)
            if not reply.ok:
                raise TncaError(f"upload refused: {reply}",
                                code='BUFFER_INVALID')
            buf.valid = True
            return len(data)

    def download(self, buf, nbytes=None, offset=0):
        self._check_usable()
        with self._lock:
            self._validate_buffer(buf, why='download')
            nbytes = buf.size - offset if nbytes is None else nbytes
            if offset < 0 or offset + nbytes > buf.size:
                raise BufferError(
                    f"download of {nbytes} bytes at offset {offset} runs "
                    f"past {buf.name} ({buf.size} bytes)")
            reply, data = self.dev.download(buf.addr + offset, nbytes)
            if not reply.ok:
                raise TncaError(f"download refused: {reply}",
                                code='BUFFER_INVALID')
            return data

    def free(self, buf):
        self._check_usable()
        with self._lock:
            self._validate_buffer(buf, why='free')
            buf.alive = False
            self._heap.release(buf.addr, buf.reserved)
            return True

    # ---- modules ----------------------------------------------------------
    @property
    def deadline_cycles(self):
        """What a launch is given before it is declared stuck."""
        return self._deadline

    def task_of(self, event):
        """The task record behind an event, or None.

        Events are handed to the caller; tasks are not. This is the one way
        back, and it is here so `failure.capture` does not have to reach
        into the queue.
        """
        # Not a search through the queue: a finished task is popped from
        # it, and a failure is captured after the failure. The event keeps
        # the reference, and the session still checks the event is its own
        # -- events are not portable between sessions.
        task = getattr(event, '_task', None)
        if task is None or getattr(event, 'epoch', None) != self.epoch:
            return None
        return task

    def capture_failure(self, event, outdir, *, note=None):
        """Save this launch as a replayable bundle. See tnca.failure."""
        from . import failure
        return failure.capture(self, event, outdir, note=note)

    def load(self, path):
        """Check an artifact against THIS device, then place its code."""
        self._check_usable()
        man, segments = artifact.load(path)

        want = man['identity']['abi_version']
        if not abi.supports(tuple(want)):
            raise VersionError(
                f"artifact wants argument ABI {want[0]}.{want[1]}, and this "
                f"runtime implements {abi.ABI_VERSION[0]}.{abi.ABI_VERSION[1]}",
                where=f"{os.path.basename(path)} "
                      f"({man['identity'].get('kernel_name', '?')})",
                detail={'artifact': list(want),
                        'runtime': list(abi.ABI_VERSION)},
                remedy='recompile the kernel with this toolchain. A higher '
                       'major means the argument block is laid out '
                       'differently, and reading it anyway would put the '
                       'wrong bytes in every parameter')
        # A memory mode is a condition of the run, not another machine:
        # 'core-sim/slow' loads artifacts compiled for 'core-sim'. The
        # architecture, the resources and the fingerprint are the base
        # profile's, and the comparison is about those.
        from . import device as _device
        from . import profiles as _profiles
        _profiles.check_executes(_device.base_profile(self.profile), man,
                                 where=os.path.basename(path))

        params = [(p['name'], p['type'])
                  for p in man['parameters']['signature']]
        code = segments.get('text')
        if not code:
            raise ArtifactError('the artifact has no code segment')

        with self._lock:
            addr = self._code_next
            if addr + len(code) > self._heap.base:
                raise BufferError('compiled code overlaps the device heap; increase heap_base')
            reply = self.dev.upload(addr, code)
            if not reply.ok:
                raise TncaError(f"could not place code: {reply}",
                                code='ARTIFACT_INVALID')
            self._code_next = (addr + len(code) + 63) // 64 * 64
            mod = Module(path, man, addr, params, self.epoch)
            self._modules.append(mod)
            return mod

    # ---- submission -------------------------------------------------------
    def submit(self, module, args, *, grid=(1, 1, 1), depends_on=()):
        self._check_usable()
        with self._lock:
            if module.epoch != self.epoch:
                raise AdmissionError(
                    'that module was loaded in a previous epoch')
            if len(self.unfinished) >= MAX_UNFINISHED:
                raise QueueFull(
                    f"{MAX_UNFINISHED} tasks are already unfinished",
                    detail={'limit': MAX_UNFINISHED},
                    remedy='wait for one to finish; this does not block and '
                           'does not discard an older task')

            gx, gy, gz = (int(g) for g in grid)
            for axis, g in zip('xyz', (gx, gy, gz)):
                if g < 0 or g > 0xFFFFFFFF:
                    raise AdmissionError(
                        f"grid.{axis} = {g} is outside 0..2**32-1")
            total = gx * gy * gz
            if total > 4096:
                raise AdmissionError(
                    f"{total} programs exceeds the 4096 limit",
                    where=f"{module.manifest['identity']['kernel_name']} "
                          f"grid {gx}x{gy}x{gz}",
                    detail={'requested': total, 'limit': 4096,
                            'grid': [gx, gy, gz]},
                    remedy='split the work across several launches; the '
                           'limit is the grid the transport encodes, '
                           'sixteen bits per axis with a 4096 cap')

            # Resolve and validate every argument BEFORE anything starts.
            resolved, leased = self._resolve(module, args)

            self._next_job += 1
            ev = Event(self._next_job, self.epoch, module.name)
            ev._to(VALIDATED)

            for dep in depends_on:
                # IDENTITY, not the job id. Job ids start at 1 in every
                # session, so `dep.job_id in self._events` was true for
                # another session's first task and a cross-session
                # dependency was accepted -- which would have made this
                # session wait on an event it does not own and cannot see
                # fail.
                if not isinstance(dep, Event) or \
                        self._events.get(dep.job_id) is not dep:
                    raise AdmissionError(
                        'a dependency must be an event this session has '
                        'already submitted',
                        detail={'given': repr(dep)[:60]},
                        remedy='events are not portable between sessions')
                if dep.epoch != self.epoch:
                    raise AdmissionError(
                        'a dependency from a previous epoch')

            scratch = module.manifest['resources'].get('scratch_bytes', 0)
            scratch = scratch if isinstance(scratch, int) and scratch > 0 else 0
            maps = []
            for name, value in args.items():
                if isinstance(value, TensorDescriptor):
                    buf, offset = value.buffer if isinstance(value.buffer, tuple) else (value.buffer, 0)
                    maps.append((name, value.pack(resolved[name], buf.size-offset)))
            slots = abi.block_size(len(module.params)) + (8 if scratch else 0)
            map_start = (slots + 255) // 256 * 256
            scratch_start = map_start + 256 * len(maps)
            arg_addr, reserved = self._heap.alloc(max(scratch_start + scratch, slots, WORD), align=256)
            block = bytearray(scratch_start if maps else slots)
            for i, (name, data) in enumerate(maps):
                offset = map_start + 256*i
                resolved[name] = arg_addr + offset
                block[offset:offset+256] = data
            block[:abi.block_size(len(module.params))] = abi.encode_block(module.params, resolved)
            if scratch:
                struct.pack_into('<Q', block, abi.block_size(len(module.params)), arg_addr+scratch_start)

            task = dict(event=ev, module=module, block=block,
                        arg_addr=arg_addr, allocation=(arg_addr, reserved), grid=(gx, gy, gz),
                        depends_on=list(depends_on), leased=leased,
                        # Kept so a failure can be saved and run again
                        # (failure.capture); the block alone is bytes, and
                        # a bundle a person cannot read is not a bundle.
                        args=dict(resolved))
            ev._task = task
            for buf in leased:
                buf.leases += 1
            self._events[ev.job_id] = ev
            self._queue.append(task)
            ev._to(QUEUED)
            return ev

    def _resolve(self, module, args):
        """Arguments -> device values, with every check done here.

        Returns (values, buffers named). Raises before anything is queued.
        """
        names = [n for n, _t in module.params]
        missing = [n for n in names if n not in args]
        extra = [k for k in args if k not in names]
        if missing or extra:
            raise ArgumentError(
                f"argument mismatch: missing {missing}, unexpected {extra}",
                detail={'signature': [f"{n}:{t}" for n, t in module.params]})

        out, leased, spans = {}, [], []
        for name, kind in module.params:
            value = args[name]
            if isinstance(value, TensorDescriptor):
                if kind != 'device_ptr64':
                    raise ArgumentError(f'{name!r} is not a descriptor pointer parameter')
                value = value.buffer
            if kind == 'device_ptr64':
                buf, offset = (value if isinstance(value, tuple)
                               else (value, 0))
                if not isinstance(buf, Buffer):
                    raise ArgumentError(
                        f"{name!r} is a device pointer and needs a buffer "
                        f"handle, got {type(buf).__name__}",
                        remedy='pass a handle from this session; a host '
                               'address is not a device address')
                if buf.session_id != self.id or buf.epoch != self.epoch \
                        or not buf.alive:
                    raise ArgumentError(
                        f"{name!r}: {buf.name} is not a live buffer of this "
                        f"session and epoch")
                if isinstance(offset, bool) or not isinstance(offset, int) \
                        or offset < 0 or offset >= buf.size:
                    raise ArgumentError(
                        f"{name!r}: byte offset {offset} is outside "
                        f"{buf.name} (0..{buf.size - 1})")
                out[name] = buf.addr + offset
                leased.append(buf)
                spans.append((name, buf.addr + offset,
                              buf.addr + buf.size, buf))
            else:
                abi.check_value(name, kind, value)
                out[name] = value

        # Undeclared overlap. The manifest lists the pairs that may alias;
        # anything else overlapping is refused here rather than left to the
        # schedule to be accidentally safe (ABI-05).
        allowed = {tuple(sorted(p)) for p in
                   module.manifest['parameters'].get('alias_allowed') or []}
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                (na, sa, ea, ba), (nb, sb, eb, bb) = spans[i], spans[j]
                if sa < eb and sb < ea and tuple(sorted((na, nb))) not in allowed:
                    raise TncaError(
                        f"{na!r} and {nb!r} overlap and the manifest does "
                        f"not declare them as aliases",
                        code='ALIAS_UNDECLARED',
                        detail={'declared': sorted(allowed)},
                        remedy='declare the alias at compile time, or use '
                               'separate buffers')
        return out, list(dict.fromkeys(leased))

    # ---- the queue --------------------------------------------------------
    def _run_queue(self):
        while not self._stop.is_set():
            task = None
            with self._lock:
                if self._queue:
                    task = self._queue[0]
            if task is None:
                time.sleep(0.002)
                continue
            self._run_one(task)
            with self._lock:
                if self._queue and self._queue[0] is task:
                    self._queue.pop(0)

    def _run_one(self, task):
        ev = task['event']
        try:
            failed_dep = next((d for d in task['depends_on']
                               if d.state == FAILED), None)
            if failed_dep is not None:
                # No role starts. The contract is explicit that a dependent
                # task does not run, rather than running and being ignored.
                with self._lock:
                    self._finish(task, FAILED, DependencyFailed(
                        f"job {failed_dep.job_id} failed, so job "
                        f"{ev.job_id} did not start",
                        detail={'predecessor': failed_dep.job_id}))
                return
            for d in task['depends_on']:
                while not d.terminal and not self._stop.is_set():
                    time.sleep(0.002)
                if d.state == FAILED:
                    with self._lock:
                        self._finish(task, FAILED, DependencyFailed(
                            f"job {d.job_id} failed, so job {ev.job_id} "
                            f"did not start"))
                    return

            gx, gy, gz = task['grid']
            if gx * gy * gz == 0:
                # An empty grid is a real task with a real terminal state and
                # no work: no role starts and the kernel writes nothing.
                with self._lock:
                    ev._to(RUNNING)
                    ev._to(DRAINING)
                    self._finish(task, SUCCEEDED)
                return

            # A kernel with no arguments has an empty block, and
            # uploading nothing is not a request. The transport refuses a
            # zero-length transfer -- both bindings, one shared rule --
            # because a transfer of nothing is a mistake everywhere
            # except here, where it is the correct amount of argument
            # data to write.
            if task['block']:
                self.dev.upload(task['arg_addr'], task['block'])
            ev._to(RUNNING)
            total = gx * gy * gz
            cycles = 0
            per_program = []
            probe = self._counters is not None
            deltas = []
            for linear in range(total):
                # The contract's mapping, inverted: linear = x + gx*(y+gy*z).
                # A program's identity is its (x, y, z), not its linear id --
                # passing the linear id as pid.x made a 2x3 grid look like a
                # 6x1 one to the kernel, and `num_programs(0)` came back as
                # the total rather than gx.
                x = linear % gx
                y = (linear // gx) % gy
                z = linear // (gx * gy)
                before = self._snapshot() if probe else None
                if probe and before is None:
                    probe = False            # this device has no op 7
                reply = self.dev.launch(task['module'].code_addr,
                                        task['arg_addr'], role_mask=1,
                                        pid=(x, y, z), npid=(gx, gy, gz))
                cycles += reply.r0
                per_program.append(reply.r0)
                if probe and reply.status == T.ST_OK:
                    after = self._snapshot()
                    deltas.append(dict(
                        {k: (after[k] - before[k]) & 0xFFFFFFFFFFFFFFFF
                         for k in after},
                        done_cycles=reply.r0, done_retired=reply.r1))
                if reply.status == T.ST_FAULT:
                    ev._to(FAILING)
                    with self._lock:
                        self.dev.quiesce()
                        ev._to(DRAINING)
                        self._finish(task, FAILED, TncaError(
                            f"kernel trapped: code {reply.r0:#04x} at pc "
                            f"{reply.r1:#x}", code='KERNEL_FAULT',
                            detail={'trap': reply.r0, 'pc': reply.r1,
                                    'program': linear}))
                    return
                if reply.status == T.ST_TIMEOUT:
                    ev._to(FAILING)
                    self.poisoned = True
                    with self._lock:
                        self._finish(task, FAILED, DeviceLost(
                            f"the device deadline expired with work "
                            f"outstanding after {reply.r0} cycles",
                            detail={'program': linear}))
                    self._poison_rest()
                    return
            ev._to(DRAINING)
            self.dev.quiesce()
            ev.device_cycles = cycles
            ev.program_cycles = per_program
            if self._counters is not None:
                ev.counters = (_counter_summary(deltas) if probe
                               else {'available': False})
            with self._lock:
                self._finish(task, SUCCEEDED)
        except DeviceLost as e:
            self.poisoned = True
            with self._lock:
                if not ev.terminal:
                    if ev.state not in (FAILING,):
                        ev._to(FAILING)
                    self._finish(task, FAILED, e)
            self._poison_rest()
        except TncaError as e:
            with self._lock:
                if not ev.terminal:
                    self._finish(task, FAILED, e)

    def _finish(self, task, state, error=None):
        ev = task['event']
        if not ev.terminal:
            ev._to(state, error)
        allocation = task.pop('allocation', None)
        if allocation is not None:
            self._heap.release(*allocation)
        for buf in task['leased']:
            buf.leases = max(0, buf.leases - 1)
            if state == FAILED:
                # Not a rollback -- the contract does not promise one. A
                # marker, so an ordinary read cannot consume it as a result.
                buf.valid = False

    def _poison_rest(self):
        with self._lock:
            for task in list(self._queue):
                ev = task['event']
                if not ev.terminal:
                    self._finish(task, FAILED, DeviceLost(
                        'the device was lost before this task ran',
                        detail={'epoch': self.epoch}))
            self._queue.clear()

    # ---- waiting ----------------------------------------------------------
    def wait(self, event, timeout=None):
        """Block until the task reaches a terminal state.

        Idempotent, and a timeout is NOT a verdict: it says the host stopped
        looking. The task is not cancelled and is certainly not successful.
        """
        if not isinstance(event, Event):
            raise TncaError('wait() takes an event', code='ARGUMENT_INVALID')
        if event.done.wait(timeout if timeout is not None
                           else self._host_timeout):
            return event
        raise Timeout(
            f"job {event.job_id} is still {event.state} after "
            f"{timeout if timeout is not None else self._host_timeout}s",
            detail={'job': event.job_id, 'state': event.state},
            remedy='the task has not been cancelled and has not succeeded; '
                   'wait again, or let the device deadline decide')

    def drain(self, timeout=None):
        for ev in list(self._events.values()):
            if not ev.terminal:
                self.wait(ev, timeout)
        return True

    # ---- recovery and shutdown --------------------------------------------
    def recover(self):
        """A new epoch: a new device, and every old handle invalid."""
        with self._lock:
            self._stop.set()
        if self._worker:
            self._worker.join(timeout=30)
        try:
            self.dev.stop()
        except Exception:
            pass
        with self._lock:
            for ev in self._events.values():
                if not ev.terminal:
                    ev._to(FAILED, DeviceLost(
                        'the session was recovered before this task ran'))
            self.epoch += 1
            self._events.clear()
            self._queue.clear()
            self._buffers.clear()
            self._modules.clear()
            self._next_job = 0
            self._code_next = 0
            self._heap = _Allocator(self._heap.base, self._heap.limit)
            self.poisoned = False
            self.dev = _open_device(self.backend_name, self.bench,
                                    workdir=self._workdir,
                                    deadline_cycles=self._deadline,
                                    host_timeout=self._host_timeout,
                                    trace=self._trace, epoch=self.epoch)
            self.dev.start()
        self._start_worker()
        return self.epoch

    def close(self):
        """Idempotent. Refuses while work is unfinished, unless lost."""
        if self.closed:
            return True
        if self.unfinished and not self.poisoned:
            raise SessionBusy(
                f"{len(self.unfinished)} task(s) are unfinished",
                detail={'jobs': [e.job_id for e in self.unfinished]},
                remedy='wait for them, or accept the loss and recover(); '
                       'close() does not discard a task')
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=30)
        try:
            self.dev.stop()
        except Exception:
            pass
        self.closed = True
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.drain(timeout=5)
        except Exception:
            pass
        try:
            self.close()
        except SessionBusy:
            self.poisoned = True
            self.close()
        return False
