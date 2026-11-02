"""Predict code residency from the preceding repeated decode step.

This only changes which completed modules occupy the code store.  It never
changes a kernel, its arguments, or the tensors on the device.  The forecast
uses the current first-fit allocator state and enables the policy only when
its predicted upload bytes are substantially below LRU. A changed launch
sequence immediately returns to LRU and becomes the next forecast.
"""
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import copy
from math import sqrt
from threading import local

_current = local()


def active():
    return getattr(_current, 'trace', None)


def _resident_identity(module, digest):
    # A module's executable identity belongs to the bytes that were loaded,
    # not to whatever may later replace its pathname. It also avoids hundreds
    # of file-stat calls when forecasting or sorting a decode's residents.
    identity = getattr(module, '_tnca_execution_identity', None)
    return identity if identity is not None else digest(module.path) or module.path


class RepeatedCodeTrace:
    def __init__(self):
        self.previous = []
        self.entries = []
        self.position = -1
        self.positions = {}
        self.predict = False
        self.decision = None
        self._session_key = None
        self._forecast_cache = OrderedDict()

    @contextmanager
    def step(self):
        prior = active()
        self.entries, self.position, self.predict = [], -1, False
        self.decision, self._session_key = None, None
        positions = defaultdict(list)
        for i, (key, _size) in enumerate(self.previous):
            positions[key].append(i)
        self.positions = positions
        _current.trace = self
        try:
            yield
        except BaseException:
            raise
        else:
            if self.entries:
                self.previous = self.entries
        finally:
            _current.trace = prior

    def _score(self, key, size, position):
        points = self.positions.get(key)
        if not points:
            return float('inf')
        after = bisect_right(points, position)
        distance = ((points[after] - position) if after < len(points)
                    else (len(self.previous) - position + points[0]))
        return distance / sqrt(max(size, 1))

    def _simulate(self, session, modules, policy):
        from tnca.errors import BufferError
        alloc = copy(session._code)
        alloc.free_list = list(alloc.free_list)
        resident = OrderedDict((key, (module.code_addr, module.code_size))
                               for key, module in modules.items())
        uploaded = 0
        for index, (key, size) in enumerate(self.previous):
            if key in resident:
                resident.move_to_end(key)
                continue
            while True:
                try:
                    address, allocated = alloc.alloc(size, align=64)
                    break
                except BufferError:
                    if not resident:
                        raise
                    if policy == 'lru':
                        victim = next(iter(resident))
                    else:
                        victim = max(resident, key=lambda k: self._score(k, resident[k][1], index))
                    address, released = resident.pop(victim)
                    alloc.release(address, released)
            resident[key] = (address, allocated)
            uploaded += size
        return uploaded

    def _forecast(self, session, modules):
        """Reuse a forecast only for exactly the same trace and allocator state.

        Replaying a 565-launch decode trace twice at the start of every token
        wastes host time once the code store reaches a repeating state. The
        complete key includes ordering and addresses: an eviction, new code,
        changed trace, or recovery epoch forces a fresh simulation.
        """
        allocator = session._code
        key = (session.id, session.epoch, tuple(self.previous),
               tuple((identity, module.code_addr, module.code_size)
                     for identity, module in modules.items()),
               allocator.base, allocator.limit, allocator.next,
               allocator.reserved, tuple(allocator.free_list))
        cached = self._forecast_cache.get(key)
        if cached is not None:
            self._forecast_cache.move_to_end(key)
            return cached
        result = (self._simulate(session, modules, 'lru'),
                  self._simulate(session, modules, 'lookahead'))
        self._forecast_cache[key] = result
        if len(self._forecast_cache) > 4:
            self._forecast_cache.popitem(last=False)
        return result

    def request(self, session, key, loaded, digest):
        self.position += 1
        session_key = (session.id, session.epoch)
        if self.position == 0:
            self._session_key = session_key
            if self.previous and all(k[:2] == session_key for k, _ in self.previous):
                modules = OrderedDict()
                for cache_key, module in loaded.items():
                    if cache_key[:2] != session_key:
                        continue
                    module_key = session_key + (_resident_identity(module, digest),)
                    modules[module_key] = module
                    modules.move_to_end(module_key)
                try:
                    lru, candidate = self._forecast(session, modules)
                    self.decision = (lru, candidate)
                    self.predict = candidate * 5 <= lru * 4
                except Exception:  # A forecast must never prevent a normal load.
                    # A custom or test session without allocator metadata uses LRU.
                    self.predict = False
        if (session_key != self._session_key or self.position >= len(self.previous)
                or self.previous[self.position][0] != key):
            self.predict = False

    def record(self, key, module):
        size = getattr(module, 'code_size', 0)
        if size:
            self.entries.append((key, size))
        else:
            # Incomplete metadata cannot train a future prediction.
            self.previous = []

    def victims(self, session, loaded, digest):
        keys = [key for key in loaded if key[:2] == (session.id, session.epoch)]
        if not self.predict:
            return keys
        def score(key):
            module = loaded[key]
            identity = (session.id, session.epoch, _resident_identity(module, digest))
            return self._score(identity, module.code_size, self.position)
        return sorted(keys, key=score, reverse=True)
