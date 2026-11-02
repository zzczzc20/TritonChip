"""Admission and memory backpressure for native device command queues."""
from tnca.errors import BufferError as TncaBufferError, QueueFull
from tnca.runtime import MAX_UNFINISHED


def _count_pending(session):
    try:
        return session.unfinished_count
    except AttributeError:
        return len(session.unfinished)


def _oldest_pending(session):
    try:
        return session.oldest_unfinished
    except AttributeError:
        events = session.unfinished
        return events[0] if events else None


def _raise_failed_predecessor(label, event):
    from tnca.triton_backend.driver import TNCALaunchError
    raise TNCALaunchError(f'{label} predecessor failed: {event.error}')


def submit_queued(session, module, bound, grid, *, label='queued'):
    """Preserve FIFO dependencies while bounding outstanding argument memory.

    The default queue retains its original depth and error behavior. When an
    explicitly larger queue temporarily exhausts the argument heap, wait for
    a real predecessor to release its allocation and then retry admission.
    """
    limit = getattr(session, 'max_unfinished', MAX_UNFINISHED)
    drained_races = 0
    while True:
        while _count_pending(session) >= limit:
            oldest = _oldest_pending(session)
            if oldest is None:  # A predecessor completed between the two reads.
                continue
            session.wait(oldest)
            if not oldest.ok:
                _raise_failed_predecessor(label, oldest)
        previous = session.latest_event
        try:
            return session.submit(module, bound, grid=grid,
                                  depends_on=[previous] if previous is not None else [])
        except (TncaBufferError, QueueFull) as error:
            oldest = _oldest_pending(session)
            if isinstance(error, QueueFull):
                # Another producer can fill the queue between our count and
                # Session.submit. A concurrent drain can also finish before
                # we identify the oldest event; retry that race only twice.
                if oldest is None:
                    if drained_races < 2:
                        drained_races += 1
                        continue
                    raise
            elif (limit <= MAX_UNFINISHED or oldest is None or
                  'out of device memory' not in str(error)):
                raise
            session.wait(oldest)
            if not oldest.ok:
                _raise_failed_predecessor(label, oldest)
            drained_races = 0
