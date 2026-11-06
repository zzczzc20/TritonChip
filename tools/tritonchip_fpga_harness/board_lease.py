"""One advisory lease for board tools that share a physical device."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
from typing import Iterator

R41_DEVICE = '0000:04:00.1'


def lease_path(device: str, home: Path | None = None) -> Path:
    home = Path.home() if home is None else home
    if not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', device):
        raise ValueError(f'invalid PCI device address: {device!r}')
    if device.lower() == R41_DEVICE:
        # Existing R41 chat and candidate queues already use this exact inode.
        return home / 'r41run3' / 'candidate_queue.lock'
    return home / '.tritonchip-fpga-harness' / 'boards' / (device.lower() + '.lock')


@contextmanager
def board_lease(device: str, *, blocking: bool = True, home: Path | None = None) -> Iterator[None]:
    path = lease_path(device, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(lock, flags)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def shared_r41_device(board: str, bdf: str | None, home: Path | None = None) -> bool:
    """The historical R41 queue marks hosts whose default U280 is shared."""
    home = Path.home() if home is None else home
    return (board == 'u280' and (bdf or R41_DEVICE).lower() == R41_DEVICE
            and (home / 'r41run3').is_dir())
