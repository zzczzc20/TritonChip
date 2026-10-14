"""Exec a device process with parent-death cleanup, outside fork callbacks."""
import ctypes
import os
import signal
import sys


def child_argv(argv):
    # Run this file directly, avoiding package imports in the short-lived
    # launcher. The parent uses its thread-safe exec path without preexec_fn.
    return [sys.executable, os.path.abspath(__file__), str(os.getpid()), *argv]


def main():
    parent = int(sys.argv[1])
    if sys.platform.startswith('linux'):
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
        except (AttributeError, OSError):
            pass  # Cleanup remains best-effort on unsupported platforms.
    # The owner may have died before this process installed its signal.
    # Check after prctl: a death after this check is handled by the kernel.
    if os.getppid() != parent:
        return 125
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == '__main__':
    sys.exit(main())
