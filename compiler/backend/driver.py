"""
The TNCA driver: how a compiled kernel is launched.

sec 15 C1 says the first closure uses AOT plus a simulation launcher, and that
a full PyTorch device integration is deliberately not a prerequisite. So this
driver targets the existing RTL benches: it writes the assembled image where
tb/core/tb_tnca_core.v expects it and runs iverilog, which makes the compiler's
output testable against the same environment the hardware is tested in
(sec 16.2, "编译产物 RTL 差分").
"""
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))


class TNCASimLauncher:
    """Run a kernel image on the Icarus model of the core."""

    def __init__(self, *, bench='tb_core', max_cycles=100000):
        self.bench = bench
        self.max_cycles = max_cycles

    def command(self, hex_path, extra_args=()):
        vvp = os.path.join(_ROOT, 'sim', 'build', f"{self.bench}.vvp")
        return ['vvp', vvp, f"+hex={hex_path}",
                f"+maxcycles={self.max_cycles}", *extra_args]


def get_launcher(**kw):
    return TNCASimLauncher(**kw)
