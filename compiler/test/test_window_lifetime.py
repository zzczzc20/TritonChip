"""Awaited window lifetimes must include aliases and loop back edges."""
import unittest
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.codegen import CodeGen
from tnca_cc.target import TargetInfo
from tnca_cc.diagnostics import ResourceError


class TestWindowLifetime(unittest.TestCase):
    def build(self):
        k = make_kernel('window_lifetime')
        return k, Builder(k, k.role(0).region)

    def fill(self, b, w, wait=True):
        d = b.descriptor(b.const(0x2000))
        bar = b.barrier()
        token = b.barrier_arm(bar, arrivals=0, ops=1)
        b.dma_load(w, d, [b.const(0)], token, dtype='f32')
        if wait:
            b.barrier_wait(token)

    def reserve(self, k):
        cg = CodeGen(k, TargetInfo.default(), check_protocol=False, window_reuse=True)
        cg.assign_shared()
        return cg

    def test_completed_windows_share_registers(self):
        k, b = self.build()
        windows = []
        for _ in range(3):
            w = b.window(32)
            windows.append(w)
            self.fill(b, w)
        cg = self.reserve(k)
        self.assertEqual(cg.window_registers, 32)
        self.assertEqual(len({w.reg for w in windows}), 1)
        cg.run()

    def test_outstanding_writer_cannot_share(self):
        k, b = self.build()
        for _ in range(3):
            self.fill(b, b.window(32), wait=False)
        with self.assertRaises(ResourceError):
            self.reserve(k)

    def test_reuse_is_opt_in(self):
        k, b = self.build()
        for _ in range(3):
            self.fill(b, b.window(32))
        cg = CodeGen(k, TargetInfo.default(), check_protocol=False)
        with self.assertRaises(ResourceError):
            cg.assign_shared()

    def test_resource_ledger_counts_implicit_window_registers(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
        from tnca.aot import measure_resources
        k, b = self.build()
        for _ in range(3):
            self.fill(b, b.window(32))
        cg = self.reserve(k)
        cg.run()
        measured = measure_resources(cg.b.text(), cg.target, include_windows=True)
        self.assertEqual(measured['vector_regs_used'], 32)

    def test_shared_roles_are_refused(self):
        k = make_kernel('shared_windows', num_roles=2)
        with self.assertRaisesRegex(ResourceError, 'single IR role'):
            self.reserve(k)

    def test_window_free_control_flow_needs_no_window_analysis(self):
        k, b = self.build()
        b.branch_nz(b.const(1), 'skip')
        b.label('skip')
        b.exit()
        cg = self.reserve(k)
        self.assertEqual(cg.window_registers, 0)
        cg.run()

    def test_alias_read_extends_the_window(self):
        k, b = self.build()
        first = b.window(32)
        self.fill(b, first)
        alias = b.window_reg(first)
        second = b.window(32)
        self.fill(b, second)
        b.store(b.const(0x4000), alias)
        cg = self.reserve(k)
        self.assertEqual(cg.window_registers, 64)
        self.assertNotEqual(first.reg, second.reg)

    def test_outer_window_stays_live_across_iterations(self):
        k, b = self.build()
        outer = b.window(32)
        self.fill(b, outer)
        with b.loop(b.const(0), 2, 'again'):
            alias = b.window_reg(outer)
            inner = b.window(32)
            self.fill(b, inner)
            b.store(b.const(0x4000), alias)
        cg = self.reserve(k)
        self.assertEqual(cg.window_registers, 64)
        self.assertNotEqual(outer.reg, inner.reg)

    def test_conditional_wait_cannot_establish_a_lifetime(self):
        k, b = self.build()
        self.fill(b, b.window(32))
        b.branch_nz(b.const(1), 'skip')
        self.fill(b, b.window(32))
        b.label('skip')
        with self.assertRaises(ResourceError):
            self.reserve(k)

    def test_scalar_loop_entry_guard_is_supported(self):
        k, b = self.build()
        b.branch_nz(b.const(0), 'done')
        with b.loop(b.const(0), 2, 'again'):
            self.fill(b, b.window(32))
            self.fill(b, b.window(32))
        b.label('done')
        self.assertEqual(self.reserve(k).window_registers, 32)

    def test_break_after_async_issue_is_not_a_loop_entry_guard(self):
        k, b = self.build()
        with b.loop(b.const(0), 2, 'again'):
            self.fill(b, b.window(32), wait=False)
            b.branch_nz(b.const(1), 'done')
        b.label('done')
        with self.assertRaises(ResourceError):
            self.reserve(k)


if __name__ == '__main__':
    unittest.main()
