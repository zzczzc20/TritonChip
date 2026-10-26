"""Unit tests for generic graph-level locality/DCO consumer."""
import unittest
from tnca.locality import StorageAnalyzer, LocalityController, GraphLocalityPlan, LocalityRefusal


class MockSession:
    def __init__(self):
        self.poisoned = False
        self.mmio_log = []

    def mmio_write_sequence(self, seq):
        if self.poisoned:
            raise RuntimeError('Device poisoned')
        self.mmio_log.extend(seq)


class MockStorage:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class MockTensor:
    def __init__(self, data_ptr, nbytes, storage=None):
        self._ptr = data_ptr
        self._nbytes = nbytes
        self._storage = storage or MockStorage(data_ptr)

    def untyped_storage(self):
        return self._storage

    def data_ptr(self):
        return self._ptr

    def nelement(self):
        return self._nbytes // 2

    def element_size(self):
        return 2


class LocalityConsumerTests(unittest.TestCase):
    def test_temporary_read_once_gets_bypass(self):
        analyzer = StorageAnalyzer()
        t1 = MockTensor(1000, 128)
        analyzer.register_tensor(t1, is_read=True, is_write=True)
        plan = analyzer.finalize_plan()
        desc = list(plan.storages.values())[0]
        self.assertTrue(desc.bypass_eligible)
        self.assertEqual(plan.policy_flags & 0x2, 0x2)

    def test_graph_io_refuses_bypass(self):
        analyzer = StorageAnalyzer()
        t_in = MockTensor(2000, 128)
        t_out = MockTensor(3000, 128)
        analyzer.register_tensor(t_in, is_input=True, is_read=True)
        analyzer.register_tensor(t_out, is_output=True, is_write=True)
        plan = analyzer.finalize_plan()
        for d in plan.storages.values():
            self.assertFalse(d.bypass_eligible)

    def test_overlapping_alias_refuses_bypass(self):
        analyzer = StorageAnalyzer()
        common_storage = MockStorage(4000)
        # Same underlying storage 4000, overlapping slices [4000, 4064) and [4032, 4096)
        slice1 = MockTensor(4000, 64, storage=common_storage)
        slice2 = MockTensor(4032, 64, storage=common_storage)
        analyzer.register_tensor(slice1, is_read=True)
        analyzer.register_tensor(slice2, is_read=True)
        plan = analyzer.finalize_plan()
        desc = plan.storages[4000]
        self.assertTrue(desc.has_unsafe_alias)
        self.assertFalse(desc.bypass_eligible)
        self.assertTrue(any('overlapping' in r for r in plan.refusals))

    def test_fault_propagates_session_poison(self):
        session = MockSession()
        controller = LocalityController(session)
        plan = GraphLocalityPlan()

        def fail_fn():
            raise RuntimeError('Hardware bus error')

        with self.assertRaises(RuntimeError):
            controller.execute_graph_with_locality(plan, fail_fn)
        self.assertTrue(session.poisoned)

        # Subsequent invocation raises LocalityRefusal
        with self.assertRaises(LocalityRefusal):
            controller.execute_graph_with_locality(plan, lambda: 1)


if __name__ == '__main__':
    unittest.main()
