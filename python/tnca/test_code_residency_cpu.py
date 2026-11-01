"""CPU-only checks for repeated decode residency forecasts."""
from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest import mock

from tnca.code_residency import RepeatedCodeTrace
from tnca.runtime import _Allocator


class ResidencyForecastTests(unittest.TestCase):
    def setUp(self):
        self.trace = RepeatedCodeTrace()
        self.session = SimpleNamespace(id=5, epoch=1, _code=_Allocator(0, 256))
        first, size = self.session._code.alloc(64)
        self.a = (5, 1, 'a')
        self.b = (5, 1, 'b')
        self.c = (5, 1, 'c')
        self.modules = OrderedDict([(self.a, SimpleNamespace(code_addr=first, code_size=size))])
        self.trace.previous = [(self.a, 64), (self.b, 64), (self.c, 64), (self.a, 64)]
        with self.trace.step():
            pass

    def test_only_identical_store_state_reuses_a_prediction(self):
        with mock.patch.object(self.trace, '_simulate', wraps=self.trace._simulate) as simulate:
            first = self.trace._forecast(self.session, self.modules)
            self.assertEqual(first, self.trace._forecast(self.session, self.modules))
            self.assertEqual(simulate.call_count, 2)
            addr, size = self.session._code.alloc(64)
            self.modules[self.b] = SimpleNamespace(code_addr=addr, code_size=size)
            changed = self.trace._forecast(self.session, self.modules)
            self.assertEqual(simulate.call_count, 4)
            self.assertEqual(changed, (self.trace._simulate(self.session, self.modules, 'lru'),
                                       self.trace._simulate(self.session, self.modules, 'lookahead')))
            self.session._code.release(addr, size)
            self.modules.pop(self.b)
            self.assertEqual(first, self.trace._forecast(self.session, self.modules))
            self.session.epoch += 1
            self.trace._forecast(self.session, self.modules)
            self.assertEqual(simulate.call_count, 8)

    def test_changed_launch_sequence_recomputes_the_forecast(self):
        with mock.patch.object(self.trace, '_simulate', wraps=self.trace._simulate) as simulate:
            self.trace._forecast(self.session, self.modules)
            self.trace.previous.append((self.c, 64))
            with self.trace.step():
                pass
            self.trace._forecast(self.session, self.modules)
            self.assertEqual(simulate.call_count, 4)


if __name__ == '__main__':
    unittest.main()
