import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('run2_auto', Path(__file__).with_name('auto.py'))
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


def row(uuid='1c36f315-c7c1-c7f5-0de1-8c3e8d80e1dd', total=0, contexts=0):
    return {'uuid': uuid, 'total execs': total, 'contexts': contexts, 'outstanding execs': 0, 'client pid': ''}


class IdleTests(unittest.TestCase):
    def test_900_seconds_and_activity_reset(self):
        t = auto.IdleTracker(seconds=900, max_gap=15)
        for n in range(0, 901, 5):
            self.assertEqual(t.observe(n, row()), n == 900)
        self.assertFalse(t.observe(905, row(total=1)))
        self.assertFalse(t.observe(910, row(contexts=1)))
        self.assertFalse(t.observe(915, row(total=1)))

    def test_gap_and_image_change_reset(self):
        t = auto.IdleTracker(seconds=20, max_gap=10)
        self.assertFalse(t.observe(0, row()))
        self.assertFalse(t.observe(11, row()))
        self.assertFalse(t.observe(16, row(uuid=auto.IMAGE_UUID)))
        self.assertFalse(t.observe(21, row(uuid=auto.IMAGE_UUID)))

    def test_sysfs_missing_or_busy_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp)/'0000:04:00.1'
            dev.mkdir()
            (dev/'xclbinuuid').write_text(auto.IMAGE_UUID)
            (dev/'kdsstat').write_text('contexts: 0\noutstanding execs: 0\ntotal execs: 4\nclient pid:\n 1234\n')
            self.assertFalse(auto.idle(auto.probe(tmp)))
            (dev/'kdsstat').write_text('contexts: 0\n')
            with self.assertRaises(ValueError):
                auto.probe(tmp)

    def test_incomplete_plan_never_reaches_programming(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'bad.xclbin'
            image.write_bytes(b'not the approved image')
            with self.assertRaises(ValueError):
                auto.validate_plan({'schema': 'tnca-r41-run2-board-plan/1', 'stages': []}, image)


if __name__ == '__main__':
    unittest.main()
