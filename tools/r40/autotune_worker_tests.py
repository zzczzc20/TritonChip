import json
from pathlib import Path
import sys
import tempfile
import unittest
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from tnca.autotune import (
    CandidateFailed,
    CandidateTimedOut,
    GraphBudget,
    run_measurement,
)


class TestAutotuneWorker(unittest.TestCase):
    def test_descendants_do_not_outlive_successful_measurement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root=Path(tmpdir);pidfile=root/'child.pid'
            code=("import subprocess,sys,json\n"
                  "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])\n"
                  "open(sys.argv[2],'w').write(str(p.pid))\n"
                  "json.dump([dict(clock='mmio-device-clock/1',epoch=1,session_id=1,start=10,end=20)],open(sys.argv[1],'w'))\n")
            run_measurement([sys.executable,'-c',code,str(root/'result'),str(pidfile)],
                budget=GraphBudget(max_seconds=3),result_path=root/'result',log_path=root/'log')
            stat=Path('/proc')/pidfile.read_text().strip()/'stat'
            for _ in range(30):
                if not stat.exists() or stat.read_text().split()[2]=='Z':break
                time.sleep(.01)
            else:self.fail('measurement descendant still running')

    def test_invalid_endpoint_json_is_not_a_measurement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root=Path(tmpdir)
            code="import sys;open(sys.argv[1],'w').write('[{\"host_seconds\":1}]')"
            with self.assertRaises(CandidateFailed):
                run_measurement([sys.executable,'-c',code,str(root/'result')],
                    budget=GraphBudget(max_seconds=3),result_path=root/'result',log_path=root/'log')

    def test_successful_child_writes_valid_record_and_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result_path = tmp / "result.json"
            log_path = tmp / "output.log"
            budget = GraphBudget(max_seconds=10)
            expected = [{"clock": "mmio-device-clock/1", "epoch": 1, "session_id": 1, "start": 10, "end": 20}]
            code = (
                "import json, sys\n"
                "records = [{'clock': 'mmio-device-clock/1', 'epoch': 1, 'session_id': 1, 'start': 10, 'end': 20}]\n"
                "with open(sys.argv[1], 'w') as f:\n"
                "    json.dump(records, f)\n"
                "print('sample stdout log')\n"
            )
            cmd = [sys.executable, "-c", code, str(result_path)]
            out = run_measurement(cmd, budget=budget, result_path=result_path, log_path=log_path)
            self.assertEqual(out, expected)
            self.assertIn("sample stdout log", log_path.read_text())
            self.assertFalse(budget.exceeded)

    def test_nonzero_child_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result_path = tmp / "result.json"
            log_path = tmp / "output.log"
            budget = GraphBudget(max_seconds=10)
            cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
            with self.assertRaises(CandidateFailed):
                run_measurement(cmd, budget=budget, result_path=result_path, log_path=log_path)

    def test_timeout_child_sleep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result_path = tmp / "result.json"
            log_path = tmp / "output.log"
            budget = GraphBudget(max_seconds=0.2)
            cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
            with self.assertRaises(CandidateTimedOut):
                run_measurement(cmd, budget=budget, result_path=result_path, log_path=log_path)
            self.assertTrue(budget.exceeded)

    def test_existing_result_file_refused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result_path = tmp / "result.json"
            log_path = tmp / "output.log"
            prior_data = "untouched initial content"
            result_path.write_text(prior_data)
            budget = GraphBudget(max_seconds=10)
            cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
            with self.assertRaises(ValueError):
                run_measurement(cmd, budget=budget, result_path=result_path, log_path=log_path)
            self.assertEqual(result_path.read_text(), prior_data)


if __name__ == "__main__":
    unittest.main()
