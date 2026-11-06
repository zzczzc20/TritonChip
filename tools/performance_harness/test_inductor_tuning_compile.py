"""Real generated Inductor candidates compile in bounded device-free workers."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'), str(ROOT/'tools/performance_harness/systemc')]


class CompileWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import triton
        import tlm_backend
        tlm_backend.register()
        import tnca.torch_device as device
        device.register()
        from torch._inductor.utils import run_and_get_code
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        cls.torch, cls.triton = torch, triton
        cls.a, cls.b = torch.randn(33).to('tnca'), torch.randn(33).to('tnca')
        graph = torch.compile(lambda a, b: (a+b)*.5, backend='inductor',
                              fullgraph=True, dynamic=False)
        _, code = run_and_get_code(graph, cls.a, cls.b)
        module = PyCodeCache.load(code[0])
        cls.tuner, = [value for value in vars(module).values()
                      if isinstance(value, CachingAutotuner)]

    def test_real_candidate_compiles_and_executes_through_common_launcher(self):
        from tnca.autotune import GraphBudget
        from tnca.inductor_tuning import compile_candidate
        from tnca.triton_backend.driver import launch_log
        directory = Path(os.environ['R41_TUNE_TEST_OUT'])/'compiled'
        result = compile_candidate(self.tuner, self.triton.Config({'XBLOCK': 8}, num_warps=1),
                                   budget=GraphBudget(max_seconds=120), directory=directory)
        launcher = result.make_launcher()
        output = self.torch.empty_like(self.a)
        with launch_log(isolated=True) as launches:
            launcher(self.a, self.b, output, 33, stream=0)
        self.torch.testing.assert_close(output.cpu(), (self.a.cpu()+self.b.cpu())*.5)
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0]['upload_bytes'], 0)
        self.assertEqual(launches[0]['download_bytes'], 0)

    def test_timeout_kills_real_worker_and_descendants(self):
        from tnca.autotune import GraphBudget, CandidateTimedOut
        from tnca.inductor_tuning import compile_candidate
        from types import SimpleNamespace
        import json
        import time
        root = Path(os.environ['R41_TUNE_TEST_OUT'])
        # The trusted source intentionally creates a descendant and hangs
        # during import; it must not survive the original graph deadline.
        child = root/'descendant.pid'
        source = root/'hung-source.py'
        source.write_text('import subprocess,time\n'
                          'from pathlib import Path\n'
                          'p=subprocess.Popen(["sleep","60"])\n'
                          f'Path({str(child)!r}).write_text(str(p.pid))\n'
                          'time.sleep(60)\n')
        fake = SimpleNamespace(filename=str(source), fn=SimpleNamespace(__name__='unused'))
        directory = root/'timeout'
        # Importing Torch may take seconds before the injected source runs.
        with self.assertRaises(CandidateTimedOut):
            compile_candidate(fake, self.triton.Config({'XBLOCK': 8}, num_warps=1),
                              budget=GraphBudget(max_seconds=8), directory=directory)
        record = json.loads((directory/'compile.json').read_text())
        self.assertEqual(record['status'], 'timed-out')
        self.assertTrue(child.exists(), 'exercise the actual descendant, not startup alone')
        pid = int(child.read_text())
        for _ in range(100):
            try:
                state = Path(f'/proc/{pid}/stat').read_text().split()[2]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            time.sleep(.01)
        else:
            self.fail('timed-out candidate left a running compiler descendant')


if __name__ == '__main__':
    unittest.main()
