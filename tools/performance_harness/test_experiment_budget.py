"""A total experiment deadline bounds multiple real child commands."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
import experiment
class TotalBudget(unittest.TestCase):
    def test_later_step_gets_only_remaining_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'source').write_text('fixed')
            spec=dict(id='budget',allowed_hosts=[socket.gethostname()],hypothesis='bounded total',
                competing_explanations=['per-step budgets accumulate'],expected_observations=['second step timeout'],
                sources=['source'],budget_seconds=2,steps=[dict(id=str(i),kind='analytical',
                    argv=[sys.executable,'-c','import time; time.sleep(0.12)'],timeout_seconds=2) for i in range(2)])
            path=root/'spec.json';path.write_text(json.dumps(spec))
            result=experiment.run(path,root,root/'out',python=sys.executable,budget_seconds=.18)
            self.assertEqual(result['status'],'stopped-on-failure')
            self.assertEqual(result['steps'][0]['status'],'completed')
            self.assertEqual(result['steps'][1]['status'],'timeout')
            self.assertLess(result['steps'][1]['host_wall_seconds'],.2)
if __name__=='__main__':unittest.main()
