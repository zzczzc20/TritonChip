"""Shared-card waiting rule: fake clocks and probes, no card access."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import experiment
spec=importlib.util.spec_from_file_location('queue',Path(__file__).parent/'launches/acceptance_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
IDENTITY=dict(image_uuid='49964b17-e0b4-5bd2-dd40-7fc37b1205d3',clock_hz=100000000,image_sha256='approved')

def idle(**kwargs):return dict(loaded_uuid=IDENTITY['image_uuid'],clock_counter_mhz=[100],contexts=0,outstanding_execs=0,client_pid=None,**kwargs)

class WaitingRule(unittest.TestCase):
    def test_python_symlink_is_not_dereferenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'system-python';target.touch()
            torch_python=Path(tmp)/'torchenv-python';torch_python.symlink_to(target)
            self.assertEqual(queue.absolute_path(torch_python),torch_python)
    def run_wait(self,rows):
        now=[0];history=[]
        def probe():
            row=rows[min(len(history),len(rows)-1)];history.append(now[0]);return row
        def sleep(t):now[0]+=t
        queue.wait_idle(probe,IDENTITY,seconds=900,poll_seconds=60,clock=lambda:now[0],sleep=sleep)
        return now[0]
    def test_fifteen_elapsed_minutes_not_fifteen_instant_samples(self):
        self.assertEqual(self.run_wait([idle()]),900)
    def test_busy_client_restarts_entire_idle_interval(self):
        busy=idle();busy['client_pid']='42'
        self.assertEqual(self.run_wait([idle()]*10+[busy]+[idle()]),1560)
    def test_outstanding_exec_restarts_interval(self):
        busy=idle();busy['outstanding_execs']=1
        self.assertEqual(self.run_wait([busy,idle()]),960)
    def test_unreadable_counters_never_count_as_idle(self):
        bad=idle();bad['outstanding_execs']=None
        self.assertEqual(self.run_wait([bad,idle()]),960)
    def test_wrong_image_refused_without_opening(self):
        bad=idle();bad['loaded_uuid']='other'
        with self.assertRaises(RuntimeError):self.run_wait([bad])
    def test_multiline_client_and_outstanding_are_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'0000:04:00.1';base.mkdir()
            (base/'kdsstat').write_text('contexts: 0\noutstanding execs: 3\nclient pid:\n  1234\n')
            # Redirect only this probe's sysfs root, never real sysfs.
            real=experiment.Path
            with mock.patch.object(experiment,'Path',side_effect=lambda x:real(tmp) if str(x)=='/sys/bus/pci/devices' else real(x)):
                row=experiment.sysfs_probe({'device':'0000:04:00.1'})
            self.assertEqual(row['outstanding_execs'],3)
            self.assertEqual(row['client_pid'],'1234')
            self.assertIsNotNone(experiment.board_refusal(IDENTITY,row))

if __name__=='__main__':unittest.main()
