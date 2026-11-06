"""Shared-card waiting rule: fake clocks and probes, no card access."""
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import experiment
spec=importlib.util.spec_from_file_location('queue',Path(__file__).parent/'launches/acceptance_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
IDENTITY=dict(image_uuid='49964b17-e0b4-5bd2-dd40-7fc37b1205d3',clock_hz=100000000,image_sha256='approved')

def idle(**kwargs):return dict(loaded_uuid=IDENTITY['image_uuid'],clock_counter_mhz=[100],contexts=0,outstanding_execs=0,total_execs=0,client_pid=None,**kwargs)

class QueueBoardIdentity(unittest.TestCase):
    def board(self):
        return dict(image='image.xclbin', image_sha256='abc', image_uuid=IDENTITY['image_uuid'],
                    clock_evidence='clock.json', clock_hz=IDENTITY['clock_hz'],
                    memory_map={'window_base': 0}, device='0000:04:00.1', mgmt_device='0000:04:00.0')

    def test_same_identity_and_memory_window_override_are_allowed(self):
        board=self.board()
        spec=dict(board=board, steps=[dict(id='one', kind='fpga',
                                          board={'memory_map': {'window_base': 8}})])
        self.assertEqual(queue.queue_board(spec),board)

    def test_step_cannot_override_the_leased_board_or_image(self):
        for key,value in (('device','0000:05:00.1'), ('mgmt_device','0000:05:00.0'),
                          ('image_sha256','different'), ('image_uuid','00000000-0000-0000-0000-000000000001'),
                          ('clock_hz',90000000)):
            with self.subTest(key=key):
                spec=dict(board=self.board(),steps=[dict(id='other',kind='fpga',board={key:value})])
                with self.assertRaisesRegex(ValueError,'changes the queued board identity'):
                    queue.queue_board(spec)

    def test_incomplete_top_level_identity_is_refused_before_waiting(self):
        board=self.board();del board['device']
        with self.assertRaisesRegex(ValueError,'queue board identity lacks'):
            queue.queue_board(dict(board=board,steps=[dict(kind='fpga')]))


class QueuePreflight(unittest.TestCase):
    def manifest(self):
        board=QueueBoardIdentity().board()
        board['image_sha256']='30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
        return dict(id='sample', allowed_hosts=['a-different-host'], hypothesis='h',
                    competing_explanations=['c'], expected_observations=['o'], sources=['source'],
                    expected_counter_changes=['counter'], board=board,
                    steps=[dict(id='run',kind='fpga',argv=['python','run.py'],timeout_seconds=1,
                                timing_boundary='device',cache_state='warm')])

    def invoke(self, spec, *, check_only):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'manifest.json';path.write_text(json.dumps(spec))
            out=root/'out'
            argv=['queue','--root',str(root),'--out',str(out),'--manifest',str(path)]
            if check_only: argv.append('--check-only')
            with mock.patch.object(sys,'argv',argv), mock.patch.object(experiment,'sysfs_probe',
                    side_effect=AssertionError('preflight must not access a board')):
                with self.assertRaises(ValueError): queue.main()
            self.assertFalse(out.exists())

    def test_bad_step_is_rejected_by_check_only_before_board_access(self):
        spec=self.manifest();spec['steps'][0]['argv']=[]
        self.invoke(spec,check_only=True)

    def test_wrong_host_is_rejected_before_the_idle_wait(self):
        self.invoke(self.manifest(),check_only=False)


class WaitingRule(unittest.TestCase):
    def test_python_symlink_is_not_dereferenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'system-python';target.touch()
            torch_python=Path(tmp)/'torchenv-python';torch_python.symlink_to(target)
            self.assertEqual(queue.absolute_path(torch_python),torch_python)
    def run_wait(self,rows,claim=None):
        now=[0];history=[]
        def probe():
            row=rows[min(len(history),len(rows)-1)];history.append(now[0]);return row
        def sleep(t):now[0]+=t
        queue.wait_idle(probe,IDENTITY,seconds=900,poll_seconds=60,clock=lambda:now[0],sleep=sleep,claim=claim)
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
    def test_execution_between_polls_restarts_idle_interval(self):
        changed=idle();changed['total_execs']=1
        self.assertEqual(self.run_wait([idle()]*10+[changed]),1500)
    def test_unreadable_total_execs_never_counts_as_idle(self):
        bad=idle();bad['total_execs']=None
        self.assertEqual(self.run_wait([bad,idle()]),960)
    def test_busy_lease_does_not_restart_a_continuous_idle_interval(self):
        attempts=[]
        def claim(_row):
            attempts.append(True)
            return len(attempts) > 1
        self.assertEqual(self.run_wait([idle()],claim=claim),960)
        self.assertEqual(len(attempts),2)
    def test_observed_busy_recheck_restarts_the_interval(self):
        attempts=[]
        def claim(_row):
            attempts.append(True)
            # The locked recheck saw a context that disappeared before the
            # next sample, without incrementing the execution counter.
            return None if len(attempts) == 1 else True
        self.assertEqual(self.run_wait([idle()],claim=claim),1860)
        self.assertEqual(len(attempts),2)
    def test_activity_while_waiting_for_lease_restarts_the_interval(self):
        changed=idle();changed['total_execs']=1
        attempts=[]
        def claim(_row):
            attempts.append(True)
            return len(attempts) > 1
        self.assertEqual(self.run_wait([idle()]*16+[changed],claim=claim),1860)
        self.assertEqual(len(attempts),2)
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
