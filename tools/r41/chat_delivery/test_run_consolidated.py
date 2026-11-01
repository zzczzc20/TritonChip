#!/usr/bin/env python3
"""Check frozen-plan guards and campaign deadlines without a board."""
import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
from unittest import mock

import consolidated_plan as plan
import run_consolidated as runner
import test_consolidated_plan as fixtures


class RunnerTests(unittest.TestCase):
    setUp = fixtures.PlanTests.setUp
    write = staticmethod(fixtures.PlanTests.write)
    write_json = fixtures.PlanTests.write_json
    def prepared(self):
        overlay = self.base / 'overlay'
        data = plan.prepare(self.root, self.templates, self.refs, overlay)
        data.update(status='frozen', frozen_commit='a' * 40)
        for item in data['models']:
            for key in ('manifest', 'reference'):
                source, target = overlay / item[key], self.root / item[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return overlay, data

    def test_rejects_changed_compiler_and_pre_freeze_start(self):
        overlay, data = self.prepared()
        runner.verify_inputs(self.root, overlay, data)
        provisional = dict(data, status='provisional', frozen_commit=None)
        with self.assertRaisesRegex(ValueError, 'not frozen'):
            runner.verify_inputs(self.root, overlay, provisional)
        with mock.patch.object(runner, 'now', return_value=runner.utc('2026-11-01T17:49:59Z')):
            with self.assertRaisesRegex(ValueError, 'freeze time'):
                runner.run(self.root, overlay, self.base / 'out', data)
        self.write(self.root / 'compiler/python/new.py', 'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            runner.verify_inputs(self.root, overlay, data)

    def test_late_start_caps_remaining_budget(self):
        overlay, data = self.prepared()
        source = overlay / data['models'][0]['registration']
        target = self.base / 'bounded.json'
        self.assertEqual(runner.effective_registration(source, target, 90), target)
        self.assertEqual(json.loads(target.read_text())['experiments'][0]['budget_seconds'], 60)
        self.assertIsNone(runner.effective_registration(source, self.base / 'none.json', 30))
        self.assertIsNone(runner.effective_registration(source, self.base / 'short.json', 30.5))

    def test_rejects_registration_without_script_gate(self):
        overlay, data = self.prepared()
        item = data['models'][0]
        path = overlay / item['registration']
        registration = json.loads(path.read_text())
        registration['experiments'][0]['checks'] = [check for check in registration['experiments'][0]['checks']
                                                   if check['field'] != ['scripted_chat', 'passed']]
        self.write_json(path, registration)
        item['registration_sha256'] = plan.sha(path)
        with self.assertRaisesRegex(ValueError, 'required session checks'):
            runner.verify_inputs(self.root, overlay, data)

    def test_rejects_changed_gate_and_compiler_build_input(self):
        overlay, data = self.prepared()
        item = data['models'][0]
        path = overlay / item['registration']
        registration = json.loads(path.read_text())
        checks = registration['experiments'][0]['checks']
        next(check for check in checks if check['field'] == ['status'])['equals'] = 'failed'
        self.write_json(path, registration)
        item['registration_sha256'] = plan.sha(path)
        with self.assertRaisesRegex(ValueError, 'required session checks'):
            runner.verify_inputs(self.root, overlay, data)
        next(check for check in checks if check['field'] == ['status'])['equals'] = 'pass'
        self.write_json(path, registration)
        item['registration_sha256'] = plan.sha(path)
        self.write(self.root / 'compiler/CMakeLists.txt', 'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            runner.verify_inputs(self.root, overlay, data)

    def test_generated_registration_runs_through_stage_with_mocked_results(self):
        overlay, data = self.prepared()
        folder = Path(__file__).resolve().parents[1] / 'run2_board'
        sys.path.insert(0, str(folder))
        self.addCleanup(sys.path.remove, str(folder))
        spec = importlib.util.spec_from_file_location('delivery_stage', folder / 'stage.py')
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        for item in data['models']:
            model = item['model']
            registration = json.loads((overlay / item['registration']).read_text())
            out = self.base / f'{model}-stage'
            def fake_run(command, **kwargs):
                target = Path(command[command.index('--out') + 1])
                target.mkdir(parents=True)
                self.write_json(target / 'report.json', {'status': 'completed-experiment'})
                self.write_json(target / f'{model}-quality/quality.json', dict(status='pass',
                    model_scope='full-checkpoint', reduced_proof=False,
                    prompts=[{}] * plan.MODELS[model][1], warm_repeat={'passed': True},
                    scripted_chat={'passed': True}))
            with mock.patch.object(stage.subprocess, 'run', side_effect=fake_run) as run:
                stage.run(self.root, out, '/usr/bin/python3', registration)
                self.assertIn(str(plan.MODEL_BUDGET_SECONDS[model]), run.call_args.args[0])
            self.assertEqual(json.loads((out / 'stage.json').read_text())['status'], 'pass')

    def test_idle_wait_reports_busy_then_acquisition_without_board_access(self):
        idle = dict(uuid=plan.IMAGE_UUID, contexts=0)
        busy = dict(uuid=plan.IMAGE_UUID, contexts=1)
        class Tracker:
            def __init__(self, **kwargs):
                pass
            def observe(self, stamp, row):
                return row['contexts'] == 0
        auto = types.SimpleNamespace(IdleTracker=Tracker, probe=mock.Mock(side_effect=[busy, idle, idle]),
                                     idle=lambda row: row['contexts'] == 0, clock_mhz=lambda _: 100)
        states = []
        deadline = runner.utc('2026-11-02T00:50:00Z')
        with (self.base / 'lock').open('w') as lock, \
                mock.patch.object(runner, 'now', return_value=runner.utc('2026-11-01T18:00:00Z')), \
                mock.patch.object(runner.time, 'sleep'), mock.patch.object(runner.fcntl, 'flock'):
            row = runner.wait_idle(auto, self.base, self.base, lock, deadline, plan.IMAGE_UUID, 100000000,
                                   monitor=False, on_state=lambda state, board: states.append(state))
        self.assertEqual(row, idle)
        self.assertEqual(states, ['busy', 'acquired'])

    def test_frozen_git_archive_preserves_source_binding(self):
        def git(*args):
            return subprocess.run(['git', '-C', str(self.root), *args], check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()
        git('init', '-q')
        git('add', '--all')
        git('-c', 'user.name=R41 Test', '-c', 'user.email=r41-test@example.invalid',
            'commit', '-q', '-m', 'synthetic source snapshot')
        commit = git('rev-parse', 'HEAD')
        git('update-ref', 'refs/remotes/origin/r41', commit)
        changed = self.root / 'python/tnca/untracked.py'
        self.write(changed, 'untracked')
        with self.assertRaisesRegex(ValueError, 'source is dirty'):
            plan.prepare(self.root, self.templates, self.refs, self.base / 'dirty', commit)
        changed.unlink()
        overlay = self.base / 'frozen-overlay'
        data = plan.prepare(self.root, self.templates, self.refs, overlay, commit)
        archive = self.base / 'source.tar'
        git('archive', '--format=tar', '--output', str(archive), commit)
        exported = self.base / 'exported'
        exported.mkdir()
        subprocess.run(['tar', '-xf', str(archive)], cwd=exported, check=True)
        for item in data['models']:
            for key in ('manifest', 'reference'):
                destination = exported / item[key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(overlay / item[key], destination)
        runner.verify_inputs(exported, overlay, data)


if __name__ == '__main__':
    unittest.main()
