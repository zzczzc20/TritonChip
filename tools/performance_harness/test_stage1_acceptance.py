"""The Stage 1 acceptance checker refuses every section 9 shortcut."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import experiment  # noqa: E402
import stage1_acceptance as acceptance  # noqa: E402

FE1B = REPO/'evidence/r40/v3/R40-04.hbm/board_run_fe1b'
NEW_TREE = 'files:' + '1'*64


def good_record(point='model.gpt2_block/medium/fp16-d160-h5-b1-s64'):
    rec = json.loads((FE1B/'native/u280_medium.jsonl').read_text().splitlines()[0])
    assert rec['point_id'] == point
    rec['platform']['software_tree'] = NEW_TREE
    return rec


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.real = json.loads((REPO/'evidence/r41/stage1/required_index.json').read_text())

    def index(self, items):
        index = dict(copy.deepcopy(self.real), items=items)
        path = self.dir/f'index-{len(list(self.dir.glob("index-*")))}.json'
        path.write_text(json.dumps(index))
        return path

    def evidence(self, name, rows=None, text=None):
        path = self.dir/name
        if rows is not None:
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        else:
            path.write_text(text)
        return path

    def entry(self, path, kind='board', **extra):
        return dict(dict(path=str(path), sha256=experiment.digest(path), kind=kind, binds={}), **extra)

    def status(self, item, item_id='S1-03.device_drain_recovery'):
        result = acceptance.evaluate(REPO, self.index({item_id: item}))
        row = next(r for r in result['items'] if r['id'] == item_id)
        return row['status'], ' | '.join(row['reasons'])

    def board_item(self, rows, **entry):
        path = self.evidence(f'records-{len(list(self.dir.glob("records-*")))}.jsonl', rows)
        return dict(claim='pass', requires=['board'], evidence=[self.entry(path, format='archbench-jsonl', **entry)])

    def test_current_index_passes_no_board_item(self):
        result = acceptance.evaluate(REPO)
        self.assertEqual(result['required'], 61)
        rows = {r['id']: r for r in result['items']}
        self.assertEqual(set(rows), set(acceptance.required_ids(json.loads((REPO/'spec/r41/stage1/plan.json').read_text()))))
        for item_id, item in self.real['items'].items():
            if 'board' in (item.get('requires') or []):
                self.assertNotEqual(rows[item_id]['status'], 'pass', item_id)
        self.assertEqual(rows['S1-05.all_stage1_required']['status'], 'blocked')
        self.assertEqual(result['status'], 'not-accepted')
        self.assertIn('S1-03.device_drain_recovery', acceptance.table(result))

    def test_a_good_board_record_passes(self):
        self.assertEqual(self.status(self.board_item([good_record()])), ('pass', ''))

    def test_section_9_record_refusals(self):
        cases = []
        def mutate(message, change, **entry):
            rec = good_record()
            change(rec)
            cases.append((message, rec, entry))
        mutate('presented as a board measurement', lambda r: r.update(evidence_level='model-prediction-uncalibrated'))
        mutate('presented as a board measurement', lambda r: r['platform'].update(backend='systemc-tlm-grid'))
        mutate('CPU numerical fallback', lambda r: r.update(fallbacks=['aten.add on cpu']))
        mutate('truncated layers or replaced weights', lambda r: r.update(point_sha256='0'*64))
        mutate('truncated layers or replaced weights', lambda r: r.update(model_identity=dict(layers=2)),
               expected_model=dict(layers=4))
        mutate('wrong image', lambda r: r['platform'].update(image_sha256='b'*64))
        mutate('qualified clock evidence states 90000000', lambda r: r['platform'].update(clock_hz=100000000))
        mutate('presented as 100000000 Hz', lambda r: None, claimed_clock_hz=100000000)
        mutate('wrong configuration', lambda r: r['platform']['backend_args'].update(memory_ports=1))
        mutate('wrong configuration', lambda r: None, knobs=dict(TNCA_TRANSFER_POLICY='per-kernel'))
        mutate('descriptors disabled', lambda r: r['platform']['inductor'].update(tensor_descriptors=False))
        mutate('without a reason', lambda r: None, descriptor_path=dict(enabled=True, taken=False))
        mutate('submission time', lambda r: r['timings']['device-graph-makespan'].update(
            boundaries='first accepted launch to last submission'))
        mutate('tuning samples inside formal samples', lambda r: (r['platform']['knobs'].update(TNCA_TUNE='1'),
                                                                 r['time'].update(launches=40)))
        mutate('tuning samples used as formal', lambda r: None, samples='tuning')
        mutate('copied historical PASS', lambda r: r['platform'].update(
            software_tree='files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75'))
        for message, rec, entry in cases:
            status, reason = self.status(self.board_item([rec], **entry))
            self.assertEqual(status, 'fail', message)
            self.assertIn(message, reason)
        # A descriptor path not taken is admissible with its reason.
        status, _ = self.status(self.board_item([good_record()], descriptor_path=dict(
            enabled=True, taken=False, reason='pointer lowering for a gather')))
        self.assertEqual(status, 'pass')

    def test_missing_not_run_changed_and_model_only_evidence(self):
        path = self.evidence('report.json', text='{}')
        missing = dict(claim='pass', requires=['doc'], evidence=[dict(path=str(self.dir/'absent.json'),
                                                                     sha256='0'*64, kind='doc')])
        self.assertEqual(self.status(missing)[0], 'not-measured')
        for state in ('not-run', 'skipped', 'registered-not-run'):
            item = dict(claim='pass', requires=['doc'], evidence=[self.entry(path, kind='doc', status=state)])
            self.assertEqual(self.status(item)[0], 'not-measured', state)
        changed = dict(claim='pass', requires=['doc'], evidence=[dict(self.entry(path, kind='doc'), sha256='0'*64)])
        self.assertEqual(self.status(changed), ('fail', f'{path}: evidence changed since it was indexed'))
        model_only = dict(claim='pass', requires=['board'], evidence=[self.entry(path, kind='systemc')])
        status, reason = self.status(model_only)
        self.assertEqual(status, 'not-measured')
        self.assertIn('cannot stand for a board measurement', reason)
        unformatted = dict(claim='pass', requires=['board'], evidence=[self.entry(path, kind='board')])
        self.assertEqual(self.status(unformatted)[0], 'not-measured')
        historical = 'evidence/r40/v3/R40-04.hbm/board_run_fe1b/hbm14.jsonl'
        copied = dict(claim='pass', requires=['board'], evidence=[dict(path=historical, kind='board',
                      sha256=experiment.digest(REPO/historical), format='archbench-jsonl')])
        status, reason = self.status(copied)
        self.assertEqual(status, 'fail')
        self.assertIn('copied historical PASS', reason)

    def test_final_identity_binds_the_current_software_tree(self):
        path = self.evidence('report.md', text='stage 1 report')
        tree = experiment.software_tree(REPO)
        item = dict(claim='pass', requires=['doc'], final_identity=True,
                    evidence=[self.entry(path, kind='doc', binds=dict(software_tree='files:'+'2'*64))])
        status, reason = self.status(item, 'S1-05.stage1_report')
        self.assertEqual(status, 'not-measured')
        self.assertIn('final tree', reason)
        item['evidence'][0]['binds'] = dict(software_tree=tree)
        self.assertEqual(self.status(item, 'S1-05.stage1_report')[0], 'pass')
        stale = good_record()
        board = self.board_item([stale])
        board['final_identity'] = True
        self.assertEqual(self.status(board, 'S1-05.frozen_board_anchor_reproduction')[0], 'not-measured')
        stale['platform']['software_tree'] = tree
        board = self.board_item([stale])
        board['final_identity'] = True
        self.assertEqual(self.status(board, 'S1-05.frozen_board_anchor_reproduction')[0], 'pass')

    def test_missing_original_large_point(self):
        template = good_record()
        plan = json.loads((REPO/'spec/r41/stage1/plan.json').read_text())
        rows, seen = [], set()
        for suite in plan['original_suites']:
            for point in json.loads((REPO/suite['path']).read_text())['points']:
                if point['id'] in seen:
                    continue
                seen.add(point['id'])
                rec = copy.deepcopy(template)
                rec.update(point_id=point['id'], point_sha256=point['sha256'])
                rows.append(rec)
        self.assertEqual(len(rows), 113)
        item = self.board_item(rows)
        item['coverage'] = 'original_suites'
        self.assertEqual(self.status(item, 'S1-05.original_full_suite_on_90mhz')[0], 'pass')
        dropped = [r for r in rows if r['point_id'] != 'model.gpt2_block/large/fp16-d1024-h16-b1-s512']
        item = self.board_item(dropped)
        item['coverage'] = 'original_suites'
        status, reason = self.status(item, 'S1-05.original_full_suite_on_90mhz')
        self.assertEqual(status, 'fail')
        self.assertIn('large: model.gpt2_block/large', reason)

    def test_fitting_on_the_holdout(self):
        holdout = json.loads((REPO/'spec/r40/suites/R40-04.model-holdout.json').read_text())['points'][0]['id']
        fit = self.evidence('fit.json', text=json.dumps(dict(fit_rows='calibration-only',
                                                             rows=[dict(point=holdout)])))
        item = dict(claim='pass', requires=['systemc'], evidence=[self.entry(fit, kind='systemc', format='fit-report')])
        status, reason = self.status(item, 'S1-01.calibration_validation_holdout_separation')
        self.assertEqual(status, 'fail')
        self.assertIn('fitting on the holdout', reason)

    def test_board_reports_and_derived_items(self):
        residual = self.evidence('residual.json', text=json.dumps(dict(status='not-comparable')))
        item = dict(claim='pass', requires=['board'], evidence=[self.entry(residual, format='prospective-residual')])
        self.assertEqual(self.status(item, 'S1-01.prospective_prediction_board_residual')[0], 'not-measured')
        failed = self.evidence('tune.json', text=json.dumps(dict(status='fail')))
        item = dict(claim='pass', requires=['board'], evidence=[self.entry(failed, format='tune-check')])
        self.assertEqual(self.status(item, 'S1-03.device_endpoint_ranking')[0], 'fail')
        items = {'S1-00.budgets': dict(claim='fail', note='budget exceeded'),
                 'S1-05.all_stage1_required': dict(derived_from=['S1-00'])}
        result = acceptance.evaluate(REPO, self.index(items))
        rows = {r['id']: r for r in result['items']}
        self.assertEqual(rows['S1-05.all_stage1_required']['status'], 'fail')
        self.assertEqual(rows['S1-00.carryover']['status'], 'not-measured')

    def test_index_must_bind_the_frozen_plan(self):
        bad = dict(self.real, plan_sha256='0'*64)
        path = self.dir/'bad.json'
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'frozen plan'):
            acceptance.evaluate(REPO, path)
        path.write_text(json.dumps(dict(self.real, items={'S1-09.invented': {}})))
        with self.assertRaisesRegex(ValueError, 'outside the frozen plan'):
            acceptance.evaluate(REPO, path)


if __name__ == '__main__':
    unittest.main()
