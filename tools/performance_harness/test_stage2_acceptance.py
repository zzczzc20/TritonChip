"""Refusal tests for the audit ledger; no hardware or Torch dependency."""
import copy
import json
from pathlib import Path
import unittest
import stage2_acceptance as acceptance

class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads((acceptance.ROOT/acceptance.INDEX).read_text())

    def check(self, index):
        return acceptance.evaluate(acceptance.ROOT, index)

    def test_current_audit_is_valid_but_product_is_not_accepted(self):
        result = self.check(self.index)
        self.assertEqual(result['integrity'], 'passed')
        self.assertEqual(result['status'], 'not accepted')

    def test_missing_required_item_is_refused(self):
        index = copy.deepcopy(self.index)
        index['items'].pop()
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_duplicate_item_is_refused(self):
        index = copy.deepcopy(self.index)
        index['items'].append(index['items'][0])
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_changed_evidence_is_refused(self):
        index = copy.deepcopy(self.index)
        index['evidence_sha256']['evidence/r41/acceptance/board-records.tgz'] = '0'*64
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_unhashed_evidence_is_refused(self):
        index = copy.deepcopy(self.index)
        index['items'][0]['evidence'].append('unregistered-model-result.json')
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_missing_closure_plan_is_refused(self):
        index = copy.deepcopy(self.index)
        next(row for row in index['items'] if row['status'] != 'passed')['closure_plan'] = None
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_review_only_entry_must_not_also_be_hashed(self):
        index = copy.deepcopy(self.index)
        name = next(iter(index['evidence_review_only']))
        index['evidence_sha256'][name] = '0'*64
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_review_only_entry_needs_reviewed_hash_and_note(self):
        index = copy.deepcopy(self.index)
        name = next(iter(index['evidence_review_only']))
        index['evidence_review_only'][name].pop('note')
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_descoping_cannot_erase_a_historical_failure(self):
        index = copy.deepcopy(self.index)
        row = next(row for row in index['items'] if row['status'] == 'descoped')
        row['prior_status'] = 'failed'
        self.assertEqual(self.check(index)['integrity'], 'failed')

    def test_chat_cannot_pass_without_checks_and_hashed_evidence(self):
        index = copy.deepcopy(self.index)
        index['author_scope']['models'][0]['status'] = 'passed'
        self.assertEqual(self.check(index)['integrity'], 'failed')

if __name__ == '__main__':
    unittest.main()
