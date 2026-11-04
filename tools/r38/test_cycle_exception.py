#!/usr/bin/env python3
"""Exercise the narrow FP8 exception and the profiler's counter controls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cycle_exception as X


class CycleExceptionTests(unittest.TestCase):
    def setUp(self):
        self.scope = X.load_rule()
        self.cases = sorted(X.APPROVED_CASES)
        self.reference = X.ROOT / 'spec/r40/b39/core_sim_cycles.json'

    def test_only_the_recorded_one_cycle_is_accepted(self):
        for case in self.cases:
            for reference in X.APPROVED_REFERENCES:
                self.assertEqual('exact', X.cycle_match(case, 'core-sim', 2, 10897, 10897, reference, self.scope))
                self.assertEqual('authorized', X.cycle_match(case, 'core-sim', 2, 10897, 10898, reference, self.scope))
            for frozen, actual in [(10897, 10899), (10897, 10896), (10896, 10897), (None, 10898),
                                   (10897, None), (True, 1)]:
                self.assertEqual('mismatch', X.cycle_match(case, 'core-sim', 2, frozen, actual,
                                                           self.reference, self.scope))
            for profile, level, reference in [('core-vrf-sync-v1', 2, self.reference),
                                               ('core-sim', 1, self.reference),
                                               ('core-sim', 2, Path('/tmp/unapproved.json'))]:
                self.assertEqual('mismatch', X.cycle_match(case, profile, level, 10897, 10898,
                                                           reference, self.scope))
        self.assertEqual('mismatch', X.cycle_match('another-case', 'core-sim', 2, 10897, 10898,
                                                   self.reference, self.scope))

    def test_both_frozen_references_are_unchanged(self):
        for reference in X.APPROVED_REFERENCES:
            frozen = json.loads(reference.read_text())
            self.assertEqual('core-sim', frozen['profile'])
            self.assertEqual(2, frozen['level'])
            self.assertEqual([10897, 10897], [frozen['cycles'][case] for case in self.cases])

    def test_both_runs_and_entire_inventory_remain_required(self):
        frozen = {self.cases[0]: 10897, self.cases[1]: 10897, 'other': 42}
        current = {self.cases[0]: 10898, self.cases[1]: 10898, 'other': 42}

        def compare(first, second):
            return X.compare_cycles(frozen, first, second, 'core-sim', 2, self.reference, self.scope)

        problems, accepted = compare(current, current)
        self.assertFalse(problems)
        self.assertEqual(set(self.cases), {case for case, _, _ in accepted})
        for first, second in [({**current, 'other': 43}, current),
                              (current, {**current, self.cases[0]: 10897}),
                              ({**current, self.cases[0]: 10899}, current),
                              ({k: v for k, v in current.items() if k != 'other'}, current),
                              ({**current, 'unexpected': 42}, current)]:
            self.assertTrue(compare(first, second)[0])

    def test_counter_controls_require_passing_identical_program_cycles(self):
        on = {case: {'case': case, 'ok': True, 'cycles': 10, 'program_cycles': [4, 6]}
              for case in self.cases}
        off = [copy.deepcopy(on[case]) for case in self.cases]
        check = lambda rows: X.counter_control_problems(on, rows, self.cases)
        self.assertFalse(check(off))
        mutations = [off[:1], off + [copy.deepcopy(off[0])],
                     off + [{'case': 'other', 'ok': True}], copy.deepcopy(off),
                     copy.deepcopy(off), copy.deepcopy(off), copy.deepcopy(off)]
        mutations[3][0]['ok'] = False
        mutations[4][0]['program_cycles'] = [5, 5]  # equal total does not suffice
        mutations[5][0]['cycles'] = 11
        mutations[6][0]['program_cycles'] = [True, 9]
        for rows in mutations:
            self.assertTrue(check(rows))

    def test_malformed_or_expanded_rule_is_rejected(self):
        rule = json.loads(X.RULE.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rule.json'
            for change in ('extra_cycle', 'frozen', 'profile', 'case', 'authority'):
                invalid = copy.deepcopy(rule)
                if change == 'extra_cycle':
                    invalid['scope']['allowed_extra_cycles'] = 2
                elif change == 'frozen':
                    invalid['scope']['frozen_cycles'] = 10896
                elif change == 'profile':
                    invalid['scope']['profile'] = 'core-vrf-sync-v1'
                elif change == 'case':
                    invalid['scope']['cases'].append('another-case')
                else:
                    invalid['authority']['quote'] = ''
                path.write_text(json.dumps(invalid))
                with patch.object(X, 'RULE', path), self.assertRaises(ValueError):
                    X.load_rule()


if __name__ == '__main__':
    unittest.main()
