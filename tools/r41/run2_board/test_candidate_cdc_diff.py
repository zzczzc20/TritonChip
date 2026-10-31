import unittest
import candidate_cdc_diff


class CdcDiffTests(unittest.TestCase):
    def test_exact_rule_and_endpoints_and_duplicate_rejection(self):
        old = '1 CDC-10 Critical depth-2 source-a destination\n'
        new = '2 CDC-10 Critical depth-2 source-b destination\n'
        self.assertEqual(candidate_cdc_diff.endpoints(old), {('CDC-10', 'source-a', 'destination')})
        self.assertNotEqual(candidate_cdc_diff.endpoints(old), candidate_cdc_diff.endpoints(new))
        with self.assertRaises(ValueError):
            candidate_cdc_diff.endpoints(old + old)
        with self.assertRaises(ValueError):
            candidate_cdc_diff.endpoints('CDC report unavailable')


if __name__ == '__main__':
    unittest.main()
