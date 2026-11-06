"""A failed exact result or unstable warm/cold result must stop a run."""
import unittest
from fragment_case import require_matching_bits
class BitwiseRefusal(unittest.TestCase):
    def test_both_checks_are_required(self):
        for exact,cold in ((True,True),(False,True),(True,False),(False,False)):
            row={'bitwise_reference':{'equal':exact},'cold_bitwise_same':cold}
            if exact and cold:require_matching_bits(row)
            else:
                with self.assertRaises(ValueError):require_matching_bits(row)
if __name__=='__main__':unittest.main()
