from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.tritonchip_fpga_harness.budget import BudgetError, TokenLedger


class BudgetTests(unittest.TestCase):
    def test_reserve_settle_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = TokenLedger(Path(td) / "ledger.json", 100)
            ledger.reserve("req-1", 60)
            ledger.reserve("req-1", 60)
            with self.assertRaises(BudgetError):
                ledger.reserve("req-2", 50)
            value = ledger.settle("req-1", 40)
            self.assertEqual(value["spent"], 40)
            self.assertEqual(value["reserved"], 0)

    def test_unknown_usage_keeps_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = TokenLedger(Path(td) / "ledger.json", 100)
            value = ledger.reserve("req-1", 60)
            self.assertEqual(value["reserved"], 60)
            value = ledger.settle("req-1", None)
            self.assertEqual(value["reserved"], 60)
            self.assertEqual(value["entries"]["req-1"]["state"], "UNKNOWN")
            with self.assertRaises(BudgetError):
                ledger.reserve("req-2", 41)


if __name__ == "__main__":
    unittest.main()
