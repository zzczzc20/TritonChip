from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.worker import dispatch_manifest


class WorkerTests(unittest.TestCase):
    def test_dispatch_validates_board_and_returns_remote_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "ssh"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "request=json.load(sys.stdin)\n"
                "assert request['board']=='u280'\n"
                "print(json.dumps({'schema_version':1,'ok':True,'operation':'submit','board':'u280','result':{'status':'PASS'}}))\n",
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            result = dispatch_manifest(
                host="u280-worker",
                remote_repo="/srv/tritonchip",
                operation="submit",
                adapter="u280",
                ssh_binary=str(fake),
                manifest={"candidate_id": "w-1", "target": "u280-r38-pr5", "mode": "full"},
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["remote"]["result"]["status"], "PASS")
        self.assertIn("worker", " ".join(result["argv"]))


if __name__ == "__main__":
    unittest.main()
