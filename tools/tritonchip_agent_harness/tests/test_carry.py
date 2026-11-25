"""carry-baseline (2026-11-09): a base's kept results move to a platform commit only when no measurement reads
what changed."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import board, carry, evaluator  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402
from tritonchip_agent_harness.workspace import git  # noqa: E402


def commit(repo: Path, files: dict, message: str) -> str:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git("-C", str(repo), "add", "-A")
    git("-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
        "commit", "-q", "-m", message)
    return git("-C", str(repo), "rev-parse", "HEAD").decode().strip()


def task(platform: str) -> Task:
    return Task(task_id="t", kind="cross-layer", base_commit="a" * 40, design=("compiler/",), evaluator_owned=(),
                hide=(), instruction="", reference=None, wall_seconds=60,
                evaluation={"profile": "p", "validation": ["v"], "final": ["f"], "device": "d",
                            "platform": {"commit": platform, "paths": ["python/tnca/backends/board_relay.py"]}},
                services={"fpga_slow": {"base_image_sha256": "c" * 64}}, path=HERE, sha256="")


class Carry(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        git("init", "-q", str(self.repo))
        self.old = commit(self.repo, {"python/tnca/backends/board_relay.py": "race\n", "python/tnca/x.py": "1\n"}, "a")
        self.relay = commit(self.repo, {"python/tnca/backends/board_relay.py": "fixed\n"}, "relay")
        self.other = commit(self.repo, {"python/tnca/x.py": "2\n"}, "other")
        self.state = Path(tempfile.mkdtemp())
        old = task(self.old)
        for root, key in ((self.state / "baselines", evaluator.baseline_key(old)),
                          (self.state / "board-baselines", board.baseline_key(old))):
            root.mkdir(parents=True)
            (root / f"{key}.json").write_text(json.dumps({"points": {"v": {"cycles": 100}}}))

    def test_a_relay_only_change_carries_both_measurements_with_their_provenance(self):
        new = task(self.relay)
        out = carry.carry(self.state, self.repo, new, self.old)
        self.assertEqual((out["simulation"]["status"], out["card"]["status"]), ("carried", "carried"))
        self.assertEqual(out["changed"], ["python/tnca/backends/board_relay.py"])
        sim = self.state / "baselines" / f"{evaluator.baseline_key(new)}.json"
        self.assertEqual(json.loads(sim.read_text()), {"points": {"v": {"cycles": 100}}})
        prov = json.loads((self.state / "baselines" / f"{evaluator.baseline_key(new)}.carried.json").read_text())
        self.assertEqual((prov["from_platform"], prov["to_platform"]), (self.old, self.relay))
        self.assertEqual(carry.carry(self.state, self.repo, new, self.old)["simulation"]["status"], "already there")

    def test_any_other_change_is_measured_again(self):
        with self.assertRaises(carry.CarryError) as got:
            carry.carry(self.state, self.repo, task(self.other), self.old)
        self.assertIn("python/tnca/x.py", str(got.exception))
        with self.assertRaises(carry.CarryError):
            carry.carry(self.state, self.repo, task(self.old), self.old)


if __name__ == "__main__":
    unittest.main()
