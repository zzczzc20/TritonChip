"""A device profile built as the image: the parameters, the manifest, the record, the board's judgement."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.core import plan, submit
from tritonchip_fpga_harness.image import ImageError, applied, image_parameters, write_image_config
from tritonchip_fpga_harness.schema import SchemaError, validate_manifest

V5 = {"FE_NOBUBBLE": 1, "FPU_OVERLAP": 1, "HAS_ATOMICS": 1, "HAS_SFU": 1, "HAS_SYS_ATOMICS": 0,
      "HAS_TENSOR_XPOSE": 1, "LSU_INFLIGHT": 2, "MTX_ABUF": 1, "NUM_RCTX": 4, "RCTX_SWITCH": 2, "RCTX_W": 2,
      "RPAIR": 1, "RSLICE": 1, "USE_BANKED_VRF": 1, "VREGS_PER_RCTX": 64, "VRF_BYPASS": 1, "VRF_PREFETCH": 1}
FILES = ("rtl/board/tnca_u50_xo.v", "rtl/common/tnca_uarch_defaults.vh", "spec/r1_profiles.json",
         "flow/fpga/package_xo.tcl")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(all((REPO / f).is_file() for f in FILES), "needs this checkout's flow and profiles")
class Image(unittest.TestCase):
    def test_v5_is_the_signed_off_parameters_and_v2_the_wrapper_defaults(self) -> None:
        self.assertEqual(image_parameters(REPO, "core-vrf-sync-v5"), V5)
        v2 = image_parameters(REPO, "core-vrf-sync-v2")
        self.assertEqual((v2["NUM_RCTX"], v2["RCTX_SWITCH"], v2["MTX_ABUF"], v2["USE_BANKED_VRF"]), (2, 0, 0, 1))

    def test_what_the_image_cannot_carry_is_refused_not_dropped(self) -> None:
        with self.assertRaisesRegex(ImageError, "cores"):
            image_parameters(REPO, "system-2x2")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for f in FILES:
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / f, root / f)
            spec = json.loads((root / "spec/r1_profiles.json").read_text())
            spec["profiles"]["core-vrf-sync-v5"]["elaboration"]["VLEN"] = 32
            (root / "spec/r1_profiles.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ImageError, "VLEN"):
                image_parameters(root, "core-vrf-sync-v5")

    def test_the_record_and_the_log_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            record = write_image_config(REPO, "core-vrf-sync-v5", Path(td))
            tcl = (Path(td) / "image_config.tcl").read_text()
            self.assertTrue(tcl.startswith("set TNCA_IMAGE_PARAMS [dict create FE_NOBUBBLE 1 "))
            self.assertEqual(record["parameters"], V5)
        log = "\n".join(f"TNCA_IMAGE_PARAM {k}={v}" for k, v in V5.items() if k != "MTX_ABUF") + "\nTNCA_IMAGE_PARAM MTX_ABUF=0\n"
        self.assertEqual(applied(log, V5), ["MTX_ABUF"])

    def test_the_manifest(self) -> None:
        base = {"candidate_id": "i", "target": "u280", "mode": "full"}
        validate_manifest(dict(base, image={"profile": "core-vrf-sync-v5"},
                               board={"backend": "fpga-u280", "profile": "core-vrf-sync-v5"}))
        for bad in ({"profile": ""}, {"profile": "x", "hbm": True}, "core-vrf-sync-v5"):
            with self.assertRaises(SchemaError):
                validate_manifest(dict(base, image=bad))
        with self.assertRaisesRegex(SchemaError, "image's profile"):
            validate_manifest(dict(base, image={"profile": "core-vrf-sync-v5"},
                                   board={"backend": "fpga-u280", "profile": "core-vrf-sync-v2"}))


@unittest.skipUnless(all((REPO / f).is_file() for f in FILES), "needs this checkout's flow and profiles")
class ImageRun(unittest.TestCase):
    def test_a_played_image_build_keeps_its_record_and_the_board_judges_by_it(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            for f in FILES:
                (repo / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / f, repo / f)
            git(repo, "init", "-q")
            git(repo, "add", "-A")
            git(repo, "commit", "-q", "-m", "c")
            state = Path(td) / "state"
            conformance = [sys.executable, "-c", "import json,os,pathlib; r=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR']);"
                           " (r/'board').mkdir(exist_ok=True); (r/'board'/'conformance.json').write_text(json.dumps([{'id':'X','passed':True}]))"]
            manifest = {"candidate_id": "img", "target": "u280", "mode": "full", "image": {"profile": "core-vrf-sync-v5"},
                        "board": {"backend": "fpga-u280", "profile": "core-vrf-sync-v5", "program": False,
                                  "conformance": {"command": conformance}}}
            self.assertEqual(plan(HERE, repo, manifest, "mock")["image"]["parameters"], V5)
            result = submit(HERE, repo, manifest, state, "mock")
            self.assertEqual(result["status"], "PASS", result.get("measurement_error"))
            run_dir = state / "runs" / result["run_id"]
            self.assertTrue((run_dir / "image_config.tcl").is_file())
            self.assertEqual(result["measurement"]["image_profile"]["source"], "image")


if __name__ == "__main__":
    unittest.main()
