"""A device description built as the image (target u280-multicore).

The description (spec/devices) is the machine. The target must build its
platform, the run takes its kernel clock, the packager is given the kernel
variant and every parameter the description generates, and the generator and
the description are the checkout's own -- never an installed package's, nor a
stale staged copy of the profiles.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
TOOLS = HERE.parent
for path in (TOOLS, REPO / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tritonchip_fpga_harness.adapters.u280 import U280FlowAdapter
from tritonchip_fpga_harness.core import plan, run_profile
from tritonchip_fpga_harness.image import ImageError, applied, device_parameters, write_device_config
from tritonchip_fpga_harness.schema import SchemaError, validate_manifest
from tnca import devices

DEVICE = "u280-3core-r40"
FILES = ("python/tnca/__init__.py", "python/tnca/devices.py", "python/tnca/profiles.py", "python/tnca/errors.py",
         "spec/r1_profiles.json", f"spec/devices/{DEVICE}.json", "flow/fpga/package_xo.tcl")


def manifest(device: str = DEVICE, **extra) -> dict:
    return dict({"candidate_id": "d", "target": "u280-multicore", "mode": "full", "image": {"device": device}}, **extra)


def checkout(td: str) -> Path:
    """A copy of what the generator reads, to change."""
    root = Path(td) / "repo"
    for f in FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, root / f)
    return root


@unittest.skipUnless(all((REPO / f).is_file() for f in FILES), "needs this checkout's generator and descriptions")
class DeviceImage(unittest.TestCase):
    def test_an_image_is_a_profile_or_a_device(self) -> None:
        validate_manifest(manifest())
        for bad in ({"device": ""}, {"device": DEVICE, "profile": "core-vrf-sync-v5"}, {"devices": DEVICE},
                    {"device": 3}):
            with self.subTest(image=bad), self.assertRaises(SchemaError):
                validate_manifest(dict(manifest(), image=bad))

    def test_the_run_takes_the_descriptions_clock(self) -> None:
        profile = run_profile(HERE, REPO, manifest())
        self.assertEqual((profile.expected_clock, profile.data["device_clock_mhz"], profile.data["device"]),
                         (90.0, 90.0, DEVICE))

    def test_the_packager_and_the_link_are_given_the_machine(self) -> None:
        impl = U280FlowAdapter()._impl(REPO, run_profile(HERE, REPO, manifest()))
        self.assertEqual(impl.package_command(Path("/build"))[-2:], ("xcu280-fsvh2892-2L-e", "multicore-hbm"))
        env = impl.command_environment()
        self.assertEqual((env["TNCA_KERNEL_MHZ"], env["TNCA_BOARD"]), ("90", "u280"))

    def test_plan_is_the_descriptions_image(self) -> None:
        p = plan(HERE, REPO, manifest(), "mock")
        self.assertEqual((p["image"]["device"], p["image"]["profile"]), (DEVICE, "core-vrf-sync-v5"))
        self.assertEqual(p["image"]["parameters"], devices.image_params(devices.load(DEVICE)))
        self.assertEqual(len(p["image"]["parameters"]), 41)
        self.assertEqual((p["clock_mhz"], p["profile_contract"]["kernel_clock_mhz"]), (90, 90))
        with self.assertRaisesRegex(SchemaError, "clock_mhz 100 does not match target 90"):
            plan(HERE, REPO, manifest(clock_mhz=100), "mock")

    def test_the_record_and_the_package_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            record = write_device_config(REPO, DEVICE, Path(td))
            tcl = (Path(td) / "image_config.tcl").read_text(encoding="utf-8")
        self.assertEqual(devices.parse_image_tcl(tcl), record["parameters"])
        self.assertEqual((record["device"], record["profile"], record["fingerprint"]),
                         (DEVICE, "core-vrf-sync-v5", devices.fingerprint(devices.load(DEVICE))))
        self.assertEqual(record["platform"]["kernel_clock_mhz"], 90)
        log = "".join(f"TNCA_IMAGE_PARAM {k}={v}\n" for k, v in record["parameters"].items() if k != "SETS")
        self.assertEqual(applied(log, record["parameters"]), ["SETS"])

    def test_a_described_target_builds_only_descriptions(self) -> None:
        with self.assertRaisesRegex(SchemaError, "needs image.device"):
            run_profile(HERE, REPO, {"candidate_id": "d", "target": "u280-multicore", "mode": "full"})

    def test_a_machine_the_target_does_not_build_is_refused(self) -> None:
        with self.assertRaisesRegex(SchemaError, "platform.variant is 'multicore-hbm', the target's is 'none'"):
            run_profile(HERE, REPO, manifest(target="u280"))
        with tempfile.TemporaryDirectory() as td:
            root = checkout(td)
            d = json.loads((root / f"spec/devices/{DEVICE}.json").read_text(encoding="utf-8"))
            d.update(name="two-core")
            d["platform"]["core_count"] = 2
            d["uncore"]["MEM_PORTS"] = 2
            (root / "spec/devices/two-core.json").write_text(json.dumps(d), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "platform.core_count is 2, the target's is 3"):
                run_profile(HERE, root, manifest("two-core"))

    def test_an_unknown_or_misnamed_device_is_refused(self) -> None:
        with self.assertRaisesRegex(SchemaError, "no device description"):
            plan(HERE, REPO, manifest("no-such-machine"), "mock")
        with tempfile.TemporaryDirectory() as td:
            root = checkout(td)
            shutil.copy2(root / f"spec/devices/{DEVICE}.json", root / "spec/devices/copy.json")
            with self.assertRaisesRegex(SchemaError, "a description is named for its file"):
                run_profile(HERE, root, manifest("copy"))

    def test_the_generator_and_the_description_are_the_checkouts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = checkout(td)
            path = root / f"spec/devices/{DEVICE}.json"
            d = json.loads(path.read_text(encoding="utf-8"))
            d["uncore"]["SETS"] = 64
            path.write_text(json.dumps(d), encoding="utf-8")
            params = device_parameters(root, DEVICE)
        self.assertEqual((params["SETS"], params["SET_W"]), (64, 6))
        self.assertEqual(device_parameters(REPO, DEVICE)["SETS"], 32)

    def test_a_stale_staged_profile_file_is_refused(self) -> None:
        # profiles.py reads python/tnca/data first; git ignores it.
        with tempfile.TemporaryDirectory() as td:
            root = checkout(td)
            spec = json.loads((root / "spec/r1_profiles.json").read_text(encoding="utf-8"))
            spec["profiles"]["core-vrf-sync-v5"]["elaboration"]["MTX_ABUF"] = 0
            staged = root / "python/tnca/data/spec/r1_profiles.json"
            staged.parent.mkdir(parents=True)
            staged.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ImageError, "restage it"):
                device_parameters(root, DEVICE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
