#!/usr/bin/env python3
"""Device descriptions (spec/devices, python/tnca/devices.py).

The one test that matters most: R40's description generates the 39
parameters of the image R40 delivered (attempt 3), which until now existed
only as a hand-written evidence file, and states the two that image left at
the top's defaults. The rest hold the generator to the same per-core rule as
tools/r40/image_profile.py and check that a description that could not be
built is refused, naming why.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

from tnca import devices, profiles  # noqa: E402

ATTEMPT03 = REPO / "evidence/r40/v2/convergence/attempt03/image.tcl"


def _image_profile():
    spec = importlib.util.spec_from_file_location("image_profile", REPO / "tools/r40/image_profile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class R40Description(unittest.TestCase):
    def setUp(self) -> None:
        self.desc = devices.load("u280-3core-r40")

    def test_generates_the_delivered_image(self) -> None:
        # Attempt 3 set 39 parameters and left PHYS_VREGS and PHYS_W at the
        # multicore top's defaults, which are what v5's four contexts of 64
        # registers make. The description states them: the same machine.
        delivered = devices.parse_image_tcl(ATTEMPT03.read_text(encoding="utf-8"))
        self.assertEqual(len(delivered), 39)
        top = (REPO / "rtl/board/tnca_multicore_hbm_xo.v").read_text(encoding="utf-8")
        defaults = {k: int(v) for k, v in re.findall(r"\b(PHYS_VREGS|PHYS_W)=(\d+)", top)}
        self.assertEqual(defaults, dict(PHYS_VREGS=256, PHYS_W=8))
        self.assertEqual(devices.image_params(self.desc), dict(delivered, **defaults))

    def test_image_tcl_round_trips(self) -> None:
        text = devices.image_tcl(self.desc)
        self.assertEqual(devices.parse_image_tcl(text), devices.image_params(self.desc))
        self.assertTrue(text.startswith("set TNCA_IMAGE_PARAMS [dict create "))

    def test_backend_and_model_follow_the_description(self) -> None:
        args = devices.backend_args(self.desc, xclbin="/x/tnca_u280.xclbin")
        self.assertEqual(args, dict(image_profile="core-vrf-sync-v5", core_count=3, memory_ports=3,
                                    hbm_dma=True, hbm_bytes=4 << 30, device_clock=True,
                                    xclbin="/x/tnca_u280.xclbin"))
        self.assertEqual(devices.systemc_overrides(self.desc),
                         dict(nx=3, ny=1, sets=32, ways=8, outlets=3, active_outlets=3, outstanding=8,
                              clock_hz=90_000_000))


class PerCoreRule(unittest.TestCase):
    def test_same_rule_as_image_profile(self) -> None:
        ip = _image_profile()
        singles = [n for n, p in profiles.spec()["profiles"].items() if p.get("num_cores", 1) == 1]
        self.assertGreater(len(singles), 5)
        for name in singles:
            with self.subTest(profile=name):
                self.assertEqual(devices.core_params(name), ip.parameters(name))

    def test_a_multicore_profile_is_not_a_core(self) -> None:
        with self.assertRaisesRegex(devices.DeviceError, "a device names one core"):
            devices.core_params("system-2x2")


class Refusals(unittest.TestCase):
    def setUp(self) -> None:
        self.desc = devices.load("u280-3core-r40")

    def refused(self, desc, *needles):
        with self.assertRaises(devices.DeviceError) as cm:
            devices.validate(desc)
        for needle in needles:
            self.assertIn(needle, str(cm.exception))

    def test_each_problem_is_named(self) -> None:
        d = copy.deepcopy(self.desc)
        d["uncore"]["SETS"] = 48
        d["uncore"]["SET_W"] = 6
        d["uncore"]["MEM_PORTS"] = 4
        del d["platform"]["kernel_clock_mhz"]
        d["extra"] = 1
        self.refused(d, "uncore.SETS=48", "power of two", "uncore.SET_W is derived",
                     "uncore.MEM_PORTS=4", "platform.kernel_clock_mhz is missing", "unknown field 'extra'")

    def test_ports_cannot_exceed_cores(self) -> None:
        d = copy.deepcopy(self.desc)
        d["platform"]["core_count"] = 2
        self.refused(d, "MEM_PORTS=3 exceeds platform.core_count=2")

    def test_unknown_and_multicore_profiles(self) -> None:
        d = copy.deepcopy(self.desc)
        d["per_core_profile"] = "no-such-profile"
        self.refused(d, "is not a profile")
        d["per_core_profile"] = "system-2x2"
        self.refused(d, "a device names one core")

    def test_booleans_are_not_integers(self) -> None:
        d = copy.deepcopy(self.desc)
        d["uncore"]["NUCA_CACHE_DATA_PIPELINE"] = True
        self.refused(d, "uncore.NUCA_CACHE_DATA_PIPELINE=True")

    def test_platform_is_checked_against_the_owner(self) -> None:
        locked = dict(self.desc["platform"])
        self.assertEqual(devices.check_platform(self.desc, locked), [])
        d = copy.deepcopy(self.desc)
        d["platform"]["kernel_clock_mhz"] = 100
        d["platform"]["core_count"] = 3
        self.assertEqual(devices.check_platform(d, locked),
                         ["platform.kernel_clock_mhz: 100, fixed at 90"])


class Derived(unittest.TestCase):
    def test_widths_follow_their_fields(self) -> None:
        for out_n, id_w in ((1, 1), (2, 1), (3, 2), (8, 3), (9, 4), (16, 4)):
            u = dict(SETS=64, WAYS=4, AXI_OUT_N=out_n)
            self.assertEqual(devices.derived(u), dict(SET_W=6, WAY_W=2, AXI_ID_W=id_w))

    def test_the_register_file_follows_the_core(self) -> None:
        # The top would otherwise build a two-context core with v5's 256.
        d = devices.load("u280-3core-r40")
        d["per_core_profile"] = "core-vrf-sync-v2"
        p = devices.image_params(devices.validate(d))
        self.assertEqual((p["NUM_RCTX"], p["VREGS_PER_RCTX"], p["PHYS_VREGS"], p["PHYS_W"]), (2, 64, 128, 7))

    def test_the_model_cannot_route_to_some_outlets(self) -> None:
        d = devices.load("u280-3core-r40")
        for ports, ok in ((1, True), (2, False), (3, True)):
            d["uncore"]["MEM_PORTS"] = ports
            if ok:
                self.assertEqual(devices.systemc_overrides(d)["active_outlets"], ports)
            else:
                with self.assertRaisesRegex(devices.DeviceError, "one outlet or to all"):
                    devices.systemc_overrides(d)

    def test_fingerprint_is_the_machine(self) -> None:
        a = devices.load("u280-3core-r40")
        b = json.loads(json.dumps(a))
        b["description"] = "the same machine, described differently"
        self.assertEqual(devices.fingerprint(a), devices.fingerprint(b))
        b["uncore"]["SETS"] = 64
        self.assertNotEqual(devices.fingerprint(a), devices.fingerprint(b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
