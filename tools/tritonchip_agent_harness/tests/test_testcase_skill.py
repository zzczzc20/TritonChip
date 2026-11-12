"""The test-case skill (skills/testcase-design): equal to the evaluation's own code, free of
the benchmark, and delivered to every run the same way.

Equivalence: the skill's copies of the protocol's `compare`, the reference, the bound and
`spec_peak` give the same answers as ArchBench's originals on the same fixtures, and each
benchmark case (benchmark_cases/) is its point bit for bit, so what an agent measures with
the skill is what the evaluation measures. Leak: the skill gives the points in its own
INDEX.json and the classes in classes.json; nothing in it names any other point, its
parameters, its actual input shapes or a tier size, no class it does not list, and no
worked example is a hidden point's workload. No task that gives the skill by default hides
a given point. The leak check draws its catalog from the live registry, so a point added
later is covered. Delivery:
the skill never enters a workspace (it reaches the container by mount), every CLI has a
skills directory to link it into, and the prompt names it at the mounted path.

The checks that need torch (compare, reference, demand, actual shapes, the examples)
are skipped where torch is not installed and must be run where it is.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SKILL = HERE / "skills" / "testcase-design"
R38 = ROOT / "tools" / "r38" / "archbench"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT))

from tritonchip_agent_harness import adapters, runner, skills  # noqa: E402
from tritonchip_agent_harness.siteconfig import Account  # noqa: E402
from tritonchip_agent_harness.task import POINT_SETS, load_task, render  # noqa: E402

SKILL_PATH = f"{adapters.SKILLS_DIR}/testcase-design"
from tritonchip_agent_harness.workspace import hidden  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


skill_bound = load("skill_bound", SKILL / "scripts" / "bound.py")
r38_bounds = load("r38_bounds", R38 / "bounds.py")
r38_peaks = load("r38_peaks", R38 / "peaks.py")

PROFILES = list(json.loads((ROOT / "spec" / "r1_profiles.json").read_text())["profiles"])


def single_core_profiles():
    out = []
    for p in PROFILES:
        try:
            r38_peaks.spec_peak(p)
        except ValueError:
            continue
        out.append(p)
    return out


def suite_points():
    for suite in sorted((ROOT / "archbench" / "suites").glob("*.json")):
        if suite.name.startswith("."):
            continue
        for pt in json.loads(suite.read_text(encoding="utf-8"))["points"]:
            yield pt


_BUILT = None


def built_points():
    """Every registry point built once (make(0)), shared by the torch-dependent leak checks."""
    global _BUILT
    if _BUILT is None:
        from archbench import registry
        _BUILT = [(p, *p.make(0)) for p in registry.points()]
    return _BUILT


CASES_DIR = SKILL / "benchmark_cases"
INDEX = json.loads((CASES_DIR / "INDEX.json").read_text(encoding="utf-8"))
VISIBLE = {c["point"] for c in INDEX["cases"]}
CLASSES = json.loads((SKILL / "classes.json").read_text(encoding="utf-8"))["classes"]


def skill_text() -> str:
    return "\n".join(f.read_text(encoding="utf-8", errors="replace")
                     for f in sorted(SKILL.rglob("*"))
                     if f.is_file() and "__pycache__" not in f.parts and not f.name.startswith("."))


def core_only(peak: dict) -> dict:
    """The evaluator's one-core spec_peak without its memory port: the skill keeps one core's compute rates
    only, and takes memory from the whole chip (2026-11-11)."""
    out = dict(peak)
    out["rates"] = {k: v for k, v in peak["rates"].items() if k != "mem_bytes"}
    out["sources"] = {k: v for k, v in peak["sources"].items() if k != "mem_bytes"}
    return out


DEVICES = sorted(p.stem for p in (ROOT / "spec" / "devices").glob("*.json"))


def card_roof(profile: str, device: str, points=()) -> dict:
    """What the card's roofline code (roofline.SCRIPT, as the harness runs it in the toolchain container)
    computes for this checkout: the whole chip's rates per operand format, and each point's bound."""
    import subprocess
    from tritonchip_agent_harness import roofline
    with tempfile.TemporaryDirectory() as out:
        script = roofline.SCRIPT.replace('"/work', f'"{ROOT}').replace('"/out/', f'"{out}/')
        args = {"profile": profile, "device": f"spec/devices/{device}.json", "points": list(points)}
        proc = subprocess.run([sys.executable, "-c", script, json.dumps(args)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(proc.stderr[-2000:])
        return json.loads(Path(out, "roofline.json").read_text())


class BoundEquivalence(unittest.TestCase):
    def test_spec_peak_equals_the_evaluators(self):
        profiles = single_core_profiles()
        self.assertTrue(profiles)
        for p in profiles:
            self.assertEqual(skill_bound.spec_peak(p), core_only(r38_peaks.spec_peak(p)), p)
        multi = set(PROFILES) - set(profiles)
        for p in multi:
            with self.assertRaises(ValueError):
                skill_bound.spec_peak(p)

    def workloads(self, mod):
        return [
            mod.gemm("g1", "mm", 24, 48, 40, in_dtype="fp16", out_dtype="fp16"),
            mod.gemm("g2", "bmm", 16, 16, 32, batch=3),
            mod.gemm("g3", "mm32", 32, 32, 48, in_dtype="fp32", out_dtype="fp32"),
            mod.elementwise("e1", "silu", 1234, inputs=1, outputs=1, fp_per_elem=2, sfu_per_elem=1, dtype="fp16"),
            mod.elementwise("e2", "add", 777, inputs=2, outputs=1, fp_per_elem=1, int_per_elem=0, dtype="fp32"),
            mod.row_reduction("r1", "softmax", 5, 37, fp_per_elem=3, sfu_per_elem=1, writes_per_elem=1,
                              writes_per_row=0, dtype="fp16"),
            mod.copy("c1", "transpose", 40 * 24, dtype="fp32"),
            mod.conv("v1", "conv", 8 * 8, 9, 10 * 10, 3 * 3 * 4 * 8, in_dtype="fp32", out_dtype="fp32",
                     in_channels=4, out_channels=8),
            mod.Workload("d1", "dep", "x", fp_ops=100, read_bytes=64, dep_cycles=3000.0),
        ]

    def test_hbm_profile_peak_equals_the_evaluators(self):
        """archbench/bounds.py tnca_peak: an `/hbm` profile is its base profile's core on another memory."""
        for p in single_core_profiles():
            theirs = core_only(dict(r38_peaks.spec_peak(p), profile=p + "/hbm"))
            self.assertEqual(skill_bound.spec_peak(p + "/hbm"), theirs, p)
        try:
            sys.path.insert(0, str(ROOT / "python"))
            from archbench import bounds as live
        except Exception as exc:  # archbench needs torch or tnca here
            self.skipTest(f"archbench.bounds not importable here: {exc}")
        for p in single_core_profiles():
            try:
                theirs = live.tnca_peak(p + "/hbm")
            except ImportError as exc:  # tnca.device, for the /hbm name, needs the toolchain
                self.skipTest(f"archbench.bounds.tnca_peak of an /hbm profile not available here: {exc}")
            self.assertEqual(skill_bound.spec_peak(p + "/hbm"), core_only(theirs), p)

    def test_lower_bound_equals_the_evaluators(self):
        for p in single_core_profiles():
            peak = r38_peaks.spec_peak(p)
            for mine, theirs in zip(self.workloads(skill_bound), self.workloads(r38_bounds)):
                for scoped in (False, True):
                    self.assertEqual(skill_bound.lower_bound(mine, peak, scoped=scoped),
                                     r38_bounds.lower_bound(theirs, peak, scoped=scoped), (p, mine.id, scoped))

    def test_unavailable_and_empty_refused_alike(self):
        peak = r38_peaks.spec_peak("core-sim")
        with self.assertRaises(skill_bound.BoundUnavailable):
            skill_bound.lower_bound(skill_bound.unavailable("u", "x", "sort", "no bound") if hasattr(skill_bound, "unavailable")
                                    else skill_bound.Workload("u", "x", "sort", bound_available=False), peak)
        with self.assertRaises(skill_bound.BoundUnavailable):
            skill_bound.lower_bound(skill_bound.Workload("z", "x", "y"), peak)

    def test_demand_bound_is_the_cards_formula(self):
        """A Demand goes to the bound the way the card sends a point's (roofline.SCRIPT): MACs count once as FP
        ops, against the whole chip's rates."""
        d = skill_bound.Demand(macs=24 * 40 * 48, fp_ops=24 * 48, sfu_ops=5, read_bytes=1000, write_bytes=500)
        w = r38_bounds.Workload("x", "f", "k", fp_ops=d.macs + d.fp_ops, macs=d.macs, sfu_ops=d.sfu_ops,
                                read_bytes=d.read_bytes, write_bytes=d.write_bytes)
        for dev in DEVICES:
            for p in single_core_profiles()[:4]:
                chip = card_roof(p, dev)["chip"]
                self.assertEqual(skill_bound.bound(d, p, device=dev),
                                 r38_bounds.lower_bound(w, {"rates": chip["any"]}), (p, dev))
                self.assertEqual(skill_bound.bound(d, p, device=dev, mma_dtype="fp16"),
                                 r38_bounds.lower_bound(w, {"rates": chip["fp16"]}), (p, dev))
        self.assertEqual(skill_bound.eta(50.0, 100), 0.5)


class ChipRoof(unittest.TestCase):
    """One eta (2026-11-11): the skill's L for a case is the card's L, the whole chip of the base machine.
    The one-core bound it replaces took memory as a 64-bit port per core; R41's are 512 bits."""

    def test_the_chip_rates_are_the_cards_for_every_device_and_format(self):
        self.assertTrue(DEVICES)
        for dev in DEVICES:
            for p in single_core_profiles():
                card = card_roof(p, dev)
                for fmt in [None, *sorted(skill_bound.MMA_GROUP)]:
                    mine = skill_bound.chip_peak(p, mma_dtype=fmt, device=dev)
                    self.assertEqual(mine["rates"], card["chip"][fmt or "any"], (dev, p, fmt))
                self.assertEqual(mine["cores"], card["cores"], (dev, p))

    def test_r41_memory_is_three_512_bit_outlets(self):
        card = card_roof("core-vrf-sync-v5-x8-g2c4-dp-board", "u280-3core-r41")
        self.assertEqual((card["cores"], card["mem_bytes_per_cycle"]), (3, 192))
        self.assertEqual(card["chip"]["fp16"]["mem_bytes"], 192)

    def test_a_runs_roof_is_used_as_given(self):
        """In a run the harness's TNCA_CHIP_ROOF wins over the profile and the device the shell names: the roof
        stays the base machine's, whatever the agent changes."""
        from unittest import mock
        from tritonchip_agent_harness import roofline
        table = card_roof("core-vrf-sync-v5-x8-g2c4-dp-board", "u280-3core-r41")
        table.update(version=roofline.VERSION, profile="core-vrf-sync-v5-x8-g2c4-dp-board")
        roof = json.dumps(roofline.agent_roof(table))
        d = skill_bound.Demand(read_bytes=65536, write_bytes=32768, fp_ops=16384)
        env = {skill_bound.ROOF_ENV: roof, "TNCA_BACKEND_ARGS": json.dumps({"device": "u280-3core-r40"})}
        with mock.patch.dict("os.environ", env):
            peak = skill_bound.chip_peak("core-sim", mma_dtype="fp16")
            self.assertEqual(peak["rates"], table["chip"]["fp16"])
            L, binding, _ = skill_bound.bound(d, "core-sim", mma_dtype="fp16")
        self.assertEqual((L, binding), (98304 / 192, "mem_bytes"))
        got = skill_bound.achieved(d, 11168, peak)["mem_bytes"]
        self.assertAlmostEqual(got["per_cycle"], 98304 / 11168)
        self.assertEqual(got["rate"], 192)

    def test_no_roof_and_no_device_is_no_bound(self):
        from unittest import mock
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(skill_bound.BoundUnavailable):
                skill_bound.chip_peak("core-sim")
        with mock.patch.dict("os.environ", {"TNCA_BACKEND_ARGS": json.dumps({"device": "u280-3core-r41"})},
                             clear=True):
            self.assertEqual(skill_bound.chip_peak("core-sim")["rates"]["mem_bytes"], 192)

    def test_no_one_core_memory_constant_is_left(self):
        text = (SKILL / "scripts" / "bound.py").read_text(encoding="utf-8")
        self.assertNotIn("MEM_DW_BITS", text)
        self.assertNotIn("mem_bytes", skill_bound.spec_peak("core-sim")["rates"])

    @unittest.skipUnless(HAVE_TORCH, "torch is not installed here; run where it is")
    def test_every_scored_points_bound_is_the_cards(self):
        """The R41 task's scored points: the skill's L from each point's own demand equals the card's table."""
        from archbench import registry
        task = load_task(HERE / "tasks" / "r41-dp-formal")
        ev = task.evaluation
        points = list(dict.fromkeys(list(ev["validation"]) + list(ev.get("board_validation") or [])))
        card = card_roof(ev["profile"], ev["device"], points)["points"]
        checked = 0
        for pid in points:
            if "error" in card[pid]:
                continue
            p = registry.get(pid)
            d = skill_bound.Demand(macs=p.demand.macs, fp_ops=p.demand.fp_ops, sfu_ops=p.demand.sfu_ops,
                                   read_bytes=p.demand.read_bytes, write_bytes=p.demand.write_bytes)
            L, binding, _ = skill_bound.bound(d, ev["profile"], mma_dtype=p.dtype, device=ev["device"])
            self.assertEqual((L, binding), (card[pid]["cycles"], card[pid]["binding"]), pid)
            checked += 1
        self.assertGreater(checked, 0)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed here; run where it is")
class ProtocolEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill = load("skill_protocol", SKILL / "scripts" / "protocol.py")
        from archbench.platforms import common
        cls.theirs = common
        import torch
        cls.torch = torch

    def fixtures(self):
        torch = self.torch
        g = torch.Generator().manual_seed(1)
        r = torch.randn(7, 9, generator=g)
        cases = [
            ((r.clone(),), (r,), 1e-6),
            ((r.half(),), (r,), 2e-2),
            (((r + 1e-3 * torch.randn(7, 9, generator=g)),), (r,), 1e-4),
            ((r.clone(),), (r.t(),), 1e-3),                                   # shape mismatch
            ((r.clone(), r.clone()), (r,), 1e-3),                             # output count
            ((torch.tensor([1, 2, 3]),), (torch.tensor([1, 2, 4]),), 0.0),    # integers
            ((torch.tensor([1.0, float("nan")]),), (torch.tensor([1.0, 2.0]),), 1e-3),
            ((torch.tensor([1.0, float("nan")]),), (torch.tensor([1.0, float("nan")]),), 1e-3),
            ((torch.zeros(3),), (torch.zeros(3),), 0.0),
            ((torch.full((4,), 1e-9),), (torch.zeros(4),), 1e-6),            # scale floor at 1
            ((torch.tensor(2.0),), (torch.tensor(1.0),), 0.5),
        ]
        return cases

    def test_compare_equals_the_evaluators(self):
        for got, ref, tol in self.fixtures():
            self.assertEqual(self.skill.compare(got, ref, tol), self.theirs.compare(got, ref, tol))
            self.assertEqual(self.skill.compare(list(got), ref, tol), self.theirs.compare(list(got), ref, tol))

    def test_reference_equals_the_evaluators(self):
        torch = self.torch
        g = torch.Generator().manual_seed(2)
        lin = torch.nn.Linear(40, 24).half()
        x = torch.randn(5, 40, generator=g).half()

        class P:
            def make(self, seed):
                return lin, (x,)
        mine = self.skill.reference(lin, (x,))
        self.assertEqual(lin.weight.dtype, torch.float16, "the skill's reference leaves the caller's module as it was")
        theirs = self.theirs.reference(P(), 0)      # ArchBench's converts the module it is given, in place
        self.assertEqual(len(mine), len(theirs))
        for a, b in zip(mine, theirs):
            self.assertTrue(torch.equal(a, b))

    def test_reference_of_every_fp8_point_equals_the_evaluators(self):
        # The evaluator keeps OCP FP8 operands FP8 (torch._scaled_mm takes nothing else); a skill that
        # upcast them would refuse every FP8 case. Each registered FP8 point, both ways.
        torch = self.torch
        from archbench import registry
        fp8 = [p for p in registry.points() if p.dtype in ("e4m3", "e5m2")]
        self.assertTrue(fp8, "the registry has FP8 points")
        for p in fp8:
            m, ins = p.make(0)
            mine = self.skill.reference(m, ins)
            theirs = self.theirs.reference(p, 0)
            self.assertEqual(len(mine), len(theirs), p.id)
            for a, b in zip(mine, theirs):
                self.assertTrue(torch.equal(a, b), p.id)

    def test_tolerances_are_the_registrys(self):
        from archbench import registry, models
        self.assertEqual(self.skill.TOLERANCE, registry.TOLERANCE)
        self.assertEqual(self.skill.MODEL_TOLERANCE, models.MODEL_TOL)
        self.assertEqual(self.skill.TORCH_DTYPE, registry.TORCH_DTYPE)
        self.assertEqual(self.skill.classify(NotImplementedError("x")), self.theirs.classify(NotImplementedError("x")))
        self.assertEqual(self.skill.classify(RuntimeError("heap exhausted")), self.theirs.classify(RuntimeError("heap exhausted")))

    def test_measured_demand_agrees_with_declared(self):
        """The skill's demand_of judged the way selftest judges a declared demand, on the GEMM points."""
        from archbench import registry
        pts = [p for p in registry.points(tier="tiny") if p.family in ("matmul", "linear", "gemv", "bmm")]
        self.assertTrue(pts)
        for p in pts:
            m, ins = p.make(0)
            d = skill_bound.demand_of(m, ins)
            self.assertAlmostEqual(d.macs, p.demand.macs, delta=1e-6 * max(1, p.demand.macs), msg=p.id)
            self.assertAlmostEqual(d.write_bytes, p.demand.write_bytes, delta=0.5, msg=p.id)
            self.assertGreaterEqual(d.read_bytes + 0.5, p.demand.read_bytes, p.id)


class Leak(unittest.TestCase):
    """Nothing in the skill directory names a point it does not give. The skill gives the points in
    benchmark_cases/ (INDEX.json) and the classes in classes.json; every other point is hidden, and so is
    every class not listed. The hidden set is derived from the skill's own index: no test names it."""

    TIER_CONSTANTS = ("384 KiB", "384 KB", "352 KiB", "352 KB", "512 KiB", "500 target cycles", "500 cycles",
                      "~500", "400,000", "400 000")

    def setUp(self):
        text = skill_text()
        # What the skill may name: the given points' ids and variant strings, removed before looking for the rest.
        for pid in sorted(VISIBLE, key=len, reverse=True):
            variant = pid.rsplit("/", 1)[1]
            text = text.replace(pid, " ").replace(variant, " ")
        self.text = text

    def test_no_hidden_point_id_or_parameter_string(self):
        given = set()
        for pid in VISIBLE:
            variant = pid.rsplit("/", 1)[1]
            given |= {variant, variant.split("-", 1)[1] if "-" in variant else variant}
        hidden = [pt for pt in suite_points() if pt["id"] not in VISIBLE]
        self.assertTrue(hidden)
        for pt in hidden:
            pid = pt["id"]
            self.assertNotIn(pid, self.text, pid)
            variant = pid.rsplit("/", 1)[1]                      # e.g. fp16-m16-k32-n16
            params = variant.split("-", 1)[1] if "-" in variant else variant
            if variant not in given:
                self.assertNotIn(variant, self.text, pid)
            if re.search(r"[a-z]", params) and params not in given:
                self.assertNotIn(params, self.text, pid)

    def test_no_tier_constant(self):
        for c in self.TIER_CONSTANTS:
            self.assertNotIn(c, self.text, c)

    def test_no_class_that_is_not_listed(self):
        """A class of the benchmark the skill does not list is not named anywhere in it."""
        listed = {c["class"] for c in CLASSES}
        # The tiers' classes. The membound sweep's are another suite's (casts, sums, maxima: common words) and
        # not split into given and held out.
        families = set()
        for suite in sorted((ROOT / "archbench" / "suites").glob("*.json")):
            data = json.loads(suite.read_text(encoding="utf-8"))
            if data.get("tier") != "membound":
                families |= {(pt["kind"], pt["family"]) for pt in data["points"]}
        unlisted = [(k, f) for k, f in families if f"{k}.{f}" not in listed]
        self.assertTrue(unlisted, "every class is listed: nothing is held out")
        # A PyTorch call of the same name (torch.bmm in the batch-dimension example) is the public API, not the
        # benchmark's class: it is not counted.
        text = re.sub(r"\b(?:torch|F|nn)\.\w+", " ", self.text)
        for kind, fam in unlisted:
            self.assertNotIn(f"{kind}.{fam}", self.text, fam)
            self.assertIsNone(re.search(rf"\b{re.escape(fam)}\b", text), fam)

    @unittest.skipUnless(HAVE_TORCH, "torch is not installed here; run where it is")
    def test_no_hidden_point_input_shape(self):
        given, spellings = set(), []
        for p, m, ins in built_points():
            for t in list(ins) + list(m.parameters()):
                shape = tuple(t.shape)
                if len(shape) < 2:
                    continue
                if p.id in VISIBLE:
                    given.add(shape)
                    continue
                spellings.append((p.id, shape, [str(shape), "(" + ", ".join(map(str, shape)) + ")",
                                                "x".join(map(str, shape)), "×".join(map(str, shape)),
                                                "[" + ", ".join(map(str, shape)) + "]"]))
        self.assertTrue(spellings)
        for pid, shape, forms in spellings:
            if shape in given:
                continue
            for f in forms:
                if re.fullmatch(r"[\d]+(?:[x×][\d]+)+", f):
                    # 16x16 is a shape; the 16x16 inside 16x160 or 1x128 is not
                    self.assertIsNone(re.search(rf"(?<!\d){re.escape(f)}(?!\d)", self.text), f"{pid}: {shape} as {f!r}")
                else:
                    self.assertNotIn(f, self.text, f"{pid}: {shape} as {f!r}")

    @unittest.skipUnless(HAVE_TORCH, "torch is not installed here; run where it is")
    def test_no_example_is_a_hidden_point(self):
        """A worked example's inputs (shapes and dtypes) may not equal a hidden point's."""
        import torch

        def signature(m, ins):
            return tuple((tuple(t.shape), str(t.dtype)) for t in ins if isinstance(t, torch.Tensor)) + \
                tuple((tuple(t.shape), str(t.dtype)) for t in m.parameters())
        points, given = {}, set()
        for p, m, ins in built_points():
            sig = signature(m, ins)
            if p.id in VISIBLE:
                given |= {sig, sig[:len(ins)]}
                continue
            points.setdefault(sig, []).append(p.id)
            points.setdefault(sig[:len(ins)], []).append(p.id + " (inputs)")
        examples = sorted((SKILL / "examples").glob("*.py"))
        self.assertTrue(examples)
        for ex in examples:
            mod = load(f"ex_{ex.stem}", ex)
            m, ins = mod.make()
            sig = signature(m, tuple(ins))
            if sig not in given:
                self.assertNotIn(sig, points, f"{ex.name} is the workload of {points.get(sig)}")
            if sig[:len(ins)] not in given:
                self.assertNotIn(sig[:len(ins)], points, f"{ex.name} has the inputs of {points.get(sig[:len(ins)])}")


class BenchmarkCases(unittest.TestCase):
    """The points the skill gives (benchmark_cases/): listed once, each its point bit for bit, none hidden by a
    task that gives the skill by default."""

    def test_index_files_and_classes_agree(self):
        files = {c["file"] for c in INDEX["cases"]}
        self.assertEqual(files, {f.name for f in CASES_DIR.glob("*.py")})
        self.assertEqual(len(files), len(INDEX["cases"]))
        listed = {c["class"]: c for c in CLASSES}
        for c in INDEX["cases"]:
            src = (CASES_DIR / c["file"]).read_text(encoding="utf-8")
            self.assertIn(f"POINT = {c['point']!r}", src, c["file"])
            cls = f"{c['kind']}.{c['family']}"
            self.assertIn(cls, listed, c["point"])
            self.assertIn(c["file"], listed[cls]["cases"], c["point"])
        for cls, entry in listed.items():
            for f in entry["cases"]:
                self.assertIn(f, files, f"{cls}: {f}")

    def test_no_task_that_gives_the_skill_hides_a_given_point(self):
        for task_dir in sorted((HERE / "tasks").iterdir()):
            task = load_task(task_dir)
            if SKILL.name not in task.skills:
                continue
            hidden = {p for s in POINT_SETS for p in task.evaluation.get(s, [])}
            self.assertFalse(VISIBLE & hidden, f"{task.task_id} hides {sorted(VISIBLE & hidden)}")

    @unittest.skipUnless(HAVE_TORCH, "torch is not installed here; run where it is")
    def test_each_case_is_its_point(self):
        import torch
        from archbench import registry
        if str(SKILL / "scripts") not in sys.path:
            sys.path.insert(0, str(SKILL / "scripts"))     # a case imports bound, as tnca_case.py lets it

        def same(a, b):
            if isinstance(a, torch.Tensor):
                bits = (lambda t: t.view(torch.uint8) if t.element_size() == 1 else t)
                return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and \
                    torch.equal(bits(a), bits(b))
            if isinstance(a, (tuple, list)):
                return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            return a == b
        for c in INDEX["cases"]:
            with self.subTest(case=c["file"]):
                try:
                    p = registry.get(c["point"])
                except KeyError:
                    self.assertTrue(c["draft"], f"{c['point']} is not in this tree's registry and not a draft")
                    self.skipTest(f"{c['point']}: a draft point, not in this tree's registry")
                mod = load(f"case_{Path(c['file']).stem}", CASES_DIR / c["file"])
                m, ins = mod.make()
                pm, pins = p.make(INDEX["seed"])
                self.assertEqual(type(m).__name__, type(pm).__name__)
                self.assertTrue(same(tuple(ins), tuple(pins)), "inputs")
                self.assertTrue(same(tuple(m.state_dict().values()), tuple(pm.state_dict().values())), "parameters")
                with torch.no_grad():
                    self.assertTrue(same(m(*ins), pm(*pins)), "output")
                self.assertEqual(mod.TOL, p.tolerance)
                self.assertEqual(mod.DEMAND.as_dict(), p.demand.as_dict())
                self.assertEqual(bool(getattr(mod, "MODEL", False)), p.kind == "model")


class Delivery(unittest.TestCase):
    """The skill reaches the agent by mount, identically for every task, run and CLI."""

    def test_skill_never_enters_a_workspace(self):
        files = [p for p in SKILL.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        self.assertTrue(files)
        for task_dir in sorted((HERE / "tasks").iterdir()):
            task = load_task(task_dir)
            for f in files:
                self.assertTrue(hidden(f.relative_to(ROOT).as_posix(), task.hide), f"{task.task_id} exports {f}")

    def test_mounted_read_only_at_the_path_the_prompt_names(self):
        self.assertEqual(runner.SKILLS_SOURCE, SKILL.parent)
        self.assertEqual(skills.catalog()[SKILL.name].mounted, SKILL_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            staged = skills.stage(skills.select([SKILL.name]), Path(tmp) / "skills")
            self.assertEqual([p.name for p in staged.iterdir()], [SKILL.name])
            mounts = runner.agent_mounts(Path("/tmp/sock"), Path("/tmp/ws"), skills=staged)
            mounted = [m for m in mounts if m.target == adapters.SKILLS_DIR]
            self.assertEqual([(m.source, m.read_only) for m in mounted], [(staged, True)])
        self.assertFalse([m for m in runner.agent_mounts(Path("/tmp/sock"), Path("/tmp/ws"))
                          if m.target == adapters.SKILLS_DIR])

    def test_skill_md_is_an_agent_skill(self):
        head = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---\n")
        self.assertEqual(head[0], "")
        front = dict(line.split(": ", 1) for line in head[1].strip().splitlines())
        self.assertEqual(front["name"], SKILL.name)
        self.assertTrue(len(front["description"]) > 40)
        for script in ("tnca_case.py", "protocol.py", "bound.py"):
            self.assertTrue((SKILL / "scripts" / script).is_file(), script)

    def test_paths_in_the_skill_text_exist_as_mounted(self):
        for ref in set(re.findall(re.escape(adapters.SKILLS_DIR) + r"/[\w./*-]+", skill_text())):
            rel = ref[len(adapters.SKILLS_DIR) + 1:]
            name, _, rest = rel.partition("/")
            self.assertEqual(name, SKILL.name, ref)
            target = SKILL / rest.rstrip("/")
            self.assertTrue(target.exists() or ("*" in rest and target.parent.is_dir()), ref)

    def test_every_cli_has_a_skills_directory(self):
        for cli in ("claude-code", "codex", "opencode"):
            account = Account(label="t", cli=cli, auth="x", credential=Path("/tmp/none"), hosts=("h",),
                              models={"main": "m"}, options={"provider": "p", "base_url": "u"})
            launch = adapters.launch(account, "p", "offline", Path(tempfile.mkdtemp()), {})
            self.assertTrue(launch.skills_dir.startswith("/"), cli)
            self.assertNotEqual(launch.skills_dir, adapters.SKILLS_DIR, cli)

    def test_manifest_names_the_skill_and_follows_its_bytes(self):
        manifest = skills.manifest(skills.select([SKILL.name]))
        self.assertEqual(list(manifest), [SKILL.name])
        self.assertEqual(manifest, skills.manifest(skills.select([SKILL.name])))
        self.assertEqual(len(manifest[SKILL.name]), 64)

    def test_render_names_the_skill_and_drops_the_control_example(self):
        task = load_task(HERE / "tasks" / "r41-dp-formal")
        self.assertIn("<!-- without-skill -->", task.instruction)
        self.assertIn(SKILL.name, task.skills)
        with_skill = render(task, "offline", skills=task.skills)
        without = render(task, "offline")
        self.assertIn(f"`{SKILL_PATH}/SKILL.md`", with_skill)
        self.assertIn(skills.catalog()[SKILL.name].description, with_skill)
        self.assertNotIn("assert_close", with_skill)
        self.assertNotIn("<!--", with_skill)
        self.assertIn("assert_close", without)
        self.assertNotIn(SKILL_PATH, without)
        self.assertNotIn("<!--", without)
        # The control arm's text is the instruction as written, minus only the markers.
        self.assertIn(task.instruction.replace("<!-- without-skill -->\n", "").replace("<!-- /without-skill -->\n", "")
                      .rstrip(), without)

    def test_every_task_marks_its_example_and_keeps_its_machine(self):
        """The skill replaces the syllabus's measuring example in every task; what selects the
        machine (a backend, its arguments) is the environment's and stays in both arms. Both arms
        render whatever the task's default set is: a task may default to none (r41-dp-pilot, whose
        baseline runs without skills) and a run give it the skill with --skills."""
        for task_dir in sorted((HERE / "tasks").iterdir()):
            task = load_task(task_dir)
            self.assertIn("assert_close", task.instruction, task.task_id)
            self.assertEqual(task.instruction.count("<!-- without-skill -->"), 1, task.task_id)
            with_skill, without = render(task, "offline", skills=(SKILL.name,)), render(task, "offline")
            self.assertTrue(task.skills == () or SKILL.name in task.skills, task.task_id)
            self.assertNotIn("assert_close", with_skill, task.task_id)
            self.assertIn("assert_close", without, task.task_id)
            for env in re.findall(r"TNCA_BACKEND\S*", task.instruction):
                self.assertIn(env, with_skill, task.task_id)
            self.assertNotIn("<!--", with_skill + without, task.task_id)


if __name__ == "__main__":
    unittest.main()


class ModelCasesOnePerProcess(unittest.TestCase):
    """The Event Model keeps one model, and its caches, for a whole Python process (2026-11-09: a second case in
    the same process came out up to a fifth short), so on a model backend tnca_case.py runs each case alone."""

    def runner_module(self):
        import types
        from unittest import mock
        stubs = {"protocol": types.ModuleType("protocol"), "bound": types.ModuleType("bound")}   # they need torch
        with mock.patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location("_tnca_case_under_test", SKILL / "scripts" / "tnca_case.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def run_main(self, backend, cases):
        from unittest import mock
        mod = self.runner_module()
        record = {"status": "pass", "case": "x", "time": None, "correctness": None, "eta": None, "problems": []}
        with mock.patch.dict("os.environ", {"TNCA_BACKEND": backend}), \
                mock.patch.object(mod.subprocess, "run", return_value=mock.Mock(returncode=0)) as run, \
                mock.patch.object(mod, "run_case", return_value=record) as in_process, \
                mock.patch.object(mod, "summary", return_value=""):
            code = mod.main(cases + ["--profile", "p", "--opt-level", "2", "--json", "/tmp/out.jsonl"])
        return code, run, in_process

    def test_several_cases_on_a_model_run_one_process_each(self):
        code, run, in_process = self.run_main("systemc-tlm-grid", ["a.py", "b.py"])
        self.assertEqual(code, 0)
        self.assertEqual(in_process.call_count, 0)
        self.assertEqual([c.args[0][2] for c in run.call_args_list], ["a.py", "b.py"])
        for c in run.call_args_list:
            argv = c.args[0]
            self.assertEqual(argv[3:], ["--profile", "p", "--opt-level", "2", "--kernels", "8",
                                        "--json", "/tmp/out.jsonl"])

    def test_a_case_of_several_launches_is_listed_by_kernel(self):
        """A model is the sum of its kernels: grouped by (kernel, grid), largest first, shares adding to one;
        the summary prints the top rows and what the rest add up to, and the binding rate beside eta."""
        mod = self.runner_module()
        per_launch = ([{"kernel": "attn", "grid": [1158, 1, 1], "cycles": 1300}] * 2
                      + [{"kernel": "kv", "grid": [12, 1, 1], "cycles": 400}] * 2
                      + [{"kernel": f"k{i}", "grid": [4, 1, 1], "cycles": 10} for i in range(10)])
        rows = mod.kernels(per_launch)
        self.assertEqual([(r["kernel"], r["launches"], r["cycles"]) for r in rows[:2]],
                         [("attn", 2, 2600), ("kv", 2, 800)])
        self.assertAlmostEqual(sum(r["share"] for r in rows), 1.0, places=3)
        total = sum(e["cycles"] for e in per_launch)
        rec = {"status": "pass", "case": "m.py", "tol": 0.05, "problems": [],
               "correctness": {"max_rel_err": 1e-3},
               "time": {"device_cycles": total, "launches": len(per_launch), "per_launch": per_launch,
                        "kernels": rows},
               "eta": 0.0015, "bound": {"cycles": 5.1, "binding": "mem_bytes"},
               "achieved": {"mem_bytes": {"per_cycle": 0.25, "rate": 192}}}
        text = mod.summary(rec, 3)
        self.assertIn("memory binds: 0.25 of 192 B/cycle", text)
        self.assertIn("whole chip", text)
        self.assertIn("1158x1x1", text)
        self.assertIn("... 9 more kernel(s)", text)
        self.assertEqual(len(text.splitlines()), 6)       # the case, the header, three rows, the rest
        one = dict(rec, time=dict(rec["time"], launches=1, per_launch=per_launch[:1], kernels=rows[:1]))
        self.assertNotIn("share", mod.summary(one))

    def test_the_simulator_and_a_single_case_stay_in_process(self):
        for backend, cases in (("", ["a.py", "b.py"]), ("systemc-tlm-grid", ["a.py"])):
            code, run, in_process = self.run_main(backend, cases)
            self.assertEqual((code, run.call_count, in_process.call_count), (0, 0, len(cases)), backend)
