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


class BoundEquivalence(unittest.TestCase):
    def test_spec_peak_equals_the_evaluators(self):
        profiles = single_core_profiles()
        self.assertTrue(profiles)
        for p in profiles:
            self.assertEqual(skill_bound.spec_peak(p), r38_peaks.spec_peak(p), p)
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
            theirs = dict(r38_peaks.spec_peak(p), profile=p + "/hbm")
            self.assertEqual(skill_bound.spec_peak(p + "/hbm"), theirs, p)
        try:
            sys.path.insert(0, str(ROOT / "python"))
            from archbench import bounds as live
        except Exception as exc:  # archbench needs torch or tnca here
            self.skipTest(f"archbench.bounds not importable here: {exc}")
        for p in single_core_profiles():
            self.assertEqual(skill_bound.spec_peak(p + "/hbm"), live.tnca_peak(p + "/hbm"), p)

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

    def test_demand_bound_is_archbenchs_formula(self):
        """A Demand goes to the bound the way archbench/bounds.py sends a point's: MACs count once as FP ops."""
        d = skill_bound.Demand(macs=24 * 40 * 48, fp_ops=24 * 48, sfu_ops=5, read_bytes=1000, write_bytes=500)
        for p in single_core_profiles():
            w = r38_bounds.Workload("x", "f", "k", fp_ops=d.macs + d.fp_ops, macs=d.macs, sfu_ops=d.sfu_ops,
                                    read_bytes=d.read_bytes, write_bytes=d.write_bytes)
            self.assertEqual(skill_bound.bound(d, p), r38_bounds.lower_bound(w, r38_peaks.spec_peak(p)), p)
        self.assertEqual(skill_bound.eta(50.0, 100), 0.5)


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
        self.assertEqual(task.skills, (SKILL.name,))
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
            self.assertIn(task.skills, ((), (SKILL.name,)), task.task_id)
            self.assertNotIn("assert_close", with_skill, task.task_id)
            self.assertIn("assert_close", without, task.task_id)
            for env in re.findall(r"TNCA_BACKEND\S*", task.instruction):
                self.assertIn(env, with_skill, task.task_id)
            self.assertNotIn("<!--", with_skill + without, task.task_id)


if __name__ == "__main__":
    unittest.main()
