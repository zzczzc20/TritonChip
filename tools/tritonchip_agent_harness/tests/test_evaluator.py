"""Scoring a set of points: each point on its own, failures counted, never hidden."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
from unittest import mock  # noqa: E402

import subprocess  # noqa: E402

from tritonchip_agent_harness.evaluator import (  # noqa: E402
    device_check, hardware_changed, observed_multicore_point, observed_point, read_multicore_observer,
    read_observer, rtl_check, score)
from tritonchip_agent_harness import evaluator  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402

BASE = {p: {"status": "pass", "cycles": 1000} for p in ("a", "b", "c", "d")}


class Score(unittest.TestCase):
    def test_a_wrong_point_fails_alone(self):
        got = {"a": {"status": "pass", "cycles": 500}, "b": {"status": "fail-numerics", "cycles": 10},
               "c": {"status": "pass", "cycles": 2000}, "d": {"status": "pass", "cycles": 1000}}
        s = score(BASE, got, ["a", "b", "c", "d"])
        self.assertEqual((s["n"], s["passed"]), (4, 3))
        self.assertEqual([r["failure"] for r in s["points"]], [None, "fail-numerics", None, None])
        self.assertAlmostEqual(s["speedup_geomean"], 1.0)     # 2.0, 0.5, 1.0 over the three that passed
        self.assertAlmostEqual(s["fast_1"], 0.25)             # only a beat the base, out of all four

    def test_fallback_and_missing_are_failures(self):
        got = {"a": {"status": "pass", "cycles": 500, "fallbacks": ["aten.mm"]},
               "c": {"status": "pass", "cycles": None}}
        s = score(BASE, got, ["a", "b", "c"])
        self.assertEqual([r["failure"] for r in s["points"]], ["fallback", "missing", "no-cycles"])
        self.assertEqual(s["passed"], 0)
        self.assertIsNone(s["speedup_geomean"])
        self.assertEqual(s["fast_1"], 0.0)

    def test_breaking_a_slow_point_does_not_raise_fast_1(self):
        slow = {"a": {"status": "pass", "cycles": 500}, "b": {"status": "pass", "cycles": 4000}}
        broken = {"a": {"status": "pass", "cycles": 500}, "b": {"status": "fail-numerics"}}
        s_slow, s_broken = score(BASE, slow, ["a", "b"]), score(BASE, broken, ["a", "b"])
        self.assertGreater(s_broken["speedup_geomean"], s_slow["speedup_geomean"])
        self.assertEqual(s_broken["fast_1"], s_slow["fast_1"])


def candidate(files: dict) -> Path:
    d = Path(tempfile.mkdtemp())
    for path, text in files.items():
        (d / "files" / path).parent.mkdir(parents=True, exist_ok=True)
        (d / "files" / path).write_text(text)
    (d / "candidate.json").write_text(json.dumps({"taken": [{"path": p, "status": "modified"} for p in files]}))
    return d


class RtlCheck(unittest.TestCase):
    def test_plain_rtl_passes(self):
        c = candidate({"rtl/core/x.v": 'module x; initial if ($value$plusargs("t=%d", t)) $display("hi"); endmodule\n',
                       "compiler/a.py": "fopen = 1\n"})
        self.assertEqual(rtl_check(c), [])

    def test_file_and_system_tasks_and_outside_includes_are_refused(self):
        c = candidate({"rtl/core/x.v": 'module x; integer f; initial begin f = $fopen("/obs/observe.log"); '
                                       '$system("id"); end\n`include "../../etc/passwd"\nendmodule\n'})
        self.assertEqual(rtl_check(c), ['rtl/core/x.v: $fopen', 'rtl/core/x.v: $system',
                                        'rtl/core/x.v: `include "../../etc/passwd"'])

    def test_a_task_named_only_in_a_comment_is_not_refused(self):
        self.assertEqual(rtl_check(candidate({"rtl/core/x.v": "// never $fopen here\n/* nor $system */ module x; endmodule\n"})), [])


LOG = ("cmd op=1 seq=5 obs=107 done=101 status=0 inside=28 after_done=0 outside=0 extra=0\n"
       "cmd op=1 seq=9 obs=207 done=201 status=0 inside=40 after_done=0 outside=0 extra=0\n"
       "cmd op=2 seq=10 obs=2 done=0 status=0 inside=0 after_done=0 outside=0 extra=0\n"
       "end outside=0 extra=0\n")


class Observer(unittest.TestCase):
    def log(self, text):
        p = Path(tempfile.mkdtemp()) / "observe.log"
        p.write_text(text)
        return read_observer(p)

    def test_the_score_is_the_platforms_count(self):
        obs = self.log(LOG)
        self.assertEqual((obs["launches"], obs["cycles"], obs["done_cycles"], obs["ended"]), (2, 314, 302, True))
        point = observed_point({"status": "pass", "time": {"device": 302}}, obs)
        self.assertEqual((point["status"], point["cycles"]), ("pass", 314))

    def test_a_record_the_device_never_reported_fails(self):
        point = observed_point({"status": "pass", "time": {"device": 150}}, self.log(LOG))
        self.assertEqual(point["status"], "record-disagrees")

    def test_activity_outside_a_launch_fails(self):
        bad = LOG.replace("end outside=0", "end outside=3")
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, self.log(bad))["status"],
                         "activity-outside-launch")
        late = LOG.replace("after_done=0", "after_done=1", 1)
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, self.log(late))["status"],
                         "activity-outside-launch")

    def test_no_observation_fails(self):
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, None)["status"], "no-observation")
        cut = LOG.replace("end outside=0 extra=0\n", "")
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, self.log(cut))["status"],
                         "no-observation")

    def test_a_failed_point_keeps_its_own_status(self):
        self.assertEqual(observed_point({"status": "fail-numerics", "time": {"device": 302}}, self.log(LOG))["status"],
                         "fail-numerics")


MC_LOG = ("cmd op=1 seq=1 obs=2450 done=2395 status=0 settle=3 quiet=1 rd=0 wr=0\n"
          "cmd op=1 seq=2 obs=1050 done=1001 status=0 settle=40 quiet=1 rd=0 wr=0\n")


class MulticoreObserver(unittest.TestCase):
    def log(self, text):
        p = Path(tempfile.mkdtemp()) / "observe.log"
        p.write_text(text)
        return read_multicore_observer(p)

    def test_the_score_is_the_endpoints(self):
        obs = self.log(MC_LOG)
        self.assertEqual((obs["launches"], obs["cycles"], obs["done_cycles"], obs["max_settle"], obs["not_quiet"]),
                         (2, 3500, 3396, 40, 0))
        point = observed_multicore_point({"status": "pass", "time": {"device": 3396}}, obs, None)
        self.assertEqual((point["status"], point["cycles"], point["record_cycles"]), ("pass", 3500, 3396))

    def test_a_record_the_device_never_reported_fails(self):
        point = observed_multicore_point({"status": "pass", "time": {"device": 1700}}, self.log(MC_LOG), None)
        self.assertEqual(point["status"], "record-disagrees")

    def test_done_before_the_work_drained_fails(self):
        early = MC_LOG.replace("settle=40 quiet=1", "settle=9 quiet=0")
        self.assertEqual(observed_multicore_point({"status": "pass", "time": {"device": 3396}},
                                                  self.log(early), None)["status"], "not-drained")
        # A write-back cache may write behind a launch: a long settle fails only
        # past the bound calibrated on the base.
        self.assertEqual(observed_multicore_point({"status": "pass", "time": {"device": 3396}},
                                                  self.log(MC_LOG), 64)["status"], "pass")
        self.assertEqual(observed_multicore_point({"status": "pass", "time": {"device": 3396}},
                                                  self.log(MC_LOG), 39)["status"], "not-drained")

    def test_a_device_fault_or_a_record_without_cycles_fails(self):
        faulted = MC_LOG.replace("status=0", "status=1", 1)
        self.assertEqual(observed_multicore_point({"status": "pass", "time": {"device": 3396}},
                                                  self.log(faulted), None)["status"], "device-fault")
        self.assertEqual(observed_multicore_point({"status": "pass", "time": {}}, self.log(MC_LOG), None)["status"],
                         "record-disagrees")

    def test_no_observation_fails_and_a_failed_point_keeps_its_status(self):
        rec = {"status": "pass", "time": {"device": 3396}}
        self.assertEqual(observed_multicore_point(rec, None, None)["status"], "no-observation")
        self.assertEqual(observed_multicore_point(rec, self.log(""), None)["status"], "no-observation")
        self.assertEqual(observed_multicore_point({"status": "fail-numerics"}, self.log(MC_LOG), None)["status"],
                         "fail-numerics")


DESC = {"schema": "tnca-device/1", "name": "d", "per_core_profile": "core-vrf-sync-v5",
        "platform": {"board": "u280", "core_count": 3, "kernel_clock_mhz": 90},
        "uncore": {"SETS": 32, "MEM_PORTS": 3}}


def device_repo() -> Path:
    """A repository whose one commit has the description spec/devices/d.json."""
    repo = Path(tempfile.mkdtemp())
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q")
    (repo / "spec/devices").mkdir(parents=True)
    (repo / "spec/devices/d.json").write_text(json.dumps(DESC))
    run("add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return repo


def device_task(repo: Path, **evaluation) -> Task:
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    ev = dict({"profile": "core-vrf-sync-v5", "device": "d", "validation": [], "final": [], "jobs": 2,
               "timeout_seconds": 60}, **evaluation)
    return Task(task_id="t", kind="k", base_commit=commit, design=(), evaluator_owned=(), hide=(),
                instruction="", reference=None, wall_seconds=60, evaluation=ev, services={},
                path=Path("task.json"), sha256="0" * 64)


class BaselineKey(unittest.TestCase):
    def test_a_multicore_base_is_kept_per_machine(self):
        repo = device_repo()
        one = device_task(repo, validation=["p"], platform={"commit": "c" * 40, "paths": []})
        other = device_task(repo, validation=["p"], platform={"commit": "c" * 40, "paths": []}, device="e")
        bigger = device_task(repo, validation=["p"], platform={"commit": "c" * 40, "paths": []}, memory_bytes=1 << 26)
        drained = device_task(repo, validation=["p"], platform={"commit": "c" * 40, "paths": []}, drain_cycles=500)
        keys = {evaluator.baseline_key(t) for t in (one, other, bigger, drained)}
        self.assertEqual(len(keys), 4)
        # A single-core task's key is what it was, so the bases kept for it stay valid.
        single = dataclasses.replace(one, evaluation={k: v for k, v in one.evaluation.items() if k != "device"})
        self.assertEqual(evaluator.baseline_key(single),
                         f"{single.base_commit[:12]}-core-vrf-sync-v5-"
                         f"{hashlib.sha256(json.dumps([['p'], []]).encode()).hexdigest()[:8]}-obs{'c' * 12}")

    def test_another_machine_is_measured_not_given_a_kept_base(self):
        # Two tasks alike but for the machine they name: the second is measured
        # on its own machine and kept in a file of its own, never handed the
        # first one's cycles; the first, asked again, is read back.
        repo = device_repo()
        one = device_task(repo, validation=["p"], platform={"commit": "c" * 40, "paths": []})
        other = dataclasses.replace(one, evaluation={**one.evaluation, "device": "e"})
        cycles, measured = {"d": 1000, "e": 3000}, []

        def trees(repo, task, candidate, work):
            for tree in ("host-tree", "device-tree"):
                (work / tree).mkdir(parents=True)
            return work / "host-tree", work / "device-tree"

        def measure(site, host, dev, task, *_):
            measured.append(task.evaluation["device"])
            return {"exit": 0, "missing": [],
                    "points": {"p": {"status": "pass", "cycles": cycles[task.evaluation["device"]]}}}

        site = types.SimpleNamespace(state_dir=Path(tempfile.mkdtemp()))
        with mock.patch.object(evaluator, "_trees", side_effect=trees), \
                mock.patch.object(evaluator, "_measure_trees", side_effect=measure):
            got = [evaluator.baseline(site, repo, t) for t in (one, other, one)]
        self.assertEqual([g["points"]["p"]["cycles"] for g in got], [1000, 3000, 1000])
        self.assertEqual(measured, ["d", "e"])
        self.assertEqual(sorted(p.name for p in (site.state_dir / "baselines").glob("*.json")),
                         sorted(f"{evaluator.baseline_key(t)}.json" for t in (one, other)))


class DeviceCheck(unittest.TestCase):
    def setUp(self):
        self.repo = device_repo()
        self.task = device_task(self.repo)

    def tree(self, desc):
        t = Path(tempfile.mkdtemp())
        (t / "spec/devices").mkdir(parents=True)
        (t / "spec/devices/d.json").write_text(json.dumps(desc))
        return t

    def test_the_uncore_may_change_the_platform_may_not(self):
        uncore = json.loads(json.dumps(DESC))
        uncore["uncore"]["SETS"] = 64
        self.assertEqual(device_check(self.repo, self.task, self.tree(uncore)), [])
        clock = json.loads(json.dumps(DESC))
        clock["platform"]["kernel_clock_mhz"] = 100
        clock["platform"]["extra"] = 1
        self.assertEqual(device_check(self.repo, self.task, self.tree(clock)),
                         ["spec/devices/d.json: platform.kernel_clock_mhz: 100, fixed at 90",
                          "spec/devices/d.json: platform.extra is not the base's"])

    def test_the_scored_profile_is_fixed(self):
        other = json.loads(json.dumps(DESC))
        other["per_core_profile"] = "core-vrf-sync-v4"
        self.assertIn("is not the scored profile", device_check(self.repo, self.task, self.tree(other))[0])
        renamed = json.loads(json.dumps(DESC))
        renamed["name"] = "another-machine"
        self.assertIn("name 'another-machine' is not the base's", device_check(self.repo, self.task, self.tree(renamed))[0])

    def test_a_changed_description_is_a_changed_machine(self):
        changed = json.loads(json.dumps(DESC))
        changed["uncore"]["SETS"] = 64
        c = candidate({"spec/devices/d.json": json.dumps(changed)})
        hw = hardware_changed(self.repo, self.task, c)
        self.assertTrue(hw["changed"])
        self.assertEqual(hw["device"], {"uncore.SETS": [32, 64]})


class ObserversOutOfReach(unittest.TestCase):
    """The candidate's code runs in the host container: what it can write is never the count."""

    def setUp(self):
        self.calls = []

        def docker(*args, **kw):
            self.calls.append(list(args))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        self.patches = [mock.patch.object(evaluator.provider, "docker", side_effect=docker),
                        mock.patch.object(evaluator.provider, "inspect_state", return_value={"Status": "exited"}),
                        mock.patch.object(evaluator.provider, "remove")]
        for p in self.patches:
            p.start()
        self.site = types.SimpleNamespace(images={"toolchain": "img"}, cpus=2, memory="4g", pids=256)
        self.out = Path(tempfile.mkdtemp()) / "out"

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def mounts(self, writable=False):
        """{container name: {target: source}} for every `docker run` made (only its writable mounts)."""
        found = {}
        for args in self.calls:
            if args[:1] != ["run"]:
                continue
            name = args[args.index("--name") + 1]
            found[name] = {}
            for i, a in enumerate(args):
                if a == "--mount" and not (writable and args[i + 1].endswith(",readonly")):
                    fields = dict(f.split("=", 1) for f in args[i + 1].split(",") if "=" in f)
                    found[name][fields["target"]] = Path(fields["source"])
        return found

    def assert_host_cannot_reach(self, host, guarded):
        """No mount of the host's reaches what a device writes (a writable one could plant a link there)."""
        for target, source in host.items():
            for g in guarded:
                self.assertFalse(source == g or source in g.parents or g in source.parents,
                                 f"host mount {target} -> {source} reaches {g}")

    def test_multicore(self):
        repo = device_repo()
        task = device_task(repo, validation=["p0", "p1"])
        dev = Path(tempfile.mkdtemp())
        (dev / "spec/devices").mkdir(parents=True)
        (dev / "spec/devices/d.json").write_text(json.dumps(DESC))
        evaluator._measure_multicore(self.site, Path(tempfile.mkdtemp()), dev, task, self.out, "ev", repo)
        found, writable = self.mounts(), self.mounts(writable=True)
        observers = [self.out / "observer-0", self.out / "observer-1"]
        outboxes = [self.out / "mailbox/out/0", self.out / "mailbox/out/1"]
        self.assertEqual(found["ev-dev0"]["/obs"], observers[0])
        self.assertEqual(found["ev"]["/out"], self.out / "host")
        # The observers are out of the host's sight, and the outboxes it reads
        # are not writable to it: the device writes nothing the host can aim.
        self.assert_host_cannot_reach({t: s for t, s in found["ev"].items() if t != "/mb/out"}, observers)
        self.assert_host_cannot_reach(writable["ev"], observers + outboxes)
        self.assertNotIn("/mb/in", writable["ev-dev0"])
        self.assertEqual(writable["ev-dev0"]["/mb/out"], outboxes[0])
        # The runtime takes the host's wait from its environment; a backend
        # argument of the same name is passed twice and every point fails.
        self.assertNotIn("host_timeout", json.loads((self.out / "host" / "backend-0.json").read_text()))
        self.assertIn("TNCA_HOST_TIMEOUT=60", (self.out / "host" / "evaluate.sh").read_text())

    def test_a_host_that_runs_too_long_leaves_no_container(self):
        def docker(*args, **kw):
            self.calls.append(list(args))
            if args[:1] == ("run",) and "-d" not in args:
                raise subprocess.TimeoutExpired(list(args), kw.get("timeout"))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        removed = []
        with mock.patch.object(evaluator.provider, "docker", side_effect=docker), \
                mock.patch.object(evaluator.provider, "remove", side_effect=removed.append):
            repo = device_repo()
            task = device_task(repo, validation=["p0", "p1"])
            dev = Path(tempfile.mkdtemp())
            (dev / "spec/devices").mkdir(parents=True)
            (dev / "spec/devices/d.json").write_text(json.dumps(DESC))
            got = evaluator._measure_multicore(self.site, Path(tempfile.mkdtemp()), dev, task, self.out, "ev", repo)
        self.assertIsNone(got["exit"])
        self.assertEqual(got["missing"], ["p0", "p1"])
        # Each device, and the host that timed out, removed -- after any
        # container of the same name a run that died left behind.
        self.assertEqual(removed.count("ev-dev0"), 2)
        self.assertEqual(removed.count("ev-dev1"), 2)
        self.assertEqual(removed.count("ev"), 2)

    def test_single_core(self):
        task = device_task(device_repo(), validation=["p0"])
        (self.out / "device-build").mkdir(parents=True)
        (self.out / "device-build" / "bench.txt").write_text("/work/bench.vvp\n")
        evaluator._measure_observed(self.site, Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()), task, self.out, "ev")
        found = self.mounts()
        self.assertEqual(found["ev"]["/out"], self.out / "host")
        self.assert_host_cannot_reach(found["ev"], [self.out / "observer-0"])


if __name__ == "__main__":
    unittest.main()

