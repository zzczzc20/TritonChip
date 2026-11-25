"""Routing congestion as the agent sees it after a link (congestion.py).

The data are trimmed copies of two real links of 2026-11-20/21 on the cluster, same platform and flow: b3 of
R41-HPC-demo, which failed in routing (1928 node overlaps), and the base, which routed at 100 MHz; and Vivado's
`report_design_analysis -congestion` of each run's last checkpoint. The batch-queue path runs against the fake
cluster of the FPGA harness's own tests (fake ssh, bsub, bjobs), with a fake `vivado` that writes the report.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import congestion, fpga_slow  # noqa: E402
from tritonchip_fpga_harness.tests.test_lsf_host import cluster  # noqa: E402
from tritonchip_fpga_harness.tests.test_split_hosts import PY  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "congestion"
HOME_PIPE = "device/system/shell/u_system/g_tile[1].u_tile/g_home_pipe.u_home_pipe"


def read(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


class Log(unittest.TestCase):
    def test_a_link_that_did_not_route_says_where_and_how_badly(self):
        log = congestion.parse_log(read("b3_runme.log"))
        self.assertEqual(log["post_place_estimate"], {"wns_ns": -0.863, "tns_ns": -120.367})
        self.assertEqual(log["placer"]["device"]["long"], {"level": 7, "direction": "north"})
        self.assertEqual(log["placer"]["SLR0"]["long"]["level"], 7)
        self.assertEqual(log["router_initial"]["long"]["level"], 6)
        route = log["route"]
        self.assertEqual((route["routed"], route["overlaps"]), (False, 1928))
        self.assertEqual(len(route["worst_nodes"]), 10)
        first = route["worst_nodes"][0]
        self.assertEqual((first["tile"], first["clock_region_row"]), ("INT_X91Y259", 4))
        self.assertIn("reset_sync[1]", first["nets"])
        modules = dict(map(tuple, route["modules_on_worst_nodes"]))
        self.assertEqual(modules[HOME_PIPE], 9)
        self.assertEqual(modules["(kernel top)"], 9)

    def test_a_link_that_routed_says_so_with_its_estimates(self):
        log = congestion.parse_log(read("base_runme.log"))
        self.assertEqual(log["route"], {"routed": True})
        self.assertEqual(log["post_place_estimate"]["wns_ns"], 0.004)
        self.assertEqual(log["placer"]["device"]["long"]["level"], 6)

    def test_a_log_that_stopped_before_placement_says_nothing(self):
        self.assertEqual(congestion.parse_log("Command: link_design\nCommand: opt_design\n"), {})

    def test_names_are_cut_to_the_kernel_or_the_shell(self):
        self.assertEqual(congestion.short("level0_i/ulp/tnca_u50_1/inst/device/system/x"), "device/system/x")
        self.assertEqual(congestion.short("level0_i/ulp/hmss_0/inst/hbm_inst/inst/TWO_STACK.u_hbm_top/a/b"),
                         "platform/ulp/hmss_0/inst/hbm_inst")
        self.assertEqual(congestion.module_of("level0_i/ulp/tnca_u50_1/inst/reset_sync[1]"), "(kernel top)")
        self.assertEqual((congestion.level("64x64"), congestion.level("128x128"), congestion.level("8x16")),
                         (6, 7, None))


class Windows(unittest.TestCase):
    def test_the_placers_worst_window_names_the_changed_path_and_the_routers_the_failure(self):
        log = congestion.parse_log(read("b3_runme.log"))
        seen = congestion.visible({"log": log, "windows": congestion.parse_design_analysis(
            read("b3_design_analysis.rpt"))})
        placer, router = seen["windows"]["placer"], seen["windows"]["router"]
        self.assertEqual(placer[0]["level"], 7)
        self.assertTrue(placer[0]["cells"][0][0].endswith("g_tile[0].u_tile/u_core/u_txf"))
        # At one level, the windows holding the router's worst nodes come first: tile 1's home-cache pipe.
        hot = [w for w in router if w.get("at_worst_nodes")]
        self.assertTrue(hot)
        self.assertTrue(all(w["cells"][0][0] == HOME_PIPE and w["cells"][0][1] >= 85 for w in hot))
        # One window reported for several directions or types appears once.
        windows = [w["window"] for w in router]
        self.assertEqual(len(windows), len(set(windows)))
        self.assertLessEqual(len(placer), congestion.WINDOWS)
        self.assertIn("UG949", seen["note"])

    def test_the_base_never_reaches_level_7_and_its_placer_windows_hold_no_transfer_engine(self):
        windows = congestion.parse_design_analysis(read("base_design_analysis.rpt"))
        self.assertEqual(max(w["level"] for w in windows["placer"]), 6)
        self.assertFalse(any("u_txf" in c[0] for w in windows["placer"] for c in w["cells"]))

    def test_nothing_read_is_nothing_shown(self):
        self.assertEqual(congestion.visible({}), {})
        self.assertEqual(congestion.visible({"log": {}, "analysis": {"why": "x"}}), {})


class Visible(unittest.TestCase):
    def test_the_name_ordered_conflict_list_is_not_passed_on(self):
        line = ("ERROR: [VPL 18-1000] Routing results verification failed due to partially-conflicted nets "
                "(Up to first 10 of violated nets):  level0_i/ulp/tnca_u50_1/inst/device/control/a1_reg_n_0_[63]")
        site = types.SimpleNamespace(fpga_pipeline={})
        pipe = {"status": "BUILD_FAIL", "build": {"status": "BUILD_FAIL", "log_errors": [line, "ERROR: route_design"]}}
        seen = fpga_slow._build_visible(site, pipe, "design", {}, {"note": "n", "route": {"overlaps": 3}})
        self.assertFalse(any("a1_reg" in e for e in seen["errors"]))
        self.assertTrue(any("see congestion" in e for e in seen["errors"]))
        self.assertEqual(seen["congestion"]["route"], {"overlaps": 3})


FAKE_VIVADO = """#!{python}
import shutil, sys
a = sys.argv[1:]
dcp, out = a[a.index("-tclargs") + 1:a.index("-tclargs") + 3]
assert dcp.endswith(".dcp"), dcp
shutil.copy({report!r}, out)
"""


class Collect(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="tc-congestion-")
        self.tmp = Path(self._tmp.name)
        self.ssh, self.profile = cluster(self.tmp)
        vitis = self.tmp / "vitis"
        (vitis / "bin").mkdir(parents=True)
        tool = vitis / "bin" / "vivado"
        tool.write_text(FAKE_VIVADO.format(python=PY, report=str(DATA / "b3_design_analysis.rpt")), encoding="utf-8")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        (vitis / "settings64.sh").write_text(f'PATH="{vitis}/bin:$PATH"; export PATH\n', encoding="utf-8")
        self.run_id = "tnca-0123456789abcdef"
        impl = (self.tmp / "hpc-state" / "runs" / self.run_id / "artifacts" / "tmp" / "link" / "vivado" / "vpl"
                / "prj" / "prj.runs" / "impl_1")
        impl.mkdir(parents=True)
        (impl / "runme.log").write_text(read("b3_runme.log"), encoding="utf-8")
        (impl / "level0_wrapper_routed_error.dcp").write_bytes(b"checkpoint")
        hosts = {"schema_version": 1, "hosts": {
            "hpc": {"ssh": "login", "roles": ["build"], "python": PY, "workspace": str(self.tmp / "hpc"),
                    "state_dir": str(self.tmp / "hpc-state"), "vitis_root": str(vitis),
                    "lsf": {"profile": str(self.profile), "queue": "cpu", "slots": 8}},
            "box": {"ssh": "login", "roles": ["build"], "python": PY, "workspace": str(self.tmp / "hpc"),
                    "state_dir": str(self.tmp / "hpc-state"), "vitis_root": str(vitis)}}}
        path = self.tmp / "hosts.json"
        path.write_text(json.dumps(hosts), encoding="utf-8")
        self.site = types.SimpleNamespace(fpga_pipeline={"hosts": path})

    def tearDown(self):
        self._tmp.cleanup()

    def collect(self, host, **kw):
        return congestion.collect(self.site, self.run_id, host, ssh_binary=str(self.ssh), poll_s=0.1, **kw)

    def test_a_failed_route_on_the_cluster_is_attributed_by_a_job_of_its_own(self):
        found = self.collect("hpc", analyse=True)
        self.assertNotIn("error", found)
        self.assertEqual(found["log"]["route"]["overlaps"], 1928)
        self.assertEqual(found["analysis"]["stat"], "DONE")
        self.assertEqual(found["windows"]["placer"][0]["level"], 7)
        opts = (self.tmp / "lsf-jobs" / "101.opts").read_text()
        self.assertIn(f"{self.run_id}-congestion", opts)
        self.assertIn("('-n', '4')", opts)
        # Asked again (a retried read), the report on disk is used and no second job is queued.
        again = self.collect("hpc", analyse=True)
        self.assertEqual(again["windows"], found["windows"])
        self.assertFalse((self.tmp / "lsf-jobs" / "102.name").exists())

    def test_a_plain_machine_gives_the_log_only(self):
        found = self.collect("box", analyse=True)
        self.assertIn("batch queue", found["analysis"]["why"])
        self.assertNotIn("windows", found)
        self.assertEqual(congestion.visible(found)["windows_not_attributed"], found["analysis"]["why"])

    def test_a_flow_that_did_not_fail_in_placement_or_routing_queues_nothing(self):
        found = self.collect("hpc", analyse=False)
        self.assertEqual(set(found), {"log"})
        self.assertFalse((self.tmp / "lsf-jobs" / "101.name").exists())

    def test_a_run_with_no_implementation_log_gives_nothing(self):
        self.assertEqual(congestion.collect(self.site, "tnca-ffffffffffffffff", "hpc", analyse=True,
                                            ssh_binary=str(self.ssh)), {})

    def placement(self):
        return congestion.placement(self.site, self.run_id, "hpc", ssh_binary=str(self.ssh))

    def test_a_running_link_reports_its_placement_once_placement_has_ended(self):
        impl = self.tmp / "hpc-state" / "runs" / self.run_id / congestion.IMPL_LOG
        full = read("b3_runme.log")
        seen = self.placement()
        self.assertEqual(seen["post_place_estimate"]["wns_ns"], -0.863)
        self.assertEqual(seen["placer"]["SLR0"]["long"]["level"], 7)
        # The log has moved past placement and the link left no report after it (a platform without the hook).
        self.assertEqual(set(seen), {"note", "post_place_estimate", "placer", "windows_not_attributed"})
        self.assertIn("no report after placement", seen["windows_not_attributed"])
        # Placement still running: the tables are not there yet, or not closed.
        impl.write_text(full.split("INFO: [Place 30-612]")[0], encoding="utf-8")
        self.assertEqual(self.placement(), {})
        impl.write_text(full.split("Command: phys_opt_design")[0], encoding="utf-8")
        self.assertEqual(self.placement(), {})
        impl.unlink()
        self.assertEqual(self.placement(), {})

    def test_the_report_after_placement_names_the_cells_of_each_congested_window(self):
        # 2026-11-24, the user: the cause of a congested placement, not only its level, while the link routes.
        run = self.tmp / "hpc-state" / "runs" / self.run_id
        full = read("b3_runme.log")
        (run / congestion.PLACED_REPORT).write_text(read("b3_design_analysis.rpt"), encoding="utf-8")
        seen = self.placement()
        self.assertNotIn("windows_not_attributed", seen)
        top = seen["windows"][0]
        self.assertEqual(top["level"], 7)
        self.assertIn("u_txf", " ".join(c[0] for c in top["cells"]))
        self.assertTrue(all(w["level"] >= congestion.MIN_LEVEL for w in seen["windows"]))
        self.assertLessEqual(len(seen["windows"]), congestion.WINDOWS)
        self.assertIn("report_design_analysis", seen["note"])
        # Placement has ended (the placer closed its task) but the report is not whole yet: nothing, and the next
        # read finds it.
        (run / congestion.PLACED_REPORT).unlink()
        (run / congestion.IMPL_LOG).write_text(
            full.split("Command: phys_opt_design")[0] + "Ending Placer Task | Checksum: 1ee0d8d53\n", encoding="utf-8")
        self.assertEqual(self.placement(), {})
        # The hook could not write it: why, from the platform's own note.
        (run / congestion.PLACED_FAILED).write_text("ERROR: [Common 17-53] User Exception: No open project.\n",
                                                    encoding="utf-8")
        self.assertIn("No open project", self.placement()["windows_not_attributed"])
        (run / congestion.PLACED_FAILED).unlink()
        # A placement with no window at level 5 or above says so; a report with no placer table is not read as one.
        (run / congestion.PLACED_REPORT).write_text("1. Placer Final Level Congestion Reporting\n\n", encoding="utf-8")
        self.assertEqual(self.placement()["windows"], "none at level 5 or above")
        (run / congestion.PLACED_REPORT).write_text("Design Analysis\n", encoding="utf-8")
        self.assertIn("no placer table", self.placement()["windows_not_attributed"])

    def test_an_unknown_machine_is_not_an_error_for_the_flow(self):
        self.assertIn("error", self.collect("nowhere", analyse=True))
        self.assertEqual(congestion.collect(self.site, self.run_id, None, analyse=True), {})


class Watch(unittest.TestCase):
    def test_the_wait_looks_at_the_link_between_polls_and_says_when_it_outlived_its_cap(self):
        class Proc:
            def __init__(self, ends_after):
                self.calls, self.ends_after = 0, ends_after

            def wait(self, timeout):
                self.calls += 1
                if self.calls > self.ends_after:
                    return 0
                raise subprocess.TimeoutExpired("link", timeout)

        looked = []
        self.assertFalse(fpga_slow._wait(Proc(3), 10.0, lambda: looked.append(1), 0.01))
        self.assertEqual(len(looked), 3)
        self.assertTrue(fpga_slow._wait(Proc(10 ** 6), 0.05, lambda: None, 0.01))

    def test_a_placement_at_level_7_is_flagged_with_the_agents_two_choices(self):
        b3 = congestion.parse_log(read("b3_runme.log"))
        base = congestion.parse_log(read("base_runme.log"))
        warning = fpga_slow.placement_warning(b3)
        self.assertEqual((warning["level"], warning["where"]), (7, "device, long congestion, north"))
        self.assertNotIn("windows", warning["why"])
        attributed = {**b3, "windows": congestion.placed_windows(read("b3_design_analysis.rpt"))}
        self.assertIn("`windows` names the cells", fpga_slow.placement_warning(attributed)["why"])
        self.assertIsNone(fpga_slow.placement_warning(base))
        self.assertEqual(fpga_slow.placement_warning(base, level=6)["level"], 6)
        job = {"id": "b3"}
        with mock.patch.object(fpga_slow.congestion_, "placement", return_value={"note": "n", **b3}):
            fpga_slow._watch_placement(None, job, "tnca-1", "hpc-1", 14400)
        self.assertIn("tnca-service abandon b3", job["placement"]["warning"]["choices"])
        self.assertIn("--to-end", job["placement"]["warning"]["choices"])
        job = {"id": "base"}
        with mock.patch.object(fpga_slow.congestion_, "placement", return_value={"note": "n", **base}):
            fpga_slow._watch_placement(None, job, "tnca-2", "hpc-1", 14400)
        self.assertNotIn("warning", job["placement"])

    def test_placement_is_read_until_found_then_left_alone(self):
        job = {"id": "b3"}
        found = {"note": "n", "placer": {"device": {"long": {"level": 7, "direction": "north"}}}}
        with mock.patch.object(fpga_slow.congestion_, "placement", side_effect=[{}, found]) as read_:
            fpga_slow._watch_placement(None, job, "tnca-1", "hpc-1", 600)
            self.assertNotIn("placement", job)
            fpga_slow._watch_placement(None, job, "tnca-1", "hpc-1", 14400)
            fpga_slow._watch_placement(None, job, "tnca-1", "hpc-1", 15000)
        self.assertEqual(read_.call_count, 2)
        self.assertEqual(job["placement"]["seen_after_minutes"], 240)
        job2 = {"id": "b4", "_cancel_requested": True}
        with mock.patch.object(fpga_slow.congestion_, "placement") as read_:
            fpga_slow._watch_placement(None, job2, "tnca-2", "hpc-1", 600)
        read_.assert_not_called()



class PlacementText(unittest.TestCase):
    """2026-11-25: what the task text says about a running flow's placement is the task's: the windows only where
    the pinned platform reports them (64d7679b on), the base's own placement only as the task states it."""

    def test_only_a_task_whose_platform_reports_the_windows_promises_them(self):
        from tritonchip_agent_harness.task import load_task, render
        tasks = HERE / "tasks"
        direction = render(load_task(tasks / "r41-dp-dir" / "task.json"), "open", island=0)
        self.assertIn("(`placement.windows`). The base's placement: level 6 at most", direction)
        layer = render(load_task(tasks / "r41-dp-formal" / "task.json"), "open", island=0)
        self.assertIn("its estimated routing congestion. A placement Vivado puts at congestion level 7", layer)
        self.assertNotIn("placement.windows", layer)
        self.assertNotIn("The base's placement", layer)

    def test_the_placement_keys_are_checked(self):
        from tritonchip_agent_harness.task import TaskError, load_task
        src = HERE / "tasks" / "r41-dp-dir"
        for key, value in (("placement_windows", "yes"), ("base_placement", ""), ("base_placement", 6)):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as tmp:
                import shutil
                dst = Path(tmp) / "t"
                shutil.copytree(src, dst)
                data = json.loads((dst / "task.json").read_text())
                data["services"]["fpga_slow"][key] = value
                (dst / "task.json").write_text(json.dumps(data))
                with self.assertRaisesRegex(TaskError, key):
                    load_task(dst / "task.json")


if __name__ == "__main__":
    unittest.main()
