"""Building on one server and measuring on another.

The hosts file, the tool roots a build host overrides, the light packet and
its import, the detached worker steps, and a whole pipeline -- sync, doctor,
build, export, transfer, import, measure -- run between two "servers" that are
two directories on this machine, reached through an ``ssh`` that runs its
command locally.  No vendor tool, card or network is involved.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.adapters.u50 import U50Adapter, U50Config
from tritonchip_fpga_harness.adapters.u280 import U280FlowAdapter
from tritonchip_fpga_harness.core import HarnessError, export_evidence, import_run, load_profile, submit
from tritonchip_fpga_harness.hosts import host_for, parse_hosts
from tritonchip_fpga_harness.pipeline import run_pipeline
from tritonchip_fpga_harness.schema import SchemaError, load_json, validate_worker_request
from tritonchip_fpga_harness.toolroots import PLATFORMS_ENV, VITIS_ENV, tool_roots, vitis_version
from tritonchip_fpga_harness.worker import remote_status, worker_main

PY = sys.executable
RUN = "import json,os,pathlib; run=pathlib.Path(os.environ['TNCA_HARNESS_RUN_DIR']); (run/'board').mkdir(parents=True,exist_ok=True); "
RECORD = {
    "schema": "archbench-torch-record/1", "point_id": "op.add/tiny/fp32-n1", "family": "add", "tier": "tiny",
    "status": "pass", "evidence_level": "measured-board", "platform": {"backend": "fpga-u50", "kind": "tnca"},
    "time": {"unit": "cycles", "device": 123, "clock_hz": 100_000_000, "launches": 1},
    "counters": {"done_cycles": 123, "done_retired": 9, "perf_cycles": 120}, "problems": [],
}
CONFORMANCE = [PY, "-c", RUN + "(run/'board'/'conformance.json').write_text(json.dumps([{'id':'R36-TEST-01','passed':True}]))"]
ARCHBENCH = [PY, "-c", RUN + "(run/'board'/'archbench.jsonl').write_text(json.dumps(" + repr(RECORD) + ")+'\\n')"]

# Runs the command after the destination locally, as OpenSSH does on the
# server: the words are joined with spaces and given to a shell. Two
# directories stand for two servers.
FAKE_SSH = """#!{python}
import os, sys
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "-T":
        i += 1
    elif args[i] in ("-p", "-i", "-o"):
        i += 2
    elif args[i] == "--":
        i += 1
        break
    else:
        break
os.environ["COPYFILE_DISABLE"] = "1"   # no AppleDouble entries from macOS tar
os.execvp("sh", ["sh", "-c", " ".join(args[i + 1:])])
"""


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), "-c", "user.name=harness-test",
                          "-c", "user.email=harness-test@example.invalid", *args],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def candidate_repo(root: Path) -> tuple[Path, str]:
    """A committed checkout that carries this harness, as a real one does."""
    repo = root / "source"
    shutil.copytree(HERE, repo / "tools" / "tritonchip_fpga_harness",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The real checkout ignores Python bytecode.  The worker imports from its
    # synced checkout before checking source identity, so the fixture must too.
    (repo / ".gitignore").write_text("__pycache__/\n*.py[cod]\n", encoding="utf-8")
    (repo / "rtl").mkdir(parents=True)
    (repo / "rtl" / "core.v").write_text("module core; endmodule\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "candidate")
    return repo, git(repo, "rev-parse", "HEAD")


class HostsFile(unittest.TestCase):
    def good(self) -> dict:
        return {"schema_version": 1, "hosts": {
            "build": {"ssh": "build-worker", "roles": ["build"], "python": "/opt/py/bin/python3",
                      "workspace": "/data/tc", "state_dir": "/data/tc-state", "vitis_root": "/opt/Xilinx/Vitis/2022.2"},
            "card": {"ssh": "card-host", "roles": ["board"], "workspace": "/home/f/tc", "state_dir": "/home/f/state",
                     "environment": {"TORCH_PYTHON": "/home/f/venv/bin/python"}},
        }}

    def test_parse_paths_roles_and_environment(self) -> None:
        hosts = parse_hosts(self.good())
        build = host_for(hosts, "build", "build")
        self.assertEqual(build.checkout("abc"), "/data/tc/checkouts/abc")
        self.assertEqual(build.cli("abc"), "/data/tc/checkouts/abc/tools/tritonchip_fpga_harness/cli.py")
        self.assertEqual(build.remote_env(), {VITIS_ENV: "/opt/Xilinx/Vitis/2022.2"})
        self.assertEqual(host_for(hosts, "card", "board").remote_env(), {"TORCH_PYTHON": "/home/f/venv/bin/python"})
        with self.assertRaises(SchemaError):
            host_for(hosts, "card", "build")

    def test_the_example_is_a_valid_hosts_file(self) -> None:
        hosts = parse_hosts(load_json(HERE / "examples" / "hosts.example.json"))
        self.assertEqual({name: host.roles for name, host in hosts.items()}, {"build": ("build",), "card": ("board",)})
        self.assertIn("PATH", hosts["build"].remote_env())

    def test_refuses_what_could_reach_a_shell(self) -> None:
        for path, value in ((("build", "workspace"), "relative/path"), (("build", "ssh"), "host; rm -rf /"),
                            (("card", "state_dir"), "/home/f/../etc"), (("build", "roles"), ["compile"]),
                            (("card", "environment"), {"lower": "x"}), (("build", "python"), "python3 -c x")):
            data = self.good()
            data["hosts"][path[0]][path[1]] = value
            with self.subTest(field=path, value=value), self.assertRaises(SchemaError):
                parse_hosts(data)
        data = self.good()
        data["hosts"]["build"]["password"] = "x"
        with self.assertRaises(SchemaError):
            parse_hosts(data)


class ToolRoots(unittest.TestCase):
    def test_host_override_wins_and_is_recorded(self) -> None:
        data = {"vitis_root": "/tools/Xilinx/Vitis/2023.2"}
        self.assertEqual(tool_roots(data, {})["vitis_source"], "target")
        roots = tool_roots(data, {VITIS_ENV: "/opt/Xilinx/Vitis/2022.2", PLATFORMS_ENV: "/opt/p"})
        self.assertEqual((roots["vitis_root"], roots["vitis_source"], roots["vitis_version"], roots["platforms_root"]),
                         ("/opt/Xilinx/Vitis/2022.2", "host", "2022.2", "/opt/p"))
        self.assertIsNone(vitis_version("/opt/vitis-latest"))

    def test_u280_adapter_links_with_the_host_installation(self) -> None:
        profile = load_profile(HERE, "u280")
        with mock.patch.dict(os.environ, {VITIS_ENV: "/opt/Xilinx/Vitis/2022.2"}):
            config = U280FlowAdapter()._impl(HERE.parents[1], profile).config
        self.assertEqual(config.environment["XILINX_VITIS"], "/opt/Xilinx/Vitis/2022.2")
        self.assertEqual(set(config.accepted_vitis_versions), {"2022.2", "2023.2"})

    def test_preflight_refuses_a_vitis_the_card_does_not_accept(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            for version, ok in (("2022.2", True), ("2024.1", False)):
                root = Path(td) / "Vitis" / version
                root.mkdir(parents=True)
                adapter = U50Adapter(U50Config(repo=Path(td), vitis_root=root, accepted_vitis_versions=("2022.2", "2023.2")))
                checks = {c.name: c for c in adapter.preflight(Path(td) / "build", dry_run=True).checks}
                self.assertEqual(checks["vitis_version"].ok, ok, checks["vitis_version"].detail)


class PacketAndImport(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tc-split-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo, _ = candidate_repo(self.tmp)
        self.build_state = self.tmp / "build-state"
        manifest = {"candidate_id": "split-1", "target": "u50", "mode": "full"}
        self.result = submit(HERE, self.repo, manifest, self.build_state, "mock")
        self.run_id = self.result["run_id"]
        heavy = self.build_state / "runs" / self.run_id / "artifacts" / "tmp"
        (heavy / "link" / "vivado").mkdir(parents=True)
        (heavy / "link" / "vivado" / "huge.dcp").write_bytes(b"x" * 1000)
        (heavy / "reports" / "link" / "imp").mkdir(parents=True)
        (heavy / "reports" / "link" / "imp" / "impl_1_timing_summary_routed.rpt").write_text("WNS 0.138\n", encoding="utf-8")

    def packet_tar(self, mutate=None) -> io.BytesIO:
        packet = export_evidence(self.build_state, self.run_id, self.tmp / "outgoing", light=True)
        if mutate:
            mutate(packet)
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            archive.add(packet, arcname=self.run_id)
        data.seek(0)
        return data

    def test_light_packet_keeps_the_image_and_reports_not_the_working_tree(self) -> None:
        packet = export_evidence(self.build_state, self.run_id, self.tmp / "outgoing", light=True)
        files = load_json(packet / "evidence-manifest.json")["files"]
        self.assertIn("artifacts/tnca_u50.xclbin", files)
        self.assertIn("artifacts/tmp/reports/link/imp/impl_1_timing_summary_routed.rpt", files)
        self.assertFalse(any("huge.dcp" in name for name in files))

    def test_import_installs_a_verified_run_once(self) -> None:
        card = self.tmp / "card-state"
        value = import_run(card, self.packet_tar())
        self.assertEqual((value["run_id"], value["already_present"]), (self.run_id, False))
        self.assertEqual(value["xclbin_sha256"], self.result["artifacts"]["artifacts/tnca_u50.xclbin"])
        self.assertTrue((card / "runs" / self.run_id / "imported.json").is_file())
        self.assertTrue(import_run(card, self.packet_tar())["already_present"])

    def test_import_refuses_a_changed_file(self) -> None:
        def tamper(packet: Path) -> None:
            (packet / "artifacts" / "tnca_u50.xclbin").write_bytes(b"another image\n")
        with self.assertRaisesRegex(HarnessError, "does not match its hash"):
            import_run(self.tmp / "card-state", self.packet_tar(tamper))
        self.assertFalse((self.tmp / "card-state" / "runs" / self.run_id).exists())

    def test_import_refuses_paths_that_leave_the_packet(self) -> None:
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            info = tarfile.TarInfo(f"{self.run_id}/../../escape.txt")
            payload = b"out"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        data.seek(0)
        with self.assertRaisesRegex(HarnessError, "plain relative path"):
            import_run(self.tmp / "card-state", data)
        self.assertFalse((self.tmp / "escape.txt").exists())

    def test_import_refuses_a_build_that_did_not_pass(self) -> None:
        def fail(packet: Path) -> None:
            result = load_json(packet / "result.json")
            result["status"] = "TIMING_FAIL"
            (packet / "result.json").write_text(json.dumps(result), encoding="utf-8")
            listed = load_json(packet / "evidence-manifest.json")
            import hashlib
            listed["files"]["result.json"] = hashlib.sha256((packet / "result.json").read_bytes()).hexdigest()
            (packet / "evidence-manifest.json").write_text(json.dumps(listed), encoding="utf-8")
        with self.assertRaisesRegex(HarnessError, "only a passed full build"):
            import_run(self.tmp / "card-state", self.packet_tar(fail))


class DetachedWorker(unittest.TestCase):
    def test_detached_submit_is_polled_to_its_verdict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tc-detach-") as td:
            repo, commit = candidate_repo(Path(td))
            state = Path(td) / "state"
            request = {"schema_version": 1, "operation": "submit", "board": "u50", "adapter": "mock", "detach": True,
                       "skip_board": True, "state_dir": str(state),
                       "manifest": {"candidate_id": "d-1", "target": "u50", "mode": "full", "source_commit": commit}}
            out = io.StringIO()
            self.assertEqual(worker_main(package_dir=repo / "tools/tritonchip_fpga_harness", root=repo,
                                         stream=io.StringIO(json.dumps(request)), output=out), 0)
            started = json.loads(out.getvalue())["result"]
            run_id = started["run_id"]
            for _ in range(200):
                value = remote_status(state, run_id)
                if value["status"] not in ("STARTING", "PENDING", "RUNNING"):
                    break
                time.sleep(0.1)
            self.assertEqual(value["status"], "PASS", value)
            self.assertEqual(value["result"]["status"], "PASS")
            # In a worker the child outlives its parent; here this process
            # is the parent and has to reap it.
            os.waitpid(started["pid"], 0)
            self.assertEqual(started["attempt"], f"{run_id}.submit.1")
            # The build's PASS does not answer for a board stage nobody asked for.
            self.assertEqual(remote_status(state, run_id, operation="measure")["status"], "UNKNOWN")

    def test_status_of_a_child_that_died_before_its_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            (state / "requests").mkdir()
            log = state / "requests" / "r1.submit.1.log"
            (state / "requests" / "r1.submit.1.json").write_text(json.dumps({
                "pid": 999999, "pgid": 999999, "pid_starttime": "1", "boot_id": "x", "log": str(log)}))
            log.write_text("harness: working tree is dirty\n")
            value = remote_status(state, "r1")
            self.assertEqual(value["status"], "FAILED_TO_START")
            self.assertIn("working tree is dirty", value["log_tail"][-1])

    def test_requests_name_their_run(self) -> None:
        with self.assertRaises(SchemaError):
            validate_worker_request({"operation": "status", "board": "u280", "adapter": "u280",
                                     "manifest": {"candidate_id": "x", "target": "u280", "mode": "full"}})


class Pipeline(unittest.TestCase):
    def test_build_on_one_server_measure_on_another(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tc-pipeline-") as td:
            tmp = Path(td)
            repo, commit = candidate_repo(tmp)
            ssh = tmp / "ssh"
            ssh.write_text(FAKE_SSH.format(python=PY), encoding="utf-8")
            ssh.chmod(ssh.stat().st_mode | stat.S_IXUSR)
            hosts = parse_hosts({"schema_version": 1, "hosts": {
                "builder": {"ssh": "builder", "roles": ["build"], "python": PY,
                            "workspace": str(tmp / "builder"), "state_dir": str(tmp / "builder-state"),
                            "vitis_root": "/opt/Xilinx/Vitis/2022.2"},
                "card": {"ssh": "card", "roles": ["board"], "python": PY,
                         "workspace": str(tmp / "card"), "state_dir": str(tmp / "card-state")},
            }})
            manifest = {"candidate_id": "split-e2e", "target": "u50", "mode": "full",
                        "board": {"backend": "fpga-u50", "profile": "core-vrf-sync-v1", "program": False,
                                  "conformance": {"command": CONFORMANCE}, "archbench": {"command": ARCHBENCH}}}
            record = run_pipeline(local_repo=repo, hosts=hosts, manifest=manifest, build_host="builder",
                                  board_host="card", adapter="mock", poll_s=0.2, ssh_binary=str(ssh))
            self.assertEqual(record["status"], "PASS", json.dumps(record, indent=1)[-3000:])
            self.assertEqual([s["name"] for s in record["steps"]],
                             ["sync:builder", "sync:card", "doctor:builder", "doctor:card", "build:submit", "build",
                              "export", "transfer", "board:measure", "board"])
            run_id = record["run_id"]
            # Built on one, measured on the other, the same image throughout.
            self.assertTrue((tmp / "builder-state" / "runs" / run_id / "artifacts" / "tnca_u50.xclbin").is_file())
            self.assertTrue((tmp / "card-state" / "runs" / run_id / "imported.json").is_file())
            self.assertEqual(record["board"]["xclbin_sha256"], record["build"]["xclbin_sha256"])
            self.assertEqual(record["board"]["software"]["relation"], "same-commit")
            self.assertEqual(record["board"]["archbench"]["status"], "PASS")
            self.assertEqual(record["steps"][0]["detail"]["action"], "created")
            # The build server ran with its own Vitis root, and said so.
            build_run = load_json(tmp / "builder-state" / "runs" / run_id / "run.json")
            self.assertEqual(build_run["environment"]["tool_roots"]["vitis_source"], "host")
            # A second sync finds the clean checkout and leaves it.
            again = run_pipeline(local_repo=repo, hosts=hosts, manifest=dict(manifest, candidate_id="split-e2e"),
                                 build_host="builder", board_host="card", adapter="mock", poll_s=0.2,
                                 ssh_binary=str(ssh), skip_board=True)
            self.assertEqual(again["status"], "PASS")
            self.assertEqual(again["steps"][0]["detail"]["action"], "present")

    def test_a_board_block_needs_a_card_server(self) -> None:
        hosts = parse_hosts({"schema_version": 1, "hosts": {
            "builder": {"ssh": "b", "roles": ["build"], "workspace": "/w", "state_dir": "/s"}}})
        manifest = {"candidate_id": "x", "target": "u50", "mode": "full", "board": {"backend": "fpga-u50", "profile": "p"}}
        with self.assertRaisesRegex(RuntimeError, "board-host"):
            run_pipeline(local_repo=HERE, hosts=hosts, manifest=manifest, build_host="builder")


class MeasurementMarker(unittest.TestCase):
    """An imported build's PASS, whatever its clock says, is never the board stage's verdict."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tc-marker-")
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        run = self.state / "runs" / "r1"
        run.mkdir(parents=True)
        # Built on a server whose clock runs an hour ahead.
        (run / "run.json").write_text(json.dumps({"status": "PASS", "finished_at": time.time() + 3600,
                                                   "manifest": {"candidate_id": "c"}}))
        (run / "result.json").write_text(json.dumps({"status": "PASS"}))
        (self.state / "requests").mkdir()

    def attempt(self, pid: int) -> None:
        (self.state / "requests" / "r1.measure.1.json").write_text(json.dumps(
            {"pid": pid, "pgid": pid, "pid_starttime": None, "boot_id": None,
             "log": str(self.state / "requests" / "r1.measure.1.log")}))

    def test_a_live_measurement_that_has_not_named_itself_is_starting(self) -> None:
        self.attempt(os.getpid())
        self.assertEqual(remote_status(self.state, "r1", operation="measure")["status"], "STARTING")

    def test_a_dead_one_never_started(self) -> None:
        self.attempt(999999)
        self.assertEqual(remote_status(self.state, "r1", operation="measure")["status"], "FAILED_TO_START")

    def test_once_named_the_record_answers(self) -> None:
        self.attempt(999999)
        run = self.state / "runs" / "r1" / "run.json"
        value = json.loads(run.read_text())
        value.update(status="BOARD_FAIL", measure_request="r1.measure.1")
        run.write_text(json.dumps(value))
        self.assertEqual(remote_status(self.state, "r1", operation="measure")["status"], "BOARD_FAIL")


class MeasureRecord(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tc-measure-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo, self.commit = candidate_repo(self.tmp)
        self.state = self.tmp / "state"
        self.manifest = {"candidate_id": "m-1", "target": "u50", "mode": "full",
                         "board": {"backend": "fpga-u50", "profile": "core-vrf-sync-v1", "program": False,
                                   "conformance": {"command": CONFORMANCE}}}
        self.run_id = submit(HERE, self.repo, self.manifest, self.state, "mock", skip_board=True)["run_id"]
        self.run_json = self.state / "runs" / self.run_id / "run.json"

    def test_a_measure_that_cannot_take_the_lease_leaves_the_record_alone(self) -> None:
        from tritonchip_fpga_harness.core import lease, measure_existing
        before = self.run_json.read_text()
        with lease(self.state, "full"):
            with self.assertRaisesRegex(HarnessError, "no full slot"):
                measure_existing(HERE, self.repo, self.state, self.run_id)
        self.assertEqual(self.run_json.read_text(), before)

    def test_a_change_to_accepted_versions_does_not_make_a_build_stale(self) -> None:
        import hashlib
        from tritonchip_fpga_harness.core import measure_existing
        run = load_json(self.run_json)
        run["profile"] = dict(run["profile"], vitis_versions=["2019.1"])
        run["profile_hash"] = hashlib.sha256(json.dumps(run["profile"], sort_keys=True).encode()).hexdigest()
        self.run_json.write_text(json.dumps(run))
        result = measure_existing(HERE, self.repo, self.state, self.run_id, request="r.measure.1")
        self.assertEqual(result["status"], "PASS", result.get("measurement_error"))
        self.assertIn("vitis_versions", result["profile_note"])
        self.assertEqual(load_json(self.run_json)["measure_request"], "r.measure.1")
        # A change to what the image is still refuses.
        run["profile"] = dict(run["profile"], part="xcu55c-fsvh2892-2L-e")
        self.run_json.write_text(json.dumps(run))
        with self.assertRaisesRegex(HarnessError, "profile changed"):
            measure_existing(HERE, self.repo, self.state, self.run_id)

    def test_reconcile_orphans_a_lost_measurement(self) -> None:
        from tritonchip_fpga_harness.core import reconcile
        run = load_json(self.run_json)
        run.update(status="MEASURING", controller={"pid": 999999, "pid_starttime": "1", "boot_id": "x"})
        self.run_json.write_text(json.dumps(run))
        self.assertEqual(reconcile(self.state, self.run_id)["status"], "ORPHANED")


class Hardening(unittest.TestCase):
    def test_run_ids_are_names_not_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(HarnessError):
                export_evidence(Path(td), "../runs/x", Path(td) / "out")
            with self.assertRaises(HarnessError):
                export_evidence(Path(td), "/abs/run", Path(td) / "out")
        with self.assertRaises(SchemaError):
            validate_worker_request({"operation": "export", "board": "u280", "adapter": "u280", "run_id": "/data/runs/x",
                                     "manifest": {"candidate_id": "x", "target": "u280", "mode": "full"}})

    def test_no_value_can_become_an_ssh_option(self) -> None:
        from tritonchip_fpga_harness.worker import DispatchError, ssh_prefix
        argv = ssh_prefix(host="card-host", env={"A": "1"})
        self.assertEqual(argv[argv.index("--") + 1], "card-host")
        with self.assertRaises(DispatchError):
            ssh_prefix(host="-oProxyCommand=x")
        data = HostsFile().good()
        data["hosts"]["card"]["python"] = "-oProxyCommand=/tmp/x"
        with self.assertRaises(SchemaError):
            parse_hosts(data)

    def test_nested_readme_is_hashed_and_imported(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tc-readme-") as td:
            repo, _ = candidate_repo(Path(td))
            state = Path(td) / "state"
            run_id = submit(HERE, repo, {"candidate_id": "r", "target": "u50", "mode": "full"}, state, "mock")["run_id"]
            run_dir = state / "runs" / run_id
            (run_dir / "artifacts" / "kernel").mkdir(parents=True)
            (run_dir / "artifacts" / "kernel" / "README.txt").write_text("kept\n")
            impl = run_dir / "artifacts/tmp/link/vivado/vpl/prj/prj.runs/impl_1"
            impl.mkdir(parents=True)
            (impl / "runme.log").write_text("DRC finished with 0 Errors\n")
            packet = export_evidence(state, run_id, Path(td) / "out", light=True)
            files = load_json(packet / "evidence-manifest.json")["files"]
            self.assertIn("artifacts/kernel/README.txt", files)
            self.assertIn("artifacts/tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log", files)
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as archive:
                archive.add(packet, arcname=run_id)
            data.seek(0)
            self.assertEqual(import_run(Path(td) / "card", data)["files"], len(files))

    def test_u50_links_with_the_root_it_records(self) -> None:
        from tritonchip_fpga_harness.core import U50FlowAdapter, environment_snapshot
        profile = load_profile(HERE, "u50")
        with mock.patch.dict(os.environ, {VITIS_ENV: "/opt/Xilinx/Vitis/2022.2"}):
            config = U50FlowAdapter()._impl(HERE.parents[1], profile).config
            recorded = environment_snapshot(HERE.parents[1], profile, "u50")["tool_roots"]
        self.assertEqual(config.environment["XILINX_VITIS"], "/opt/Xilinx/Vitis/2022.2")
        self.assertEqual((recorded["vitis_root"], recorded["vitis_source"]), ("/opt/Xilinx/Vitis/2022.2", "host"))


class PipelineRecovery(unittest.TestCase):
    """What --build-run is for: repeating a board stage, and refusing the wrong run."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tc-recover-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo, self.commit = candidate_repo(self.tmp)
        self.ssh = self.tmp / "ssh"
        self.ssh.write_text(FAKE_SSH.format(python=PY), encoding="utf-8")
        self.ssh.chmod(self.ssh.stat().st_mode | stat.S_IXUSR)
        self.hosts = parse_hosts({"schema_version": 1, "hosts": {
            "builder": {"ssh": "builder", "roles": ["build"], "python": PY,
                        "workspace": str(self.tmp / "builder"), "state_dir": str(self.tmp / "builder-state")},
            "card": {"ssh": "card", "roles": ["board"], "python": PY,
                     "workspace": str(self.tmp / "card"), "state_dir": str(self.tmp / "card-state")},
            "both": {"ssh": "both", "roles": ["build", "board"], "python": PY,
                     "workspace": str(self.tmp / "both"), "state_dir": str(self.tmp / "both-state")},
        }})

    def manifest(self, conformance_rc: int = 0, candidate: str = "rec-1") -> dict:
        command = [PY, "-c", RUN + "(run/'board'/'conformance.json').write_text(json.dumps([{'id':'R36-TEST-01','passed':True}])); "
                   + f"raise SystemExit({conformance_rc})"]
        return {"candidate_id": candidate, "target": "u50", "mode": "full",
                "board": {"backend": "fpga-u50", "profile": "core-vrf-sync-v1", "program": False,
                          "conformance": {"command": command}}}

    def pipe(self, manifest: dict, **options) -> dict:
        options.setdefault("build_host", "builder")
        options.setdefault("board_host", "card")
        return run_pipeline(local_repo=self.repo, hosts=self.hosts, manifest=manifest, adapter="mock",
                            poll_s=0.2, ssh_binary=str(self.ssh), **options)

    def test_a_failed_board_stage_is_repeated_without_building_again(self) -> None:
        first = self.pipe(self.manifest(conformance_rc=3))
        self.assertEqual(first["status"], "BOARD_FAIL", first.get("error"))
        second = self.pipe(self.manifest(), build_run=first["run_id"])
        self.assertEqual(second["status"], "PASS", second.get("error"))
        self.assertNotIn("build:submit", [s["name"] for s in second["steps"]])
        self.assertEqual(second["board"]["attempt"], f"{first['run_id']}.measure.2")
        result = load_json(self.tmp / "card-state" / "runs" / first["run_id"] / "result.json")
        self.assertEqual(result["board_config_source"], "measure")

    def test_build_run_must_be_this_candidate(self) -> None:
        first = self.pipe(self.manifest(), skip_board=True)
        other = self.pipe(self.manifest(candidate="rec-2"), build_run=first["run_id"])
        self.assertEqual(other["status"], "PIPELINE_FAIL")
        self.assertIn("is not this candidate", other["error"])
        bad = self.pipe(self.manifest(), build_run="../../etc")
        self.assertEqual(bad["status"], "PIPELINE_FAIL")

    def test_a_flaky_connection_is_retried(self) -> None:
        # Every third connection fails as a slow jump host does.
        flaky = self.tmp / "flaky-ssh"
        counter = self.tmp / "ssh-count"
        flaky.write_text(FAKE_SSH.format(python=PY).replace(
            'os.environ["COPYFILE_DISABLE"] = "1"',
            f'import pathlib\nc = pathlib.Path({str(counter)!r})\nn = int(c.read_text()) + 1 if c.exists() else 1\n'
            'c.write_text(str(n))\nif n % 3 == 0:\n    sys.stderr.write("Connection timed out during banner exchange\\n"); sys.exit(255)\n'
            'os.environ["COPYFILE_DISABLE"] = "1"'), encoding="utf-8")
        flaky.chmod(flaky.stat().st_mode | stat.S_IXUSR)
        with mock.patch("tritonchip_fpga_harness.pipeline.SSH_RETRY_WAITS", (0.0, 0.0, 0.0)):
            record = run_pipeline(local_repo=self.repo, hosts=self.hosts, manifest=self.manifest(), adapter="mock",
                                  build_host="builder", board_host="card", poll_s=0.2, ssh_binary=str(flaky))
        self.assertEqual(record["status"], "PASS", record.get("error"))
        self.assertGreater(int(counter.read_text()), 20)

    def test_one_server_with_both_roles_measures_in_place(self) -> None:
        record = self.pipe(self.manifest(), build_host="both", board_host="both")
        self.assertEqual(record["status"], "PASS", record.get("error"))
        self.assertNotIn("transfer", [s["name"] for s in record["steps"]])


if __name__ == "__main__":
    unittest.main()
