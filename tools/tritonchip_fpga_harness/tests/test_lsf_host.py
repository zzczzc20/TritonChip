"""A build host that is a batch cluster: the link as an LSF job.

The "cluster" is this machine: ``bsub``, ``bjobs`` and ``bkill`` are small
programs that start the job in the background and keep its state in a
directory, put on the PATH by the profile the hosts file names. The login node
and the card server are directories reached through the fake ``ssh`` of
test_split_hosts. No vendor tool, card, scheduler or network is involved.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness import lsf
from tritonchip_fpga_harness.hosts import parse_hosts
from tritonchip_fpga_harness.pipeline import run_pipeline
from tritonchip_fpga_harness.schema import SchemaError, load_json
from tritonchip_fpga_harness.tests.test_split_hosts import ARCHBENCH, CONFORMANCE, FAKE_SSH, PY, candidate_repo

# bsub starts the job detached and returns; a watcher records how it ended.
BSUB = """#!{python}
import os, subprocess, sys
d = os.environ["FAKE_LSF_DIR"]
a = sys.argv[1:]
opts = {{}}
while a and a[0].startswith("-"):
    opts[a[0]] = a[1]
    a = a[2:]
n = len([f for f in os.listdir(d) if f.endswith(".name")]) + 101
open(os.path.join(d, f"{{n}}.name"), "w").write(opts["-J"])
open(os.path.join(d, f"{{n}}.opts"), "w").write(repr(sorted(opts.items())))
open(os.path.join(d, f"{{n}}.stat"), "w").write("RUN")
watch = ("import subprocess,sys,os\\n"
         "log=open(sys.argv[2],'ab')\\n"
         "p=subprocess.Popen(sys.argv[3:],stdout=log,stderr=subprocess.STDOUT)\\n"
         "open(sys.argv[1]+'.pid','w').write(str(p.pid))\\n"
         "c=p.wait()\\n"
         "open(sys.argv[1]+'.stat','w').write('DONE' if c==0 else 'EXIT')\\n")
subprocess.Popen([sys.executable, "-c", watch, os.path.join(d, str(n)), opts["-o"], *a],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(f"Job <{{n}}> is submitted to queue <{{opts['-q']}}>.")
"""
BJOBS = """#!{python}
import os, sys
d = os.environ["FAKE_LSF_DIR"]
name = sys.argv[sys.argv.index("-J") + 1]
found = False
for f in sorted(os.listdir(d)):
    if f.endswith(".name") and open(os.path.join(d, f)).read() == name:
        n = f[:-5]
        print(n, open(os.path.join(d, n + ".stat")).read(), "8*node07")
        found = True
if not found:
    print(f"Job <{{name}}> is not found", file=sys.stderr)
    sys.exit(255)
"""
BKILL = """#!{python}
import os, signal, sys
d = os.environ["FAKE_LSF_DIR"]
name = sys.argv[sys.argv.index("-J") + 1]
for f in sorted(os.listdir(d)):
    if f.endswith(".name") and open(os.path.join(d, f)).read() == name:
        n = f[:-5]
        try:
            os.kill(int(open(os.path.join(d, n + ".pid")).read()), signal.SIGTERM)
            print(f"Job <{{n}}> is being terminated")
        except (OSError, ValueError):
            pass
"""


def cluster(tmp: Path) -> tuple[Path, Path]:
    """The fake ssh and LSF's profile: (ssh, profile)."""
    ssh = tmp / "ssh"
    ssh.write_text(FAKE_SSH.format(python=PY), encoding="utf-8")
    ssh.chmod(ssh.stat().st_mode | stat.S_IXUSR)
    binaries, jobs = tmp / "lsf-bin", tmp / "lsf-jobs"
    binaries.mkdir()
    jobs.mkdir()
    for name, text in (("bsub", BSUB), ("bjobs", BJOBS), ("bkill", BKILL)):
        tool = binaries / name
        tool.write_text(text.format(python=PY), encoding="utf-8")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    profile = tmp / "profile.lsf"
    profile.write_text(f'PATH="{binaries}:$PATH"; FAKE_LSF_DIR="{jobs}"; export PATH FAKE_LSF_DIR\n', encoding="utf-8")
    return ssh, profile


def hosts_for(tmp: Path, profile: Path):
    return parse_hosts({"schema_version": 1, "hosts": {
        "hpc": {"ssh": "login", "roles": ["build"], "python": PY, "workspace": str(tmp / "hpc"),
                "state_dir": str(tmp / "hpc-state"), "vitis_root": "/opt/Xilinx/Vitis/2022.2",
                "lsf": {"profile": str(profile), "queue": "cpu", "slots": 8}},
        "card": {"ssh": "card", "roles": ["board"], "python": PY, "workspace": str(tmp / "card"),
                 "state_dir": str(tmp / "card-state")},
    }})


MANIFEST = {"candidate_id": "lsf-e2e", "target": "u50", "mode": "full",
            "board": {"backend": "fpga-u50", "profile": "core-vrf-sync-v1", "program": False,
                      "conformance": {"command": CONFORMANCE}, "archbench": {"command": ARCHBENCH}}}


class HostsFile(unittest.TestCase):
    def entry(self, **lsf_block):
        return {"schema_version": 1, "hosts": {"hpc": {
            "ssh": "login", "roles": ["build"], "workspace": "/w", "state_dir": "/s", "lsf": lsf_block}}}

    def test_a_cluster_names_its_profile_and_queue(self) -> None:
        host = parse_hosts(self.entry(profile="/opt/lsf/conf/profile.lsf", queue="cpu"))["hpc"]
        self.assertEqual(host.lsf, {"profile": "/opt/lsf/conf/profile.lsf", "queue": "cpu", "slots": 8})
        self.assertEqual(host.summary()["lsf"], {"queue": "cpu", "slots": 8})

    def test_what_is_refused(self) -> None:
        for block in ({"queue": "cpu"}, {"profile": "/p", "queue": "cpu", "slots": 0},
                      {"profile": "relative", "queue": "cpu"}, {"profile": "/p", "queue": "a b"},
                      {"profile": "/p", "queue": "cpu", "extra": 1}):
            with self.assertRaises(SchemaError, msg=block):
                parse_hosts(self.entry(**block))
        card = self.entry(profile="/p", queue="cpu")
        card["hosts"]["hpc"]["roles"] = ["board"]
        with self.assertRaises(SchemaError):
            parse_hosts(card)


class Job(unittest.TestCase):
    def test_the_script_runs_the_checkouts_submit_and_stops_its_tools(self) -> None:
        host = parse_hosts({"schema_version": 1, "hosts": {"hpc": {
            "ssh": "login", "roles": ["build"], "workspace": "/w", "state_dir": "/s", "python": "/py",
            "vitis_root": "/x/Vitis/2023.2", "environment": {"TNCA_BOARD": "u280"},
            "lsf": {"profile": "/p", "queue": "cpu"}}}})["hpc"]
        text = lsf.script(host, "abc123", "run-1", "u280", 3600)
        self.assertIn("export FPGA_HARNESS_VITIS_ROOT=/x/Vitis/2023.2\n", text)
        self.assertIn("export TNCA_BOARD=u280\n", text)
        self.assertIn("/py /w/checkouts/abc123/tools/tritonchip_fpga_harness/cli.py --repo /w/checkouts/abc123 "
                      "submit /s/requests/run-1.lsf.manifest.json --state-dir /s --run-id run-1 --adapter u280 "
                      "--skip-board --timeout 3600 &", text)
        self.assertIn("reconcile run-1 --state-dir /s --terminate", text)

    def test_a_run_id_is_a_name(self) -> None:
        host = hosts_for(Path("/tmp"), Path("/tmp/p"))["hpc"]
        with self.assertRaises(Exception):
            lsf.paths(host, "../x")


class Pipeline(unittest.TestCase):
    def test_the_link_is_a_job_and_the_card_measures_its_image(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tc-lsf-") as td:
            tmp = Path(td)
            repo, commit = candidate_repo(tmp)
            ssh, profile = cluster(tmp)
            hosts = hosts_for(tmp, profile)
            record = run_pipeline(local_repo=repo, hosts=hosts, manifest=MANIFEST, build_host="hpc",
                                  board_host="card", adapter="mock", poll_s=0.2, ssh_binary=str(ssh), run_id="lsf-run-1")
            self.assertEqual(record["status"], "PASS", json.dumps(record, indent=1)[-3000:])
            self.assertEqual([s["name"] for s in record["steps"]],
                             ["sync:hpc", "sync:card", "doctor:hpc", "doctor:card", "build:submit", "build:lsf",
                              "build", "export", "transfer", "board:measure", "board"])
            job = record["build"]["lsf"]
            self.assertEqual((job["job_id"], job["stat"], job["exec_host"]), (101, "DONE", "node07"))
            # LSF was asked for the queue, the slots on one node, and the run's name.
            asked = dict(eval((tmp / "lsf-jobs" / "101.opts").read_text()))
            self.assertEqual((asked["-q"], asked["-n"], asked["-R"], asked["-J"]),
                             ("cpu", "8", "span[hosts=1]", "lsf-run-1"))
            # The job ran the checkout's own submit, with the host's Vitis root.
            run = load_json(tmp / "hpc-state" / "runs" / "lsf-run-1" / "run.json")
            self.assertEqual(run["environment"]["tool_roots"]["vitis_source"], "host")
            self.assertEqual(record["board"]["xclbin_sha256"], record["build"]["xclbin_sha256"])
            self.assertEqual(record["hosts"]["build"]["lsf"], {"queue": "cpu", "slots": 8})
            # Asked again under the same name, the job that ran is found: nothing is queued twice.
            again = lsf.submit(hosts["hpc"], commit, {"manifest": MANIFEST, "adapter": "mock"}, "lsf-run-1",
                               ssh_binary=str(ssh))
            self.assertTrue(again["attached"])
            self.assertEqual(len(list((tmp / "lsf-jobs").glob("*.name"))), 1)

    def test_a_job_that_dies_before_its_run_is_a_failed_build_with_its_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tc-lsf-") as td:
            tmp = Path(td)
            repo, _commit = candidate_repo(tmp)
            ssh, profile = cluster(tmp)
            value = {"schema_version": 1, "hosts": {
                "hpc": {"ssh": "login", "roles": ["build"], "python": PY, "workspace": str(tmp / "hpc"),
                        "state_dir": str(tmp / "hpc-state"), "lsf": {"profile": str(profile), "queue": "cpu"}}}}
            hosts = parse_hosts(value)
            # The job's interpreter is not there: the script fails before any run record exists.
            broken = parse_hosts({**value, "hosts": {"hpc": {**value["hosts"]["hpc"], "python": "/no/such/python"}}})
            good = run_pipeline(local_repo=repo, hosts=hosts, manifest=MANIFEST, build_host="hpc", adapter="mock",
                                poll_s=0.2, ssh_binary=str(ssh), skip_board=True, run_id="lsf-ok")
            self.assertEqual(good["status"], "PASS", json.dumps(good, indent=1)[-2000:])
            lost = lsf.submit(broken["hpc"], good["commit"], {"manifest": MANIFEST, "adapter": "mock"}, "lsf-lost",
                              ssh_binary=str(ssh))
            self.assertFalse(lost["attached"])
            ended = lsf.wait(hosts["hpc"], "lsf-lost", poll_s=0.2, ssh_binary=str(ssh))
            self.assertEqual(ended["stat"], "EXIT")
            self.assertTrue(any("/no/such/python" in line for line in lsf.log_tail(hosts["hpc"], "lsf-lost",
                                                                               ssh_binary=str(ssh))))

    def test_a_job_lsf_no_longer_lists_has_ended(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tc-lsf-") as td:
            tmp = Path(td)
            ssh, profile = cluster(tmp)
            ended = lsf.wait(hosts_for(tmp, profile)["hpc"], "never-queued", poll_s=0.1, ssh_binary=str(ssh))
            self.assertEqual(ended["stat"], "GONE")

    def test_kill_ends_the_job_by_its_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tc-lsf-") as td:
            tmp = Path(td)
            ssh, profile = cluster(tmp)
            host = hosts_for(tmp, profile)["hpc"]
            env = dict(os.environ, FAKE_LSF_DIR=str(tmp / "lsf-jobs"))
            import subprocess
            subprocess.run([str(tmp / "lsf-bin" / "bsub"), "-q", "cpu", "-J", "slow-1", "-o", str(tmp / "slow.log"),
                            "sleep", "60"], env=env, check=True, capture_output=True)
            for _ in range(50):
                if (tmp / "lsf-jobs" / "101.pid").exists():
                    break
                time.sleep(0.1)
            self.assertEqual(lsf.job(host, "slow-1", ssh_binary=str(ssh))["stat"], "RUN")
            self.assertEqual(lsf.kill(host, "slow-1", ssh_binary=str(ssh))["exit"], 0)
            self.assertEqual(lsf.wait(host, "slow-1", poll_s=0.2, ssh_binary=str(ssh))["stat"], "EXIT")


if __name__ == "__main__":
    unittest.main()
