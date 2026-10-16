#!/usr/bin/env python3
"""Command-line entry point for the TritonChip FPGA harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tritonchip_fpga_harness.budget import BudgetError, TokenLedger
    from tritonchip_fpga_harness.core import HarnessError, doctor, export_evidence, import_run, load_profile, measure_existing, plan, reconcile, repo_root, status, submit, validate_existing
    from tritonchip_fpga_harness.hosts import load_hosts
    from tritonchip_fpga_harness.pipeline import PipelineError, resolve_commit, run_pipeline, sync
    from tritonchip_fpga_harness.worker import DispatchError, dispatch_manifest, worker_main
    from tritonchip_fpga_harness.schema import SchemaError, load_json
else:
    from .budget import BudgetError, TokenLedger
    from .core import HarnessError, doctor, export_evidence, import_run, load_profile, measure_existing, plan, reconcile, repo_root, status, submit, validate_existing
    from .hosts import load_hosts
    from .pipeline import PipelineError, resolve_commit, run_pipeline, sync
    from .worker import DispatchError, dispatch_manifest, worker_main
    from .schema import SchemaError, load_json


HERE = Path(__file__).resolve().parent


def _root(value: str | None) -> Path:
    return repo_root(Path(value or "."))


def _state(value: str | None, root: Path) -> Path:
    # Keep generated logs/binaries outside the source checkout.  A stale
    # artifact must never be mistaken for an output of the current candidate.
    return (Path(value) if value else root.parent / ".tritonchip-fpga-harness").resolve()


def _adapter(asked: str | None, target: str) -> str:
    """The adapter a target is built with, unless another (`mock`) is asked for."""
    return asked or load_profile(HERE, target).expected_adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tritonchip-fpga-harness")
    parser.add_argument("--repo", help="repository root or path inside it")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--target", help="the target to check; taken from --manifest if that is given")
    d.add_argument("--adapter", choices=("u50", "u280", "mock"), help="default: the target's own")
    d.add_argument("--manifest", type=Path, help="also check that this manifest's board stage can start on this host")
    d.add_argument("--json", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("manifest", type=Path)
    p.add_argument("--adapter", choices=("u50", "u280", "mock"), help="default: the target's own")
    p.add_argument("--json", action="store_true")
    s = sub.add_parser("submit")
    s.add_argument("manifest", type=Path)
    s.add_argument("--adapter", choices=("u50", "u280", "mock"), help="default: the target's own")
    s.add_argument("--state-dir")
    s.add_argument("--allow-dirty", action="store_true")
    s.add_argument("--timeout", type=int)
    s.add_argument("--run-id")
    s.add_argument("--skip-board", action="store_true", help="build/link only; retain board config for a later measure RUN_ID")
    s.add_argument("--json", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("run_id")
    st.add_argument("--state-dir")
    st.add_argument("--json", action="store_true")
    m = sub.add_parser("measure", help="program the recorded XCLBIN and run board conformance/ArchBench")
    m.add_argument("run_id")
    m.add_argument("--state-dir")
    m.add_argument("--timeout", type=int)
    m.add_argument("--manifest", type=Path, help="board configuration for a run submitted without one; must name the same candidate, target and mode")
    m.add_argument("--allow-dirty", action="store_true", help="measure from a checkout with uncommitted changes; recorded in the result")
    m.add_argument("--allow-source-mismatch", action="store_true", help="measure from a checkout that differs from the implemented commit outside the harness; recorded in the result")
    m.add_argument("--json", action="store_true")
    rd = sub.add_parser("dispatch", help="submit a manifest to a prepared U50/U280 SSH worker")
    rd.add_argument("host")
    rd.add_argument("manifest", type=Path)
    rd.add_argument("--remote-repo", required=True)
    rd.add_argument("--board", choices=("u50", "u280"))
    rd.add_argument("--remote-python", default="python3")
    rd.add_argument("--remote-cli", default="tools/tritonchip_fpga_harness/cli.py")
    rd.add_argument("--state-dir", dest="remote_state_dir")
    rd.add_argument("--port", type=int)
    rd.add_argument("--identity-file")
    rd.add_argument("--connect-timeout", type=int)
    rd.add_argument("--ssh-timeout", type=int)
    rd.add_argument("--adapter", choices=("u50", "u280"))
    rd.add_argument("--timeout", type=int)
    rd.add_argument("--run-id")
    rd.add_argument("--allow-dirty", action="store_true")
    rd.add_argument("--skip-board", action="store_true", help="build/link only; run measure RUN_ID later")
    rd.add_argument("--ssh-binary", default="ssh")
    rd.add_argument("--output", type=Path)
    rd.add_argument("--json", action="store_true")
    w = sub.add_parser("worker", help="run one JSON worker request from stdin")
    w.add_argument("--stdin", action="store_true", required=True)
    rc = sub.add_parser("reconcile")
    rc.add_argument("run_id")
    rc.add_argument("--state-dir")
    rc.add_argument("--terminate", action="store_true")
    rc.add_argument("--json", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("run_id")
    e.add_argument("--state-dir")
    e.add_argument("--output", required=True)
    e.add_argument("--light", action="store_true", help="leave out v++'s working tree: the packet a card host measures from")
    e.add_argument("--json", action="store_true")
    im = sub.add_parser("import", help="install a run built on another host, from its light packet")
    im.add_argument("--state-dir")
    im.add_argument("--from-tar", required=True, help="the packet as a tar file, or '-' for a tar stream on stdin")
    im.add_argument("--json", action="store_true")
    pl = sub.add_parser("pipeline", help="build on one server and measure on another, as a hosts file names them")
    pl.add_argument("manifest", type=Path)
    pl.add_argument("--hosts", required=True, type=Path, help="the operator's hosts file (hosts.py)")
    pl.add_argument("--build-host", required=True)
    pl.add_argument("--board-host", help="required when the manifest has a board block, unless --skip-board")
    pl.add_argument("--adapter", choices=("u50", "u280", "mock"), help="default: the target's own; mock plays the build")
    pl.add_argument("--timeout", type=int, help="per remote step (the link, the board stage), in seconds")
    pl.add_argument("--poll", type=float, default=60.0, help="seconds between status polls")
    pl.add_argument("--skip-board", action="store_true", help="build only")
    pl.add_argument("--build-run", help="a build already started or finished on the build host; skip the submit")
    pl.add_argument("--no-sync", action="store_true", help="the checkouts are already on the servers")
    pl.add_argument("--ssh-binary", default="ssh")
    pl.add_argument("--output", type=Path, help="also write the record here")
    pl.add_argument("--json", action="store_true")
    sy = sub.add_parser("sync", help="put a clean checkout of a commit on a server of the hosts file")
    sy.add_argument("host")
    sy.add_argument("--hosts", required=True, type=Path)
    sy.add_argument("--commit", default="HEAD")
    sy.add_argument("--ssh-binary", default="ssh")
    sy.add_argument("--json", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--target", required=True)
    v.add_argument("--build-dir", required=True, type=Path)
    v.add_argument("--output", type=Path)
    v.add_argument("--historical", action="store_true", help="inspect an artifact without Harness lineage; result is marked HISTORICAL")
    v.add_argument("--json", action="store_true")
    b = sub.add_parser("budget")
    bsub = b.add_subparsers(dest="budget_command", required=True)
    br = bsub.add_parser("reserve")
    br.add_argument("request_id")
    br.add_argument("amount", type=int)
    br.add_argument("--ledger", required=True)
    br.add_argument("--limit", type=int, required=True)
    bs = bsub.add_parser("settle")
    bs.add_argument("request_id")
    bs.add_argument("--actual", type=int)
    bs.add_argument("--ledger", required=True)
    bs.add_argument("--limit", type=int, required=True)
    bv = bsub.add_parser("status")
    bv.add_argument("--ledger", required=True)
    bv.add_argument("--limit", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        root = _root(args.repo)
        if args.command == "doctor":
            board = load_json(args.manifest) if args.manifest else None
            target = args.target or (board or {}).get("target")
            if not target:
                raise HarnessError("doctor needs --target or --manifest")
            value = doctor(HERE, root, target, _adapter(args.adapter, target), board)
        elif args.command == "plan":
            manifest = load_json(args.manifest)
            value = plan(HERE, root, manifest, _adapter(args.adapter, str(manifest.get("target", ""))))
        elif args.command == "submit":
            manifest = load_json(args.manifest)
            value = submit(HERE, root, manifest, _state(args.state_dir, root), _adapter(args.adapter, str(manifest.get("target", ""))), args.allow_dirty, args.timeout, args.run_id, args.skip_board)
        elif args.command == "status":
            value = status(_state(args.state_dir, root), args.run_id)
        elif args.command == "measure":
            value = measure_existing(
                HERE, root, _state(args.state_dir, root), args.run_id, args.timeout,
                board_manifest=load_json(args.manifest) if args.manifest else None,
                allow_dirty=args.allow_dirty,
                allow_source_mismatch=args.allow_source_mismatch,
            )
        elif args.command == "dispatch":
            value = dispatch_manifest(
                host=args.host,
                remote_repo=args.remote_repo,
                manifest=load_json(args.manifest),
                operation="submit",
                board=args.board,
                adapter=args.adapter,
                remote_python=args.remote_python,
                remote_cli=args.remote_cli,
                state_dir=args.remote_state_dir,
                port=args.port,
                identity_file=args.identity_file,
                connect_timeout=args.connect_timeout,
                ssh_timeout=args.ssh_timeout,
                timeout=args.timeout,
                run_id=args.run_id,
                allow_dirty=args.allow_dirty,
                skip_board=args.skip_board,
                ssh_binary=args.ssh_binary,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.command == "worker":
            return worker_main(package_dir=HERE, root=root)
        elif args.command == "import":
            state = _state(args.state_dir, root)
            if args.from_tar == "-":
                value = import_run(state, sys.stdin.buffer)
            else:
                with open(args.from_tar, "rb") as packet:
                    value = import_run(state, packet)
        elif args.command == "pipeline":
            value = run_pipeline(
                local_repo=root, hosts=load_hosts(args.hosts), manifest=load_json(args.manifest),
                build_host=args.build_host, board_host=args.board_host, adapter=args.adapter,
                timeout=args.timeout, poll_s=args.poll, skip_board=args.skip_board,
                build_run=args.build_run, do_sync=not args.no_sync, ssh_binary=args.ssh_binary,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.command == "sync":
            hosts = load_hosts(args.hosts)
            if args.host not in hosts:
                raise SchemaError(f"no host {args.host!r} in {args.hosts}")
            value = dict(sync(hosts[args.host], resolve_commit(root, args.commit), root, ssh_binary=args.ssh_binary), ok=True)
        elif args.command == "reconcile":
            value = reconcile(_state(args.state_dir, root), args.run_id, terminate=args.terminate)
        elif args.command == "validate":
            value = validate_existing(HERE, root, args.target, args.build_dir, historical=args.historical)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            if args.command == "export":
                destination = export_evidence(_state(args.state_dir, root), args.run_id, Path(args.output).resolve(), light=args.light)
                value = {"run_id": args.run_id, "evidence_dir": str(destination)}
            else:
                ledger = TokenLedger(Path(args.ledger), args.limit)
                if args.budget_command == "reserve":
                    value = ledger.reserve(args.request_id, args.amount)
                elif args.budget_command == "settle":
                    value = ledger.settle(args.request_id, args.actual)
                else:
                    value = ledger.status()
        print(json.dumps(value, indent=2, sort_keys=True) if getattr(args, "json", False) else json.dumps(value, indent=2, sort_keys=True))
        return 0 if value.get("ok", value.get("status", "PASS") in ("PASS", "PENDING", "RUNNING")) else 1
    except (HarnessError, SchemaError, BudgetError, DispatchError, PipelineError, OSError) as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
