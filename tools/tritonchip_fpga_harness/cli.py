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
    from tritonchip_fpga_harness.core import HarnessError, doctor, export_evidence, measure_existing, plan, reconcile, repo_root, status, submit, validate_existing
    from tritonchip_fpga_harness.worker import DispatchError, dispatch_manifest, worker_main
    from tritonchip_fpga_harness.schema import SchemaError, load_json
else:
    from .budget import BudgetError, TokenLedger
    from .core import HarnessError, doctor, export_evidence, measure_existing, plan, reconcile, repo_root, status, submit, validate_existing
    from .worker import DispatchError, dispatch_manifest, worker_main
    from .schema import SchemaError, load_json


HERE = Path(__file__).resolve().parent


def _root(value: str | None) -> Path:
    return repo_root(Path(value or "."))


def _state(value: str | None, root: Path) -> Path:
    # Keep generated logs/binaries outside the source checkout.  A stale
    # artifact must never be mistaken for an output of the current candidate.
    return (Path(value) if value else root.parent / ".tritonchip-fpga-harness").resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tritonchip-fpga-harness")
    parser.add_argument("--repo", help="repository root or path inside it")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--target", help="target profile; defaults to the manifest's, else u50-legacy")
    d.add_argument("--adapter", default="u50", choices=("u50", "u280", "mock"))
    d.add_argument("--manifest", type=Path, help="also check that this manifest's board stage can start on this host")
    d.add_argument("--json", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("manifest", type=Path)
    p.add_argument("--adapter", default="u50", choices=("u50", "u280", "mock"))
    p.add_argument("--json", action="store_true")
    s = sub.add_parser("submit")
    s.add_argument("manifest", type=Path)
    s.add_argument("--adapter", default="u50", choices=("u50", "u280", "mock"))
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
    e.add_argument("--json", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--target", default="u50-legacy")
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
            value = doctor(HERE, root, args.target or (board or {}).get("target") or "u50-legacy", args.adapter, board)
        elif args.command == "plan":
            value = plan(HERE, root, load_json(args.manifest), args.adapter)
        elif args.command == "submit":
            value = submit(HERE, root, load_json(args.manifest), _state(args.state_dir, root), args.adapter, args.allow_dirty, args.timeout, args.run_id, args.skip_board)
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
        elif args.command == "reconcile":
            value = reconcile(_state(args.state_dir, root), args.run_id, terminate=args.terminate)
        elif args.command == "validate":
            value = validate_existing(HERE, root, args.target, args.build_dir, historical=args.historical)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            if args.command == "export":
                destination = export_evidence(_state(args.state_dir, root), args.run_id, Path(args.output).resolve())
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
    except (HarnessError, SchemaError, BudgetError, DispatchError, OSError) as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
