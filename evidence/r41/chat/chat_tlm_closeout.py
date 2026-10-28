#!/usr/bin/env python3
"""Validate and summarize archived reduced-depth SystemC/TLM chat runs."""
import hashlib
import json
import math
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The teacher-forced gate is the fixed relative-L2 limit registered before the
# runs. The floor-based limit max(0.05, 2 x CPU floor) was introduced after
# q35-v1 failed, so it is not used for any run here (AUDIT.md R6).
FIXED_MAX_REL_L2 = 0.05
RUNS = (
    ("vl-greedy", "tlm-vl-greedy-v8.tgz", "tlm-vl-v8/report.json", True),
    ("vl-sampled", "tlm-vl-sample-v1.tgz", "tlm-vl-sample-v1/report.json", False),
)
EXTRAS = (
    ("q35-v1-failed-fixed-limit", "tlm-q35-v1.tgz", "tlm-q35-v1/report.json", "failure"),
    ("q35-v2-failed-fixed-limit", "tlm-q35-v2.tgz", "tlm-q35-v2/report.json", "rejudged"),
    ("q35-cpu-scale-floor", "q35-cpu-floor-v1.tgz", "q35-cpu-floor-v1/report.json", "floor"),
)


def fixed_limit_passed(check):
    return all(row["rel_l2"] <= FIXED_MAX_REL_L2 for row in check["steps"])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_archive(filename, report_name):
    archive = HERE / filename
    members = {}
    report = None
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile():
                data = tar.extractfile(member).read()
                members[member.name] = dict(size=len(data), sha256=sha(data))
                if member.name == report_name:
                    report = json.loads(data)
    assert report_name in members and report is not None
    index = dict(archive_sha256=sha(archive.read_bytes()), members=members)
    return report, index


def slim_step(row):
    transfers = row["transfers_inside_step"]
    return {k: row[k] for k in ("step", "kind", "input_tokens", "seconds", "new_graphs",
                                "token", "candidate_token", "greedy_token")} | {
        "transfers_inside_step": {k: transfers[k] for k in (
            "host_read_calls", "host_read_bytes", "host_write_calls", "host_write_bytes",
            "launches", "staging_upload_bytes", "staging_download_bytes", "device_cycles")}}


def main():
    summaries = []
    indices = {}
    for label, filename, report_name, teacher_required in RUNS:
        report, index = load_archive(filename, report_name)
        indices[filename] = index
        assert report["status"] == "pass" and report["device"] == "tnca"
        assert report["model_scope"] == "reduced-depth-proof"
        assert report["standard_torch_fullgraph"] and not report["cpu_numerical_fallback"]
        assert report["fp8_route_in_use"]["kind"] == "official"
        assert len(report["steps"]) == len(report["generated_tokens"]) >= 2
        decode = [row for row in report["steps"] if row["kind"] == "decode"]
        assert decode and decode[0]["new_graphs"] == 1
        assert all(row["new_graphs"] == 0 for row in decode[1:])
        assert all(row["transfers_inside_step"]["host_read_calls"] == 0 for row in decode[1:])
        statistics = report["statistics"]
        decode_seconds = sum(row["seconds"] for row in decode)
        assert math.isclose(statistics["decode_seconds"], decode_seconds)
        assert math.isclose(statistics["mean_decode_seconds_per_token"], decode_seconds / len(decode))
        assert math.isclose(statistics["decode_tokens_per_second"], len(decode) / decode_seconds)
        bound = report["theoretical_core_port_bound"]
        assert math.isclose(bound["weight_stream_ceiling_tokens_per_second"],
                            bound["bytes_per_second"] / report["decode_weight_bytes"])
        check = report.get("teacher_forced_cpu_w8a8")
        if teacher_required:
            assert check and check["passed"] and check["checked_tokens"] > 0
            assert len(check["steps"]) == check["checked_tokens"]
            assert check["max_allowed_rel_l2"] == FIXED_MAX_REL_L2 and fixed_limit_passed(check)
        summaries.append(dict(label=label, archive=filename, archive_sha256=index["archive_sha256"],
            report_sha256=index["members"][report_name]["sha256"], model=report["model"], repo=report["repo"],
            revision=report["revision"], model_scope=report["model_scope"],
            decoder_layers=report["decoder_layers"], prompt_tokens=report["prompt_tokens"],
            bucket=report["bucket"], sampling=report["sampling"],
            generated_tokens=report["generated_tokens"], generated_text=report["generated_text"],
            statistics=statistics, steps=[slim_step(row) for row in report["steps"]],
            teacher_forced_cpu_w8a8=check, decode_weight_bytes=report["decode_weight_bytes"],
            theoretical_core_port_bound=bound))
    failures = []
    floors = []
    for label, filename, report_name, kind in EXTRAS:
        report, index = load_archive(filename, report_name)
        indices[filename] = index
        if kind == "failure":
            assert report["status"] == "failed-teacher-forced-tolerance"
            assert not report["teacher_forced_cpu_w8a8"]["passed"]
            assert not fixed_limit_passed(report["teacher_forced_cpu_w8a8"])
            failures.append(dict(label=label, archive=filename, status=report["status"],
                statistics=report["statistics"], teacher_forced_cpu_w8a8=report["teacher_forced_cpu_w8a8"]))
        elif kind == "rejudged":
            # The archived report says pass only under the post-hoc floor rule.
            check = report["teacher_forced_cpu_w8a8"]
            assert not fixed_limit_passed(check)
            failures.append(dict(label=label, archive=filename,
                status="failed-teacher-forced-tolerance (fixed limit 0.05)",
                archived_report_status=report["status"],
                note="The archived report's pass applied max(0.05, 2 x CPU floor), introduced after "
                     "q35-v1 failed; under the registered fixed limit the run fails (AUDIT.md R6).",
                fixed_max_rel_l2=FIXED_MAX_REL_L2, statistics=report["statistics"],
                teacher_forced_cpu_w8a8=check))
        else:
            assert report["status"] == "measured" and report["model_scope"] == "reduced-depth-proof"
            floors.append(dict(label=label, archive=filename, model=report["model"],
                decoder_layers=report["decoder_layers"], comparison=report["comparison"],
                interpretation=report["interpretation"]))
    result = dict(schema="r41-chat-tlm-closeout/2",
                  status="reduced-depth-tlm: VL greedy and sampled pass; Qwen3.5 greedy failed",
                  fixed_max_rel_l2=FIXED_MAX_REL_L2,
                  execution_domain="SystemC/TLM; not physical-U280 performance or full-model evidence",
                  runs=summaries, preserved_failures=failures, cpu_floor_diagnostics=floors)
    (HERE / "tlm-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "tlm-index.json").write_text(json.dumps(indices, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], runs=[dict(label=row["label"],
        statistics=row["statistics"]) for row in summaries])))


if __name__ == "__main__":
    main()
