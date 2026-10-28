#!/usr/bin/env python3
"""Summarise a board chat queue archive (domain: board) from its raw files.

  python3 evidence/r41/chat/chat_board_closeout.py ARCHIVE.tgz [--out SUMMARY.json]

Reads every step's result.json (experiment record with before/after image and clock probes) and, where the
step produced one, the chat report.json/transcript.json. Nothing is typed in: every value printed is read from
the archive. A failed step is reported as failed with its stderr tail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def chat_summary(report):
    steps = report.get("steps", [])
    decode = [s for s in steps if s["kind"] == "decode"]
    steady = [s for s in decode if s.get("new_graphs") == 0]
    out = dict(status=report.get("status"), model=report.get("model"), model_scope=report.get("model_scope"),
               decoder_layers=report.get("decoder_layers"), prompt_tokens=report.get("prompt_tokens"),
               generated_tokens=report.get("generated_tokens"), generated_text=report.get("generated_text"),
               statistics=report.get("statistics"),
               steady_decode_seconds=[s["seconds"] for s in steady],
               steps=[dict(step=s["step"], kind=s["kind"], seconds=s["seconds"], new_graphs=s["new_graphs"],
                           device_cycles=(s.get("transfers_inside_step") or {}).get("device_cycles"),
                           launches=(s.get("transfers_inside_step") or {}).get("launches"),
                           host_read_calls=(s.get("transfers_inside_step") or {}).get("host_read_calls"),
                           host_write_bytes=(s.get("transfers_inside_step") or {}).get("host_write_bytes"),
                           code_store=s.get("code_store")) for s in steps],
               teacher_forced=report.get("teacher_forced_cpu_w8a8"),
               persistent_session_check={k: v for k, v in (report.get("persistent_session_check") or {}).items()
                                         if k not in ("other", "repeat")} or None,
               device_memory=report.get("device_memory"), decode_weight_bytes=report.get("decode_weight_bytes"))
    if steady:
        out["steady_decode_tokens_per_second"] = len(steady) / sum(s["seconds"] for s in steady)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    summary = dict(domain="board", archive=str(a.archive),
                   archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(), steps=[])
    with tarfile.open(a.archive) as tar:
        names = tar.getnames()
        read = lambda n: tar.extractfile(n).read().decode()  # noqa: E731
        for name in sorted(n for n in names if n.endswith("/result.json") and "/steps/" in n):
            r = json.loads(read(name))
            board = r.get("board") or {}
            before, after = board.get("probe_before") or {}, board.get("probe_after") or {}
            row = dict(step=r["id"], status=r["status"], returncode=r.get("returncode"),
                       host_wall_seconds=r.get("host_wall_seconds"), started_utc=r.get("started_utc"),
                       ended_utc=r.get("ended_utc"), image_sha256=board.get("image_sha256"),
                       image_uuid=board.get("image_uuid"), clock_hz=board.get("clock_hz"),
                       uuid_before=before.get("loaded_uuid"), uuid_after=after.get("loaded_uuid"),
                       clock_mhz_before=(before.get("clock_counter_mhz") or [None])[0],
                       clock_mhz_after=(after.get("clock_counter_mhz") or [None])[0])
            base = name.split("/steps/")[0]
            for kind in ("report.json", "transcript.json"):
                path = f"{base}/{r['id']}/{kind}"
                if path in names:
                    data = json.loads(read(path))
                    if kind == "report.json":
                        row["chat"] = chat_summary(data)
                        if row["clock_hz"]:
                            for s in row["chat"]["steps"]:
                                if s["device_cycles"] is not None:
                                    s["device_seconds_at_clock"] = s["device_cycles"] / row["clock_hz"]
                    else:
                        row["transcript"] = [dict(user=t["user"], assistant=t.get("assistant"),
                                                  status=(t.get("report") or {}).get("status", t.get("status")),
                                                  chat=chat_summary(t["report"]) if t.get("report") else None)
                                             for t in data["turns"]]
            stderr = name.replace("result.json", "stderr.log")
            if r["status"] != "completed" and stderr in names:
                row["stderr_tail"] = read(stderr)[-600:]
            summary["steps"].append(row)
    text = json.dumps(summary, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
