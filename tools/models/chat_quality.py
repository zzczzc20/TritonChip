#!/usr/bin/env python3
"""Registered chat-quality check (spec/r41/chat/quality-v1.json): CPU reference answers, then device scoring.

  # 1. before any device run: the CPU W8A8 reference model's own answers
  python tools/models/chat_quality.py --model vl --device cpu --spec spec/r41/chat/quality-v1.json --out REF_DIR
  # 2. the device, scored against that archived reference
  python tools/models/chat_quality.py --model vl --device tnca --spec ... --reference REF_DIR/quality.json --out OUT

Each prompt is a fresh conversation in one persistent session (multi-turn prompts keep their history). Greedy
decoding, stop at EOS or the spec's new_tokens. The scoring rule is the one written in the spec; nothing here is
tuned to a model or an answer.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat  # noqa: E402


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def score_turn(device, reference, keywords):
    dn, rn = normalize(device["text"]), normalize(reference["text"])
    required = [k for k in keywords if k.lower() in reference["text"].lower()]
    missing_in_reference = [k for k in keywords if k not in required]
    keyword_ok = all(k.lower() in device["text"].lower() for k in required)
    contained = bool(rn) and (rn in dn or (dn and dn in rn))
    a, b = device["tokens"], reference["tokens"]
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return dict(passed=bool(keyword_ok and contained), required_keywords=required,
                keywords_missing_in_reference=missing_in_reference, keywords_ok=keyword_ok, containment=contained,
                tokens_equal=a == b, common_prefix_tokens=prefix, device_tokens=len(a), reference_tokens=len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("vl", "q35"), required=True)
    ap.add_argument("--device", choices=("tnca", "cpu"), default="tnca")
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--reference", type=Path, help="archived CPU reference quality.json (required for tnca)")
    ap.add_argument("--root", type=Path, default=HERE.parents[1], help="tree that image paths are relative to")
    ap.add_argument("--layers", type=int, help="reduced-depth proof only")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tnca-tree")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    if a.device == "tnca" and a.reference is None:
        raise SystemExit("--reference (the archived CPU answers) is required on the device")
    reference = json.loads(a.reference.read_text()) if a.reference else None
    args = chat.parser().parse_args(["--model", a.model, "--device", a.device, "--new-tokens", str(spec["new_tokens"]),
                                     "--bucket", str(spec["bucket"]), "--threads", str(a.threads)]
                                    + (["--layers", str(a.layers)] if a.layers else [])
                                    + (["--tnca-tree", a.tnca_tree] if a.tnca_tree else []))
    app = chat.ChatApplication(args)
    a.out.mkdir(parents=True, exist_ok=True)
    result = dict(schema="tnca-chat-quality-result/1", spec=spec["id"], spec_path=str(a.spec), model=app.spec.name,
                  device=a.device, model_scope="reduced-depth-proof" if a.layers else "full-checkpoint",
                  reference=str(a.reference) if a.reference else None, prompts=[], status="running")
    try:
        for prompt in spec["prompts"]:
            if a.model not in prompt.get("models", ["vl", "q35"]):
                continue
            app.reset_conversation()
            images = chat.open_images([a.root / p for p in prompt.get("images", [])])
            turns = []
            for i, text in enumerate(prompt["turns"]):
                report = app.generate(text, images if i == 0 else (), keep_history=True)
                turns.append(dict(user=text, text=report["generated_text"], tokens=report["generated_tokens"],
                                  statistics=report["statistics"], session=report["session"],
                                  steps=[{k: s.get(k) for k in ("kind", "seconds", "new_graphs", "code_store",
                                                                 "reused_cached_tokens")} for s in report["steps"]]))
                print(json.dumps(dict(prompt=prompt["id"], turn=i, user=text, assistant=report["generated_text"])),
                      flush=True)
            row = dict(id=prompt["id"], kind=prompt["kind"], keywords=prompt["keywords"], turns=turns)
            if reference is not None:
                ref = {p["id"]: p for p in reference["prompts"]}[prompt["id"]]
                scores = [score_turn(t, r, prompt["keywords"] if j == len(turns) - 1 else [])
                          for j, (t, r) in enumerate(zip(turns, ref["turns"], strict=True))]
                row.update(scores=scores, passed=all(s["passed"] for s in scores))
            else:
                row["reference_keywords_present"] = [k for k in prompt["keywords"]
                                                     if k.lower() in turns[-1]["text"].lower()]
            result["prompts"].append(row)
            (a.out / "quality.json").write_text(json.dumps(result, indent=1) + "\n")
        if reference is not None:
            result["passed"] = all(p["passed"] for p in result["prompts"])
            result["status"] = "pass" if result["passed"] else "failed-chat-quality"
        else:
            result["status"] = "reference-recorded"
    except BaseException:
        result["status"] = "failed-preserved"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["versions"] = chat.L.versions()
        result["load"] = app.load_report
        (a.out / "quality.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "model", "device")} | {"passed": result.get("passed")}))
    return 0 if result["status"] in ("pass", "reference-recorded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
