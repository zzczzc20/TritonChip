#!/usr/bin/env python3
"""Measure a reduced-depth chat model's CPU W8A8 scale-rounding floor."""
import argparse
import json
from pathlib import Path

import chat
import qwen_fp8_lib as L


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vl", "q35"), required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--prompt", default="Say hi.")
    parser.add_argument("--new-tokens", type=int, default=2)
    parser.add_argument("--bucket", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = chat.parser().parse_args(["--model", args.model, "--device", "cpu",
        "--layers", str(args.layers), "--new-tokens", str(args.new_tokens),
        "--bucket", str(args.bucket), "--prompt", args.prompt, "--threads", str(args.threads)])
    app = chat.ChatApplication(base)
    exact = app.generate(args.prompt)
    inputs, text = chat.make_inputs(app.spec, app.processor, args.prompt, None)
    L.install_fp8_route("reference", dot="exact", scale_div="recip")
    reciprocal = chat.run_generation(app.model, app.processor, app.spec, inputs, text, "cpu",
        bucket=args.bucket, new_tokens=args.new_tokens, forced_tokens=exact["generated_tokens"])
    rows = chat.compare_logits(exact["logits"], reciprocal["logits"])
    args.out.mkdir(parents=True, exist_ok=False)
    from safetensors.torch import save_file
    for name, report in (("exact", exact), ("reciprocal", reciprocal)):
        save_file({"step_%04d" % i: x.contiguous() for i, x in enumerate(report["logits"])},
                  str(args.out / f"{name}-logits.safetensors"))
    result = dict(schema="tnca-qwen-chat-numerical-floor/1", status="measured",
        model=app.spec.name, repo=app.spec.repo, revision=app.spec.revision,
        model_scope="reduced-depth-proof", decoder_layers=exact["decoder_layers"],
        prompt_tokens=exact["prompt_tokens"], bucket=args.bucket,
        reference_tokens=exact["generated_tokens"], reciprocal_candidate_tokens=reciprocal["candidate_tokens"],
        exact_statistics=exact["statistics"], reciprocal_statistics=reciprocal["statistics"],
        comparison=rows, load=exact["load"], versions=exact["versions"],
        interpretation="CPU W8A8 IEEE-scale versus reciprocal-scale rounding floor; not a TNCA error allowance by itself")
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
