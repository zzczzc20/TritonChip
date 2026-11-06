#!/usr/bin/env python3
"""Check that running the vision tower once per image equals one joint call over all images.

The official tower takes several images as one packed sequence with per-image attention segments
(`cu_seqlens`); every other operation is per patch. The chat driver therefore calls it once per image, so each
compiled call has a single static segment. This CPU check compares the two forms bit for bit on the official
checkpoint and processor, for the images given (default: the fixed test image and its rotation).

  python tools/models/vision_split_check.py --model vl [--image A --image B ...] --out OUT.json
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import qwen_fp8_lib as L  # noqa: E402
from chat import image_rows, make_inputs, open_images, spec_of, user_message  # noqa: E402


def run(stage, visual, pixel_values, grid):
    pre = L.vision_precompute(visual, grid)
    with torch.no_grad():
        return stage(pixel_values, grid, pre["interp_indices"], pre["interp_weights"], pre["position_ids"],
                     pre["cu_seqlens"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("vl",), default="vl")
    ap.add_argument("--image", action="append")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L.set_threads(a.threads)
    spec = spec_of(a.model)
    model, _ = L.load_model(spec, "fp8", layers=1)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(L.snapshot(spec))
    if a.image:
        images = open_images([Path(p) for p in a.image])
    else:
        base = L.make_image()
        images = [base, base.rotate(90), base.resize((320, 256))]
    inputs, _ = make_inputs(spec, processor, [user_message("Describe.", images)])
    visual = L.vision_tower(model)
    stage = L.VisionStage(visual).eval()
    grid, pixels = inputs["image_grid_thw"], inputs["pixel_values"]
    joint = run(stage, visual, pixels, grid)
    rows, start, parts = image_rows(grid), 0, []
    for i, n in enumerate(rows):
        parts.append(run(stage, visual, pixels[start:start + n], grid[i:i + 1]))
        start += n
    result = dict(model=spec.name, grid_thw=grid.tolist(), outputs=[])
    passed = True
    for j, whole in enumerate(joint):
        split = torch.cat([p[j] for p in parts], dim=0)
        equal = bool(torch.equal(whole, split))
        passed &= equal
        result["outputs"].append(dict(index=j, shape=list(whole.shape), dtype=str(whole.dtype), bitwise_equal=equal,
                                      max_abs=float((whole.float() - split.float()).abs().max())))
    result["passed"] = passed
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
