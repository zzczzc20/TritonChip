#!/usr/bin/env python3
"""The holdout cases of the R41 calibration (decided 2026-11-08): agent-visible shapes of the given classes.

    python3 make_holdout.py SKILL_DIR OUT_DIR

Each case is a given point's file from the test-case skill (benchmark_cases/), with its
dimensions and seed changed and its POINT and DEMAND dropped (tnca_case.py then measures the
demand itself). The holdout is never used to fit: it judges the fitted model (absolute error)
and, measured on every variant, the model's response to a change (direction, size, no false
gains). Together with the skill's eight worked examples it is the holdout set.

Why each case is here:
  gemv, rope        the two mechanisms the R41 base error showed wrong (GEMV ~2x slow, rope ~2x fast)
  linear m48 W64K   a reused weight between the base cache (32 sets: 48 KiB over three home slices)
                    and the 64-set cache (96 KiB): sensitive to SETS if the cores re-read it
  linear m48 W32K   the same with a weight that fits either cache: the control (no SETS change)
  add, silu, transpose  long streams: sensitive to LSU_INFLIGHT (the author: copy fp32 +9.8 % on the board)
  attention         prefill (the author: +9..20 % from LSU_INFLIGHT) and decode
  matmul            the matrix path: sensitive to MTX_VRF_REG
  FP8               the two FP8 linear forms the agent is given
  rmsnorm, softmax  the reduction kernels
"""
import re
import sys
from pathlib import Path

CASES = [
    # (case id, template file, kwargs, seed)
    ("gemv-k192-n320", "gemv_tiny_fp16_k64_n64.py", dict(k=192, n=320), 11),
    ("gemv-k96-n200", "gemv_tiny_fp16_k64_n64.py", dict(k=96, n=200), 12),
    ("rope-b1-h4-s48-d64", "rope_medium_fp16_b1_h8_s128_d32.py", dict(b=1, h=4, s=48, d=64), 13),
    ("rope-b2-h2-s40-d32", "rope_medium_fp16_b1_h8_s128_d32.py", dict(b=2, h=2, s=40, d=32), 14),
    ("attention_decode-b1-h4-l48-d32", "attention_decode_tiny_fp16_b1_h2_l16_d16.py", dict(b=1, h=4, l=48, d=32), 15),
    ("attention_prefill-b1-h2-s32-d32", "attention_prefill_tiny_fp16_b1_h2_s16_d16.py", dict(b=1, h=2, s=32, d=32), 16),
    ("linear-m48-k128-n256", "linear_small_fp16_m64_k256_n256.py", dict(m=48, k=128, n=256), 17),
    ("linear-m48-k128-n128", "linear_small_fp16_m64_k256_n256.py", dict(m=48, k=128, n=128), 18),
    ("matmul-m32-k128-n48", "matmul_tiny_fp16_m16_k32_n16.py", dict(m=32, k=128, n=48), 19),
    ("add-fp32-n12288", "add_tiny_fp32_n256.py", dict(n=12288), 20),
    ("silu-n8192", "silu_tiny_fp16_n256.py", dict(n=8192), 21),
    ("transpose-r96-c128", "transpose_medium_fp32_r160_c256.py", dict(r=96, c=128), 22),
    ("rmsnorm-r12-d320", "rmsnorm_tiny_fp16_r4_d64.py", dict(r=12, d=320), 23),
    ("softmax-r24-c320", "softmax_medium_fp16_r80_c512.py", dict(r=24, c=320), 24),
    ("wfp8_linear-m8-k128-n96", "wfp8_linear_tiny_bf16_m16_k64_n64_we4m3.py", dict(m=8, k=128, n=96), 25),
    ("w8a8_block_linear-m2-k256-n256", "w8a8_block_linear_small_e4m3_to_bf16_m1_k128_n384.py", dict(m=2, k=256, n=256), 26),
]


def make_case(template: str, cid: str, kwargs: dict, seed: int) -> str:
    body = template.split('"""', 2)[2]                       # drop the point's docstring
    body = re.sub(r"^POINT = .*$", "POINT = None", body, count=1, flags=re.M)
    body = re.sub(r"^DEMAND = .*$", "DEMAND = None  # measured by tnca_case.py", body, count=1, flags=re.M)
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    body, n = re.subn(r"def make\(\):\n    return _point_make\(0\)\n?", f"def make():\n    return _point_make({seed}, {args})\n", body)
    assert n == 1, cid
    head = (f'"""Calibration holdout case {cid} (seed {seed}): a given class with dimensions of its own.\n\n'
            f'Not a point of the benchmark. Made by tools/performance_harness/systemc/r41cal/make_holdout.py\n'
            f'from a given point\'s file; regenerated, not edited.\n"""')
    return head + body


def main():
    skill, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for cid, tmpl, kwargs, seed in CASES:
        text = make_case((skill / "benchmark_cases" / tmpl).read_text(), cid, kwargs, seed)
        (out / f"{cid.replace('-', '_')}.py").write_text(text)
    print(len(CASES), "cases in", out)


if __name__ == "__main__":
    main()
