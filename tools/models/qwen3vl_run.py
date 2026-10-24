"""Qwen/Qwen3-VL-2B-Instruct-FP8 end to end through the standard torch path, stage by stage.

Model code: transformers' Qwen3VLForConditionalGeneration (official), processor: AutoProcessor (official).
Checkpoint: one safetensors file, quantization_config quant_method "fp8" (e4m3, dynamic activations, weight
blocks 128x128, 245 ignored layers: lm_head and the whole vision tower stay BF16), 196 FP8Linear in the text
decoder.  FP8 handling is the format-keyed path of qwen_fp8_lib.py; nothing here is specific to the model
except the repo, the revision, the class name the checkpoint declares, and the prompt.

    run              staged vision -> prefill -> greedy decode (fixed-bucket StaticCache); --*-device cpu|tnca|cuda
    official         unmodified model.generate() (cpu, or cuda with the official Hub kernel)
    compare A B      numeric comparison of two runs (tokens, logits, per-layer hidden states, vision features)
    layer-check      every FP8Linear alone on its reference input vs the exact W8A8 value
    plan             per-stage fullgraph capture: op/dtype list, FLOPs, weights/KV/activations vs 4/6/8 GiB
    probe-quantizer  what unmodified transformers does with the checkpoint (CUDA host / no-CUDA host)

Typical (GB10, HF_HOME=~/flux-hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8):
    qwen3vl_run.py run --check-official --verify-weights --out ref/fp8          # CPU reference, W8A8
    qwen3vl_run.py run --weights bf16 --out ref/bf16                            # dequantized BF16 (tolerance floor)
    qwen3vl_run.py run --teacher-force ref/fp8 --vision-device tnca --prefill-device tnca --decode-device tnca \\
        --fp8-kernel official --out tnca/                                         # on TNCA (route a)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from qwen_fp8_lib import ModelSpec, main  # noqa: E402

SPEC = ModelSpec(
    name="qwen3vl-2b-fp8",
    repo="Qwen/Qwen3-VL-2B-Instruct-FP8",
    revision="46485250d8854c0a9be4f1adbc67ca47e5bb6fa5",
    cls="Qwen3VLForConditionalGeneration",
    prompt="Describe this image in one sentence.",
    image=True,
    new_tokens=32,
    bucket=256,
)

if __name__ == "__main__":
    main(SPEC, __doc__)
