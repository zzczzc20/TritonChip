"""lovedheart/Qwen3.5-4B-FP8 end to end through the standard torch path, stage by stage.

Model code: transformers' Qwen3_5ForConditionalGeneration (official; hybrid decoder: 24 Gated-DeltaNet linear
attention layers + 8 full attention layers, plus a 24-block vision tower), processor: AutoProcessor (official).
Checkpoint: two safetensors shards, quantization_config quant_method "fp8" (dynamic activations, weight blocks
128x128, weight_scale_inv stored as BF16), modules_to_not_convert = embeddings, norms, the vision tower and the
linear-attention A_log/conv1d/dt_bias/in_proj_a/in_proj_b/norm.  The checkpoint also carries a multi-token-
prediction head (mtp.*) that this class does not load.  FP8 handling is the format-keyed path of
qwen_fp8_lib.py; nothing here is specific to the model except the repo, the revision, the class name the
checkpoint declares, the prompt, and the chat template's documented `enable_thinking=False` switch (the
answer instead of a thinking trace within the token budget).

Subcommands and usage: as qwen3vl_run.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I does not add the script dir
from qwen_fp8_lib import ModelSpec, main  # noqa: E402

SPEC = ModelSpec(
    name="qwen3.5-4b-fp8",
    repo="lovedheart/Qwen3.5-4B-FP8",
    revision="752f65fd155dd34323cfc22d6143df1d48ce3c28",
    cls="Qwen3_5ForConditionalGeneration",
    prompt="Describe this image in one sentence.",
    image=True,
    chat_kwargs=dict(enable_thinking=False),
    new_tokens=32,
    bucket=256,
)

if __name__ == "__main__":
    main(SPEC, __doc__)
