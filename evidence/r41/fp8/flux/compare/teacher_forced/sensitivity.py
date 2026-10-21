"""How visible is a one-layer fault in a teacher-forced denoise step?  (evidence for the acceptance floor)

    python sensitivity.py REF_DIR OUT.json        (GB10 flux-venv, HF_HOME set; ~2 min on 8 threads)

Step 0 of the FP8 transformer on the reference's own inputs, with a fault in ONE FP8Linear's output
(a forward hook), compared with the reference's step-0 noise prediction.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "tools" / "models"))
import flux2_klein_run as r  # noqa: E402
import fp8_checkpoint as fc  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ref = load_file(str(Path(sys.argv[1]) / "tensors.safetensors"))
torch.set_num_threads(8)
main = r.snapshot(r.MAIN_REPO, r.MAIN_REV)
model, _ = r.load_transformer(main, "fp8", fp8_path=r.snapshot(r.FP8_REPO, r.FP8_REV) / r.FP8_FILE)
stage = r.DenoiseStage(model)
sched = r.load_scheduler(main)
pipe = r.helper_pipeline(main, sched, r.load_vae(main))
timesteps, _ = r.timesteps_for(pipe, sched, r.DEFAULT_STEPS, ref["latents_init"].shape[1], torch.device("cpu"))
dt = (sched.sigmas[1] - sched.sigmas[0]).float()
args = (ref["latents_init"], timesteps[0], dt, ref["prompt_embeds"], ref["text_ids"], ref["latent_ids"])
layer = "single_transformer_blocks.10.attn.to_out"
mod = dict(model.named_modules())[layer]
g = torch.Generator().manual_seed(0)
faults = {"none": None,
          "1% multiplicative noise": lambda y: y * (1 + 0.01 * torch.randn(y.shape, generator=g)).to(y.dtype),
          "10% multiplicative noise": lambda y: y * (1 + 0.10 * torch.randn(y.shape, generator=g)).to(y.dtype),
          "output zeroed": lambda y: torch.zeros_like(y)}
out = dict(layer=layer, reference=sys.argv[1], results={})
y_ref = ref["step0_noise_pred"].double()
for name, fault in faults.items():
    h = mod.register_forward_hook(lambda m, a, y, f=fault: f(y)) if fault else None
    with torch.no_grad():
        _, noise = stage(*args)
    if h:
        h.remove()
    d = noise.double() - y_ref
    out["results"][name] = dict(rel_l2=(d.norm() / y_ref.norm()).item(),
                                cosine=torch.nn.functional.cosine_similarity(noise.double().flatten(), y_ref.flatten(), dim=0).item(),
                                bitwise_equal=bool(torch.equal(noise, ref["step0_noise_pred"])))
    print(name, out["results"][name], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
