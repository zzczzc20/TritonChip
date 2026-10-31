"""CPU-only graph-count check: chat.Session compiled with the CPU Inductor backend (dynamic=False, as on tnca),
three text prompts of different lengths with chunked prefill. Prints new graphs per prompt."""
import sys, json
sys.path.insert(0, sys.argv[1])
import torch, qwen_fp8_lib as L, chat
orig = L.place
def place(stage, device, mode, tree=None, skip=None):
    L.prepare_for_compile(stage); L.to_device_except(stage, torch.device("cpu"), skip)
    return torch.compile(stage, fullgraph=True, dynamic=False), torch.device("cpu")
L.place = place
args = chat.parser().parse_args(["--model", "vl", "--device", "cpu", "--layers", "1", "--new-tokens", "2", "--bucket", "128",
                                 "--prefill-chunk", "8", "--threads", "4"])
app = chat.ChatApplication(args)
rows = []
for prompt in ("Say hi.", "What is the capital of France? Answer with one word.", "Tom is taller than Ann. Ann is taller than Bob. Who is the shortest?"):
    r = app.generate(prompt)
    rows.append(dict(prompt_tokens=r["prompt_tokens"], new_graphs=r["session"]["new_graphs_this_prompt"],
                     calls=[(c["kind"], c["tokens"], c["new_graphs"]) for c in r["steps"][0]["calls"]]))
print(json.dumps(rows))
