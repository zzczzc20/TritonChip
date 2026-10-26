"""Diagnostic only: where do first-call native copy launches come from?"""
import os, sys, traceback, collections
from pathlib import Path
ROOT = Path(os.environ['SRC'])
for p in (ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch, triton  # noqa
import tnca.torch_device as D
from tnca import device
from tnca.triton_backend.driver import launch_log
import tnca.torch_device.copy as C
device.prepare(os.environ['TNCA_PROFILE'])
D.register()
origins = collections.Counter()
orig = C.copy_tensor
def traced(src, dst):
    st = [f'{Path(f.filename).name}:{f.lineno}:{f.name}' for f in traceback.extract_stack(limit=40)[:-1]
          if 'torch/_inductor' in f.filename or 'torch/_dynamo' in f.filename or 'diag_copy' in f.filename]
    origins[' <- '.join(st[-6:])] += 1
    return orig(src, dst)
C.copy_tensor = traced
torch.manual_seed(42)
h, w = torch.randn(16, 16).half(), (torch.randn(16, 16) * 0.1).half()
x, dw = h.to('tnca'), w.to('tnca')
def fn(t, weight):
    return torch.softmax(torch.nn.functional.layer_norm(t, (16,)) @ weight, -1)
graph = torch.compile(fn, fullgraph=True)
os.environ['TNCA_SUBMIT'] = 'single'
with launch_log() as first:
    y = graph(x, dw)
print('FIRST', collections.Counter(e['kernel'] for e in first))
with launch_log() as second:
    y = graph(x, dw)
print('SECOND', [e['kernel'] for e in second])
for k, v in origins.most_common(10):
    print('ORIGIN', v, k)
