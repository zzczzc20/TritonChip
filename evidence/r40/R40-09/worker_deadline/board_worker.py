"""Functional worker: unchanged native Torch graph and actual board endpoints."""
import json,sys
from pathlib import Path
import torch,triton
import tnca.torch_device as D
from tnca.autotune import measure_native
D.register();torch.manual_seed(49)
x,y=torch.randn(256),torch.randn(256)
dx,dy=x.to('tnca'),y.to('tnca')
def original(a,b):return (a+b)*.5
compiled=torch.compile(original,fullgraph=True,dynamic=False)
want=original(x,y)
torch.testing.assert_close(compiled(dx,dy).cpu(),want)
outputs=[]
records=measure_native(lambda:outputs.append(compiled(dx,dy)))
for output in outputs:torch.testing.assert_close(output.cpu(),want)
out=Path(sys.argv[1]);assert not out.exists();out.write_text(json.dumps(records)+'\n')
print('BOARD NATIVE WORKER PASS: fullgraph Inductor; zero staging; actual common-clock endpoints',flush=True)
