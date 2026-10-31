#!/usr/bin/env python3
"""Build/check an isolated FPGA-host FP8 environment; never access a device.

The base venv itself is never modified. A .pth file exposes its site-packages
to the new venv so pip sees the installed Torch rather than downloading a
second version. Constraints keep Torch 2.14.0 and Triton 3.8.0 immutable.
"""
import argparse
import json
import subprocess
from pathlib import Path

PINS={'torch':'2.14.0','triton':'3.8.0','transformers':'5.20.2','kernels':'0.17.2',
      'accelerate':'1.15.0','huggingface_hub':'1.33.0','safetensors':'0.8.0',
      'torchvision':'0.29.0','pillow':'10.2.0'}
CHECK=r'''
import importlib, importlib.metadata as md, json, pathlib, sys
pins=json.loads(sys.argv[1]); base=pathlib.Path(sys.argv[2]).resolve()
versions={name:md.version(name) for name in pins}
for name,wanted in pins.items():
    if versions[name].split('+')[0]!=wanted:raise RuntimeError((name,versions[name],wanted))
import torch,triton,transformers,kernels,accelerate,safetensors
assert pathlib.Path(torch.__file__).resolve().is_relative_to(base),torch.__file__
assert pathlib.Path(triton.__file__).resolve().is_relative_to(base),triton.__file__
from transformers.integrations.finegrained_fp8 import FP8Linear
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
import torchvision,PIL
print(json.dumps(dict(status='passed',versions=versions,python=sys.executable,
    torch_file=torch.__file__,triton_file=triton.__file__,base_unchanged=True,
    torch_cuda_metadata=torch.version.cuda,actual_cuda_available=torch.cuda.is_available()),indent=2))
'''

def run(argv):return subprocess.check_output([str(x) for x in argv],text=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path.home()/'tc_env/torchenv')
    p.add_argument('--venv',type=Path,default=Path.home()/'r41s2_fp8env')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.base.resolve()==a.venv.resolve():raise ValueError('refuse to modify shared base')
    base_python=a.base/'bin/python'
    base_site=Path(run([base_python,'-c','import torch,pathlib;print(pathlib.Path(torch.__file__).resolve().parents[1])']).strip())
    original=run([base_python,'-c','import torch,triton;print(torch.__version__,triton.__version__)']).strip()
    if original.split()[0].split('+')[0]!='2.14.0' or original.split()[1]!='3.8.0':raise ValueError('base version mismatch')
    python=a.venv/'bin/python'
    if not a.check_only:
        if not python.is_file():subprocess.run([str(base_python),'-m','venv',str(a.venv)],check=True)
        site=Path(run([python,'-c','import site;print(site.getsitepackages()[0])']).strip())
        # kernels 0.17.2 uses Torch's CUDA metadata to select its Triton source
        # namespace. Match the lead's tested environment without changing the
        # actual CPU Torch build, shared files or CUDA availability.
        (site/'tnca_base_torch.pth').write_text(str(base_site)+'\nimport site; site.addsitedir('+repr(str(base_site))+')\nimport torch; setattr(torch.version, "cuda", "13.0")\n')
        constraints=a.venv/'tnca-constraints.txt'
        constraints.write_text('\n'.join(f'{k}=={v}' for k,v in PINS.items())+'\n')
        subprocess.run([str(python),'-m','pip','install','--no-input','--constraint',str(constraints),
                        *[f'{k}=={v}' for k,v in PINS.items() if k not in ('torch','triton','torchvision')],
                        'torchvision==0.29.0+cpu','--extra-index-url','https://download.pytorch.org/whl/cpu'],check=True)
    result=json.loads(run([python,'-c',CHECK,json.dumps(PINS),a.base]))
    assert run([base_python,'-c','import torch,triton;print(torch.__version__,triton.__version__)']).strip()==original
    result['base_versions_before_after']=original
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
