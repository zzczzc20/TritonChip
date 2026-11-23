#!/bin/bash
# builds the probed device bench for PROFILE from TREE into ~/r42/xfer_out/probe/device.vvp
TREE=${1:-$HOME/r42/xfer_out/xfer}; PROFILE=${2:-core-vrf-sync-v5-x8-g2c4-dp}
cd $TREE; source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
mkdir -p ~/r42/xfer_out/probe
nice -n 19 $HOME/tc_env/torchenv/bin/python - <<PY
import os, subprocess, shutil
from tnca import device
p='$PROFILE'
out=os.path.expanduser('~/r42/xfer_out/probe')
rtl=device._package_data('rtl'); tb=device._package_data('tb')
open(out+'/tnca_uarch.vh','w').write(device.uarch_header(p))
argv=[shutil.which('iverilog'),'-g2005','-I',os.path.join(rtl,'common'),'-y',os.path.join(rtl,'core'),'-y',os.path.join(rtl,'common'),
      '-y',os.path.join(rtl,'gen'),'-y',os.path.join(tb,'common'),'-y',os.path.join(rtl,'fpga')]+device.bench_params(p)+['-s','tnca_device','-s','gemm_probe','-o',out+'/device.vvp',out+'/tnca_uarch.vh',
      os.path.join(tb,'device','tnca_device.v'),os.path.join(rtl,'tnca_shell.v'),out+'/xf_probe.v']
open(out+'/cmd.txt','w').write(' '.join(argv))
r=subprocess.run(argv,capture_output=True,text=True); print(r.returncode, r.stdout[-1500:], r.stderr[-1500:])
PY
