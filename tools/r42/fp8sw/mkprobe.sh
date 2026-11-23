#!/bin/bash
# usage: mkprobe.sh TREE OUTDIR [PROFILE]   build the probed device bench (iverilog) into OUTDIR/device.vvp
TREE=$1; OUT=$2; PROFILE=${3:-core-vrf-sync-v5-x8-g2c4-dp}
cd $TREE || exit 2
. tools/r42/fp8sw/env.sh
mkdir -p $OUT
PYTHONDONTWRITEBYTECODE=1 nice -n 19 $PY - <<PY
import os, subprocess, shutil
from tnca import device
p='$PROFILE'
out='$OUT'
rtl=device._package_data('rtl'); tb=device._package_data('tb')
open(out+'/tnca_uarch.vh','w').write(device.uarch_header(p))
argv=[shutil.which('iverilog'),'-g2005','-I',os.path.join(rtl,'common'),'-y',os.path.join(rtl,'core'),'-y',os.path.join(rtl,'common'),
      '-y',os.path.join(rtl,'gen'),'-y',os.path.join(tb,'common'),'-y',os.path.join(rtl,'fpga')]+device.bench_params(p)+['-s','tnca_device','-s','issue_probe','-o',out+'/device.vvp',out+'/tnca_uarch.vh',
      os.path.join(tb,'device','tnca_device.v'),os.path.join(rtl,'tnca_shell.v'),'$TREE/tools/r42/fp8sw/issue_probe.v']
open(out+'/cmd.txt','w').write(' '.join(argv))
r=subprocess.run(argv,capture_output=True,text=True); print(r.returncode, r.stdout[-1500:], r.stderr[-1500:])
PY
