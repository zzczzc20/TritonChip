#!/usr/bin/env python3
"""Bounded functional validation for same-image memory controls; no gate claims."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/r1'))
import r1lib as R

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--native',action='store_true')
    ap.add_argument('--python',default=sys.executable)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tnca-runtime-memory-control-tests/1',gate_status='not-measured',
                contract_sha256=hashlib.sha256((ROOT/'spec/r40/v2/lock.json').read_bytes()).hexdigest(),
                results=[],source_sha256={})
    paths=['rtl/core/tnca_core.v','python/tnca/backends/u50_sim.py','rtl/fpga/tnca_memory_mode.v','rtl/fpga/tnca_line_axi.v','rtl/fpga/tnca_multicore_mmio.v',
           'rtl/fpga/tnca_multicore_axi.v','rtl/fpga/tnca_multicore_fpga.v','rtl/board/tnca_multicore_hbm_xo.v',
           'rtl/nuca/tnca_mem_arbiter.v','python/tnca/backends/multicore_mmio.py',
           'python/tnca/backends/multicore_mmio_sim.py','tb/fpga/tb_memory_mode.v',
           'tb/fpga/tb_mmio_memory_controls.v','tb/fpga/tb_line_axi.v',
           'tb/fpga/tnca_multicore_regs.v','tb/fpga/tnca_multicore_board_regs.v',
           'tb/fpga/tnca_multicore_shared_hbm.vh',
           'tools/r40/native_memory_control_tests.py','tools/r40/memory_control_host_tests.py',
           'tools/r40/memory_control_tests.py']
    report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    def step(name,argv,cap=120):
        result=R.run_step(argv,kind='build',cwd=str(ROOT),case_dir=str(out),name=name,cap_seconds=cap)
        report['results'].append(dict(result))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if result['status']!=R.PASS:
            print(name+': '+result['status'],flush=True)
            print(R.read_text(result['stdout'])[-1800:]+R.read_text(result['stderr'])[-1800:],flush=True)
            return False
        print(name+': PASS',flush=True);return True
    cases=[('memory_mode','tb_memory_mode',[],['rtl/nuca/tnca_mem_arbiter.v','rtl/fpga/tnca_memory_mode.v','tb/fpga/tb_memory_mode.v'])]
    for enabled in (0,1):
        cases.append(('mmio_'+str(enabled),'tb_mmio_memory_controls',[f'-Ptb_mmio_memory_controls.ENABLE={enabled}'],
                      ['rtl/fpga/tnca_multicore_mmio.v','tb/fpga/tb_mmio_memory_controls.v']))
    for limit in (1,8):
        for stages in (0,2):
            cases.append((f'line_limit{limit}_stages{stages}','tb_line_axi',
                [f'-Ptb_line_axi.RUNTIME_LIMIT={limit}',f'-Ptb_line_axi.STAGES={stages}','-Ptb_line_axi.WRITE_ACK=0'],
                ['rtl/common/tnca_elastic_link.v','rtl/common/tnca_line_link.v','rtl/fpga/tnca_line_axi.v','tb/fpga/tb_line_axi.v']))
    for name,top,parameters,sources in cases:
        binary=str(out/(name+'.vvp'))
        if not step(name+'_build',['iverilog','-g2012','-I','rtl/common','-y','rtl/common','-s',top,*parameters,'-o',binary,*sources]):return 1
        if not step(name,['vvp',binary],60):return 1
    if not step('host',[args.python,'tools/r40/memory_control_host_tests.py']):return 1
    if not step('legacy_options',[args.python,'tools/r40/multicore_mmio_option_tests.py']):return 1
    if args.native and not step('native',[args.python,'tools/r40/native_memory_control_tests.py'],3600):return 1
    report['status']='functional-pass'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RUNTIME MEMORY CONTROLS FUNCTIONAL PASS; board performance not-measured',flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
