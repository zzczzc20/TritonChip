"""Real parameterized multicore RTL, private code and coherent mailbox IO."""
import hashlib
import json
import operator
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from .. import device
from ..errors import DeviceLost, ProfileMismatch
from ..transport import Transport


def configuration(core_count, memory_bytes):
    core_count,memory_bytes=operator.index(core_count),operator.index(memory_bytes)
    if core_count not in (1,2,3,4,8):
        raise ValueError('core_count must be 1, 2, 3, 4 or 8')
    if memory_bytes<=0x20000 or memory_bytes%64 or memory_bytes//8>0x7fffffff:
        raise ValueError('memory_bytes must be line-aligned, exceed code space and fit the simulator word count')
    return core_count,memory_bytes


def prepare(profile='core-vrf-sync-v5', *, core_count=3, memory_bytes=8*1024*1024, memory_ports=1, **kwargs):
    if profile!='core-vrf-sync-v5':
        raise ProfileMismatch('sim-multicore requires the core-vrf-sync-v5 per-core architecture')
    core_count,memory_bytes=configuration(core_count,memory_bytes)
    memory_ports=operator.index(memory_ports)
    if not 1<=memory_ports<=core_count:raise ValueError('memory_ports must be within 1..core_count')
    rtl_path,tb_path=device._package_data('rtl'),device._package_data('tb')
    if not rtl_path or not tb_path:
        raise DeviceLost('multicore package does not contain RTL and device binding sources')
    rtl,tb=Path(rtl_path),Path(tb_path)
    overrides=dict(device.rtl_overrides())
    allowed={'VRF_BYPASS','FE_NOBUBBLE','VRF_PREFETCH','RCTX_SWITCH','MTX_ABUF','LSU_INFLIGHT','FPU_OVERLAP'}
    if set(overrides)-allowed:
        raise ValueError('RTL overrides are not supported by this multicore shell: '+', '.join(sorted(set(overrides)-allowed)))
    compiler=shutil.which('iverilog')
    if not compiler:
        raise DeviceLost('sim-multicore requires iverilog on PATH')
    version=subprocess.run([compiler,'-V'],capture_output=True,text=True,timeout=30)
    if version.returncode:
        raise DeviceLost('cannot identify the multicore RTL compiler')
    identity={'schema':3,'core_count':core_count,'memory_bytes':memory_bytes,'memory_ports':memory_ports,
              'compiler':os.path.realpath(compiler),'compiler_version':version.stdout,
              'source':device.source_fingerprint(profile),'rtl_overrides':overrides}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory=Path(device.cache_root())/('multicore-'+digest[:24]);directory.mkdir(parents=True,exist_ok=True)
    target=directory/'device.vvp'
    if target.exists():return str(target)
    # Atomic publication; concurrent processes may build the same immutable key.
    with tempfile.TemporaryDirectory(prefix='build-',dir=directory) as temp:
        output=Path(temp)/'device.vvp'
        args=[compiler,'-g2012','-I',str(rtl/'common'),'-I',str(rtl/'noc'),
              '-s','tnca_multicore_device',f'-Ptnca_multicore_device.N={core_count}',
              f'-Ptnca_multicore_device.TOTAL_WORDS={memory_bytes//8}',f'-Ptnca_multicore_device.PORTS={memory_ports}','-o',str(output)]
        args += [f'-Ptnca_multicore_device.{key}={value}' for key,value in sorted(overrides.items())]
        for library in ('common','core','gen','nuca','noc','tile'):args+=['-y',str(rtl/library)]
        args += [str(p) for p in sorted((rtl/'common').glob('*.v'))]
        args += [str(rtl/p) for p in ('tnca_top.v','tnca_grid_dispatch.v','tnca_grid_deadline.v','tnca_multicore_shell.v')]
        args += [str(tb/'common/tnca_dram_model.v'),str(tb/'device/tnca_multicore_device.v')]
        result=subprocess.run(args,capture_output=True,text=True,timeout=300)
        if result.returncode:
            raise DeviceLost('multicore RTL elaboration failed',detail={'diagnostic':result.stderr[-4000:]})
        os.replace(output,target)
        (directory/'configuration.json').write_text(json.dumps(identity,sort_keys=True,indent=2)+'\n')
    return str(target)


class MulticoreFileBackend(Transport):
    def __init__(self, bench=None, *, core_count=3, memory_bytes=8*1024*1024, memory_ports=1, **kwargs):
        self.core_count,self.memory_bytes=configuration(core_count,memory_bytes)
        self.memory_ports=operator.index(memory_ports)
        if not 1<=self.memory_ports<=self.core_count:raise ValueError('memory_ports must be within 1..core_count')
        super().__init__(bench or prepare(core_count=core_count,memory_bytes=memory_bytes,memory_ports=memory_ports),**kwargs)

    def _await(self, seq, **kwargs):
        reply=super()._await(seq,**kwargs)
        if seq==0:self.boot=reply
        return reply

    def capabilities(self):
        out=super().capabilities()
        out.update({'physical_grid':True,'memory_bytes':self.memory_bytes,
                    'completion_counters':'global-makespan-u64',
                    'fault_detail':False,'memory_ports':self.memory_ports})
        return out

    def start(self):
        try:
            super().start()
            capability=self.physical_grid_capability()
            if capability is None or capability[0]!=self.core_count or self.boot.r0*8!=self.memory_bytes:
                raise DeviceLost('multicore image does not match requested core count/memory capacity')
            if self.memory_port_capability()!=self.memory_ports:
                raise DeviceLost('multicore image does not match the memory outlet contract')
            return self
        except Exception:
            self.stop()
            raise
