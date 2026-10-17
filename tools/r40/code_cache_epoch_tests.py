#!/usr/bin/env python3
"""Execute alternating modules under code pressure without replacing the device."""
from pathlib import Path
import struct
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'python',ROOT/'compiler/python',ROOT/'tools/asm'):sys.path.insert(0,str(p))
import triton
from tnca import aot,artifact,device
from tnca.runtime import Session
from tnca.triton_backend.driver import _load

def main():
    profile='core-vrf-sync-v2'
    device.prepare(profile)
    with tempfile.TemporaryDirectory(prefix='r40-code-epoch-') as tmp:
        paths=[]
        for value in (17,29,43):
            def program(b,target,value=value):
                dest=b.sload(b.special('arg_base'),0)
                b.sstore(dest,b.const(value),size='d');b.exit()
            result=aot.compile_aot(program,kernel_name=f'code_{value}',source_kind='tnca-ir',
                                   profile=profile,signature={'out':'*i64'},output=str(Path(tmp)/f'{value}.tnca'))
            paths.append(result.path)
        # Exactly one module fits. Keep a live data handle while evicting code.
        region=max((len(artifact.load(p)[1]['text'])+63)//64*64 for p in paths)
        with Session(profile=profile,heap_base=region) as session:
            sid,epoch=session.id,session.epoch
            out=session.allocate(8)
            for index in (0,1,2,0,2,1):
                actual,module=_load(session,paths[index])
                assert actual is session and session.id==sid and session.epoch==epoch
                assert out.alive
                result=session.wait(session.submit(module,{'out':out}))
                assert result.ok
                assert struct.unpack('<Q',session.download(out))[0]==(17,29,43)[index]
            assert len(session._modules)==1
        print('CODE CACHE EPOCH PASS 6 RTL launches, one-module region, live buffer preserved',flush=True)
if __name__=='__main__':main()
