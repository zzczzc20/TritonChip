#!/usr/bin/env python3
"""Native transfer execution against an independent rank/stride/byte oracle."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import random
import struct

from run import run, sha
from validate import config
from validate_chip import assemble
from tnca_cc.target import TargetInfo
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.codegen import CodeGen

SIZES={0:2,1:2,2:4,3:1,4:2,5:4,6:8,7:1,8:1,9:8}
NANS={0:0x7e00,1:0x7fc0,2:0x7fc00000,7:0x7f,8:0x7f,9:0x7ff8000000000000}
SUFFIX={1:'b',2:'h',4:'w',8:'d'}


def descriptor(base,extent,strides,block,dtype=0,padding=0,version=7):
    data=bytearray(256)
    struct.pack_into('<Q',data,0,base)
    for d in range(len(extent)):
        struct.pack_into('<Q',data,8+8*d,extent[d])
        struct.pack_into('<q',data,48+8*d,strides[d])
        struct.pack_into('<I',data,88+4*d,block[d])
    struct.pack_into('4B',data,108,len(extent),dtype,padding,0)
    struct.pack_into('<QQ',data,112,7319,version)
    data[128:]=random.Random(version).randbytes(128)
    return bytes(data)


def copy_assembly(size,coordinates,*,transpose=False,before='',after=''):
    coords='\n'.join(f'.li s{10+d}, #{c}' for d,c in enumerate(coordinates))
    suffix=SUFFIX[size]+('.t' if transpose else '')
    return f'''
        .li s1, #0x30000
        td.load d0, [s1]
        .li s2, #0x30100
        td.load d1, [s2]
        w.def w0, #0, #8
        {before}
        mb.arm s3, b0, #0, #1
        {coords}
        t.ld.async.{suffix} w0, d0, s10, s3
        mb.wait s3
        mb.arm s4, b0, #0, #1
        .li s10, #2
        t.st.async.{SUFFIX[size]} w0, d1, s10, s4
        {after}
        mb.wait s4
        exit
    '''


def execute(binary,cfg,out,name,assembly,images,expected=None,address=0x60005,runner=run,identity=None):
    code=out/f'{name}.hex';assemble(assembly,code);(out/f'{name}.s').write_text(assembly)
    commands=f'code {code}\n'
    for index,(base,data) in enumerate(images):
        path=out/f'{name}-input{index}.bin';path.write_bytes(data);commands+=f'data {base:#x} {path}\n'
    commands+='launch 1 0 0 1 1 1\nfence\n'
    if expected is not None:
        ref=out/f'{name}-expected.bin';ref.write_bytes(expected);commands+=f'expect {address:#x} {ref}\n'
    script=out/f'{name}.commands';script.write_text(commands)
    return runner(binary,cfg,script,out/name,mode='chip',source_identity=identity)


def case(rank,dtype,padding):
    size=SIZES[dtype];extent=[3]*(rank-1)+[8//size+2];block=[2]*(rank-1)+[8//size]
    strides=[0]*rank;strides[-1]=size;span=extent[-1]*size+3
    for d in reversed(range(rank-1)):strides[d]=span;span=span*extent[d]+5
    offset=3
    if rank>1 and rank%2==0:offset+=strides[0]*(extent[0]-1);strides[0]=-strides[0]
    locations=[]
    for at in itertools.product(*(range(e) for e in extent)):
        locations.append(offset+sum(i*s for i,s in zip(at,strides)))
    data=bytearray([0xd3]*(max(locations)+size+8))
    mask=(1<<(8*size))-1
    for index,pos in enumerate(locations):data[pos:pos+size]=((index*17+0x123)&mask).to_bytes(size,'little')
    coordinates=[-1]+[0]*(rank-1)
    result=bytearray()
    for at in itertools.product(*(range(b) for b in block)):
        logical=tuple(c+i for c,i in zip(coordinates,at))
        if all(0<=i<e for i,e in zip(logical,extent)):
            pos=offset+sum(i*s for i,s in zip(logical,strides));result+=data[pos:pos+size]
        else:result+=(NANS[dtype] if padding==3 else 0).to_bytes(size,'little')
    elements=len(result)//size;destination=bytearray([0xa5]*((elements+4)*size))
    destination[2*size:(2+elements)*size]=result
    images=[(0x40000,bytes(data)),(0x60005,bytes([0xa5]*len(destination))),
        (0x30000,descriptor(0x40000+offset,extent,strides,block,dtype,padding)),
        (0x30100,descriptor(0x60005,[elements+4],[size],[elements],dtype))]
    return size,coordinates,images,bytes(destination)


def validate(binary,out,mutant=None,mutant_source=None):
    out.mkdir(parents=True,exist_ok=False);cfg=config(out/'dma.cfg',trace_events=10000)
    results={}
    for dtype in range(10):
        rank=dtype%5+1;padding=3 if dtype in NANS and dtype%2 else 0
        size,coordinates,images,expected=case(rank,dtype,padding)
        name=f'rank{rank}-dtype{dtype}-pad{padding}'
        r=execute(binary,cfg,out,name,copy_assembly(size,coordinates),images,expected)
        assert r['status']=='completed-prediction',out/name/'stderr.log'
        assert r['summary']['counters']['core_0_dma_commands_completed']==2
        results[name]=r['summary']
    # Public TNCA IR/codegen path, including its real window allocation and
    # token/coordinate lowering, not just hand-assembled instructions.
    size,coordinates,images,expected=case(3,0,0)
    kernel=make_kernel('descriptor_copy');builder=Builder(kernel,kernel.role(0).region)
    source=builder.descriptor(builder.const(0x30000));destination=builder.descriptor(builder.const(0x30100))
    window=builder.window(1);barrier=builder.barrier()
    token=builder.barrier_arm(barrier,ops=1)
    builder.dma_load(window,source,[builder.const(c) for c in coordinates],token)
    builder.barrier_wait(token)
    token=builder.barrier_arm(barrier,ops=1)
    builder.dma_store(window,destination,[builder.const(2)],token)
    builder.barrier_wait(token)
    generator=CodeGen(kernel,TargetInfo.load('tnca-sim-vrf-v5'),check_protocol=False)
    generator.assign_shared()
    assembly=generator.run().text()+'\nexit\n'
    r=execute(binary,cfg,out,'compiled-descriptor',assembly,images,expected)
    assert r['status']=='completed-prediction',out/'compiled-descriptor/stderr.log'
    results['compiled-descriptor']=r['summary']
    for rows in (4,8,16):
        columns=8;source=bytearray((rows+2)*24)
        for row in range(rows+2):
            for column in range(12):struct.pack_into('<H',source,row*24+column*2,row*100+column+1)
        values=[0 if row==0 else (row-1)*100+column+1 for column in range(columns) for row in range(rows)]
        expected=bytearray([0xa5]*((len(values)+4)*2));expected[4:-4]=struct.pack('<'+'H'*len(values),*values)
        images=[(0x40000,bytes(source)),(0x60005,bytes([0xa5]*len(expected))),
            (0x30000,descriptor(0x40000,[rows+2,12],[24,2],[rows,columns])),
            (0x30100,descriptor(0x60005,[len(values)+4],[2],[len(values)]))]
        name=f'transpose-{rows}'
        r=execute(binary,cfg,out,name,copy_assembly(2,[-1,0],transpose=True),images,bytes(expected))
        assert r['status']=='completed-prediction',out/name/'stderr.log'
        results[name]=r['summary']
    # A fully predicated-off command must not inspect an undefined descriptor or
    # window, but still completes exactly one armed operation.
    assembly='''
        mb.arm s3, b0, #0, #1
        p.frombits p1, s0
        @p1 t.ld.async.h w0, d0, s10, s3
        mb.wait s3
        mb.test s4, s3
        .li s5, #0x60000
        s.st.d [s5, #0], s4
        exit
    '''
    r=execute(binary,cfg,out,'masked',assembly,[],struct.pack('<Q',1),address=0x60000)
    assert r['status']=='completed-prediction',out/'masked/stderr.log'
    counters=r['summary']['counters'];assert counters['core_0_dma_commands_completed']==1 and counters.get('core_0_dma_read_words',0)==0
    results['masked']=r['summary']
    # Geometry/lifetime faults must preserve an incomplete trace and counters.
    bad_descriptor=descriptor(0x40000,[2],[2],[8],0,2)
    prefix='''.li s1, #0x30000
td.load d0, [s1]
w.def w0, #0, #1
mb.arm s3, b0, #0, #1
.li s10, #0
t.ld.async.h w0, d0, s10, s3
'''
    failures={}
    for name,desc,tail,error in [
        ('fault-padding',bad_descriptor,'mb.wait s3\nexit\n','out-of-bounds fault padding'),
        ('release-live',descriptor(0x40000,[32],[2],[32]),'td.release d0\nmb.wait s3\nexit\n','descriptor still referenced'),
        ('overwrite-live',descriptor(0x40000,[32],[2],[32]),'v.movi v0, #7\nmb.wait s3\nexit\n','read before asynchronous destination completion'),
        ('overwrite-store-source',descriptor(0x40000,[32],[2],[32]),'mb.wait s3\nmb.arm s3, b0, #0, #1\nt.st.async.h w0, d0, s10, s3\nv.movi v0, #7\nmb.wait s3\nexit\n','asynchronous source release')]:
        r=execute(binary,cfg,out,name,prefix+tail,[(0x30000,desc),(0x40000,bytes(64))])
        assert r['status']=='failed' and not r['timeout'] and error in (out/name/'stderr.log').read_text(),out/name
        partial=json.loads((out/name/'partial.json').read_text())
        assert partial['status']=='incomplete'
        if name=='fault-padding':assert partial['counters']['core_0_dma_read_words']==32
        assert json.loads((out/name/'trace.json').read_text())['metadata']['complete'] is False
        failures[name]=dict(error=error,partial=partial)
    report=dict(schema='tnca-native-dma-validation/1',status='passed',results=results,expected_failures=failures,
        performance_gate='not-measured',whole_chip_qualified=False,
        scope='native descriptor/window/load/store semantics; no measured architecture speedup or physical memory qualification')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(validate(a.binary,a.out)['status'])
