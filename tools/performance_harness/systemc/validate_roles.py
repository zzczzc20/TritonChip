#!/usr/bin/env python3
"""Validate role isolation, explicit RF aliasing, barrier generations and faults."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct

from run import run, sha
from validate import config
from validate_chip import assemble

PAIR_PROGRAM = '''
    s.rdspecial s1, role_id
    br.z s1, producer
    .li s5, #0x10000
    mb.wait s5
    v.mov v2, v1
    s.rdspecial s2, pid.x
    s.shli s2, s2, #6
    .li s3, #0x50000
    s.add s3, s3, s2
    v.st.w [s3, #0], v2
    exit
producer:
    .li s4, #40
delay:
    s.subi s4, s4, #1
    br.nz s4, delay
    mb.arm s5, b0, #1, #0
    .li s2, #0x3f800000
    v.splat v0, s2
    v.exp2.approx v1, v0
    mb.arrive b0
    exit
'''


def fixture(out, name, assembly, *, roles=3, pair=1, gx=1, expected=None, address=0x50000):
    code=out/f'{name}.hex';assemble(assembly,code)
    (out/f'{name}.s').write_text(assembly)
    script=out/f'{name}.commands'
    text=f'code {code}\nlaunch_roles 1 0 0 {gx} 1 1 {roles} {pair}\nfence\n'
    if expected is not None:
        reference=out/f'{name}-expected.bin';reference.write_bytes(expected)
        text+=f'expect {address:#x} {reference}\n'
    script.write_text(text)
    return script


def validate(binary,out,mutant=None,mutant_source=None):
    out.mkdir(parents=True,exist_ok=False)
    cfg=config(out/'roles.cfg',trace_events=10000)
    results={}
    script=fixture(out,'pair',PAIR_PROGRAM,gx=9,expected=struct.pack('<I',0x40000000)*(9*16))
    r=run(binary,cfg,script,out/'pair',mode='chip')
    assert r['status']=='completed-prediction',out/'pair/stderr.log'
    counters=r['summary']['counters']
    for core in range(3):
        for role in (0,1):assert counters.get(f'core_{core}_role_{role}_completed',0)>0
    assert sum(v for k,v in counters.items() if k.endswith('_barrier_sleep_cycles'))>0
    results['pair']=r['summary']
    isolated='''
        s.rdspecial s1, role_id
        .li s2, #11
        v.splat v0, s2
        .li s2, #22
        v.splat v1, s2
        s.cmp.eq s2, s1, s0
        s.sub s2, s0, s2
        p.frombits p1, s2
        @p1 v.sel v2, v0, v1
        s.rdspecial s3, pid.x
        s.shli s3, s3, #7
        s.shli s1, s1, #6
        s.add s3, s3, s1
        .li s4, #0x60000
        s.add s3, s3, s4
        v.st.w [s3, #0], v2
        exit
    '''
    expected=b''.join(struct.pack('<16I',*([value]*16)) for _ in range(6) for value in (11,22))
    script=fixture(out,'isolated',isolated,pair=0,gx=6,expected=expected,address=0x60000)
    r=run(binary,cfg,script,out/'isolated',mode='chip')
    assert r['status']=='completed-prediction',out/'isolated/stderr.log'
    results['isolated']=r['summary']
    generation='''
        .li s1, #70000
loop:
        mb.arm s2, b0, #0, #0
        mb.wait s2
        s.subi s1, s1, #1
        br.nz s1, loop
        .li s3, #0x50000
        s.st.d [s3, #0], s2
        exit
    '''
    script=fixture(out,'generation',generation,roles=4,pair=0,expected=struct.pack('<Q',70000<<16))
    r=run(binary,cfg,script,out/'generation',mode='chip')
    assert r['status']=='completed-prediction',out/'generation/stderr.log'
    results['generation']=r['summary']
    fault_program='''
        s.rdspecial s1, role_id
        br.z s1, producer
        .li s5, #0x10000
        mb.wait s5
        exit
producer:
        .li s4, #40
delay:
        s.subi s4, s4, #1
        br.nz s4, delay
        ACTION
        exit
    '''
    failures={}
    for name,assembly,roles,error in [
        ('init_waiter',fault_program.replace('ACTION','mb.init b0'),3,'init under a sleeping waiter'),
        ('invalidation',fault_program.replace('ACTION','mb.inval b0'),3,'invalidation woke waiter'),
        ('underflow','mb.arm s1, b0, #1, #0\nmb.arrive b0\nmb.arrive b0\nexit\n',1,'arrival underflow'),
        ('deadlock','.li s5, #0x10000\nmb.wait s5\nexit\n',2,'deadlocked'),
        ('bad_token','.li s5, #0x10020\nmb.wait s5\nexit\n',1,'token index')]:
        script=fixture(out,name,assembly,roles=roles,pair=0)
        r=run(binary,cfg,script,out/name,mode='chip')
        assert r['status']=='failed' and not r['timeout'] and error in (out/name/'stderr.log').read_text(),out/name
        failures[name]=dict(returncode=r['returncode'],expected_error=error)
    if mutant:
        assert mutant_source is not None and mutant_source.is_file()
        spec=importlib.util.spec_from_file_location('_tnca_mutation_runner',mutant_source.parents[1]/'run.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        script=fixture(out,'early_arrive',PAIR_PROGRAM,gx=1,expected=struct.pack('<I',0x40000000)*16)
        r=runner.run(mutant,cfg,script,out/'early_arrive',mode='chip',source_identity=dict(test_mutation='early-arrive',mutated_source_sha256=sha(mutant_source)))
        assert r['status']=='failed' and 'chip numerical mismatch' in (out/'early_arrive/stderr.log').read_text()
        failures['early_arrive']=dict(returncode=r['returncode'],binary_sha256=sha(mutant),mutated_source_sha256=sha(mutant_source),scope='release wait removed from a separately preserved source clone')
    result=dict(schema='tnca-native-role-validation/1',status='passed',results=results,
        expected_failures=failures,early_arrive_mutation_checked=mutant is not None,
        physical_register_writeback='timestamped, lane-masked commit; private per-role scoreboard',
        performance_gate='not-measured',whole_chip_qualified=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--early-arrive-mutant',type=Path)
    p.add_argument('--early-arrive-source',type=Path)
    a=p.parse_args();print(validate(a.binary,a.out,a.early_arrive_mutant,a.early_arrive_source)['status'])
