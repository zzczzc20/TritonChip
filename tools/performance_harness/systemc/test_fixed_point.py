#!/usr/bin/env python3
"""Real slow-path convergence, bounded inconclusive execution and refusal."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

from prediction_domain import prediction_status, timing_status
from run import run
from validate_chip import assemble


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    values=dict(line.split('#')[0].strip().split('=',1) for line in args.config.read_text().splitlines()
                if '=' in line.split('#')[0])
    values.update(memory_bytes=str(1<<24),dram_path='1')
    image=out/'copy.hex'
    assemble('.li s1, #0x400000\nv.ld.w v0, [s1, #0]\n.li s2, #0x410000\nv.st.w [s2, #0], v0\nexit\n',image)
    data=out/'input.bin';data.write_bytes(struct.pack('<16I',*range(16)))
    commands=out/'copy.commands'
    commands.write_text(f'code {image}\ndata 0x400000 {data}\nlaunch 1 0 0 1 1 1\nfence\nexpect 0x410000 {data}\n')
    rows=[]
    for passes,expected in ((1,'inconclusive-fixed-point'),(8,'uncalibrated-fixed-point-checked')):
        config=out/f'passes-{passes}.cfg'
        config.write_text(''.join(f'{k}={v}\n' for k,v in dict(values,dram_max_passes=str(passes)).items()))
        result=run(args.binary,config,commands,out/f'passes-{passes}',timeout=30,mode='chip')
        assert result['returncode']==0 and result['summary']['checked_bytes']==64,result
        assert result['model_timing_status']==expected,result
        assert result['summary']['prediction_status']==expected,result
        records=[json.loads(line) for line in (out/f'passes-{passes}/dram_fixed_point.jsonl').read_text().splitlines()]
        traffic=[r for r in records if r['requests']]
        assert traffic and all(r['replays']>=1 for r in traffic),records
        assert all(r['prediction_status']==expected for r in traffic),records
        if passes==8:
            assert all(r['converged'] and r['same_requests'] and r['latencies_stable'] and
                       r['reproduced_last_pass'] and r['unmatched']==0 for r in traffic),records
        else:
            assert all(not r['converged'] for r in traffic),records
        row=dict(point=f'copy-passes-{passes}',key=f'passes-{passes}',status='pass',knobs={},
                 model_timing_status=result['model_timing_status'],
                 model_graph=dict(graph_cycles=result['summary']['visible_cycles'],launches=[],device_cycles=0))
        rows.append(row)
    # The actual numerically correct one-pass result must not enter residual
    # analysis, while the independently validated fixed point remains usable
    # only as an uncalibrated prediction.
    source=out/'results.json';source.write_text(json.dumps(dict(points=rows)))
    destination=out/'predictions.jsonl'
    check=subprocess.run([sys.executable,str(Path(__file__).with_name('predictions.py')),
        '--results',str(source),'--config',str(out/'passes-8.cfg'),'--binary',str(args.binary),
        '--out',str(destination)],text=True,capture_output=True,check=True)
    (out/'predictions-stderr.log').write_text(check.stderr)
    predictions=[json.loads(line) for line in destination.read_text().splitlines()]
    assert [p['point'] for p in predictions]==['copy-passes-8'],predictions
    assert 'no prediction: passes-1 (inconclusive-fixed-point)' in check.stderr
    assert prediction_status(dict(dram=dict(dram_slow_commands=1)))=='inconclusive-fixed-point'
    assert timing_status(dict(dram_slow_commands=2,dram_slow_fixed_point_checked=2,
                              dram_slow_unmatched=1),slow=True)=='inconclusive-fixed-point'
    # A real unsupported operation in a fork pass refuses the model. Its
    # failure must not be hidden by retrying a final numeric execution.
    bad=out/'unsupported.hex';assemble('.li s2, #0x400000\natom.add.d s1, [s2], s3\nexit\n',bad)
    commands=out/'unsupported.commands';commands.write_text(f'code {bad}\nlaunch 1 0 0 1 1 1\n')
    refused=run(args.binary,out/'passes-8.cfg',commands,out/'refused',timeout=30,mode='chip')
    assert refused['status']=='failed' and refused['returncode']!=0,refused
    assert 'child pass failed; prediction unavailable' in (out/'refused/stderr.log').read_text()
    log=json.loads((out/'refused/dram_fixed_point.jsonl').read_text().splitlines()[-1])
    assert log['child_failed'] and log['prediction_status']=='refused-child-pass',log
    report=dict(status='pass',checks=['actual-stable-request-latency-fixed-point','bounded-inconclusive-numeric-pass',
                 'residual-exclusion','legacy-unchecked-refusal','unmatched-refusal','actual-child-failure-refusal'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
