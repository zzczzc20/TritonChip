"""Read reviewed operating-clock evidence without inferring absent measurements."""
import hashlib
import json
import math
import re
from pathlib import Path


# A qualification name is a reviewed operating path, bound to its image where
# the review was of one image. R40 v2: the user-authorized <=91 MHz path on
# any reviewed image. R41 Stage 2 run 1: the 100 MHz routed image 30e79769...
# (evidence/r41/stage2/run1), its raw reports and a counter capture after load
# (tools/performance_harness/clock_qualify.py assembles and re-checks them).
QUALIFICATIONS={
    'conditional-r40v2-20261018':dict(image_sha256=None,max_hz=91_000_000),
    'r41s2-run1-100mhz':dict(image_sha256='30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de',
                             max_hz=100_000_000,exact_hz=100_000_000),
    'r41-run2-100mhz-hbm441':dict(image_sha256='1bf769fdbfbeb93586be86561664d43e785af3a01a739d19a3882647132d2d13',
                                  max_hz=100_000_000,exact_hz=100_000_000),
}

RUN2_RAW={
    'timing':'cd4c6254c960f5f368ea38f4a9e86b606747fba5d467fc4ac07c00f17971f577',
    'route':'a9f8adc3cd14885894ad79341aa1ba3673432cd5fac136d778c8e2fd141c4980',
    'cdc':'35617dbb86a0ddb86718ea622492634bf91ff3c51effe52e19b7a299841785c5',
    'placement':'eb5a884a26b7d2cab5fed2b26121aa044a257bcdb3551aaa94e5f29e02fd657f',
}


def qualified_clock(path,image_sha256):
    path=Path(path).resolve();data=json.loads(path.read_text())
    if data.get('schema')!='tnca-qualified-operating-clock/1':raise ValueError('unknown clock evidence schema')
    if data.get('image_sha256')!=image_sha256:raise ValueError('clock evidence belongs to another image')
    rule=QUALIFICATIONS.get(data.get('qualification'))
    if rule is None:raise ValueError('explicit conditional clock review required')
    if rule['image_sha256'] not in (None,image_sha256):raise ValueError('this clock review is registered for another image')
    hz=data.get('clock_hz')
    if type(hz) is not int or not 0<hz<=rule['max_hz']:raise ValueError(f"conditional clock must be positive and <={rule['max_hz']//1_000_000} MHz")
    if rule.get('exact_hz') not in (None,hz):raise ValueError(f"this review qualifies exactly {rule['exact_hz']} Hz")
    run2=data.get('qualification')=='r41-run2-100mhz-hbm441'
    if run2 and (data.get('setup_scope')!='kernel-clock' or data.get('original_design_setup_wns_ns')!=-0.045
                 or data.get('hbm_metadata_mhz')!=441 or data.get('setup_wns_ns')!=0.038
                 or data.get('hold_whs_ns')!=0.001 or data.get('pulse_width_wpws_ns')!=0.0):
        raise ValueError('run-2 requires the explicit kernel-only setup and original HBM clock exception')
    for field in ('setup_wns_ns','hold_whs_ns'):
        v=data.get(field)
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('negative or absent '+field)
    if data.get('route_legal') is not True or data.get('cdc_review')!='passed':raise ValueError('route/CDC review incomplete')
    if data.get('physical_core_slrs')!={'0':0,'1':1,'2':2}:raise ValueError('three physical cores must occupy their reviewed SLRs')
    if data.get('clock_source')!='measured-board-counter':raise ValueError('actual board clock measurement required')
    # These must include the reviewed route/timing/CDC and actual clock capture.
    files=data.get('evidence_files',{})
    required={'timing','route','cdc','board_clock','placement'} | ({'cdc_review'} if run2 else set())
    if set(files)!=required:raise ValueError('missing reviewed raw evidence files')
    for item in files.values():
        f=Path(item['path']);f=f if f.is_absolute() else path.parent/f
        if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('raw clock evidence changed')
    if run2:
        if any(files[k]['sha256']!=v for k,v in RUN2_RAW.items()):raise ValueError('run-2 raw reports are not the original reviewed reports')
        f=Path(files['cdc_review']['path']);f=f if f.is_absolute() else path.parent/f
        review=json.loads(f.read_text())
        if review.get('status')!='pass' or review.get('kind')!='cdc-review' or review.get('image_sha256')!=image_sha256:
            raise ValueError('run-2 CDC review is incomplete')
        if (review.get('cdc_report_sha256')!=RUN2_RAW['cdc'] or
                review.get('trace_sha256')!='ed2ec1ced819e81a3df0248ec3d4ba80bd462ef06e142fe3d4d4433ad7a65859' or
                review.get('logic_sha256')!='c2018fea38f9eac440dbf444192db5eb3828fc326fd9c3ea4e39d8f0deacf682'):
            raise ValueError('run-2 CDC review lacks the original circuit evidence')
        rows=review.get('reviewed_critical',[])
        if not isinstance(rows,list) or len(rows)!=56 or any(r.get('disposition')!='accepted-platform-scoped' for r in rows):
            raise ValueError('run-2 CDC critical findings need individual dispositions')
        reviewed={(r['rule'],r['source'],r['destination']) for r in rows}
        c=Path(files['cdc']['path']);c=c if c.is_absolute() else path.parent/c
        actual=[]
        for line in c.read_text().splitlines():
            match=re.match(r'\s*\d+\s+(CDC-\d+)\s+Critical\s',line)
            if match:
                fields=line.split();actual.append((match.group(1),fields[-2],fields[-1]))
        if len(actual)!=56 or len(reviewed)!=56 or set(actual)!=reviewed:
            raise ValueError('run-2 CDC review does not match the reported endpoints')
        f=Path(files['board_clock']['path']);f=f if f.is_absolute() else path.parent/f
        capture=json.loads(f.read_text())
        samples=capture.get('counter_samples',[])
        if (capture.get('image_sha256')!=image_sha256 or capture.get('loaded_xclbin_uuid')!='01490bbd-d831-23bf-d783-56ec51bd71ce'
                or capture.get('clock_hz')!=100_000_000 or len(samples)!=3
                or any(not x.get('frequencies_mhz') or x['frequencies_mhz'][0]!=100 for x in samples)):
            raise ValueError('run-2 board clock capture is incomplete or mismatched')
    return dict(clock_hz=hz,clock_policy=f'{hz}Hz-undivided-conditional',
                clock_qualification=data['qualification'],
                operating_clock_evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                operating_clock_evidence=data)


def from_environment(image_sha256):
    import os
    path=os.environ.get('TNCA_OPERATING_CLOCK_EVIDENCE')
    return qualified_clock(path,image_sha256) if path else {}
