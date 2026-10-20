"""Read reviewed operating-clock evidence without inferring absent measurements."""
import hashlib
import json
import math
from pathlib import Path


def qualified_clock(path,image_sha256):
    path=Path(path).resolve();data=json.loads(path.read_text())
    if data.get('schema')!='tnca-qualified-operating-clock/1':raise ValueError('unknown clock evidence schema')
    if data.get('image_sha256')!=image_sha256:raise ValueError('clock evidence belongs to another image')
    if data.get('qualification')!='conditional-r40v2-20261018':raise ValueError('explicit conditional clock review required')
    hz=data.get('clock_hz')
    if type(hz) is not int or not 0<hz<=91_000_000:raise ValueError('conditional clock must be positive and <=91 MHz')
    for field in ('setup_wns_ns','hold_whs_ns'):
        v=data.get(field)
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('negative or absent '+field)
    if data.get('route_legal') is not True or data.get('cdc_review')!='passed':raise ValueError('route/CDC review incomplete')
    if data.get('physical_core_slrs')!={'0':0,'1':1,'2':2}:raise ValueError('three physical cores must occupy their reviewed SLRs')
    if data.get('clock_source')!='measured-board-counter':raise ValueError('actual board clock measurement required')
    # These must include the reviewed route/timing/CDC and actual clock capture.
    files=data.get('evidence_files',{})
    required={'timing','route','cdc','board_clock','placement'}
    if set(files)!=required:raise ValueError('missing reviewed raw evidence files')
    for item in files.values():
        f=Path(item['path']);f=f if f.is_absolute() else path.parent/f
        if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('raw clock evidence changed')
    return dict(clock_hz=hz,clock_policy=f'{hz}Hz-undivided-conditional',
                clock_qualification=data['qualification'],
                operating_clock_evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                operating_clock_evidence=data)


def from_environment(image_sha256):
    import os
    path=os.environ.get('TNCA_OPERATING_CLOCK_EVIDENCE')
    return qualified_clock(path,image_sha256) if path else {}
