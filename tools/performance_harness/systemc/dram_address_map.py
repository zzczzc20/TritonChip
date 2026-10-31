#!/usr/bin/env python3
"""Audit an actual AXI BO/PC map and export a native-model mapping input.

PC identities come from recorded physical HBM tags and allocation ranges.
This establishes no bank/row map, controller equivalence or calibrated timing.

--plan-bo-bytes N exports a PLANNED map instead: a larger BO on the same image,
extrapolated from the recorded window's connected memory group (e.g. MBG[0:31])
with the recorded local limit, BO base and per-PC rule. It is marked planned in
map.txt and report.json, is not an audited allocation, and must be replaced by
an audit of the board run's actual record.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def number(value):
    if type(value) is str:
        value=int(value,0)
    if type(value) is not int or not 0<=value<1<<64:
        raise ValueError('mapping values must be unsigned 64-bit integers')
    return value


def pointer(document, path):
    if path=='':
        return document
    if not path.startswith('/'):
        raise ValueError('JSON pointer must start with /')
    for key in path[1:].split('/'):
        key=key.replace('~1','/').replace('~0','~')
        document=document[int(key)] if isinstance(document,list) else document[key]
    return document


def normalize(window):
    base,size,local=(number(window[k]) for k in ('base','size','local_limit'))
    if not size or base+size>=1<<64 or local+size>=1<<64 or (base|size|local)&63:
        raise ValueError('unaligned or overflowing AXI window')
    allocation=window['allocation']
    if allocation.get('mapping_error') or number(allocation['exposed_bytes'])!=size:
        raise ValueError('allocation is not a complete physical mapping of this BO')
    topology=allocation['topology_sha256']
    if not re.fullmatch('[0-9a-f]{64}',topology):
        raise ValueError('physical topology identity missing')
    banks=[]
    for raw in allocation['physical_banks']:
        pc=number(raw['pc']);start,capacity,lo,length=(number(raw[k]) for k in
            ('base','size','allocated_base','allocated_size'))
        if raw['tag']!=f'HBM[{pc}]' or raw.get('used') is not True:
            raise ValueError('PC ID is not certified by a used physical HBM tag')
        if (not capacity or not length or (start|capacity|lo|length)&63 or
                start+capacity>=1<<64 or lo<start or lo+length>start+capacity):
            raise ValueError('invalid physical PC allocation range')
        banks.append(dict(pc=pc,base=start,size=capacity,allocated_base=lo,allocated_size=length))
    check_coverage(base,size,banks)
    return dict(local_limit=local,bo_base=base,bo_size=size,banks=banks,topology_sha256=topology)


def check_coverage(base,size,banks):
    banks.sort(key=lambda b:b['allocated_base'])
    cursor=base;previous_end=0
    for bank in banks:
        if bank['allocated_base']!=cursor or bank['base']<previous_end:
            raise ValueError('BO physical coverage has a gap or overlap')
        cursor+=bank['allocated_size']
        previous_end=bank['base']+bank['size']
    if cursor!=base+size or len({b['pc'] for b in banks})!=len(banks):
        raise ValueError('BO physical coverage is incomplete or repeats a PC')


def plan(window,bo_bytes):
    """A planned BO of bo_bytes on the recorded window's connected memory group.

    The recorded window is audited first. Its connected group tag (MBG[a:b])
    names the PCs; every recorded bank must follow the group's rule
    base = group base + (pc - a) * PC size, which is then applied to the PCs a
    larger BO would occupy. Local limit and BO base are kept. Nothing here is
    measured: the result is a planned allocation, not an audited one.
    """
    recorded=normalize(window)
    bo_bytes=number(bo_bytes)
    group=window['allocation']['connected_memory']
    match=re.fullmatch(r'MBG\[(\d+):(\d+)\]',group.get('tag',''))
    if not match or group.get('used') not in (1,True):
        raise ValueError('recorded connected memory is not a used MBG[a:b] group')
    first,last=int(match.group(1)),int(match.group(2))
    group_base,group_size=number(group['base']),number(group['size'])
    sizes={b['size'] for b in recorded['banks']}
    if last<first or len(sizes)!=1:
        raise ValueError('recorded banks do not share one PC size')
    pc_bytes=sizes.pop()
    if (last-first+1)*pc_bytes!=group_size:
        raise ValueError('connected group size is not its PC count times the recorded PC size')
    for bank in recorded['banks']:
        if not first<=bank['pc']<=last or bank['base']!=group_base+(bank['pc']-first)*pc_bytes:
            raise ValueError('a recorded PC does not follow the connected group rule')
    base,local=recorded['bo_base'],recorded['local_limit']
    if not bo_bytes or bo_bytes&63 or base<group_base or base+bo_bytes>group_base+group_size or local+bo_bytes>=1<<64:
        raise ValueError('planned BO does not fit the recorded connected memory group')
    banks=[]
    for pc in range(first,last+1):
        start=group_base+(pc-first)*pc_bytes
        lo,hi=max(start,base),min(start+pc_bytes,base+bo_bytes)
        if lo<hi:
            banks.append(dict(pc=pc,base=start,size=pc_bytes,allocated_base=lo,allocated_size=hi-lo))
    check_coverage(base,bo_bytes,banks)
    return recorded,dict(local_limit=local,bo_base=base,bo_size=bo_bytes,banks=banks,topology_sha256=None)


def translate(mapping,address,nbytes=64):
    address,nbytes=number(address),number(nbytes)
    local,size=mapping['local_limit'],mapping['bo_size']
    if not nbytes or address<local or nbytes>size or address-local>size-nbytes:
        raise ValueError('address is SRAM or outside the AXI BO')
    physical=mapping['bo_base']+address-local
    for bank in mapping['banks']:
        if bank['allocated_base']<=physical and physical+nbytes<=bank['allocated_base']+bank['allocated_size']:
            return dict(pc=bank['pc'],offset=physical-bank['base'],physical_address=physical)
    raise ValueError('request crosses or misses a physical PC range')


def export(mapping,note=None):
    rows=['TNCA_DRAM_MAP_V1',*(f'# {line}' for line in (note or [])),
          f"window {mapping['local_limit']} {mapping['bo_base']} {mapping['bo_size']}"]
    rows.extend('pc {pc} {base} {size} {allocated_base} {allocated_size}'.format(**b) for b in mapping['banks'])
    return '\n'.join(rows)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--window-pointer',required=True)
    ap.add_argument('--image-pointer',required=True)
    ap.add_argument('--image-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--plan-bo-bytes',help='export a PLANNED map of this BO size instead of the recorded one')
    args=ap.parse_args()
    data=args.source.read_bytes();document=json.loads(data)
    image=pointer(document,args.image_pointer)
    if not re.fullmatch('[0-9a-f]{64}',args.image_sha256) or image!=args.image_sha256:
        raise ValueError('mapping evidence belongs to a different physical image')
    window=pointer(document,args.window_pointer)
    planned=None
    if args.plan_bo_bytes:
        recorded,mapping=plan(window,int(args.plan_bo_bytes,0))
        note=[f"PLANNED ALLOCATION, NOT MEASURED: {mapping['bo_size']}-byte BO, PCs "
              f"{mapping['banks'][0]['pc']}-{mapping['banks'][-1]['pc']}, extrapolated from the recorded",
              f"{recorded['bo_size']}-byte window of image {image[:16]}; replace with an audit of the board record."]
        text=export(mapping,note)
        recorded_text=export(recorded)
        planned=dict(recorded_mapping=recorded,recorded_map_sha256=hashlib.sha256(recorded_text.encode()).hexdigest(),
                     recorded_topology_sha256=recorded['topology_sha256'],
                     rule='recorded local limit and BO base; PC p at connected group base + (p - first) * PC size',
                     replace_with='dram_address_map.py audit (no --plan-bo-bytes) of the board run that allocates this BO')
    else:
        mapping=normalize(window)
        text=export(mapping)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'map.txt').write_text(text)
    probes=[]
    for bank in mapping['banks']:
        for physical in (bank['allocated_base'],bank['allocated_base']+bank['allocated_size']-64):
            logical=mapping['local_limit']+physical-mapping['bo_base']
            probes.append(dict(logical_address=logical,**translate(mapping,logical)))
    report=dict(schema='tnca-dram-address-map/1',status='axi-bo-pc-map-audited',
        source=str(args.source),source_sha256=hashlib.sha256(data).hexdigest(),image_sha256=image,
        window_pointer=args.window_pointer,image_pointer=args.image_pointer,mapping=mapping,
        map_sha256=hashlib.sha256(text.encode()).hexdigest(),boundary_probes=probes,
        bank_row_mapping='unverified',calibration='not-measured',
        scope='recorded BO allocation only; a new Session allocation requires its own map')
    if planned:
        report.update(status='planned-allocation-not-measured',measured=False,planned=planned,
            scope='planned BO allocation for model development; not a recorded allocation, '
                  'not board evidence; the board run must record and audit the actual map')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],physical_pcs=[b['pc'] for b in mapping['banks']],
                          local_limit=mapping['local_limit'],bank_row_mapping=report['bank_row_mapping']),indent=2))


if __name__=='__main__':
    main()
