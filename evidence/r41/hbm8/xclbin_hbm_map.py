import sys, json, hashlib, struct
from pathlib import Path
home = Path.home()
sys.path.insert(0, str(home/'tnca_src_r40_attempt03_54249f7/python'))
from tnca.backends import u50
x = str(home/'tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin')
sha = hashlib.sha256(Path(x).read_bytes()).hexdigest()
banks = [u50._xclbin_bank(x, u50.IP_NAME, arg_index=i) for i in range(3)]
sec = u50._xclbin_sections(x)
conn = sec.get(u50._CONNECTIVITY); gconn = sec.get(u50._ASK_GROUP_CONNECTIVITY)
topo = sec.get(u50._MEM_TOPOLOGY); gtopo = sec.get(u50._ASK_GROUP_TOPOLOGY)
def mems(t):
    (n,) = struct.unpack_from('<i', t, 0); out=[]
    for i in range(n):
        mtype, used, kb, base, tag = struct.unpack_from('<BB6xQQ16s', t, 8+40*i)
        out.append(dict(index=i, type=mtype, used=used, size=kb*1024, base=base, tag=tag.split(b'\0')[0].decode()))
    return out
def conns(c):
    (n,) = struct.unpack_from('<i', c, 0)
    return [struct.unpack_from('<iii', c, 4+12*i) for i in range(n)]
pm = mems(topo); gm = mems(gtopo) if gtopo else None
pc = conns(conn); gc = conns(gconn) if gconn else None
alloc8 = u50._xclbin_hbm_allocation(x, 0, 8 << 30)
alloc4 = u50._xclbin_hbm_allocation(x, 0, 4 << 30)
res = dict(image=x, image_sha256=sha, arg_group_banks=banks,
           plain_connectivity_count=len(pc),
           plain_connectivity_args=sorted({a for a, _, _ in pc}),
           plain_connectivity_per_arg={a: sorted(pm[m]['tag'] for aa, _, m in pc if aa == a) for a in sorted({a for a, _, _ in pc})},
           group_connectivity=[dict(arg=a, ip=i, mem=m, tag=gm[m]['tag'], size=gm[m]['size']) for a, i, m in gc] if gc else None,
           hbm_topology=[m for m in pm if m['tag'].startswith('HBM')],
           allocation_8gib=alloc8, allocation_4gib_pcs=[b['pc'] for b in alloc4['physical_banks']],
           scope='static parse of the xclbin file; no XRT library loaded, no device opened, no BO allocated')
print(json.dumps(res, indent=1))
