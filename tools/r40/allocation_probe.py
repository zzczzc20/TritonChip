#!/usr/bin/env python3
"""Trace allocations around an unchanged standard ArchBench point.

Diagnostic only. Instrumentation changes host overhead; these samples cannot
replace formal timing records. No model, compiler or numerical op is replaced.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point', required=True)
    parser.add_argument('--profile', default='core-vrf-sync-v5')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--timed', type=int, default=30)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():parser.error('--out must be new')
    if args.warmup < 0 or args.timed < 1:parser.error('invalid sample count')
    import triton  # noqa: F401
    from tnca.runtime import Session
    from archbench import registry
    from archbench.platforms.tnca import run
    rows=[]
    allocate, retire, submit = Session.allocate, Session.retire, Session.submit

    def alloc(session, *a, **kw):
        buf=allocate(session,*a,**kw)
        rows.append(dict(op='allocate',session=session.id,epoch=session.epoch,
            name=buf.name,address=buf.addr,bytes=buf.size,
            local_limit=(getattr(session.dev,'axi_window',None) or {}).get('local_limit'),
            memory_stats=session.memory_stats))
        return buf

    def release(session, buf):
        rows.append(dict(op='retire',session=session.id,name=buf.name,address=buf.addr,bytes=buf.size))
        return retire(session,buf)

    def launch(session, module, arguments, *a, **kw):
        with session._lock:
            rows.append(dict(op='submit',session=session.id,module=module.code_addr,
                buffers=[dict(name=buf.name,address=buf.addr,bytes=buf.size)
                         for buf in session._buffers.values() if buf.alive],memory_stats=session.memory_stats))
        return submit(session,module,arguments,*a,**kw)

    with patch.object(Session,'allocate',alloc), patch.object(Session,'retire',release), \
            patch.object(Session,'submit',launch):
        result=run(registry.get(args.point),profile=args.profile,warmup=args.warmup,timed=args.timed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(schema='tnca-allocation-probe/1',
        measured_performance=False,device=os.environ.get('TNCA_TORCH_DEVICE','cpu'),
        result=result,allocations=rows),indent=2,default=str)+'\n')
    print(f"allocation probe {result['status']}: {len(rows)} records -> {args.out}",flush=True)
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
