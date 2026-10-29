#!/usr/bin/env python3
"""Host upload path through the multicore MMIO backend on the RTL stand-in.

Uploads one payload that spans the register window and the DMA window, with
the bytes path (`transport.upload` -> `_put_payload_bytes`, the default) and
with the per-word list path it replaced (`Transport._put_payload_bytes`, kept
as the medium-independent fallback). Both must read back bit for bit from the
RTL. The Python allocation peak of each upload (tracemalloc) is reported; the
sim link's own DMA command stream is included in both and identical.

  python3 tools/r41/upload_path_tests.py [--kib 256] [--out OUT.json]
"""
from pathlib import Path
import argparse
import json
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from tnca.backends.multicore_mmio_sim import stand_in  # noqa: E402
from tnca.transport import Transport  # noqa: E402


def upload_measured(dev, addr, data):
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    reply = dev.upload(addr, data)
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return reply.ok, peak, seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kib', type=int, default=256)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    dev = stand_in(core_count=3, memory_ports=3, maintenance=True, hbm_dma=True, host_timeout=1800)
    rows = []
    try:
        dev.start()
        limit = dev.local_limit
        addr = limit - 4096                         # 512 words through the register window, the rest by DMA
        n = a.kib * 1024
        for name in ('bytes-path', 'word-list-path'):
            data = bytes((i * 131 + 7 + len(rows)) & 255 for i in range(n))
            if name == 'word-list-path':
                dev._put_payload_bytes = lambda ad, d, _dev=dev: Transport._put_payload_bytes(_dev, ad, d)
            ok, peak, seconds = upload_measured(dev, addr, data)
            if name == 'word-list-path':
                del dev._put_payload_bytes
            reply, back = dev.download(addr, n)
            rows.append(dict(path=name, payload_bytes=n, local_limit=limit, address=addr, upload_ok=ok,
                             readback_ok=bool(reply.ok and back == data), python_alloc_peak_bytes=peak,
                             upload_seconds=seconds))
    finally:
        dev.stop()
    result = dict(domain='RTL simulation (multicore MMIO stand-in, Icarus)', rows=rows,
                  passed=all(r['upload_ok'] and r['readback_ok'] for r in rows))
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text + '\n')
    print(text)
    print('UPLOAD PATH ' + ('PASS' if result['passed'] else 'FAIL'))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
