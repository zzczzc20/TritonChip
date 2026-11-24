#!/usr/bin/env python3
"""tools/r41/link_budget_rtl.py with one addition: the sha256 of every compared output tensor of a point is appended to
the file named by LE_DIGEST (one JSON line per comparison), so two arms of an RTL screen can be compared bit for bit
and not only through max_rel_err. Same arguments as link_budget_rtl.py (the TNCA_* environment is cleared there; settings go
through --set). Domain: RTL simulation (rtl-3c), never board.
"""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'tools/r41', ROOT, ROOT / 'python', ROOT / 'compiler/python', ROOT / 'tools/asm'):
    sys.path.insert(0, str(p))
DIGEST = os.environ.pop('LE_DIGEST', None)       # link_budget_rtl clears TNCA_* only; keep ours out of the way anyway


def tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from tensors(y)


def main():
    import link_budget_rtl as lb                                  # noqa: F401
    from archbench import registry
    from archbench.platforms import tnca as platform
    cur = {}
    orig_get, orig_cmp = registry.get, platform.compare

    def get(pid):
        cur['point'] = pid
        return orig_get(pid)

    def compare(got, ref, tol):
        if DIGEST:
            h = hashlib.sha256()
            shapes = []
            for t in tensors(got):
                import torch
                a = t.detach().cpu().contiguous()
                shapes.append([str(a.dtype), list(a.shape)])
                h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
            with open(DIGEST, 'a') as fh:
                fh.write(json.dumps(dict(point=cur.get('point'), output_sha256=h.hexdigest(), tensors=shapes)) + '\n')
        return orig_cmp(got, ref, tol)

    registry.get, platform.compare = get, compare
    lb.main()


if __name__ == '__main__':
    main()
