"""archbench.run with the kfuse-speed FP8 proxy points registered (tools/r42/kfuse_speed_xl_points.py).

usage: python tools/r42/fp8sw/run_with_proxies.py [archbench.run arguments]
The proxies are the Hub FP8 kernel at reduced shapes (measurement proxies of the R42 studies, not registered in any suite).
archbench.run runs each point in a child `python -m archbench.run --child`; the child is started through this script
instead so that it registers the proxies too.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import points  # noqa: E402,F401  registers the proxy points (needs archbench.xl importable)
import archbench.run as R  # noqa: E402

_run = subprocess.run


def _run_child_here(cmd, *a, **k):
    if list(cmd[:3]) == [sys.executable, '-m', 'archbench.run']:
        cmd = [sys.executable, os.path.abspath(__file__)] + list(cmd[3:])
    return _run(cmd, *a, **k)


R.subprocess.run = _run_child_here

# FP8SW_DIGEST_DIR=DIR: record the sha256 of every output tensor of every point in DIR/<point>.digest (a bit-for-bit
# comparison of two compilers on the same inputs; the record's max_rel_err is a tolerance check, not that).
if os.environ.get('FP8SW_DIGEST_DIR'):
    import hashlib
    import archbench.platforms.tnca as T
    _cur = []
    _run_point, _compare = T.run, T.compare

    def _tracked_run(point, *a, **k):
        _cur[:] = [point.id]
        return _run_point(point, *a, **k)

    def _cmp(got, ref, tol):
        import torch
        from archbench.platforms.common import flatten
        lines = []
        for i, g in enumerate(flatten(got)):
            g = g.detach().cpu().contiguous()
            lines.append(f'{i} {tuple(g.shape)} {g.dtype} '
                         + hashlib.sha256(g.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        d = os.environ['FP8SW_DIGEST_DIR']
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, _cur[0].replace('/', '_') + '.digest'), 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        return _compare(got, ref, tol)

    T.run, T.compare = _tracked_run, _cmp

if __name__ == '__main__':
    sys.exit(R.main(sys.argv[1:]))
