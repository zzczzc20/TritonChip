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

if __name__ == '__main__':
    sys.exit(R.main(sys.argv[1:]))
