"""Run R3.7 TileBench cases through the public AOT path with R38 options.

The same thing tools/r37/aot_run.py does -- compile_aot, .tnca, Session,
RTL, checked against the independent reference, multi-stage cases fed from
the preceding DEVICE outputs -- with two things added: a Session may be
given extra options (an explicit bench, counters=True), and benches can be
built from any RTL tree, so a changed or mutated design is measured
without touching the installed device cache.
"""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tools' / 'r37'))
import execute as E                                         # noqa: E402
import upstream_cases                                       # noqa: E402
import aot_run                                              # noqa: E402
from tnca import device                                     # noqa: E402

TILEBENCH = Path(os.environ.get('TILEBENCH_SOURCE', ROOT.parent / 'Tilebench'))


def cases(source=TILEBENCH):
    return upstream_cases.build(E.Case, Path(source).resolve())


def build_bench(profile, out, *, rtl=ROOT / 'rtl', tb=ROOT / 'tb'):
    """Compile the device bench the way tnca.device.prepare does, from `rtl`."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = (['iverilog', '-g2005', '-I', str(Path(rtl) / 'common'),
             '-y', str(Path(rtl) / 'core'), '-y', str(Path(rtl) / 'common'),
             '-y', str(Path(rtl) / 'gen'), '-y', str(Path(tb) / 'common')]
            + device.BENCH_PARAMS.get(profile, [])
            + ['-o', str(out), str(Path(tb) / 'device' / 'tnca_device.v'),
               str(Path(rtl) / 'tnca_shell.v')])
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'bench build failed: {r.stderr[-800:]}')
    return out


def rtl_copy(dest, patches=()):
    """A private copy of rtl/ and tb/, with (relative path, old, new) edits."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT / 'rtl', dest / 'rtl')
    shutil.copytree(ROOT / 'tb', dest / 'tb')
    for rel, old, new in patches:
        p = dest / rel
        text = p.read_text()
        if old not in text:
            raise RuntimeError(f'patch target not found in {rel}: {old!r}')
        p.write_text(text.replace(old, new, 1))
    return dest / 'rtl', dest / 'tb'


def run_cases(level, directory, *, only=None, profile='core-sim', session_kwargs=None,
              source=TILEBENCH, progress=True, stop_on_fail=False):
    """Every selected case at one level. Returns a list of rows."""
    all_cases = cases(source)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, captured = [], {}
    wanted = None
    if only:
        # A stage needs the stages that feed it, even when only it was asked for.
        wanted = set()
        for name in all_cases:
            if name in only or all_cases[name].kernel in only:
                wanted.add(name)
        changed = True
        while changed:
            changed = False
            for name in list(wanted):
                for feed in all_cases[name].feeds.values():
                    if feed[0] not in wanted:
                        wanted.add(feed[0])
                        changed = True
    for name, original in all_cases.items():
        if wanted is not None and name not in wanted:
            continue
        case = copy.copy(original)
        case.buffers = copy.deepcopy(original.buffers)
        row = dict(case=name, operator=case.kernel, level=level, ok=False)
        start = time.monotonic()
        try:
            for buf, feed in case.feeds.items():
                dtype, data = captured[feed[0]][feed[1]]
                if len(feed) == 3:
                    nr, nc = feed[2]
                    data = [data[r * nc + c] for c in range(nc) for r in range(nr)]
                case.buffers[buf] = (dtype, data)
            captured[name], detail = aot_run.run(name, case, level, profile, directory,
                                                 session_kwargs=session_kwargs)
            row.update(ok=True, **detail)
        except Exception as error:                        # noqa: BLE001
            row['detail'] = f'{type(error).__name__}: {error}'
        row['seconds'] = round(time.monotonic() - start, 2)
        rows.append(row)
        if progress:
            print(f'  O{level} {"PASS" if row["ok"] else "FAIL"} {name}', flush=True)
        if stop_on_fail and not row['ok']:
            break
    return rows


def run_points(points, level, directory, *, profile='core-sim', session_kwargs=None, progress=True,
               stop_on_fail=False):
    """Generated single-stage points (tools/r38/dev_cases.py). Returns rows."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, p in points.items():
        case = copy.copy(p['case'])
        case.buffers = copy.deepcopy(p['case'].buffers)
        row = dict(case=pid, operator=p['operator'], level=level, ok=False)
        start = time.monotonic()
        try:
            _, detail = aot_run.run(pid, case, level, profile, directory, session_kwargs=session_kwargs)
            row.update(ok=True, **detail)
        except Exception as error:                        # noqa: BLE001
            row['detail'] = f'{type(error).__name__}: {error}'
        row['seconds'] = round(time.monotonic() - start, 2)
        row['source'] = p['case'].kernel_spec[0].fn.__code__.co_filename
        rows.append(row)
        if progress:
            print(f'  O{level} {"PASS" if row["ok"] else "FAIL"} {pid} {row.get("cycles", "")} '
                  f'{"" if row["ok"] else row["detail"][:200]}', flush=True)
        if stop_on_fail and not row['ok']:
            break
    return rows
