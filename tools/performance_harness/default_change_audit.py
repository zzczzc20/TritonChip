#!/usr/bin/env python3
"""R41 Stage 1: compile-only audit of default-on compiler changes against B40-OBS.

    default_change_audit.py run --source-root TREE --software-tree files:<sha> \
        --label r40 --out DIR [--jobs 4]
    default_change_audit.py compare --control DIR/report.json \
        --candidate DIR/report.json [--ablation LABEL=DIR/report.json ...] \
        --out audit.json

`run` compiles every original point (spec/r40/suites/hbm.json, the union of
the 113 frozen points) with the delivered default compile knobs of B40-S
(tools/r40/v3_compile_preflight.py BASE), each point in a fresh process with
private Triton/Inductor/temporary directories. Tensors live on the `tnca`
PyTorch device over a host-memory stand-in Session (v3_compile_preflight's
HostSession): no board, simulator or device session is opened, and every
launch is skipped (TNCA_LAUNCH=skip).

Recorded per steady-call launch: kernel and grid. Per launched kernel: the
sealed artifact's text segment (sha256 and bytes), its launch structure
(manifest execution, entry points, semantic options, constexprs, manifest
resources) and the Triton source Inductor generated for it. Artifact hashes
are not compared: manifests gained R41 provenance and differ regardless.

`compare` reports, per point, whether any kernel's code text or the graph's
launch structure differs, and which kernels. With identical text and launch
structure a point executes the same instructions on the same image; it cannot
regress there. Not measured: numerics, execution, performance.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]/'r40'))
from v3_compile_preflight import BASE, HARNESS, RUNTIME, HostSession  # noqa: E402

SUITE = 'spec/r40/suites/hbm.json'
SUITE_SHA256 = '7cebc8d2c6e6cab8a688418ce16e36785cd603385bff1e6028a0657c89120dfd'
THREADS = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
# Provenance entries that name a selection, retile or fallback decision.
DECISIONS = ('predicate_spills', 'mma_retiles', 'register_blocks', 'rctx_spmd', 'rctx_roles',
             'schedule_slack', 'vector_spill')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def heavy(point_id):
    tier = point_id.split('/')[1]
    return tier in ('large', 'hbm')


# ---------------------------------------------------------------- child ----
def child(a):
    root = Path(a.source_root).resolve()
    out = Path(a.run_dir)
    res = dict(point=a.point, point_sha256=a.point_sha256, status='incomplete',
               knobs={k: os.environ.get(k) for k in sorted(BASE)}, pid=os.getpid())
    t_start = time.monotonic()

    def save():
        tmp = out/'result.tmp'
        tmp.write_text(json.dumps(res, indent=1, default=str)+'\n')
        tmp.replace(out/'result.json')
    save()
    try:
        sys.path[:0] = [str(root), str(root/'python'), str(root/'compiler/python'), str(root/'tools/asm')]
        import torch
        import triton
        import tnca
        import tnca.inductor
        import tnca_cc
        from tnca.triton_backend import driver
        from tnca.artifact import load
        from archbench import registry
        from archbench.provenance import software_tree
        from archbench.platforms.tnca import _wrapper_launches
        from torch._inductor.utils import run_and_get_code
        for name, mod in dict(tnca=tnca, tnca_cc=tnca_cc, driver=driver, archbench=registry).items():
            if not Path(mod.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f'{name} imported from {mod.__file__}, outside {root}')
        tree = software_tree(root, refresh=True)
        if tree != a.software_tree:
            raise RuntimeError(f'software tree {tree} != {a.software_tree}')
        res.update(software_tree=tree, toolchain=dict(torch=str(torch.__version__),
                                                      triton=str(triton.__version__)))
        point = registry.get(a.point)
        if point.definition_sha256() != a.point_sha256:
            raise RuntimeError(f'{a.point}: definition {point.definition_sha256()} != {a.point_sha256}')

        def forbidden(*args, **kwargs):
            raise RuntimeError('compile audit forbids every device session')
        driver._Sessions.get = classmethod(forbidden)
        captured = []
        original_call = driver.TNCALauncher.__call__

        def call(self, *args, **kwargs):
            assert os.environ['TNCA_LAUNCH'] == 'skip'
            captured.append(Path(args[4] if len(args) > 4 else kwargs['function']))
            return original_call(self, *args, **kwargs)
        driver.TNCALauncher.__call__ = call

        res['inductor_config'] = tnca.inductor.configure()
        import tnca.torch_device as tdev
        stub = HostSession(os.environ['TNCA_PROFILE'])

        def session(index=0):
            if index != 0:
                raise RuntimeError(f'tnca:{index}: one device')
            return stub
        tdev.MEMORY.session = session
        tdev.register()

        res['stage'] = 'make'
        save()
        torch._dynamo.reset()
        m, inputs = point.make(0)
        m = m.eval().to('tnca')
        inputs = tuple(x.to('tnca') if isinstance(x, torch.Tensor) else x for x in inputs)
        res['stage'] = 'compile'
        save()
        compiled = torch.compile(m, fullgraph=True)
        t = time.monotonic()
        with torch.no_grad(), driver.launch_log() as cold:
            _out, codes = run_and_get_code(compiled, *inputs)
        res['compile_seconds'] = round(time.monotonic()-t, 2)
        cold_paths = list(captured)
        captured.clear()
        with torch.no_grad(), driver.launch_log() as log:
            _out = compiled(*inputs)
        res['stage'] = 'audit'
        problems = []
        if tnca.inductor.host_fallbacks(codes):
            problems.append(f'host fallback: {tnca.inductor.host_fallbacks(codes)}')
        if not log or len(log) != len(captured):
            problems.append(f'{len(log)} logged launches, {len(captured)} captured')
        expected = _wrapper_launches(codes)
        if expected is not None and expected != len(log):
            problems.append(f'{len(log)} launches for {expected} in the wrapper')
        if not all(e.get('skipped') is True for e in log + cold):
            problems.append('a launch was not skipped')
        sources = {}
        for src in codes:
            for match in re.finditer(r"async_compile\.triton\('(\w+)',\s*'''(.*?)'''", src, re.S):
                # backend_hash names the compiler sources, not the kernel.
                text = re.sub(r"'backend_hash': '[0-9A-F]+'", "'backend_hash': ''", match.group(2))
                sources[match.group(1)] = _sha(text.encode())
        kernels, launches = {}, []
        for entry, path in zip(log, captured):
            name = entry['kernel']
            launches.append(dict(kernel=name, grid=list(entry['grid'])))
            manifest, segments = load(path)
            text = segments['text']
            prov = manifest['optional_metadata']['provenance']
            asm = Path(os.environ['TNCA_DUMP_DIR'])/name/f'{name}.s'
            record = dict(
                text_sha256=_sha(text), text_bytes=len(text), artifact_sha256=_sha(path.read_bytes()),
                asm=str(asm.relative_to(out)) if asm.is_file() else None,
                triton_source_sha256=sources.get(name),
                launch_structure=dict(execution=manifest['execution'],
                                      entry_points=manifest['code']['entry_points'],
                                      options=manifest['identity']['options'],
                                      constexprs=manifest['identity']['constexprs'],
                                      resources=manifest['resources']),
                decisions={k: v for k, v in sorted(prov.items())
                           if k in DECISIONS or k.endswith(('_fallback', '_refused'))})
            old = kernels.setdefault(name, record)
            if old['text_sha256'] != record['text_sha256']:
                problems.append(f'{name}: two different artifacts launched under one name')
        cold_kernels = {}
        for entry in cold:
            cold_kernels[entry['kernel']] = cold_kernels.get(entry['kernel'], 0)+1
        res.update(launches=launches, kernels=kernels, cold_call_kernels=cold_kernels,
                   cold_call_artifacts=len(set(map(str, cold_paths))),
                   block_fallbacks=list(getattr(tnca.inductor, '_BLOCK_FALLBACKS', [])),
                   triton_sources=sources, wrapper_sha256=[_sha(c.encode()) for c in codes],
                   problems=problems)
        for i, src in enumerate(codes):
            (out/f'wrapper_{i}.py').write_text(src)
        if software_tree(root, refresh=True) != a.software_tree:
            problems.append('software tree changed during the run')
        res['status'] = 'pass' if not problems else 'fail'
        res['stage'] = 'done'
    except Exception as e:                                       # noqa: BLE001
        res.update(status='fail', error=f'{type(e).__name__}: {e}'[:4000],
                   traceback=traceback.format_exc())
    res['wall_seconds'] = round(time.monotonic()-t_start, 2)
    res['peak_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1)
    save()
    print(f'AUDIT {res["status"]} {a.point} {res["wall_seconds"]}s', flush=True)
    return 0 if res['status'] == 'pass' else 1


# ----------------------------------------------------------------- run ----
def run(a):
    root = a.source_root.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root)]
    from archbench.provenance import software_tree
    tree = software_tree(root, refresh=True)
    if tree != a.software_tree:
        raise SystemExit(f'software tree {tree} != {a.software_tree}; refusing to run')
    raw = (root/SUITE).read_bytes()
    suite = json.loads(raw)
    if _sha(raw) != SUITE_SHA256 or len(suite['points']) != 113:
        raise SystemExit('original suite changed')
    points = [p for p in suite['points'] if not a.only or p['id'] in a.only]
    report = dict(schema='tnca-r41-default-change-run/1', status='running', label=a.label,
                  reverted=a.reverted,
                  software_tree=tree, source_root=str(root), host=os.uname().nodename,
                  harness=str(HERE), harness_sha256=_sha(HERE.read_bytes()),
                  suite=dict(path=SUITE, sha256=suite['sha256'], file_sha256=_sha(raw),
                             points=len(suite['points'])),
                  knobs=BASE, harness_env=dict(HARNESS, TNCA_TORCH_DEVICE='native', **RUNTIME, **THREADS),
                  method='compile-only; fresh process and private caches per point; tnca tensors over a '
                         'host-memory stand-in session; every launch skipped; device sessions forbidden',
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), points={})
    lock = threading.Lock()

    def save():
        with lock:
            tmp = a.out/'report.tmp'
            tmp.write_text(json.dumps(report, indent=1, default=str)+'\n')
            tmp.replace(a.out/'report.json')

    def one(point):
        run_dir = a.out/'runs'/point['id'].replace('/', '__')
        run_dir.mkdir(parents=True)
        for d in ('triton_cache', 'triton_home', 'inductor_cache', 'tmp', 'dump'):
            (run_dir/d).mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(('TNCA_', 'TRITON_', 'TORCHINDUCTOR_'))}
        env.update(HARNESS, TNCA_TORCH_DEVICE='native', **RUNTIME, **BASE, **THREADS)
        if a.torch_extensions_dir:
            env['TORCH_EXTENSIONS_DIR'] = str(a.torch_extensions_dir)
        env.update(PYTHONPATH=os.pathsep.join(str(x) for x in (root, root/'python', root/'compiler/python',
                                                               root/'tools/asm')),
                   TRITON_CACHE_DIR=str(run_dir/'triton_cache'), TRITON_HOME=str(run_dir/'triton_home'),
                   TORCHINDUCTOR_CACHE_DIR=str(run_dir/'inductor_cache'), TMPDIR=str(run_dir/'tmp'),
                   TNCA_DUMP_DIR=str(run_dir/'dump'))
        cmd = [sys.executable, '-u', str(HERE), 'child', '--source-root', str(root),
               '--software-tree', tree, '--point', point['id'], '--point-sha256', point['sha256'],
               '--run-dir', str(run_dir)]
        started = time.monotonic()
        with open(run_dir/'child.log', 'w') as fh:
            try:
                code = subprocess.run(cmd, cwd=run_dir, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        try:
            res = json.loads((run_dir/'result.json').read_text())
        except Exception:                                    # noqa: BLE001
            res = dict(status='fail', error='child left no result')
        if code == 'timeout' or res.get('status') == 'incomplete' or (code != 0 and res.get('status') == 'pass'):
            tail = (run_dir/'child.log').read_text(errors='replace')[-2000:]
            res.update(status='fail', error=res.get('error') or f'child exit {code}: {tail}')
        for d in ('triton_cache', 'inductor_cache', 'triton_home', 'tmp'):
            subprocess.run(['rm', '-rf', str(run_dir/d)])
        res['process_wall_seconds'] = round(time.monotonic()-started, 1)
        with lock:
            report['points'][point['id']] = res
        save()
        print(f'{time.strftime("%H:%M:%S")} {res["status"]:4s} {res["process_wall_seconds"]:7.1f}s '
              f'{point["id"]}' + ('' if res['status'] == 'pass' else f'\n    {str(res.get("error"))[:300]}'),
              flush=True)

    save()
    light = [p for p in points if not heavy(p['id'])]
    heavy_points = [p for p in points if heavy(p['id'])]
    # Heavy points one at a time, alongside the light queue.
    with cf.ThreadPoolExecutor(max_workers=1) as big, cf.ThreadPoolExecutor(max_workers=max(1, a.jobs-1)) as small:
        futures = [big.submit(one, p) for p in heavy_points] + [small.submit(one, p) for p in light]
        for f in futures:
            f.result()
    report['software_tree_after'] = software_tree(root, refresh=True)
    report['status'] = 'complete' if report['software_tree_after'] == tree else 'tree-changed'
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    report['summary'] = dict(points=len(report['points']),
                             passed=sum(r['status'] == 'pass' for r in report['points'].values()))
    save()
    print(json.dumps(report['summary']))
    return 0


# ------------------------------------------------------------- compare ----
def static_loop_split(asm_path):
    """Static instructions inside backward-branch loops and outside them."""
    body, labels = [], {}
    for line in Path(asm_path).read_text().splitlines():
        text = line.split('//')[0].strip()
        if not text or text.startswith('.'):
            continue
        if text.endswith(':'):
            labels[text[:-1]] = len(body)
            body.append(None)
        else:
            body.append(text)
    inside = [False] * len(body)
    for i, text in enumerate(body):
        if text and text.split()[0].startswith('br') and labels.get(text.split()[-1], i) < i:
            for j in range(labels[text.split()[-1]], i + 1):
                inside[j] = True
    return dict(in_loops=sum(1 for t, f in zip(body, inside) if t and f),
                outside_loops=sum(1 for t, f in zip(body, inside) if t and not f))


def _asm(run, pid, record):
    root = run.get('_dir')
    if not root or not record.get('asm'):
        return None
    path = Path(root)/'runs'/pid.replace('/', '__')/record['asm']
    return static_loop_split(path) if path.is_file() else None


def compare_points(control, candidate):
    """Per-point code-text and launch-structure comparison of two run reports."""
    if set(control['points']) != set(candidate['points']):
        raise ValueError('the two runs cover different points')
    rows = {}
    for pid in sorted(control['points']):
        c, n = control['points'][pid], candidate['points'][pid]
        if c['status'] != 'pass' or n['status'] != 'pass':
            rows[pid] = dict(status='not-compared', control=c['status'], candidate=n['status'],
                             error=(c.get('error') or n.get('error') or '')[:600])
            continue
        names = sorted(set(c['kernels']) | set(n['kernels']))
        changed = []
        for name in names:
            ck, nk = c['kernels'].get(name), n['kernels'].get(name)
            if ck is None or nk is None:
                changed.append(dict(kernel=name, change='kernel only in '+('candidate' if ck is None else 'control')))
                continue
            why = []
            if ck['text_sha256'] != nk['text_sha256']:
                why.append('code text')
            if ck['launch_structure'] != nk['launch_structure']:
                why.append('launch structure: '+', '.join(
                    k for k in ck['launch_structure'] if ck['launch_structure'][k] != nk['launch_structure'].get(k)))
            if why:
                changed.append(dict(
                    kernel=name, change='; '.join(why),
                    inductor_kind={'tem': 'template', 'poi': 'pointwise', 'red': 'reduction',
                                   'per': 'persistent reduction'}.get(name.split('_')[1], 'other')
                                  if name.startswith('triton_') else 'other',
                    text_delta_bytes=nk['text_bytes'] - ck['text_bytes'],
                    triton_source_same=ck['triton_source_sha256'] == nk['triton_source_sha256'],
                    control_text_bytes=ck['text_bytes'], candidate_text_bytes=nk['text_bytes'],
                    candidate_decisions=nk['decisions'],
                    control_static_instructions=_asm(control, pid, ck),
                    candidate_static_instructions=_asm(candidate, pid, nk)))
        launch_same = c['launches'] == n['launches']
        rows[pid] = dict(status='unchanged' if not changed and launch_same else 'changed',
                         kernels=len(names), steady_launches=len(n['launches']),
                         launch_sequence_same=launch_same, changed_kernels=changed,
                         text_sha256=sorted({k['text_sha256'] for k in n['kernels'].values()}))
    return rows


def compare(a):
    control = json.loads(a.control.read_text())
    candidate = json.loads(a.candidate.read_text())
    # Kernel dumps (TNCA_KERNEL_DUMP) beside each run report.
    control['_dir'], candidate['_dir'] = str(a.control.parent), str(a.candidate.parent)
    for r in (control, candidate):
        if r['status'] != 'complete':
            raise SystemExit(f'{r["label"]}: run not complete')
    if control['knobs'] != candidate['knobs'] or control['suite'] != candidate['suite']:
        raise SystemExit('the two runs used different knobs or suites')
    rows = compare_points(control, candidate)
    changed = sorted(p for p, r in rows.items() if r['status'] == 'changed')
    # An ablation is the candidate tree with one change reverted, run on the
    # changed points. Reverting the responsible change restores the control's
    # code text and launch structure exactly.
    ablations = {}
    for item in a.ablation or []:
        label, path = item.split('=', 1)
        abl = json.loads(Path(path).read_text())
        if abl['status'] != 'complete' or abl['knobs'] != control['knobs']:
            raise SystemExit(f'ablation {label}: incomplete run or different knobs')
        restored = compare_points(dict(points={p: control['points'][p] for p in abl['points']}), abl)
        ablations[label] = dict(software_tree=abl['software_tree'], reverted=abl.get('reverted'),
                                points=sorted(abl['points']),
                                restores_control=sorted(p for p, r in restored.items()
                                                        if r['status'] == 'unchanged'))
        for p in ablations[label]['restores_control']:
            if p in rows:
                rows[p].setdefault('restored_by_reverting', []).append(label)
    # Runs of other trees that must emit the candidate's code exactly
    # (e.g. the tree before a bookkeeping-only change).
    same = {}
    for item in a.same_code_as or []:
        label, path = item.split('=', 1)
        other = json.loads(Path(path).read_text())
        if other['status'] != 'complete' or other['knobs'] != candidate['knobs']:
            raise SystemExit(f'{label}: incomplete run or different knobs')
        rows_same = compare_points(other, candidate)
        same[label] = dict(software_tree=other['software_tree'],
                           points=len(rows_same),
                           unchanged=sum(r['status'] == 'unchanged' for r in rows_same.values()),
                           changed=sorted(p for p, r in rows_same.items() if r['status'] != 'unchanged'))
    out = dict(schema='tnca-r41-default-change-audit/1',
               control=dict(label=control['label'], software_tree=control['software_tree']),
               candidate=dict(label=candidate['label'], software_tree=candidate['software_tree']),
               suite=control['suite'], knobs=control['knobs'], harness_env=control['harness_env'],
               method=control['method'],
               summary=dict(points=len(rows), unchanged=sum(r['status'] == 'unchanged' for r in rows.values()),
                            changed=len(changed),
                            not_compared=sum(r['status'] == 'not-compared' for r in rows.values())),
               changed_points=changed, ablations=ablations, same_code_as=same, points=rows)
    a.out.write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out['summary']))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--source-root', type=Path, required=True)
    r.add_argument('--software-tree', required=True)
    r.add_argument('--label', required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--jobs', type=int, default=4)
    r.add_argument('--timeout', type=float, default=3600)
    r.add_argument('--only', nargs='*')
    r.add_argument('--torch-extensions-dir', type=Path)
    r.add_argument('--reverted', help='ablation: the one change reverted in this tree')
    c = sub.add_parser('child')
    for name in ('--source-root', '--software-tree', '--point', '--point-sha256', '--run-dir'):
        c.add_argument(name, required=True)
    m = sub.add_parser('compare')
    m.add_argument('--control', type=Path, required=True)
    m.add_argument('--candidate', type=Path, required=True)
    m.add_argument('--ablation', action='append')
    m.add_argument('--same-code-as', action='append', help='LABEL=report.json expected to equal the candidate')
    m.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    return dict(run=run, child=child, compare=compare)[a.cmd](a)


if __name__ == '__main__':
    raise SystemExit(main())
