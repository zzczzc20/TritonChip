"""Shared checks for the R40-03 / R40-07 v3 capability auditors.

Only preserved raw evidence is judged: the run manifest written by
tools/r40/v3_capgate_jobs.py and the job logs it names, copied under
evidence/r40/v3/<gate>/. The auditor re-reads every log, re-verifies its
sha256 against the manifest and re-matches the pass marker itself; the
runner's own ok flags are not trusted.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/r40')]


class Missing(Exception):
    """Evidence not (yet) present: not-measured, never a pass."""


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def need(path):
    path = Path(path)
    if not path.exists():
        raise Missing(f'missing evidence {path}')
    return path


def delivery():
    """The v3 delivery identity (frozen plan + identity amendment)."""
    import contract_v3
    contract_v3.verify()
    plan = json.loads((ROOT / 'doc/r40_v3/plan.json').read_text())
    return contract_v3.delivery(plan), contract_v3.LOCK_SHA256


def local_tree():
    from archbench.provenance import software_tree
    return software_tree(ROOT, refresh=True)


def check(name, ok, **extra):
    return dict(name=name, ok=bool(ok), **extra)


def load_run(directory, tree):
    """(manifest, {job: (record, log text)}, checks) for one runner output copy."""
    directory = need(directory)
    manifests = sorted(directory.glob('manifest_*.json'))
    if len(manifests) != 1:
        raise Missing(f'{directory}: expected one runner manifest, found {len(manifests)}')
    m = json.loads(manifests[0].read_text())
    checks = [check('runner manifest schema', m.get('schema') == 'tnca-r40-v3-capgate-run/1',
                    manifest=str(manifests[0].relative_to(ROOT)) if manifests[0].is_relative_to(ROOT)
                    else str(manifests[0]), sha256=sha_file(manifests[0]),
                    runner_sha256=m.get('harness_hashes', {}).get('tools/r40/v3_capgate_jobs.py'))]
    if m.get('status') == 'running' or 'software_tree_after' not in m:
        raise Missing(f'{directory}: run still in progress')
    checks.append(check('run closed by the runner (complete) or finalized after a deliberate stop (partial)',
                        m.get('status') in ('complete', 'partial'), status=m.get('status'),
                        jobs_not_run=m.get('jobs_not_run', [])))
    checks.append(check('run bound to the delivery product tree before and after',
                        m['software_tree_before'] == m['software_tree_after'] == tree,
                        before=m['software_tree_before'], after=m['software_tree_after'],
                        expected=tree, host=m.get('host'), root=m.get('root')))
    origins = m.get('import_origins', {})
    checks.append(check('product imports resolved inside the immutable run copy',
                        origins.get('ok') and all(str(origins.get(k, '')).startswith(m['root'] + '/')
                                                  for k in ('tnca', 'tnca_cc'))
                        and 'tnca.triton_backend' in origins.get('backends', []),
                        origins=origins))
    checks.append(check('locked toolchain (torch 2.14.0, Triton 3.8.0)',
                        str(origins.get('torch', '')).startswith('2.14.0')
                        and origins.get('triton') == '3.8.0'))
    jobs = {}
    for rec in m.get('results', []):
        log = directory / rec['log']
        if not log.exists():
            jobs[rec['name']] = (rec, None)
            continue
        raw = log.read_bytes()
        text = raw.decode(errors='replace')
        rec = dict(rec, log_sha_ok=sha_bytes(raw) == rec['log_sha256'],
                   marker_ok=re.search(rec['marker'], text) is not None)
        jobs[rec['name']] = (rec, text)
    return m, jobs, checks


def job_checks(jobs, names, *, label):
    """Every named job present, log preserved and unmodified, exit 0, marker found."""
    out = []
    for name in names:
        rec, text = jobs.get(name, (None, None))
        if rec is None:
            out.append(check(f'{label}: {name} ran', False, why='job absent from the manifest'))
            continue
        ok = text is not None and rec['exit'] == 0 and rec['log_sha_ok'] and rec['marker_ok']
        out.append(check(f'{label}: {name}', ok, exit=rec['exit'], seconds=rec['seconds'],
                         log=rec['log'], log_sha256=rec['log_sha256'], marker=rec['marker'],
                         marker_found=rec.get('marker_ok'), log_unmodified=rec.get('log_sha_ok'),
                         cmd=rec['cmd'], env=rec['env']))
    return out


def harness_same(manifest, paths):
    """The harness files the run used equal the repository's (outside the product tree)."""
    hashes = manifest.get('harness_hashes', {})
    diff = {p: dict(run=hashes.get(p), repo=sha_file(ROOT / p) if (ROOT / p).exists() else None)
            for p in paths if hashes.get(p) != (sha_file(ROOT / p) if (ROOT / p).exists() else None)}
    return check('test harness files that ran equal the repository copies', not diff,
                 files={p: hashes.get(p) for p in paths}, differ=diff)


def lines(text, pattern):
    return re.findall(pattern, text or '', flags=re.M)


# Why a first attempt may be superseded by a rerun of the same recipe line. Each is a
# harness or host condition, never a product result: a numerical, device or protocol
# failure is not in this list and cannot be superseded.
SUPERSEDE = {
    'host wait': r'\[WAIT_TIMEOUT\] job \d+ is still RUNNING after 600\.0s',
    'reference file absent from the run copy': r"No such file or directory: '[^']*evidence/r40/R40-03/contract_reference\.json'",
    'bank ledger harness: compile-time copy launches share one dump directory':
        r'bank_tests\.py", line 70, in main\n\s+assert hashlib\.sha256\(asm\.encode\(\)\)\.hexdigest\(\) == prov\[.assembly_sha256.\]',
    'pass marker carried a stale test count': r'r40-pipeline-stage-options PASS \((\d+)/\1\)',
}
PRODUCT_FAILURE = r'KERNEL_FAULT|DEVICE_LOST|DeviceLost|mismatch|Mismatched elements|LoweringException|ResourceError|TNCALaunchError'


def merge_reruns(jobs, reruns, names, *, label):
    """(effective jobs, checks): a failed first attempt is replaced by its rerun only when
    the first log shows exactly one known harness/host cause and no product failure, and
    the rerun passes. Both logs stay in the evidence."""
    out, checks = dict(jobs), []
    for name in names:
        rec, text = jobs.get(name, (None, None))
        if rec is None or text is None:
            continue
        if rec['exit'] == 0 and rec['log_sha_ok'] and rec['marker_ok']:
            continue
        causes = [k for k, pat in SUPERSEDE.items() if re.search(pat, text)]
        product = re.search(PRODUCT_FAILURE, text)
        again, again_text = reruns.get(name + '_longhost', reruns.get(name, (None, None)))
        again_ok = (again is not None and again_text is not None and again['exit'] == 0
                    and again['log_sha_ok'] and again['marker_ok'])
        if 'host wait' in causes:
            again_ok &= again is not None and again['env'].get('TNCA_HOST_TIMEOUT') == '172800'
        ok = len(causes) == 1 and not product and again_ok
        checks.append(check(f'{label}: {name} first attempt superseded ({causes[0] if causes else "no known cause"})',
                            ok, first_log=rec['log'], first_log_sha256=rec['log_sha256'],
                            first_exit=rec['exit'], causes=causes,
                            product_failure_in_first_log=bool(product),
                            rerun=again['name'] if again else None,
                            rerun_log_sha256=again['log_sha256'] if again else None,
                            rerun_env=again['env'] if again else None))
        if ok:
            out[name] = (again, again_text)
    return out, checks


def harness_same_runs(runs, paths):
    """Each harness file equals the repository copy in the run that produced the
    result used for it: `runs` maps a path to the manifest whose job used it last."""
    diff = {}
    used = {}
    for p in paths:
        m = runs.get(p)
        run_hash = (m or {}).get('harness_hashes', {}).get(p)
        repo = sha_file(ROOT / p) if (ROOT / p).exists() else None
        used[p] = run_hash
        if run_hash != repo:
            diff[p] = dict(run=run_hash, repo=repo)
    return check('test harness files behind each accepted result equal the repository copies',
                 not diff, files=used, differ=diff)
