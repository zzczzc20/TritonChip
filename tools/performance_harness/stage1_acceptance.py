#!/usr/bin/env python3
"""R41 Stage 1 acceptance: judge the frozen plan's required items from indexed evidence.

    stage1_acceptance.py [--root R] [--index evidence/r41/stage1/required_index.json] [--json OUT]

The frozen plan (spec/r41/stage1/plan.json, locked by spec/r41/stage1/
lock.json) names 61 required items. evidence/r41/stage1/required_index.json
maps each to its evidence files: path, sha256, kind (rtl, systemc, board,
compile, doc), the product identity the file binds to, and the kinds the
contract requires. Each item is reported as pass, fail, not-measured or
blocked. The index can only claim; this checker never upgrades a claim:

- missing, not-run, skipped or changed evidence is not a pass;
- an item that requires board evidence never passes on model, RTL or
  compile evidence alone;
- items marked final_identity must bind the current product software tree
  (archbench.provenance.software_tree of --root);
- board records are refused (fail) on the contract's section 9 rules: a
  missing original (large) point, a changed point definition (truncated
  layers, replaced weights), CPU numerical fallback, descriptors enabled but
  not taken without a reason, a 90 MHz record presented as 100 MHz, a model
  prediction presented as a board measurement, a submission time presented as
  a completion time, tuning samples inside formal samples, fitting on the
  holdout, a wrong image or configuration, and a copied historical PASS.

Exit status is zero only when all required items pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment  # noqa: E402

ROOT = HERE.parents[1]
STATUSES = ('pass', 'fail', 'not-measured', 'blocked')
KINDS = {'rtl', 'systemc', 'board', 'compile', 'doc'}
NOT_RUN = {'not-run', 'skipped', 'not-measured', 'pending', 'running', 'registered-not-run', 'inconclusive',
           'partial'}
FAILED = {'fail', 'failed', 'failed-preserved', 'error', 'stopped-on-failure', 'refused-board-identity',
          'stopped-on-preserved-failure'}
RECORD_FORMATS = {'archbench-jsonl', 'board-batch'}
CLOCK_KNOB = 'TNCA_OPERATING_CLOCK_EVIDENCE'


def path_digest(path):
    """sha256 of a file, or of a directory's sorted (relative path, file sha256) list."""
    if path.is_file():
        return experiment.digest(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        h.update(f'{child.relative_to(path)}\0{experiment.digest(child)}\n'.encode())
    return h.hexdigest()


def required_ids(plan):
    return [f'{package}.{name}' for package, names in plan['required'].items() for name in names]


class Context:
    def __init__(self, root, index_path):
        self.root = root
        self.plan_path = root/'spec/r41/stage1/plan.json'
        self.plan = json.loads(self.plan_path.read_text())
        self.index = json.loads(index_path.read_text())
        if self.index.get('plan_sha256') != experiment.digest(self.plan_path):
            raise ValueError('the evidence index is not bound to the frozen plan')
        self.ids = required_ids(self.plan)
        extra = set(self.index['items']) - set(self.ids)
        if extra:
            raise ValueError(f'index names items outside the frozen plan: {sorted(extra)}')
        self.tree = experiment.software_tree(root)
        self.board = self.index['board']
        historical = self.index.get('historical') or {}
        self.historical_trees = set(historical.get('software_trees') or [])
        self.historical_paths = tuple(historical.get('paths') or ())
        self.definitions, self.original = {}, {}
        suites = [s['path'] for s in self.plan['original_suites']] + list(self.plan['diagnostic_subsets']) + \
            [self.plan['selection_development']]
        for suite in suites:
            for point in json.loads((root/suite).read_text())['points']:
                self.definitions[point['id']] = point['sha256']
        for suite in self.plan['original_suites']:
            for point in json.loads((root/suite['path']).read_text())['points']:
                self.original[point['id']] = dict(sha256=point['sha256'], tier=point.get('tier'))
        self.holdout = set()
        for name in self.index.get('holdout') or []:
            self.holdout |= {p['id'] for p in json.loads((root/name).read_text())['points']}


def board_record_problems(rec, ctx, entry):
    """Section 9 refusals for one ArchBench record presented as board evidence."""
    out = []
    plat = rec.get('platform') or {}
    where = rec.get('point_id', '?')
    if rec.get('evidence_level') != 'measured-board' or not str(plat.get('backend', '')).startswith('fpga-'):
        out.append(f'{where}: model prediction or simulator record ({rec.get("evidence_level")}, '
                   f'{plat.get("backend")}) presented as a board measurement')
    if plat.get('software_tree') in ctx.historical_trees:
        out.append(f'{where}: copied historical PASS (software tree {plat.get("software_tree")})')
    if rec.get('status') != 'pass':
        out.append(f'{where}: status {rec.get("status")}')
    if rec.get('fallbacks'):
        out.append(f'{where}: CPU numerical fallback {rec.get("fallbacks")}')
    corr = rec.get('correctness') or {}
    if rec.get('status') == 'pass' and (corr.get('max_rel_err') is None or corr.get('tol') is None
                                        or corr['max_rel_err'] > corr['tol']):
        out.append(f'{where}: numerical comparison missing or out of tolerance')
    frozen = ctx.definitions.get(rec.get('point_id'))
    if frozen is None:
        out.append(f'{where}: not a frozen point')
    elif rec.get('point_sha256') != frozen:
        out.append(f'{where}: point definition changed (truncated layers or replaced weights)')
    model = rec.get('model_identity') or {}
    expected = entry.get('expected_model') or {}
    for field in ('layers', 'weights_sha256'):
        if field in expected and model.get(field) != expected[field]:
            out.append(f'{where}: model {field} {model.get(field)} differs from {expected[field]} '
                       '(truncated layers or replaced weights)')
    if plat.get('image_sha256') != ctx.board['image_sha256']:
        out.append(f'{where}: wrong image {plat.get("image_sha256")}')
    clock = plat.get('clock_hz')
    qualified = (plat.get('operating_clock_evidence') or {}).get('clock_hz')
    claimed = entry.get('claimed_clock_hz')
    if claimed is not None and claimed != clock:
        out.append(f'{where}: a {clock} Hz record presented as {claimed} Hz')
    if qualified is not None and clock != qualified:
        out.append(f'{where}: record labels {clock} Hz but its qualified clock evidence states {qualified} Hz')
    if clock != ctx.board['clock_hz']:
        out.append(f'{where}: clock {clock} Hz is not the Stage 1 {ctx.board["clock_hz"]} Hz')
    if plat.get('operating_clock_evidence_sha256') != ctx.board['clock_evidence_sha256']:
        out.append(f'{where}: clock evidence is not the qualified image clock')
    args = {k: v for k, v in (plat.get('backend_args') or {}).items() if k != 'xclbin'}
    expected_args = {k: v for k, v in ctx.board['backend_args'].items() if k != 'xclbin'}
    if plat.get('backend') != ctx.board['backend'] or args != expected_args:
        out.append(f'{where}: wrong configuration (backend {plat.get("backend")} {args})')
    knobs = plat.get('knobs') or {}
    for key, value in (entry.get('knobs') or {}).items():
        if knobs.get(key) != value:
            out.append(f'{where}: wrong configuration ({key}={knobs.get(key)!r}, registered {value!r})')
    if (plat.get('inductor') or {}).get('tensor_descriptors') is not True:
        out.append(f'{where}: tensor descriptors disabled on the formal path')
    descriptor = entry.get('descriptor_path') or {}
    if descriptor.get('enabled') and not descriptor.get('taken') and not descriptor.get('reason'):
        out.append(f'{where}: descriptor path enabled but not taken, without a reason')
    for name, timing in (rec.get('timings') or {}).items():
        boundary = str(timing.get('boundaries') or '')
        if boundary and 'completion' not in boundary:
            out.append(f'{where}: {name} boundary "{boundary}" is a submission time, not completion')
    roi = str((rec.get('time') or {}).get('roi') or '')
    if roi and not (roi.endswith('idle') or 'completion' in roi):
        out.append(f'{where}: device time roi "{roi}" does not end at completion')
    if 'submission' in str(entry.get('timing_boundary') or '').split(' to ')[-1]:
        out.append(f'{where}: registered timing boundary ends at submission')
    graph = len(((rec.get('provenance') or {}).get('launch_map') or {}).get('launches') or [])
    if knobs.get('TNCA_TUNE') == '1' and (rec.get('time') or {}).get('launches') != graph:
        out.append(f'{where}: tuning samples inside formal samples')
    if entry.get('samples') == 'tuning':
        out.append(f'{where}: tuning samples used as formal samples')
    return out


def records_of(path, fmt):
    if fmt == 'board-batch':
        summary = json.loads(path.read_text())
        if summary.get('status') != 'complete':
            return None, f'board batch {summary.get("status")}'
        path = path.parent/'formal.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows, None


def fit_points(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'point' and isinstance(item, str):
                yield item
            else:
                yield from fit_points(item)
    elif isinstance(value, list):
        for item in value:
            yield from fit_points(item)


FORMAT_STATUS = {'prospective-residual': ({'residual-recorded'}, {'not-comparable'}),
                 'tune-check': ({'tuning-board-qualified'}, set()),
                 'deadline-expiry': ({'poison-and-recovery-pass'}, set()),
                 'experiment-report': ({'completed-experiment'}, NOT_RUN)}


def judge(item_id, item, ctx):
    """(status, reasons, kinds) for one non-derived item."""
    claim = item.get('claim', 'not-measured')
    if claim not in STATUSES:
        return 'fail', [f'unknown claim {claim!r}'], []
    if claim == 'blocked':
        return 'blocked', [item.get('blocker') or 'blocked without a stated blocker'], []
    evidence = item.get('evidence') or []
    fails, missing = [], []
    kinds, records = set(), []
    for entry in evidence:
        path = ctx.root/entry['path']
        role = entry.get('role', 'evidence')
        if entry.get('kind') not in KINDS:
            fails.append(f'{entry["path"]}: unknown evidence kind {entry.get("kind")!r}')
            continue
        if not path.exists():
            missing.append(f'missing evidence {entry["path"]}')
            continue
        if path_digest(path) != entry.get('sha256'):
            fails.append(f'{entry["path"]}: evidence changed since it was indexed')
            continue
        binds = entry.get('binds') or {}
        bound = binds.get('software_tree')
        bound = set(bound if isinstance(bound, list) else [bound] if bound else [])
        if role != 'baseline-reference' and (str(entry['path']).startswith(ctx.historical_paths)
                                             or bound & ctx.historical_trees):
            fails.append(f'{entry["path"]}: copied historical PASS cannot stand for R41 evidence')
            continue
        status = entry.get('status')
        if status in NOT_RUN:
            missing.append(f'{entry["path"]}: {status}')
            continue
        if status in FAILED:
            fails.append(f'{entry["path"]}: evidence status {status}')
            continue
        if role == 'baseline-reference':
            continue
        fmt = entry.get('format')
        problems = []
        if entry['kind'] == 'board' and fmt not in RECORD_FORMATS | set(FORMAT_STATUS):
            missing.append(f'{entry["path"]}: board evidence must be records or a board report')
            continue
        if fmt in RECORD_FORMATS:
            rows, why = records_of(path, fmt)
            if why:
                missing.append(f'{entry["path"]}: {why}')
                continue
            if not rows:
                missing.append(f'{entry["path"]}: no records')
                continue
            for rec in rows:
                problems += board_record_problems(rec, ctx, entry)
                bound.add((rec.get('platform') or {}).get('software_tree'))
            records += rows
        elif fmt in FORMAT_STATUS:
            report = json.loads(path.read_text())
            good, pending = FORMAT_STATUS[fmt]
            if report.get('status') in pending:
                missing.append(f'{entry["path"]}: {report.get("status")}')
                continue
            if report.get('status') not in good:
                problems.append(f'{entry["path"]}: report status {report.get("status")}')
            if fmt == 'experiment-report' and entry['kind'] == 'board':
                boards = [s.get('board') for s in report.get('steps') or [] if s.get('kind') == 'fpga']
                if not boards or any((b or {}).get('image_sha256') != ctx.board['image_sha256'] for b in boards):
                    problems.append(f'{entry["path"]}: board experiment without the Stage 1 image identity')
            if report.get('software_tree'):
                bound.add(report['software_tree'])
            if (report.get('identity') or {}).get('software_tree'):
                bound.add(report['identity']['software_tree'])
        elif fmt == 'fit-report':
            report = json.loads(path.read_text())
            leaked = sorted(set(fit_points(report)) & ctx.holdout)
            if leaked:
                problems.append(f'{entry["path"]}: fitting on the holdout ({", ".join(leaked[:3])})')
            if report.get('fit_rows') != 'calibration-only':
                problems.append(f'{entry["path"]}: fit does not declare calibration-only rows')
        if problems:
            fails += problems
            continue
        if item.get('final_identity'):
            if not bound:
                missing.append(f'{entry["path"]}: no software tree binding for a final-identity item')
                continue
            stale = sorted(t for t in bound if t != ctx.tree)
            if stale:
                missing.append(f'{entry["path"]}: bound to {stale[0]}, final tree is {ctx.tree}')
                continue
        kinds.add(entry['kind'])
    reasons = []
    if item.get('coverage') == 'original_suites':
        passed = {r['point_id'] for r in records if r.get('status') == 'pass'}
        absent = sorted(set(ctx.original) - passed)
        large = [p for p in absent if ctx.original[p]['tier'] == 'large']
        if records and absent:
            fails.append(f'missing original point(s): {len(absent)}' + (f' (large: {", ".join(large[:3])})' if large else ''))
    if fails:
        return 'fail', fails, sorted(kinds)
    if claim == 'fail':
        return 'fail', [item.get('note') or 'indexed as failed'], sorted(kinds)
    absent_kinds = sorted(set(item.get('requires') or []) - kinds)
    if absent_kinds:
        if 'board' in absent_kinds and kinds & {'systemc', 'rtl', 'compile'}:
            reasons.append('requires board evidence; model/RTL/compile evidence cannot stand for a board measurement')
        else:
            reasons.append(f'requires {", ".join(absent_kinds)} evidence')
    if missing or absent_kinds or claim == 'not-measured':
        if claim == 'not-measured' and item.get('note'):
            reasons.append(item['note'])
        return 'not-measured', reasons + missing or ['no evidence'], sorted(kinds)
    if not evidence:
        return 'not-measured', ['no evidence'], []
    return 'pass', [], sorted(kinds)


def evaluate(root=ROOT, index=None):
    root = Path(root).resolve()
    ctx = Context(root, Path(index) if index else root/'evidence/r41/stage1/required_index.json')
    rows = {}
    derived = {}
    for item_id in ctx.ids:
        item = ctx.index['items'].get(item_id)
        if item is None:
            rows[item_id] = ('not-measured', ['no index entry'], [])
        elif item.get('derived_from'):
            derived[item_id] = item
        else:
            rows[item_id] = judge(item_id, item, ctx)
    for item_id, item in derived.items():
        deps = []
        for name in item['derived_from']:
            deps += [i for i in ctx.ids if i.startswith(name + '.')] if name in ctx.plan['required'] else [name]
        statuses = {d: rows[d][0] for d in deps}
        failed = sorted(d for d, s in statuses.items() if s == 'fail')
        open_ = sorted(d for d, s in statuses.items() if s != 'pass')
        if failed:
            rows[item_id] = ('fail', [f'{len(failed)} prerequisite(s) failed: {", ".join(failed[:3])}'], [])
        elif open_:
            rows[item_id] = ('blocked', [f'{len(open_)} of {len(deps)} prerequisites not passed'], [])
        else:
            rows[item_id] = ('pass', [], [])
    items = [dict(id=i, status=rows[i][0], kinds=rows[i][2], reasons=rows[i][1]) for i in ctx.ids]
    counts = {s: sum(1 for r in items if r['status'] == s) for s in STATUSES}
    return dict(schema='tnca-r41-stage1-acceptance/1', plan_sha256=experiment.digest(ctx.plan_path),
                index_sha256=experiment.digest(Path(index) if index else root/'evidence/r41/stage1/required_index.json'),
                software_tree=ctx.tree, required=len(items), counts=counts,
                status='accepted' if counts['pass'] == len(items) else 'not-accepted', items=items)


def table(result):
    lines = [f'{"ID":62s} {"STATUS":13s} {"KINDS":22s} REASON']
    for row in result['items']:
        lines.append(f'{row["id"]:62s} {row["status"]:13s} {",".join(row["kinds"]) or "-":22s} '
                     f'{(row["reasons"][0] if row["reasons"] else "")[:150]}')
    counts = ', '.join(f'{k} {v}' for k, v in result['counts'].items())
    lines.append(f'{result["required"]} required: {counts} -> {result["status"]} (software tree {result["software_tree"]})')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--index', type=Path)
    ap.add_argument('--json', type=Path, help='also write the machine-readable result')
    args = ap.parse_args()
    result = evaluate(args.root, args.index)
    print(table(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['status'] == 'accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
