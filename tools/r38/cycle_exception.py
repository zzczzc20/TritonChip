"""The author's narrow FP8 cycle exception; historical baselines stay fixed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULE = ROOT / 'spec/r38/fp8_cycle_exception.json'
APPROVED_REFERENCES = {
    (ROOT / 'spec/r38/baseline_cycles.json').resolve(),
    (ROOT / 'spec/r40/b39/core_sim_cycles.json').resolve(),
}
APPROVED_CASES = frozenset({
    'matmul_fp32_fp16_fp8/e4m3', 'matmul_fp32_fp16_fp8/e5m2',
})


def load_rule():
    rule = json.loads(RULE.read_text())
    scope = rule['scope']
    if (rule.get('schema') != 'tnca-r38-cycle-exception/1'
            or scope.get('profile') != 'core-sim' or scope.get('level') != 2
            or set(scope.get('cases', [])) != APPROVED_CASES
            or len(scope['cases']) != len(APPROVED_CASES)
            or type(scope.get('frozen_cycles')) is not int or scope['frozen_cycles'] != 10897
            or type(scope.get('allowed_extra_cycles')) is not int
            or scope.get('allowed_extra_cycles') != 1
            or not rule.get('authority', {}).get('quote')):
        raise ValueError('invalid or expanded FP8 cycle exception')
    return scope


def cycle_match(case, profile, level, frozen, actual, reference, scope):
    """Return exact, authorized, or mismatch; never accept a missing value."""
    if type(frozen) is not int or type(actual) is not int:
        return 'mismatch'
    if actual == frozen:
        return 'exact'
    if (Path(reference).resolve() in APPROVED_REFERENCES
            and profile == scope['profile'] and level == scope['level']
            and case in scope['cases'] and frozen == scope['frozen_cycles']
            and actual == frozen + scope['allowed_extra_cycles']):
        return 'authorized'
    return 'mismatch'


def compare_cycles(frozen, first, second, profile, level, reference, scope):
    """Compare both complete runs, retaining the exact frozen references."""
    problems, accepted = [], []
    for case in sorted(set(first) | set(second) | set(frozen)):
        a, b, original = first.get(case), second.get(case), frozen.get(case)
        if a != b:
            problems.append(f'{case}: not deterministic ({a} vs {b})')
        verdict = cycle_match(case, profile, level, original, a, reference, scope)
        if verdict == 'authorized':
            accepted.append((case, original, a))
        elif verdict == 'mismatch':
            problems.append(f'{case}: {a} cycles, frozen baseline says {original}')
    return problems, accepted


def counter_control_problems(instrumented, controls, accepted_cases):
    """Require numerically passing, per-program-identical uninstrumented runs."""
    problems = []
    wanted = set(accepted_cases)
    by_case = {}
    for row in controls:
        name = row.get('case')
        if name not in wanted or name in by_case:
            problems.append(f'unexpected or duplicate counter control: {name}')
        by_case[name] = row
    for name in sorted(wanted):
        on, off = instrumented.get(name), by_case.get(name)
        if not on or not off or not on.get('ok') or not off.get('ok'):
            problems.append(f'{name}: a passing uninstrumented control is required')
            continue
        on_programs, off_programs = on.get('program_cycles'), off.get('program_cycles')
        if (not isinstance(on_programs, list) or not on_programs
                or not isinstance(off_programs, list)
                or any(type(cycle) is not int for cycle in on_programs + off_programs)
                or on_programs != off_programs or on.get('cycles') != off.get('cycles')
                or sum(on_programs) != on.get('cycles')):
            problems.append(f'{name}: counters changed per-program or total cycles')
    return problems
