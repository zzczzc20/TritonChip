#!/usr/bin/env python3
"""Compare default-profile artifacts with the frozen B39 software reference.

References were independently compiled from spec/r40/baseline.json's commit,
using the locked Triton environment and the unchanged 149-case inventory.
This checks bytes and ABI/resources; it measures no performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/r38'))
import harness as H
from tnca import aot, device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path,
                    default=ROOT / 'evidence/r40/R40-03/contract_reference.json')
    args = ap.parse_args()
    reference = json.loads(args.reference.read_text())
    baseline = json.loads((ROOT / 'spec/r40/baseline.json').read_text())
    assert reference['schema'] == 'tnca-r40-artifact-contract-reference/1'
    assert reference['software_commit'] == baseline['software']['commit']
    assert reference['measured_performance'] is False
    cases = H.cases()
    expected = {f'{name.replace("/", "_")}-O{level}.tnca'
                for name in cases for level in reference['levels']}
    assert expected == set(reference['artifacts']), 'frozen contract inventory changed'
    matched, failures = 0, []
    with tempfile.TemporaryDirectory(prefix='r40-contract-') as temporary:
        for level in reference['levels']:
            for name, case in cases.items():
                filename = f'{name.replace("/", "_")}-O{level}.tnca'
                output = Path(temporary) / filename
                try:
                    fn, signature, constexprs = case.kernel_spec
                    aot.compile_aot(fn, signature=signature, constexprs=constexprs,
                                    profile=device.base_profile(reference['profile']),
                                    options={'opt_level': level}, output=str(output))
                    digest = hashlib.sha256(output.read_bytes()).hexdigest()
                    if digest != reference['artifacts'][filename]:
                        failures.append(f'O{level} {name}: artifact differs')
                    else:
                        matched += 1
                except Exception as exc:
                    failures.append(f'O{level} {name}: {type(exc).__name__}: {str(exc)[:200]}')
    for failure in failures[:10]:
        print(f'DIFF {failure}', flush=True)
    if len(failures) > 10:
        print(f'{len(failures)-10} additional differences', flush=True)
    print(f'r40-contract-tests {"FAIL" if failures else "PASS"} '
          f'({matched}/{len(expected)} byte-identical to B39)', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
