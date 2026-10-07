#!/usr/bin/env python3
"""Regression by identity: recompile every R3.7 case and require the same bytes.

A compiler change that leaves every existing artifact byte-identical cannot
have changed any existing kernel's behaviour or cycles, so this is the check
that a change to the compile path "does not disturb the correctness tests"
without re-simulating 447 cases. It fails on the first byte of difference and
names every case that differs.

    tools/r38/artifact_identity.py --profile core-sim --levels 0 1 2
    tools/r38/artifact_identity.py --profile core-xpose-v1 --levels 2 --ref-dir DIR

References: for core-sim, the artifact hashes recorded by the S1-01 baseline
(sim/build/r38/baseline_O<n>.json); otherwise the `.tnca` files in --ref-dir,
named as tools/r37/aot_run.py names them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import harness as H                                         # noqa: E402
from tnca import aot, device                                # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', default='core-sim')
    ap.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--ref-dir', type=Path, action='append', default=[])
    ap.add_argument('--only', nargs='+', help='case names (development; not the acceptance run)')
    args = ap.parse_args()
    cases = H.cases()
    if args.only:
        cases = {n: c for n, c in cases.items() if n in args.only}
    diffs, checked = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for level in args.levels:
            if args.profile == 'core-sim' and not args.ref_dir:
                base = json.loads((ROOT / 'sim' / 'build' / 'r38' / f'baseline_O{level}.json').read_text())
                ref = {r['case']: r['artifact_sha256'] for r in base['cases']}
            else:
                ref = {}
                for d in args.ref_dir:
                    for name in cases:
                        f = d / f'{name.replace("/", "_")}-O{level}.tnca'
                        if f.exists():
                            ref[name] = sha(f)
            for name, case in cases.items():
                if name not in ref:
                    diffs.append(f'O{level} {name}: no reference artifact')
                    continue
                fn, sig, ce = case.kernel_spec
                out = Path(tmp) / f'{name.replace("/", "_")}-O{level}.tnca'
                aot.compile_aot(fn, signature=sig, constexprs=ce, profile=device.base_profile(args.profile),
                                options={'opt_level': level}, output=str(out))
                checked += 1
                if sha(out) != ref[name]:
                    diffs.append(f'O{level} {name}: artifact differs')
    for d in diffs:
        print('DIFF', d)
    print(f'artifact identity [{args.profile}]: {checked - len([d for d in diffs if "differs" in d])}/{checked} '
          f'byte-identical; {"PASS" if not diffs else "FAIL"}')
    return 1 if diffs else 0


if __name__ == '__main__':
    sys.exit(main())
