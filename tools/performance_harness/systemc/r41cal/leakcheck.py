#!/usr/bin/env python3
"""No calibration case is a hidden point: the test-case skill's leak rules applied to case files.

    python leakcheck.py --tree BASE_TREE --index SKILL/benchmark_cases/INDEX.json \
        --classes SKILL/classes.json --hidden hidden.json CASE.py ...

Run where torch is (the toolchain image), from any directory. A case fails when
  * its inputs and parameters (shapes, dtypes), or its inputs alone, equal those of a point
    of the benchmark the skill does not give (the skill's test_no_example_is_a_hidden_point), or
  * its text names a hidden point's id, its variant string or its parameter string, or a
    class the skill does not list (test_no_hidden_point_id_or_parameter_string,
    test_no_class_that_is_not_listed).
The hidden list may name points the base's registry does not have (unfrozen drafts): their
strings are checked; their shapes cannot be, and are listed for a check by hand.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


def load(path):
    spec = importlib.util.spec_from_file_location(f"case_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--hidden", type=Path, required=True)
    ap.add_argument("--skill-scripts", type=Path, required=True)
    ap.add_argument("cases", nargs="+", type=Path)
    a = ap.parse_args()
    sys.path[:0] = [str(a.tree), str(a.tree / "python"), str(a.skill_scripts)]
    import torch
    from archbench import registry

    visible = {c["point"] for c in json.loads(a.index.read_text())["cases"]}
    listed = {c["class"] for c in json.loads(a.classes.read_text())["classes"]}
    hidden_ids = set(json.loads(a.hidden.read_text()))

    def signature(m, ins):
        return tuple((tuple(t.shape), str(t.dtype)) for t in ins if isinstance(t, torch.Tensor)) + \
            tuple((tuple(t.shape), str(t.dtype)) for t in m.parameters())

    points, given, families = {}, set(), set()
    for p in registry.points():
        m, ins = p.make(0)
        sig = signature(m, ins)
        families.add(p.id.split("/", 1)[0])
        if p.id in visible:
            given |= {sig, sig[:len(ins)]}
            continue
        points.setdefault(sig, []).append(p.id)
        points.setdefault(sig[:len(ins)], []).append(p.id + " (inputs)")
    in_registry = {p.id for p in registry.points()}
    drafts = sorted(hidden_ids - in_registry)
    unlisted = sorted(f for f in families if f not in listed)
    given_strings = set()
    for pid in visible:
        v = pid.rsplit("/", 1)[1]
        given_strings |= {v, v.split("-", 1)[1] if "-" in v else v}

    bad = 0
    for path in a.cases:
        mod = load(path)
        m, ins = mod.make()
        sig = signature(m, tuple(ins))
        text = path.read_text()
        for pid in sorted(visible, key=len, reverse=True):
            text = text.replace(pid, " ")
        found = []
        if sig not in given and sig in points:
            found.append(f"workload of {points[sig]}")
        if sig[:len(ins)] not in given and sig[:len(ins)] in points:
            found.append(f"inputs of {points[sig[:len(ins)]]}")
        for pid in sorted(hidden_ids | (in_registry - visible)):
            v = pid.rsplit("/", 1)[1]
            params = v.split("-", 1)[1] if "-" in v else v
            if pid in text:
                found.append(f"names {pid}")
            if v not in given_strings and v in text:
                found.append(f"variant string of {pid}")
            if re.search(r"[a-z]", params) and params not in given_strings and \
                    re.search(rf"(?<![\w-]){re.escape(params)}(?![\w-])", text):
                found.append(f"parameter string of {pid}")
        plain = re.sub(r"\b(?:torch|F|nn)\.\w+", " ", text)
        for fam in unlisted:
            if fam in text or re.search(rf"\b{re.escape(fam.split('.', 1)[1])}\b", plain):
                found.append(f"unlisted class {fam}")
        shapes = [tuple(t.shape) for t in ins if isinstance(t, torch.Tensor)]
        print(("LEAK " if found else "ok   ") + f"{path.name:44} inputs {shapes}" + ("".join(f"\n       {f}" for f in found)))
        bad += bool(found)
    if drafts:
        print("hidden ids not in the base registry (shapes checked by hand):", ", ".join(drafts))
    print(f"{len(a.cases) - bad} of {len(a.cases)} clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
