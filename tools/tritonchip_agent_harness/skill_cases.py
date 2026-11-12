"""Export ArchBench points as the test-case skill's benchmark cases (skills/testcase-design/benchmark_cases/).

A benchmark case is a point of the benchmark given to the agent as the evaluation runs it: the module's
class and the point's own make(), copied from archbench as code only (no docstrings or comments: the
benchmark's free text can name other points' shapes) with their defaults bound, called with seed 0 (the
evaluation's), judged at the registry's tolerance, its demand the registry's declared one. Under the same
torch the case builds bit for bit the module and inputs the point does (tests/test_testcase_skill.py checks
this where torch is installed). Which points are given is the experiment's decision (the visible set); every
other point stays hidden, and the skill's leak tests hold the skill to that.

Run where torch and archbench import, for instance in the toolchain image:

    /opt/torchenv/bin/python -m tritonchip_agent_harness.skill_cases \
        --archbench REPO_ROOT --ids VISIBLE.txt --out tools/tritonchip_agent_harness/skills/testcase-design/benchmark_cases \
        [--source-commit SHA]

The output directory is replaced: one file per point and INDEX.json.
"""

from __future__ import annotations

import argparse
import ast
import dis
import inspect
import json
import re
import shutil
import sys
import textwrap
import types
from pathlib import Path

SCHEMA = "tritonchip-skill-benchmark-cases/1"
SEED = 0   # the evaluation's (archbench/run.py --seed default)


class ExportError(RuntimeError):
    pass


def file_name(point_id: str) -> str:
    """op.matmul/tiny/fp16-m16-k32-n16 -> matmul_tiny_fp16_m16_k32_n16.py (model points keep their kind)."""
    head, rest = point_id.split(".", 1)
    stem = re.sub(r"[^a-z0-9]+", "_", (rest if head == "op" else point_id).lower()).strip("_")
    return stem + ".py"


def _drop_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def code_only(source: str) -> str:
    """A definition's code without the benchmark's free text: its docstrings, and its comments (ast.unparse
    keeps none). Free text can name other points' shapes (a block "128x128" that is also a hidden point's
    input); what a class is, classes.json says."""
    return ast.unparse(_drop_docstrings(ast.parse(textwrap.dedent(source)))).rstrip() + "\n"


def _literal(value) -> bool:
    try:
        return ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError):
        return False


class Bundle:
    """The module-level names a point's code needs, gathered from the functions that use them."""

    def __init__(self) -> None:
        self.imports: dict = {}   # local name -> import line
        self.consts: dict = {}    # name -> source
        self.defs: dict = {}      # name -> source of a function or class, in the order first needed
        self.aliases: dict = {}   # name -> the exported definition it names (written after the definitions)

    def bind(self, name: str, obj) -> None:
        """A name the code uses for a value: a literal, a module, or a function or class it refers to by another name."""
        if _literal(obj):
            self.consts[name] = repr(obj)
        elif (inspect.isclass(obj) or inspect.isfunction(obj)) and name != obj.__name__:
            self._add(obj.__name__, obj)
            self.aliases[name] = obj.__name__
        else:
            self._add(name, obj)

    def need(self, fn) -> None:
        """Every module-level name `fn` uses, its nested code (a generator expression, a comprehension) included,
        which inspect.getclosurevars does not look into."""
        # Only the names the code loads as globals: co_names also holds attribute names (`.t()`), and an
        # attribute that happens to share a module-level name would drag that value in.
        names = set()
        todo = [fn.__code__]
        while todo:
            code = todo.pop()
            names |= {i.argval for i in dis.get_instructions(code) if i.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
            todo += [c for c in code.co_consts if isinstance(c, types.CodeType)]
        for name in sorted(names):
            if name in fn.__globals__:
                self.bind(name, fn.__globals__[name])

    def _add(self, name: str, obj) -> None:
        if isinstance(obj, types.ModuleType):
            mod = obj.__name__
            self.imports[name] = f"import {mod}" if name == mod else f"import {mod} as {name}"
            return
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if not (obj.__module__ or "").startswith("archbench"):
                line = f"from {obj.__module__} import {obj.__name__}"
                self.imports[name] = line + (f" as {name}" if name != obj.__name__ else "")
                return
            if name in self.defs:
                return
            self.defs[name] = None        # placed now: a recursive reference finds it
            if inspect.isclass(obj):
                for base in obj.__bases__:
                    if (base.__module__ or "").startswith("archbench"):
                        self._add(base.__name__, base)
                for attr in vars(obj).values():
                    if inspect.isfunction(attr):
                        self.need(attr)
            else:
                self.need(obj)
            self.defs[name] = code_only(inspect.getsource(obj))
            return
        if _literal(obj):
            self.consts[name] = repr(obj)
            return
        raise ExportError(f"cannot export the global {name!r} ({type(obj).__name__})")


def _function_node(fn) -> ast.FunctionDef:
    """The definition of `fn` as a function: its own `def`, or a lambda found in its module by line and argument
    names and written as one."""
    if fn.__name__ != "<lambda>":
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        if not isinstance(node, ast.FunctionDef):
            raise ExportError(f"{fn.__qualname__}: make is not a plain function")
        return node
    code = fn.__code__
    names = list(code.co_varnames[:code.co_argcount])
    tree = ast.parse(inspect.getsource(inspect.getmodule(fn)))
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda) and n.lineno == code.co_firstlineno
             and [a.arg for a in n.args.args] == names]
    if len(found) != 1:
        raise ExportError(f"{fn.__qualname__} at line {code.co_firstlineno}: {len(found)} lambdas match")
    kw = dict(name="_point_make", args=found[0].args, body=[ast.Return(found[0].body)], decorator_list=[],
              returns=None, type_comment=None)
    if "type_params" in ast.FunctionDef._fields:
        kw["type_params"] = []
    return ast.fix_missing_locations(ast.FunctionDef(**kw))


def point_make_source(fn, bundle: Bundle) -> str:
    """The point's make(), renamed _point_make, its defaults bound to their values (they were the loop's)."""
    node = _drop_docstrings(_function_node(fn))
    node.name = "_point_make"
    node.decorator_list = []
    node.args.defaults = [ast.parse(repr(v), mode="eval").body for v in (fn.__defaults__ or ())]
    node.args.kw_defaults = [ast.parse(repr(v), mode="eval").body if v is not None else None
                             for v in [(fn.__kwdefaults__ or {}).get(a.arg) for a in node.args.kwonlyargs]]
    for name, value in inspect.getclosurevars(fn).nonlocals.items():
        bundle.bind(name, value)      # the loop's values: a size, a dtype, the module class it builds
    bundle.need(fn)
    return ast.unparse(node) + "\n"


def case_source(point, source_commit: str | None) -> str:
    bundle = Bundle()
    make = point_make_source(point.make, bundle)
    module, _ = point.make(SEED)
    if inspect.isclass(type(module)) and (type(module).__module__ or "").startswith("archbench"):
        bundle._add(type(module).__name__, type(module))
    d = point.demand
    # The point's note is not copied: it is the benchmark's free text (it can name other points' shapes);
    # classes.json says what a class is.
    note = ""
    out = f" -> {point.out_dtype}" if point.out_dtype else ""
    lines = [
        '"""' + point.id,
        "",
        textwrap.fill(f"A point of the benchmark itself ({point.kind} {point.family}, {point.dtype}{out}), given to "
                      f"you as the evaluation runs it: the module and inputs make() returns are the point's own "
                      f"(seed {SEED}), judged at TOL, and DEMAND is the demand the benchmark declares for it. "
                      f"Measure it with scripts/tnca_case.py like any case; copy it to vary a dimension.", 100)
        + note,
        "",
        "Generated from the benchmark" + (f" at {source_commit[:12]}" if source_commit else "") + "; regenerated, not edited.",
        '"""',
    ]
    imports = sorted(set(bundle.imports.values()), key=lambda s: (not s.startswith("import "), s))
    body = ["\n".join(imports), "import bound  # this skill's scripts/: tnca_case.py puts it on the path", "",
            f"POINT = {point.id!r}", f"TOL = {point.tolerance!r}",
            f"DEMAND = bound.Demand(macs={d.macs!r}, fp_ops={d.fp_ops!r}, sfu_ops={d.sfu_ops!r}, "
            f"read_bytes={d.read_bytes!r}, write_bytes={d.write_bytes!r})"]
    if point.kind == "model":
        body.append("MODEL = True")
    if bundle.consts:
        body += [""] + [f"{k} = {v}" for k, v in bundle.consts.items()]
    for src in bundle.defs.values():
        body += ["", "", src.rstrip()]
    if bundle.aliases:
        body += [""] + [f"{k} = {v}" for k, v in bundle.aliases.items()]
    body += ["", "", make.rstrip(), "", "", "def make():", f"    return _point_make({SEED})", ""]
    return "\n".join(lines) + "\n" + "\n".join(body)


def export(archbench_root: Path, ids: list, out: Path, source_commit: str | None = None) -> dict:
    sys.path.insert(0, str(archbench_root))
    from archbench import registry
    frozen = set()
    for suite in (archbench_root / "archbench" / "suites").glob("*.json"):
        frozen |= {p["id"] for p in json.loads(suite.read_text(encoding="utf-8"))["points"]}
    cases = []
    sources = {}
    for pid in ids:
        point = registry.get(pid)
        name = file_name(pid)
        if name in sources:
            raise ExportError(f"two points map to {name}")
        sources[name] = case_source(point, source_commit)
        cases.append({"point": pid, "file": name, "kind": point.kind, "family": point.family, "tier": point.tier,
                      "dtype": point.dtype, "out_dtype": point.out_dtype, "tolerance": point.tolerance,
                      "draft": pid not in frozen})
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name, src in sources.items():
        (out / name).write_text(src, encoding="utf-8")
    index = {"schema": SCHEMA, "seed": SEED, "source_commit": source_commit, "cases": cases}
    (out / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archbench", type=Path, required=True, help="a tree with archbench/ (registry and suites)")
    ap.add_argument("--ids", type=Path, required=True, help="the visible point ids, one per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-commit")
    a = ap.parse_args(argv)
    ids = [line.strip() for line in a.ids.read_text().splitlines() if line.strip()]
    index = export(a.archbench.resolve(), ids, a.out, a.source_commit)
    print(f"{len(index['cases'])} cases -> {a.out} ({sum(c['draft'] for c in index['cases'])} draft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
