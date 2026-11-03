"""The skills catalog, and which skills a run gets (decided 2026-11-03).

    skills/<name>/SKILL.md      an Agent Skill: front matter `name` (= the directory) and
                                `description`, then the method; scripts and examples beside it

Three layers, each with one job:

- the **catalog** is the directory: every skill the harness can provide, each
  self-describing through its front matter; the hash over all of it is the
  environment version of the skills;
- a **task** declares its default set (`task.json` `skills`), the skills a
  run of it gets unless told otherwise; a task that declares none gets none;
- a **run** may override the task (`run-task --skills a,b`, `--no-skills`).

The runner stages the selected skills into a directory of the run, mounts
that directory alone (read-only, at adapters.SKILLS_DIR) and links only
those into the CLI's own skills directory. The run record names the selected
set, where it came from, each skill's hash and the catalog's version: two
runs are comparable only when their skill sets are equal (README section 13).
The prompt names the selected skills from their front matter, so no task text
knows a skill by heart. Tools travel inside a skill (`scripts/`); what needs a
mount of its own is the environment's, not a skill.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .adapters import SKILLS_DIR

SOURCE = Path(__file__).resolve().parent / "skills"
NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path
    description: str
    sha256: str

    @property
    def mounted(self) -> str:
        return f"{SKILLS_DIR}/{self.name}"


def _front_matter(skill_md: Path) -> dict:
    text = skill_md.read_text(encoding="utf-8")
    parts = text.split("---\n")
    if len(parts) < 3 or parts[0] != "":
        raise SkillError(f"{skill_md}: no front matter")
    front = {}
    for line in parts[1].strip().splitlines():
        key, sep, value = line.partition(": ")
        if not sep:
            raise SkillError(f"{skill_md}: front matter line without a value: {line!r}")
        front[key.strip()] = value.strip()
    for key in ("name", "description"):
        if not front.get(key):
            raise SkillError(f"{skill_md}: front matter has no {key}")
    return front


def _files(d: Path) -> list:
    return sorted(p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and not p.name.startswith("."))


def _hash_dir(d: Path) -> str:
    h = hashlib.sha256()
    for f in _files(d):
        h.update(f.relative_to(d).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def catalog(source: Path = SOURCE) -> dict:
    """Every skill the directory holds, by name, in name order."""
    out = {}
    if not source.is_dir():
        return out
    for d in sorted(p for p in source.iterdir() if p.is_dir() and (p / "SKILL.md").is_file()):
        front = _front_matter(d / "SKILL.md")
        if front["name"] != d.name:
            raise SkillError(f"{d}: its front matter names it {front['name']!r}")
        if not NAME.match(d.name):
            raise SkillError(f"{d}: a skill's name is lower-case letters, digits and hyphens")
        out[d.name] = Skill(name=d.name, path=d, description=front["description"], sha256=_hash_dir(d))
    return out


def version(source: Path = SOURCE) -> str:
    """The catalog's version: a hash over every skill's name and hash."""
    h = hashlib.sha256()
    for name, s in catalog(source).items():
        h.update(f"{name}\0{s.sha256}\0".encode())
    return h.hexdigest()


def select(names: Iterable[str], source: Path = SOURCE) -> list:
    """The skills these names pick, in the order given; an unknown or repeated name is refused."""
    names = list(names)
    known = catalog(source)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SkillError(f"no such skill: {', '.join(unknown)} (the catalog has: {', '.join(known) or 'none'})")
    if len(set(names)) != len(names):
        raise SkillError(f"a skill is named twice: {names}")
    return [known[n] for n in names]


def stage(skills: list, dest: Path) -> Path:
    """A fresh directory holding copies of these skills and nothing else: what the run mounts."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for s in skills:
        shutil.copytree(s.path, dest / s.name, ignore=shutil.ignore_patterns("__pycache__", ".*"))
    return dest


def manifest(skills: list) -> dict:
    return {s.name: s.sha256 for s in skills}


def prompt_lines(skills: list) -> str:
    """What the prompt says about each selected skill: its name, where it is, what it is for."""
    return "".join(f"- The `{s.name}` skill (`{s.mounted}/SKILL.md`): {s.description}\n" for s in skills)


__all__ = ["NAME", "SOURCE", "Skill", "SkillError", "catalog", "manifest", "prompt_lines", "select", "stage",
           "version"]
