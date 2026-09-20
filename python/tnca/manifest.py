"""
Manifest v1 -- doc/phase2_tasks_and_acceptance.md R2-05, section 3.1.

The schema is `spec/r2_manifest_v1.schema.json` and this reads it rather than
restating it, so a field added there is a field this refuses to find missing.

Three rules here are not what a JSON Schema validator would give, and each is
in the contract for a reason:

  - a DUPLICATE JSON KEY is rejected. `json.load` keeps the last one
    silently, so `{"profile":"core-sim","profile":"system-2x2"}` parses to
    whichever the producer wrote last and the reader never learns there were
    two. A manifest is a security-relevant description of what to execute;
    "whichever came last" is not a rule anybody agreed to.
  - an UNKNOWN FIELD is rejected, at the top level and inside every section.
    An ignorable extension has exactly one home, `optional_metadata`, and the
    loader does not read it. Accepting unknown keys anywhere else means a
    typo'd `capabilties` is an artifact with no capability requirements.
  - a HIGHER MINOR is rejected, not just a higher major. A higher minor may
    add a field this reader would place at the wrong offset or fail to check,
    and a binary interface may not guess.

`canonical()` is the serialisation the hash is taken over. It is here rather
than at the call site because two environments must produce the same bytes,
and a hash over "whatever json.dumps did by default" is a hash over a Python
version.
"""
import hashlib
import json
import os

from . import _data
from .errors import ArtifactError, VersionError

_SCHEMA_PATH = _data.path('spec', 'r2_manifest_v1.schema.json',
                          env='TNCA_MANIFEST_SCHEMA')

with open(_SCHEMA_PATH) as _fh:
    SCHEMA = json.load(_fh)

FORMAT_VERSION = tuple(SCHEMA['format_version'])
SECTIONS = SCHEMA['sections']
OPTIONAL_SECTION = 'optional_metadata'


def _no_duplicates(pairs):
    """An object_pairs_hook that refuses a repeated key."""
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ArtifactError(
                f"duplicate JSON key {key!r} in the manifest",
                detail={'first': repr(seen[key])[:80],
                        'second': repr(value)[:80]},
                remedy='a manifest says what to execute; two answers to one '
                       'question is not a manifest')
        seen[key] = value
    return seen


def loads(text):
    """Parse manifest JSON. Structure only -- `validate` checks the content."""
    try:
        return json.loads(text, object_pairs_hook=_no_duplicates)
    except ArtifactError:
        raise
    except Exception as e:
        raise ArtifactError(f"manifest is not valid JSON: {e}")


def canonical(manifest):
    """The bytes the manifest hash is taken over.

    Sorted keys, no spaces, UTF-8, and no NaN or Infinity -- those are not
    JSON, and a reader that accepts them is a reader that disagrees with
    another one.
    """
    body = {k: v for k, v in manifest.items() if k != 'manifest_hash'}
    return json.dumps(body, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def manifest_hash(manifest):
    return hashlib.sha256(canonical(manifest)).hexdigest()


def check_version(manifest):
    v = manifest.get('identity', {}).get('format_version')
    if not (isinstance(v, list) and len(v) == 2
            and all(isinstance(x, int) for x in v)):
        raise ArtifactError(
            "identity.format_version must be [major, minor] of integers",
            detail={'got': repr(v)[:60]})
    major, minor = v
    if major != FORMAT_VERSION[0]:
        raise VersionError(
            f"artifact format {major}.{minor}; this runtime reads "
            f"{FORMAT_VERSION[0]}.x",
            remedy='recompile the kernel with this version of the compiler')
    if minor > FORMAT_VERSION[1]:
        raise VersionError(
            f"artifact format {major}.{minor} is newer than this runtime's "
            f"{FORMAT_VERSION[0]}.{FORMAT_VERSION[1]}",
            detail={'why': 'a higher minor may add a field this reader would '
                           'not check; guessing is not an option for a '
                           'binary interface'})


def validate(manifest, *, where='manifest'):
    """Every required field present, nothing unknown anywhere.

    Returns the list of problems rather than raising on the first, because a
    producer fixing an artifact wants all of them, and raises only if asked
    to by `require`.
    """
    problems = []

    if not isinstance(manifest, dict):
        return [f"{where} is {type(manifest).__name__}, not an object"]

    known_sections = set(SECTIONS) | {'manifest_hash'}
    for key in sorted(manifest):
        if key not in known_sections:
            problems.append(
                f"{where}: unknown top-level key {key!r}; an ignorable "
                f"extension belongs in {OPTIONAL_SECTION!r}")

    for name, spec in SECTIONS.items():
        required = spec.get('required', {})
        if name not in manifest:
            if required:
                problems.append(f"{where}: missing section {name!r}")
            continue
        section = manifest[name]
        if not isinstance(section, dict):
            problems.append(
                f"{where}.{name} is {type(section).__name__}, not an object")
            continue
        if name == OPTIONAL_SECTION:
            continue          # free-form by contract, and never read
        for field in sorted(required):
            if field not in section:
                problems.append(f"{where}.{name}: missing {field!r}")
            elif section[field] is None:
                problems.append(
                    f"{where}.{name}.{field} is null; a required field with "
                    "no value is a missing field with extra steps")
        for field in sorted(section):
            if field not in required:
                problems.append(
                    f"{where}.{name}: unknown field {field!r}")
    return problems


def require(manifest):
    """Validate, and raise on the first problem with all of them attached."""
    check_version(manifest)
    problems = validate(manifest)
    if problems:
        raise ArtifactError(
            f"manifest is not valid v{FORMAT_VERSION[0]}."
            f"{FORMAT_VERSION[1]}: {problems[0]}",
            detail={'problems': problems[:12],
                    'total': len(problems)})
    return manifest


def check_integrity(manifest):
    """The manifest's own hash, recomputed.

    Integrity and compatibility, not authentication: this says the manifest
    is the one that was built, and says nothing about who built it.
    """
    claimed = manifest.get('manifest_hash')
    if not claimed:
        raise ArtifactError("manifest has no manifest_hash")
    got = manifest_hash(manifest)
    if got != claimed:
        raise ArtifactError(
            "manifest hash does not match its contents",
            detail={'claimed': claimed, 'computed': got})
    return got


def seal(manifest):
    """Attach the hash. The last thing a producer does."""
    manifest = dict(manifest)
    manifest.pop('manifest_hash', None)
    manifest['manifest_hash'] = manifest_hash(manifest)
    return manifest
