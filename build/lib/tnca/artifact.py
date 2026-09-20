"""
The artifact container -- doc/phase2_tasks_and_acceptance.md R2-05.

Layout is `spec/r2_container_v1.json` and read from it. One file:

    magic "TNCA-AOT" | major u16 | minor u16 | header_len u32 | manifest | payload

Three decisions worth stating, because each closes a case the contract tests.

**One file, not a directory.** A directory cannot be replaced atomically, so a
reader can open a manifest while a writer is halfway through replacing the code
beside it and get a manifest describing different bytes. One `os.replace` has
no such window, and CMP-04 asks precisely that a failed compile leave nothing a
loader could mistake for a finished artifact.

**No path names anywhere.** A zip or a tar carries paths, and a path in an
archive is an invitation to write outside the destination. ART-04 asks for
nested path traversal to be refused; a format with no paths cannot express the
attack, which is a better answer than sanitising it. Segment names are labels,
and are still validated -- a producer who thinks in paths should be told so.

**Every length is a claim.** `header_length` and each segment's offset and
length come from the file being read. They are checked against its actual size
before anything is allocated, because a 4-gigabyte length field in a 200-byte
file should be a diagnostic and not a MemoryError.
"""
import hashlib
import json
import os
import re
import struct
import tempfile

from . import manifest as M
from . import _data
from .errors import ArtifactError, VersionError

_SPEC_PATH = _data.path('spec', 'r2_container_v1.json',
                        env='TNCA_CONTAINER_SPEC')

with open(_SPEC_PATH) as _fh:
    SPEC = json.load(_fh)

MAGIC = SPEC['layout'][0]['value'].encode('ascii')
FORMAT_VERSION = tuple(SPEC['format_version'])
HEADER_BYTES = SPEC['header_bytes']
SEGMENT_NAME = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')

SUFFIX = '.tnca'


def check_segment_name(name):
    if not isinstance(name, str) or not SEGMENT_NAME.match(name):
        raise ArtifactError(
            f"segment name {name!r} is not a label",
            detail={'allowed': '[A-Za-z0-9_.-], 1 to 64 characters'},
            remedy='a segment name is a label, not a path')
    if name in ('.', '..') or '/' in name or '\\' in name:
        raise ArtifactError(
            f"segment name {name!r} looks like a path",
            remedy='this container has no directories; name the segment')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(manifest, segments):
    """The container bytes.

    `segments` is an ordered mapping name -> bytes. The manifest's code and
    constant segment tables are REWRITTEN here from what is actually being
    stored, so the offsets, lengths and hashes in the manifest cannot drift
    from the payload they describe. A producer that computes them separately
    has two sources of truth and eventually one of them is wrong.
    """
    payload = bytearray()
    table = []
    for name, data in segments.items():
        check_segment_name(name)
        if not isinstance(data, (bytes, bytearray)):
            raise ArtifactError(
                f"segment {name!r} is {type(data).__name__}, not bytes")
        # 8-byte alignment between segments: the ISA's word, and it means an
        # entry point offset is never inside a word.
        while len(payload) % 8:
            payload.append(0)
        table.append(dict(name=name, offset=len(payload), length=len(data),
                          sha256=sha256(bytes(data)), alignment=8))
        payload += bytes(data)

    manifest = dict(manifest)
    code = dict(manifest.get('code') or {})
    code['segments'] = [s for s in table if s['name'] != 'constants']
    code['constant_segments'] = [s for s in table if s['name'] == 'constants']
    manifest['code'] = code
    manifest = M.seal(manifest)
    M.require(manifest)

    # The SEALED manifest, including its own hash. `M.canonical` deliberately
    # omits manifest_hash -- the hash cannot cover itself -- so the stored
    # bytes are not the canonical bytes, and a reader recomputes the
    # canonical form from what it parsed rather than hashing the file slice.
    header_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=False,
                              allow_nan=False).encode('utf-8')

    out = bytearray()
    out += MAGIC
    out += struct.pack('<HH', *FORMAT_VERSION)
    out += struct.pack('<I', len(header_bytes))
    out += header_bytes
    out += bytes(payload)
    return bytes(out), manifest


def write(path, manifest, segments):
    """Write atomically: a temp file beside the target, then os.replace.

    In the destination directory rather than /tmp, so the rename is within
    one filesystem and therefore atomic. A cross-device rename is a copy,
    and a copy has the window this exists to close.
    """
    blob, sealed = build(manifest, segments)
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tnca-', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # A failed write must not leave a partial file where a loader looks,
        # and must not damage an artifact already at that path (CMP-04).
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return sealed


def read_header(path):
    """Magic, version and manifest. Does not read the payload.

    This is what `inspect` uses: no simulator, no compiler, no payload.
    """
    size = os.path.getsize(path)
    if size < HEADER_BYTES:
        raise ArtifactError(
            f"{os.path.basename(path)} is {size} bytes; the header alone is "
            f"{HEADER_BYTES}")
    with open(path, 'rb') as fh:
        head = fh.read(HEADER_BYTES)
        if head[:8] != MAGIC:
            raise ArtifactError(
                f"{os.path.basename(path)} is not a TNCA artifact",
                detail={'first_bytes': head[:8].hex()})
        major, minor = struct.unpack('<HH', head[8:12])
        if major != FORMAT_VERSION[0]:
            raise VersionError(
                f"artifact container {major}.{minor}; this runtime reads "
                f"{FORMAT_VERSION[0]}.x")
        if minor > FORMAT_VERSION[1]:
            raise VersionError(
                f"artifact container {major}.{minor} is newer than this "
                f"runtime's {FORMAT_VERSION[0]}.{FORMAT_VERSION[1]}")
        (header_len,) = struct.unpack('<I', head[12:16])
        # Before allocating anything. A length is a claim by whoever wrote
        # the file, and the file's real size is the only thing that can
        # contradict it.
        if header_len > size - HEADER_BYTES:
            raise ArtifactError(
                f"manifest claims {header_len} bytes and only "
                f"{size - HEADER_BYTES} follow the header")
        raw = fh.read(header_len)
        if len(raw) != header_len:
            raise ArtifactError(
                f"manifest is truncated: {len(raw)} of {header_len} bytes")
        text = raw.decode('utf-8', errors='strict')
        man = M.loads(text)
        M.check_version(man)
        M.require(man)
        M.check_integrity(man)
        return man, HEADER_BYTES + header_len


def segment_table(manifest):
    code = manifest.get('code') or {}
    return list(code.get('segments') or []) + \
        list(code.get('constant_segments') or [])


def load(path, *, verify=True):
    """Manifest and segments, with every hash checked before returning.

    `verify=False` exists for a test that wants to see the bytes of a
    corrupted artifact; nothing in the product uses it, because the loader
    checking hashes AFTER handing the code to the device would be a loader
    that does not check hashes.
    """
    man, payload_at = read_header(path)
    size = os.path.getsize(path)
    payload_len = size - payload_at

    table = segment_table(man)
    for s in table:
        check_segment_name(s['name'])
        off, length = s['offset'], s['length']
        if not isinstance(off, int) or not isinstance(length, int):
            raise ArtifactError(
                f"segment {s['name']!r} has a non-integer offset or length")
        if off < 0 or length < 0:
            raise ArtifactError(
                f"segment {s['name']!r}: negative offset or length")
        if off + length > payload_len:
            raise ArtifactError(
                f"segment {s['name']!r} runs to {off + length} and the "
                f"payload is {payload_len} bytes",
                remedy='the artifact is truncated, or its manifest describes '
                       'a different payload')

    # Overlap, and bytes nobody claims. Both mean the manifest and the
    # payload disagree about what is in the file, and a loader that picks a
    # winner is a loader that executes something unexamined.
    covered = sorted((s['offset'], s['offset'] + s['length'], s['name'])
                     for s in table)
    prev_end, prev_name = 0, None
    for start, end, name in covered:
        if start < prev_end:
            raise ArtifactError(
                f"segments {prev_name!r} and {name!r} overlap at byte {start}")
        prev_end, prev_name = end, name

    with open(path, 'rb') as fh:
        fh.seek(payload_at)
        payload = fh.read()

    out = {}
    for s in table:
        data = payload[s['offset']:s['offset'] + s['length']]
        if verify:
            got = sha256(data)
            if got != s['sha256']:
                raise ArtifactError(
                    f"segment {s['name']!r} does not match its hash",
                    detail={'expected': s['sha256'], 'computed': got},
                    remedy='the artifact was modified, or a segment from '
                           'another artifact was substituted')
        out[s['name']] = data

    # Padding between segments is allowed; anything after the last one is
    # not. Trailing bytes are bytes nobody checked.
    last_end = max((s['offset'] + s['length'] for s in table), default=0)
    if payload_len > last_end:
        raise ArtifactError(
            f"{payload_len - last_end} trailing byte(s) after the last "
            "segment, covered by no hash")
    return man, out
