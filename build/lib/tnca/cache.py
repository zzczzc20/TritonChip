"""
The compile cache -- doc/phase2_tasks_and_acceptance.md R2-07.

Content-addressed: the key is a hash of everything that decides what the
artifact IS, and the artifact is stored under it. What is deliberately not in
the key -- addresses, argument values, the grid, seeds -- is the reason the
cache is useful at all: changing them has to reuse the artifact, because the
kernel loads them from a block rather than being compiled for them.

Two properties worth being careful about.

**A reader never sees a half-written artifact.** Two processes compiling the
same key at the same time is the ordinary case, not the exotic one: a build
system running two targets does it. Each writes to a private temporary file
and publishes with `os.replace`, which is atomic within a filesystem, so a
reader either sees the previous entry or the complete new one. There is no
lock, because a lock introduces a way to deadlock and this does not need one:
if both processes publish, they publish identical bytes.

**A corrupt entry is refused, not consumed.** An entry whose artifact no
longer matches its manifest is a hit that would run the wrong code, so
`get()` verifies before returning and treats a failure as a miss with a
reason. What it will NOT do is rebuild: R2-07 says an explicit compile may
repair the cache and a launch may not, because a runtime that can compile is
a runtime with a compiler in it.
"""
import json
import os
import shutil
import tempfile
import time

from . import artifact
from .errors import ArtifactError, TncaError

SUFFIX = artifact.SUFFIX


class Miss:
    """Why a lookup did not hit. Never just `None`."""

    def __init__(self, reason, detail=None):
        self.reason = reason
        self.detail = detail or {}

    def __repr__(self):
        return f"<Miss {self.reason}>"

    def __bool__(self):
        return False


class Cache:
    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.trace = []

    def _entry(self, key):
        # Two levels, like git: a directory with fifty thousand entries is
        # slow to list on every filesystem that has ever existed.
        return os.path.join(self.root, key[:2], key[2:] + SUFFIX)

    def _note(self, event, key, **kw):
        rec = dict(event=event, key=key, at=round(time.time(), 3), **kw)
        self.trace.append(rec)
        return rec

    def get(self, key):
        """The artifact for `key`, or a Miss saying why not."""
        path = self._entry(key)
        if not os.path.exists(path):
            self._note('miss', key, reason='absent')
            return Miss('absent', {'path': path})
        try:
            # Verified, not merely present. An entry that no longer matches
            # its manifest is a hit that would run the wrong code.
            artifact.load(path)
        except TncaError as e:
            self._note('miss', key, reason='corrupt', message=e.message)
            return Miss('corrupt', {'path': path, 'message': e.message})
        self._note('hit', key, path=path)
        return path

    def put(self, key, source_path):
        """Publish an artifact under `key`. Atomic, and safe to race.

        Copy then replace, rather than replace in place: the caller's file is
        theirs, and a cache that moves it away surprises everyone once.
        """
        path = self._entry(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.put-',
                                   dir=os.path.dirname(path))
        os.close(fd)
        try:
            shutil.copyfile(source_path, tmp)
            with open(tmp, 'rb') as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._note('store', key, path=path,
                   bytes=os.path.getsize(path))
        return path

    def stats(self):
        hits = sum(1 for t in self.trace if t['event'] == 'hit')
        misses = sum(1 for t in self.trace if t['event'] == 'miss')
        stores = sum(1 for t in self.trace if t['event'] == 'store')
        return dict(hits=hits, misses=misses, stores=stores,
                    reasons=sorted({t.get('reason') for t in self.trace
                                    if t['event'] == 'miss'} - {None}))

    def write_trace(self, path):
        with open(path, 'w') as fh:
            json.dump(dict(root=self.root, trace=self.trace,
                           stats=self.stats()), fh, indent=2)
