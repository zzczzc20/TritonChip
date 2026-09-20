"""
Saving a failure so it can be run again -- R2-12 OBS-03.

    "保存失败 artifact、参数元数据、输入/seed、trace 与重放命令，在同锁定环境
     复现相同语义错误；不要求复现 host PID/时间戳."

A bundle is a directory, and it is deliberately not a pickle: everything in
it is either JSON a person can read or a `.bin` they can hexdump. What goes
in is what the launch was -- the artifact, the parameters with their types,
the bytes of every input buffer, the grid, the profile, the seeds, and the
transport trace up to the failure. What stays out is anything that would
make the replay a different run: the host PID, the wall clock, the
temporary paths. The contract says those need not reproduce, and a bundle
that recorded them would invite a reader to think they should.

The replay is the test of the bundle. `replay()` starts a new session from
the bundle alone and reports what happened; if the error code and the fault
detail come back the same, the bundle captured enough.
"""
import json
import os
import shutil
import struct
import sys

from .errors import TncaError

BUNDLE_VERSION = 1
INDEX = 'bundle.json'


def _buffers_dir(bundle):
    return os.path.join(bundle, 'buffers')


def capture(session, event, outdir, *, note=None):
    """Write everything needed to run this launch again.

    Called with the event that failed. The buffers are read back from the
    device as they were at the failure, not as the host last wrote them:
    a kernel that corrupted its input is a thing worth replaying.
    """
    task = session.task_of(event)
    if task is None:
        raise TncaError(
            'this session does not know that event',
            code='ARGUMENT_INVALID',
            remedy='capture the event returned by submit(), from the '
                   'session that submitted it')

    os.makedirs(_buffers_dir(outdir), exist_ok=True)
    module = task['module']
    artifact_name = 'kernel' + os.path.splitext(module.path)[1]
    shutil.copy2(module.path, os.path.join(outdir, artifact_name))

    params = []
    for name, typ in module.params:
        given = task['args'].get(name)
        entry = {'name': name, 'type': typ}
        if typ == 'device_ptr64' and hasattr(given, 'addr'):
            blob = b''
            try:
                blob = session.download(given, given.size)
            except TncaError:
                # The device may be gone; the bundle says so rather than
                # pretending the buffer was empty.
                entry['contents'] = 'unavailable'
            if blob:
                fn = f"{name}.bin"
                with open(os.path.join(_buffers_dir(outdir), fn), 'wb') as fh:
                    fh.write(blob)
                entry['contents'] = fn
            entry['size'] = given.size
            entry['addr'] = given.addr
        else:
            entry['value'] = given
        params.append(entry)

    err = event.error
    index = {
        'bundle_version': BUNDLE_VERSION,
        'artifact': artifact_name,
        'kernel': module.manifest['identity']['kernel_name'],
        'profile': session.profile,
        'grid': list(task['grid']),
        'parameters': params,
        'seeds': dict(session.seeds),
        'deadline_cycles': session.deadline_cycles,
        'outcome': {
            'state': event.state,
            'code': getattr(err, 'code', None),
            'message': getattr(err, 'message', None),
            'detail': getattr(err, 'detail', None),
        },
        # The trace, with the sequence numbers kept: they are what makes a
        # late reply legible after the fact.
        'trace': list(session.dev.history),
        'note': note,
        'replay_command': f"tnca-sim replay {os.path.abspath(outdir)}",
    }
    with open(os.path.join(outdir, INDEX), 'w') as fh:
        json.dump(index, fh, indent=2, sort_keys=True, default=str)
        fh.write('\n')
    return outdir


def read(bundle):
    with open(os.path.join(bundle, INDEX)) as fh:
        index = json.load(fh)
    if index.get('bundle_version') != BUNDLE_VERSION:
        raise TncaError(
            f"bundle version {index.get('bundle_version')!r} is not "
            f"{BUNDLE_VERSION}",
            code='VERSION_UNSUPPORTED',
            remedy='a bundle is replayed by the toolchain that wrote it')
    return index


def replay(bundle, *, profile=None, workdir=None):
    """Run the saved launch again and report what happened.

    Returns (index, outcome). The caller compares; this function does not
    decide whether a replay "matched", because what counts as the same
    failure is the contract's question, not the transport's.
    """
    from .runtime import Session

    index = read(bundle)
    art = os.path.join(bundle, index['artifact'])
    s = Session(profile=profile or index['profile'],
                workdir=workdir,
                deadline_cycles=index.get('deadline_cycles') or 200000)
    try:
        mod = s.load(art)
        args = {}
        for p in index['parameters']:
            if p['type'] == 'device_ptr64':
                buf = s.allocate(p['size'], name=p['name'])
                blob = p.get('contents')
                if blob and blob != 'unavailable':
                    with open(os.path.join(_buffers_dir(bundle), blob),
                              'rb') as fh:
                        s.upload(buf, fh.read())
                args[p['name']] = buf
            else:
                args[p['name']] = p['value']
        ev = s.submit(mod, args, grid=tuple(index['grid']))
        s.wait(ev)
        err = ev.error
        outcome = {
            'state': ev.state,
            'code': getattr(err, 'code', None),
            'message': getattr(err, 'message', None),
            'detail': getattr(err, 'detail', None),
        }
    finally:
        s.poisoned = True
        s.close()
    return index, outcome


def same_failure(saved, replayed):
    """Whether two outcomes are the same failure, semantically.

    The state and the stable code must match. The message may differ in the
    parts the contract exempts -- a path, a pid, a sequence number -- so it
    is not compared; the fault detail is, minus those.
    """
    if saved.get('state') != replayed.get('state'):
        return False, 'the state differs'
    if saved.get('code') != replayed.get('code'):
        return False, 'the error code differs'
    a = _semantic(saved.get('detail'))
    b = _semantic(replayed.get('detail'))
    if a != b:
        return False, f"the fault detail differs: {a} vs {b}"
    return True, 'same state, same code, same fault detail'


_VOLATILE = ('pid', 'log', 'tail', 'seq', 'path', 'workdir', 'elapsed',
             'cycles', 'host_timeout')


def _semantic(detail):
    """A fault detail with the parts that cannot reproduce removed."""
    if not isinstance(detail, dict):
        return detail
    return {k: v for k, v in sorted(detail.items())
            if k not in _VOLATILE}
