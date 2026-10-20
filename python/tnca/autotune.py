"""Bounded native-device candidate selection; never rank host wall time.

The caller supplies standard-path candidates and device-event measurements.
This policy does not create kernels or substitute numeric implementations.
"""
import hashlib
import json
import math
import time
import threading
import os
from pathlib import Path
import tempfile
import sqlite3
import signal
import subprocess
from contextlib import contextmanager

from .device_timing import clock_span


IDENTITY_FIELDS = ('kernel_source', 'shape', 'stride', 'dtype', 'software_tree', 'compiler',
                   'profile', 'capabilities', 'rtl', 'memory_map', 'image', 'clock_hz')


class PersistentCache:
    """Atomic per-search records shared by independent tuning processes.

    Identity and candidate-set invalidation stays in ``select``. A damaged
    record is a miss; no unvalidated winner can bypass resource checks.
    Writers publish complete files with replace, so readers never see partial
    JSON. Concurrent writers may repeat a search but cannot corrupt a record.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('cache key must be a SHA256 digest')
        return self.directory / (key + '.json')

    def _read(self, key):
        try:
            record = json.loads(self._path(key).read_text())
            value = record['value']
            payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            if (record['schema'] != 'tnca-autotune-cache/1' or record['key'] != key
                    or record['sha256'] != hashlib.sha256(payload.encode()).hexdigest()):
                return None
            return value
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def __contains__(self, key):
        return self._read(key) is not None

    def __getitem__(self, key):
        value = self._read(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        path = self._path(key)
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        record = dict(schema='tnca-autotune-cache/1', key=key, value=value,
                      sha256=hashlib.sha256(payload.encode()).hexdigest())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as stream:
                temporary = stream.name
                json.dump(record, stream, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)


def identity_key(identity):
    missing=[key for key in IDENTITY_FIELDS if key not in identity]
    if missing:
        raise ValueError(f'tuning identity missing {missing}')
    if type(identity['clock_hz']) not in (int,float) or not math.isfinite(identity['clock_hz']) or identity['clock_hz']<=0:
        raise ValueError('tuning requires the declared device clock frequency')
    return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def event_seconds(records, clock_hz):
    if type(clock_hz) not in (int, float) or not math.isfinite(clock_hz) or clock_hz <= 0:
        raise ValueError('invalid device clock frequency')
    fields={'clock','epoch','session_id','start','end'}
    if not isinstance(records,(list,tuple)) or any(not isinstance(r,dict) or not fields<=r.keys() for r in records):
        raise ValueError('candidate timing must be device endpoint records')
    if any(r['clock']!='mmio-device-clock/1' for r in records):
        raise ValueError('candidate timing is not the common device clock')
    if any(type(r[k]) is not int or not 0 <= r[k] < 1 << 64
           for r in records for k in ('start', 'end')):
        raise ValueError('device endpoints must be unsigned 64-bit integers')
    span=clock_span(records)
    if span is None:
        raise ValueError('candidate has no complete device-event timing')
    return span['cycles']/clock_hz


def measure_native(fn, *, warmup=1):
    """Execute the original native graph and return its device endpoints.

    Tuning traffic is isolated from outer formal launch logs. Warmup is never
    part of the returned sample. No host duration is returned for ranking.
    """
    from .torch_device.graph import active_capture
    from .triton_backend.driver import launch_log
    if type(warmup) is not int or warmup < 0:
        raise ValueError('warmup must be a nonnegative integer')
    if active_capture() is not None:
        raise ValueError('tune before replay capture; candidates cannot enter the captured graph')
    with launch_log(isolated=True):
        for _ in range(warmup):
            fn()
    with launch_log(isolated=True) as launches:
        fn()
    if not launches or any(e['upload_bytes'] or e['download_bytes'] for e in launches):
        raise ValueError('autotune requires native resident launches without tensor staging')
    records=[entry['device_timing'] for entry in launches]
    event_seconds(records,1)  # Validate complete common-clock records now.
    return records


class GraphBudget:
    def __init__(self, *, max_seconds=600, max_candidates=16, clock=time.monotonic):
        if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or not 0<max_seconds<=600 or type(max_candidates) is not int
                or not 1<=max_candidates<=16):
            raise ValueError('R40 tuning limits are 600 seconds/graph and 16 candidates/kernel')
        self.clock=clock
        self.started=clock()
        self.max_seconds=max_seconds
        self.max_candidates=max_candidates
        self.counts={}
        self.exceeded=False
        self._lock=threading.Lock()

    def admit(self, kernel):
        with self._lock:
            if self.clock()-self.started>=self.max_seconds:
                self.exceeded=True
                return False
            count=self.counts.get(kernel,0)
            if count>=self.max_candidates:
                return False
            self.counts[kernel]=count+1
            return True

    def check_elapsed(self):
        with self._lock:
            self.exceeded |= self.clock()-self.started>self.max_seconds
            return self.exceeded

    def time_left(self):
        with self._lock:
            return max(0.0, self.max_seconds-(self.clock()-self.started))


class CandidateFailed(RuntimeError):
    """An isolated candidate did not produce valid device measurements."""


class CandidateTimedOut(TimeoutError):
    """The graph deadline terminated an isolated candidate and its children."""


def run_measurement(command, *, budget, result_path, log_path, cwd=None, env=None,
                    read_result=None, worker_record=None):
    """Compile/run an external worker within the original graph deadline.

    The worker publishes a JSON list of device endpoint records. Host elapsed
    time bounds the search; it never ranks candidates. Call from select's
    measurement callback after admission. A compiler-only caller may supply
    its own result decoder; the default always validates device endpoints.
    Every worker owns a process group, so a deadline also terminates children.
    """
    if not isinstance(command, (list, tuple)) or not command or any(not isinstance(a,str) for a in command):
        raise ValueError('candidate command must be a nonempty argument list')
    result_path,log_path=Path(result_path).absolute(),Path(log_path).absolute()
    if result_path==log_path or result_path.exists() or log_path.exists():
        raise ValueError('candidate result/log paths must be distinct and new')
    remaining=budget.time_left()
    if remaining<=0:
        budget.exceeded=True
        raise CandidateTimedOut('graph deadline reached before worker launch')
    result_path.parent.mkdir(parents=True,exist_ok=True)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    from ._child_exec import child_argv
    proc=None
    try:
        with log_path.open('xb') as log:
            proc=subprocess.Popen(child_argv(command),cwd=cwd,env=env,
                                  stdin=subprocess.DEVNULL,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            if worker_record is not None:
                worker_record['pid'] = proc.pid
            try:
                # Include process startup, compilation and publication in the
                # same deadline, rather than granting each worker 600 seconds.
                code=proc.wait(timeout=budget.time_left())
            except subprocess.TimeoutExpired as error:
                budget.exceeded=True
                raise CandidateTimedOut('candidate exceeded the graph deadline') from error
            if code:
                raise CandidateFailed(f'candidate worker exited {code}; see {log_path}')
        if budget.check_elapsed():
            raise CandidateTimedOut('candidate publication exceeded the graph deadline')
        try:
            if read_result is None:
                records=json.loads(result_path.read_text())
                if not isinstance(records,list):raise ValueError('expected a JSON list')
                event_seconds(records,1)
            else:
                records=read_result(result_path)
        except (OSError,ValueError,TypeError,KeyError) as error:
            raise CandidateFailed(f'invalid candidate device records: {error}') from error
        if budget.check_elapsed():
            raise CandidateTimedOut('candidate validation exceeded the graph deadline')
        return records
    finally:
        if proc is not None:
            # Kill the owned group even when the leader exited successfully:
            # a compiler child must not outlive its completed measurement.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()


class SharedGraphBudget(GraphBudget):
    """Cooperative graph budget shared by workers on the same Linux host.

    Use a new path for each graph tuning invocation. Reopening that path
    preserves the first worker's deadline and per-kernel admission counts.
    This admission policy does not terminate an already running measurement.
    """
    def __init__(self, path, *, max_seconds=600, max_candidates=16):
        super().__init__(max_seconds=max_seconds, max_candidates=max_candidates)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        with self._database() as db:
            db.execute('CREATE TABLE IF NOT EXISTS graph (id INTEGER PRIMARY KEY, boot TEXT, started REAL, seconds REAL, candidates INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS kernels (name TEXT PRIMARY KEY, count INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO graph VALUES (1, ?, ?, ?, ?)',
                       (boot, self.started, max_seconds, max_candidates))
            stored = db.execute('SELECT boot, started, seconds, candidates FROM graph WHERE id=1').fetchone()
            if stored[0] != boot or stored[2:] != (max_seconds, max_candidates):
                raise ValueError('shared tuning budget belongs to another boot or budget configuration')
            self.started = stored[1]
            self.counts = dict(db.execute('SELECT name, count FROM kernels'))

    @contextmanager
    def _database(self):
        # IMMEDIATE transactions serialize independent processes before they
        # inspect a count. Closing also rolls back any exceptional admission.
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        finally:
            db.close()

    def admit(self, kernel):
        with self._lock, self._database() as db:
            if self.clock() - self.started >= self.max_seconds:
                self.exceeded = True
                return False
            row = db.execute('SELECT count FROM kernels WHERE name=?', (kernel,)).fetchone()
            count = row[0] if row else 0
            if count >= self.max_candidates:
                return False
            db.execute('INSERT INTO kernels VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET count=excluded.count',
                       (kernel, count + 1))
            self.counts = dict(db.execute('SELECT name, count FROM kernels'))
            return True


def select(identity, candidates, *, capacities, capabilities, budget, measure, cache=None):
    """Measure only legal candidates; `measure` returns device endpoint records.

    Each candidate names knobs, required capabilities and estimated resources.
    Compiler/manifest resource validation remains mandatory before execution.
    The measured winner/cache contains the entire candidate, not its list index.
    """
    key=identity_key(identity)
    candidates=list(candidates)
    encoded=[json.dumps(c,sort_keys=True,allow_nan=False) for c in candidates]
    if len(set(encoded))!=len(encoded):
        raise ValueError('duplicate tuning candidates')
    available={k for k,v in capabilities.items() if v is True} if isinstance(capabilities,dict) else set(capabilities)
    context=json.dumps(dict(capacities=capacities,capabilities=sorted(available)),sort_keys=True,allow_nan=False)
    search_key=hashlib.sha256((key+'\n'+context+'\n'+'\n'.join(encoded)).encode()).hexdigest()
    if cache is not None:
        try:
            result=json.loads(json.dumps(cache[search_key], allow_nan=False))
        except (KeyError, TypeError, ValueError):
            result=None
        if (isinstance(result, dict) and result.get('identity') == identity
                and result.get('identity_key') == key and result.get('selected') in candidates
                and not result.get('budget_exceeded', True)
                and type(result.get('seconds')) in (int, float)
                and math.isfinite(result['seconds']) and result['seconds'] >= 0):
            candidate=result['selected']
            legal=(not (set(candidate.get('requires', ())) - available)
                   and all(k in capacities and type(v) is int and 0 <= v <= capacities[k]
                           for k, v in candidate.get('resources', {}).items()))
            if legal:
                result['cache_hit']=True
                return result
    rows=[]
    for candidate in candidates:
        missing=set(candidate.get('requires',()))-available
        resources=candidate.get('resources',{})
        illegal=any(k not in capacities or type(v) is not int or v<0 or v>capacities[k]
                    for k,v in resources.items())
        if missing or illegal:
            rows.append(dict(candidate=candidate,status='refused',reason='capability/resource'))
            continue
        if not budget.admit(identity['kernel_source']):
            rows.append(dict(candidate=candidate,status='not-measured',reason='budget'))
            continue
        try:
            records=measure(candidate)
        except CandidateTimedOut as error:
            budget.exceeded=True
            rows.append(dict(candidate=candidate,status='over-budget',reason=str(error)))
            continue
        except CandidateFailed as error:
            rows.append(dict(candidate=candidate,status='refused',reason=str(error)))
            continue
        # A measurement cannot supply a host elapsed time in place of endpoints.
        seconds=event_seconds(records,identity['clock_hz'])
        over=budget.check_elapsed()
        rows.append(dict(candidate=candidate,status='over-budget' if over else 'measured',
                         seconds=seconds,device_records=records))
    measured=[row for row in rows if row['status']=='measured']
    if not measured:
        raise ValueError('no legal measured candidate within the tuning budget')
    winner=min(measured,key=lambda row:row['seconds'])
    result=dict(identity=identity,identity_key=key,selected=winner['candidate'],
                seconds=winner['seconds'],rows=rows,cache_hit=False,
                tuning_seconds=budget.clock()-budget.started,budget_exceeded=budget.exceeded)
    if cache is not None and not budget.exceeded:
        cache[search_key]=json.loads(json.dumps(result))
    return result
