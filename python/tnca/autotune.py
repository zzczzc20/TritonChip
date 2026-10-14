"""Bounded native-device candidate selection; never rank host wall time.

The caller supplies standard-path candidates and device-event measurements.
This policy does not create kernels or substitute numeric implementations.
"""
import hashlib
import json
import math
import time
import threading

from .device_timing import clock_span


IDENTITY_FIELDS = ('kernel_source', 'shape', 'stride', 'dtype', 'software_tree', 'compiler',
                   'profile', 'capabilities', 'rtl', 'memory_map', 'image', 'clock_hz')


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
    if cache is not None and search_key in cache:
        result=json.loads(json.dumps(cache[search_key]))
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
        records=measure(candidate)
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
