#!/usr/bin/env python3
"""CPU-only measure of historical-event scans in an actual TNCA Session."""
from collections import OrderedDict
import json, platform, sys, threading, time
from tnca.runtime import Session, Event, SUCCEEDED

s=Session.__new__(Session)
s._events=OrderedDict()
s._lock=threading.RLock()
rows=[]
for n in (1000,10000,30000,60000):
    while len(s._events)<n:
        k=len(s._events)+1
        e=Event(k,0,'cpu-diagnostic')
        e.state=SUCCEEDED
        s._events[k]=e
    timings={}
    for name,fn in (('unfinished',lambda:s.unfinished),('drain',s.drain),
                    ('unload_guard',lambda:any(not e.terminal for e in s._events.values()))):
        start=time.perf_counter()
        for _ in range(100): fn()
        timings[name]=(time.perf_counter()-start)/100
    rows.append({'history_events':n,'seconds_per_scan':timings})
print(json.dumps({'domain':'CPU synthetic session-event scan; no board or model','repetitions':100,
                  'python':sys.version,'machine':platform.machine(),'rows':rows},indent=2))
