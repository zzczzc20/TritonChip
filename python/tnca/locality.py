"""Conservative graph storage/lifetime plan for the existing TMU.

This module never computes numerical tensor values. A captured interval is not
proof of the allocation's global last reader or complete writer coverage, so
it never enables dead-line or bypass hints. Hardware application is explicit.
"""
from dataclasses import dataclass,asdict
from .runtime import Buffer
from .tensor import TensorDescriptor
from .errors import ArgumentError

@dataclass(frozen=True)
class StorageRange:
    buffer_id: int
    address: int
    size: int
    epoch: int
    first_command: int
    last_command: int
    arguments: tuple
    aliases: bool
    nacc: int = 0xffffffff
    bypass: bool = False
    tile_lsb: int = 6
    complete_readers: bool = False
    complete_writers: bool = False

@dataclass(frozen=True)
class LocalityPlan:
    session_id: int
    epoch: int
    command_count: int
    ranges: tuple
    policy: int = 1  # Existing TMU LRU; dead/bypass policy bits remain clear.
    def report(self):
        result=asdict(self)
        result['scope']='captured command interval; storages retained by Graph; global reader/writer completeness unproved'
        result['aggressive_hints']=False
        return result


def plan_commands(commands, *, range_capacity=8):
    """Validate real runtime bindings and coalesce views by allocation identity.

    The same Session admission/alias resolver used by numerical launches runs
    before returning a plan. No command is submitted by this analysis.
    """
    if not commands:raise ArgumentError('locality requires a nonempty captured graph')
    session=commands[0][0];epoch=session.epoch;storages={}
    for index,command in enumerate(commands):
        current,module,bound,grid,values,signatures,captured_epoch,template=command
        if current is not session or captured_epoch!=epoch or session.closed or session.poisoned or not module.alive or module.epoch!=epoch:
            raise ArgumentError('locality graph has a stale or mixed session/module')
        session._resolve(module,bound)
        for name,kind in module.params:
            if kind!='device_ptr64':continue
            value=bound[name]
            if isinstance(value,TensorDescriptor):value=value.buffer
            buf,offset=value if isinstance(value,tuple) else (value,0)
            if not isinstance(buf,Buffer):raise ArgumentError('locality needs native runtime storage handles')
            # Session has checked offsets, epoch, validity and declared aliases.
            # Check storage identity too: a forged handle is not ownership proof.
            if session._buffers.get(buf.id) is not buf:
                raise ArgumentError('locality storage is not owned by this session')
            if buf.addr%64 or buf.size<=0:
                raise ArgumentError('locality registration needs aligned nonempty storage')
            key=buf.id
            if key not in storages:
                storages[key]=dict(buffer=buf,first=index,last=index,arguments=[],views=set(),aliases=False)
            entry=storages[key];entry['last']=index
            entry['arguments'].append((index,name,offset));entry['views'].add(offset)
            if sum(1 for i,_,_ in entry['arguments'] if i==index)>1:entry['aliases']=True
    if len(storages)>range_capacity:
        raise ArgumentError('locality range table capacity exceeded; no partial registration')
    ordered=sorted(storages.values(),key=lambda e:e['buffer'].addr)
    for left,right in zip(ordered,ordered[1:]):
        if left['buffer'].addr+left['buffer'].size>right['buffer'].addr:
            raise ArgumentError('distinct locality storage handles overlap')
    ranges=tuple(StorageRange(e['buffer'].id,e['buffer'].addr,e['buffer'].size,epoch,e['first'],e['last'],
        tuple(e['arguments']),e['aliases'] or len(e['views'])>1) for e in ordered)
    return LocalityPlan(session.id,epoch,len(commands),ranges)

# Metadata-only kernels use the common Session/AOT path. Numerical commands
# remain the captured Inductor artifacts, without a separate implementation.
def _read_controls(b,target):
    args=b.special('arg_base');out=b.sload(args,0);tmu=b.const(0x7fff0000)
    for index,offset in enumerate((0,0x20,0x28,0x30,0x38)):
        b.sstore(out,b.sload(tmu,offset),offset=index*8)
    b.fence();b.exit()


def _set_controls(b,target):
    args=b.special('arg_base');policy=b.sload(args,0);tmu=b.const(0x7fff0000)
    b.sstore(tmu,b.const(0),offset=0x18);b.fence()
    b.sstore(tmu,policy);b.fence();b.exit()


def _register_ranges(b,target):
    args=b.special('arg_base');ranges=b.sload(args,0);pid=b.program_id(0)
    row=b.sbin('add',ranges,b.sbini('mul',pid,16));tmu=b.const(0x7fff0000)
    b.sstore(tmu,b.sload(row,0),offset=0x100)
    b.sstore(tmu,b.sload(row,8),offset=0x108)
    desc=b.sbin('or',b.const((6<<40)|0xffffffff),b.sbini('shl',pid,36))
    b.sstore(tmu,desc,offset=0x110);b.fence();b.exit()


def _resources(session):
    import tempfile
    from pathlib import Path
    from . import aot,device
    saved=getattr(session,'_locality_resources',None)
    if saved is not None and saved['epoch']==session.epoch:return saved
    temp=tempfile.TemporaryDirectory(prefix='tnca-locality-');root=Path(temp.name)
    modules={}
    for name,callback,signature in [('read',_read_controls,{'out':'*i64'}),
            ('set',_set_controls,{'policy':'i64'}),('register',_register_ranges,{'ranges':'*i64'})]:
        path=aot.compile_aot(callback,kernel_name='locality_'+name,source_kind='tnca-ir',
            profile=device.base_profile(session.profile),signature=signature,output=str(root/(name+'.tnca'))).path
        modules[name]=session.load(path)
    saved=dict(epoch=session.epoch,temp=temp,modules=modules,stats=session.allocate(40,name='locality-controls'),
               table=session.allocate(8*16,name='locality-ranges'))
    session._locality_resources=saved
    return saved


def replay_with_locality(graph,execute):
    """Synchronous opt-in LRU scope on an exclusive supported Session.

    Starts and ends with policy zero and clears temporary range metadata.
    Nonzero pre-existing policy is refused rather than replaced. It does not
    promise to restore another owner's registrations. Only one physical client
    may own the device; board experiments must enforce the shared-card guard.
    """
    import struct,threading,time
    from .errors import CapabilityError,SessionBusy
    session=graph.commands[0][0]
    caps=session.dev.capabilities()
    if caps.get('tmu_locality') is not True:
        raise CapabilityError('backend does not implement graph locality controls')
    plan=graph.locality_plan()
    if len(plan.ranges)>caps['tmu_ranges']:
        raise CapabilityError('graph exceeds backend locality range capacity')
    with session._lock:
        session._check_usable()
        if session._measurement_owner is not None or session.unfinished:
            raise SessionBusy('locality requires an idle exclusively owned Session')
        session._measurement_owner=threading.get_ident()
    report=dict(plan=plan.report(),status='started',control_events=[],numerical_execution='captured native Inductor artifacts')
    graph.last_locality_report=report;enabled=False;started=time.perf_counter()
    try:
        resources=_resources(session)
        def command(name,args,grid=(1,1,1)):
            previous=session.latest_event
            event=session.wait(session.submit(resources['modules'][name],args,grid=grid,
                depends_on=[previous] if previous else []))
            report['control_events'].append(dict(operation=name,status=event.state,cycles=event.device_cycles))
            if not event.ok:raise event.error
        def read():
            command('read',{'out':resources['stats']})
            data=struct.unpack('<5Q',session.download(resources['stats']))
            return dict(zip(('policy','registered','tile_notices','dead','dropped'),data))
        before=read();report['before']=before
        if before['policy']!=0:
            raise CapabilityError('locality refuses an existing nonzero TMU policy')
        table=b''.join(struct.pack('<QQ',r.address,r.size) for r in plan.ranges)
        session.upload(resources['table'],table)
        # Mark cleanup necessary before submission, including partial faults.
        enabled=True;command('set',{'policy':plan.policy})
        command('register',{'ranges':resources['table']},grid=(len(plan.ranges),1,1))
        registered=read();report['registered']=registered
        if registered['policy']!=plan.policy or (registered['registered']-before['registered'])&0xffffffff!=len(plan.ranges):
            raise RuntimeError('locality hardware registration/readback mismatch')
        last=execute();session.wait(last)
        if not last.ok:raise last.error
        report['after']=read()
        command('set',{'policy':0});enabled=False
        report['restored']=read()
        if report['restored']['policy']!=0:raise RuntimeError('locality policy failed to restore')
        report['status']='completed';return last
    except BaseException as exc:
        report.update(status='failed-preserved',error=str(exc))
        # A failed numerical replay is never silently reusable.
        graph.invalid_reason=str(exc)
        if enabled and not session.poisoned and not session.closed:
            try:command('set',{'policy':0})
            except BaseException as cleanup:report['cleanup_error']=str(cleanup)
        raise
    finally:
        report['host_scope_seconds']=time.perf_counter()-started
        with session._lock:session._measurement_owner=None
