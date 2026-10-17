import inspect
import hashlib
import json
from pathlib import Path
import runpy
import sys
sys.path[:0]=['python','compiler/python','tools/asm','tools/r40']
import torch
import triton
from tnca.triton_backend.driver import launch_log
last={}
compile_original=torch.compile
assert_original=torch.testing.assert_close

def compile_audited(*args, **kwargs):
    compiled=compile_original(*args, **kwargs)
    def invoke(*tensors):
        with launch_log() as launches:
            result=compiled(*tensors)
        last.update(launches=launches, shapes=[list(t.shape) if t is not None else None for t in tensors],
                    addresses=[t.data_ptr() if t is not None else None for t in tensors])
        return result
    return invoke

def assert_audited(actual, expected, **kwargs):
    try:
        return assert_original(actual, expected, **kwargs)
    except AssertionError:
        state=inspect.currentframe().f_back.f_locals
        report=dict(last, case=state.get('name'), actual=actual.flatten().tolist(), expected=expected.flatten().tolist())
        matches=[]
        for name in ('q','k','v'):
            original=state[name]
            resident=state['d'+name].cpu()
            matches.append(dict(name=name, exact=torch.equal(original,resident),
                                difference=float((original.float()-resident.float()).abs().max())))
        report['inputs_after_failure']=matches
        session=state['session']
        event=list(session._events.values())[-1]
        task=session.task_of(event)
        if task is not None:
            from tnca import artifact
            import tnca.torch_device as D
            _,segments=artifact.load(task['module'].path)
            reply,code=session.dev.download(task['module'].code_addr,len(segments['text']))
            report['code_exact']=code==segments['text']
            reply,data=session.dev.download(task['arg_addr'],len(task['block']))
            report['runtime_task']=dict(code_addr=task['module'].code_addr,
                arg_addr=task['arg_addr'],initial_block=task['block'].hex(),
                actual_block=data.hex(),download_ok=reply.ok,
                memory_stats=session.memory_stats)
            bindings={name:D.MEMORY.locate(task['args'][name]) if kind=='device_ptr64' else task['args'][name]
                      for name,kind in task['module'].params}
            replay=session.submit(task['module'],bindings,grid=task['grid'])
            session.wait(replay)
            same=state['warm'].cpu().float()
            report['same_address_replay']=dict(ok=replay.ok,values=same.flatten().tolist(),
                error=float((same-expected).abs().max()))
            path=task['module'].path
            for module in list(session._modules):session.unload(module)
            relocated=session.load(path)
            replay=session.submit(relocated,bindings,grid=task['grid'])
            session.wait(replay)
            moved=state['warm'].cpu().float()
            report['relocated_replay']=dict(code_addr=relocated.code_addr,ok=replay.ok,
                values=moved.flatten().tolist(),error=float((moved-expected).abs().max()))
        Path('/tmp/r40_sdpa_failure_probe.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        raise

torch.compile=compile_audited
torch.testing.assert_close=assert_audited
sys.argv=['online_attention_tests.py','--out','/tmp/r40_sdpa_full_probe_relocated.json']
runpy.run_path('tools/r40/online_attention_tests.py',run_name='__main__')
