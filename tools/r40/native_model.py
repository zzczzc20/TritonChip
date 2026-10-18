#!/usr/bin/env python3
"""Frozen full-size model correctness and residency, never a performance pass.

The original seed-0 HBM model/prompt are taken directly from the registry.
Validation downloads happen outside the resident execution audit. No smaller
tier, layer count, prompt or generation-length override is provided.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'python'),str(ROOT/'compiler/python'),
               str(ROOT/'tools/asm'),str(ROOT/'tools/r40')]


def fresh_software_tree():
    """Rehash even on the frozen 8abf527 library, which predates refresh=True."""
    from archbench import provenance
    provenance._tree_cache.pop(provenance.ROOT, None)
    return provenance.software_tree()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-only', action='store_true')
    ap.add_argument('--software-tree-sha256', required=True)
    ap.add_argument('--xclbin', type=Path)
    ap.add_argument('--image-sha256')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--operating-clock-evidence',type=Path)
    args = ap.parse_args()
    if not args.reference_only and (args.xclbin is None or args.image_sha256 is None):
        ap.error('native execution requires --xclbin and --image-sha256')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', gate_status='not-measured', performance='not-measured',steps=[])
    try:
        import contract_v2
        contract_v2.verify()
        tree = fresh_software_tree()
        assert tree == args.software_tree_sha256, 'immutable library source differs'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        import triton
        import torch.nn.functional as F
        from archbench import registry
        from archbench.models import LlamaDecode
        from archbench.ops import rope_tables
        from archbench.platforms.common import compare, ieee_fp32
        from archbench.platforms.tnca import knobs
        workload = json.loads((ROOT/'spec/r40/v2/plan.json').read_text())['workloads']['llama_generate']
        point = registry.get(workload['points']['prefill']['id'])
        assert point.definition_sha256() == workload['points']['prefill']['sha256']
        base, (prompt,) = point.make(workload['model']['seed'])
        cfg = base.c
        for key in ('dim','n_layers','n_heads','n_kv_heads','ffn','vocab','eps'):
            assert getattr(cfg,key) == workload['model'][key], key
        assert list(prompt.shape) == [workload['model']['batch'],workload['prefill']]
        weight_bytes = sum(p.numel()*p.element_size() for p in base.parameters())
        assert weight_bytes == workload['model']['weight_bytes'] == 221267968
        weights_sha = hashlib.sha256()
        for name,value in base.named_parameters():
            weights_sha.update(name.encode()+b'\0')
            weights_sha.update(value.detach().contiguous().numpy().tobytes())
        report['identity'] = dict(software_tree=tree,
                                 harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 contract_sha256=contract_v2.LOCK_SHA256,
                                 torch=torch.__version__,triton=triton.__version__,python=sys.version,
                                 point=point.id,point_sha256=point.definition_sha256(),
                                 weights_sha256=weights_sha.hexdigest(),weight_bytes=weight_bytes,
                                 workload=workload)
        prefix = workload['prefill']
        new_tokens = workload['decode_steps']
        # Preserve the frozen public --new 64 prefix. Its first token comes
        # from prefill, so add one separately labelled validation decode to
        # prove the contract's literal >=64 decode-call requirement as well.
        report['requested_new_tokens'] = new_tokens
        report['expected_decode_calls'] = new_tokens
        cos,sin = rope_tables(prefix+new_tokens,cfg.hd,'fp16')

        class PrefillState(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, tokens):
                m = self.model
                x = F.embedding(tokens,m.tok)
                keys,values = [],[]
                for layer in m.layers:
                    x,k,v = layer(x,m.cos,m.sin)
                    keys.append(k)
                    values.append(v)
                logits = m._head(x[:,-1:,:])
                return logits,torch.stack(keys),torch.stack(values),logits.float().argmax(-1)

        class SharedDecode(LlamaDecode):
            def __init__(self, model, cosine, sine):
                # Share the original model's parameters, not another copy of
                # 211 MiB of weights or a per-step .to()/load_state_dict().
                torch.nn.Module.__init__(self)
                self.c,self.dtype = model.c,model.dtype
                self.tok,self.layers,self.norm,self.out = model.tok,model.layers,model.norm,model.out
                self.register_buffer('cos',cosine,persistent=False)
                self.register_buffer('sin',sine,persistent=False)

        class GrowState(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, token, keys, values):
                logits,nk,nv = self.decoder(token,keys,values)
                return logits,torch.cat((keys,nk),dim=3),torch.cat((values,nv),dim=3),logits.float().argmax(-1)

        reference_base = copy.deepcopy(base).float().eval()
        reference_prefill = PrefillState(reference_base).eval()
        reference_decode = SharedDecode(reference_base,cos[prefix:prefix+1].float(),
                                        sin[prefix:prefix+1].float()).eval()
        reference_grow = GrowState(reference_decode).eval()
        # Independent IEEE fp32 generation is kept separate from stepwise
        # references fed the device's actual tokens and KV state.
        reference_tokens,reference_margins = [],[]
        with torch.no_grad(),ieee_fp32():
            state = reference_prefill(prompt)
            for index in range(new_tokens+1):
                if index:
                    position = prefix+index-1
                    reference_decode.cos = cos[position:position+1].float()
                    reference_decode.sin = sin[position:position+1].float()
                    state = reference_grow(state[3],state[1],state[2])
                reference_tokens.append(state[3].tolist())
                top2 = state[0][:,-1,:].topk(2,dim=-1).values
                reference_margins.append((top2[:,0]-top2[:,1]).tolist())
        del state
        report['reference_tokens'] = reference_tokens[:new_tokens]
        report['supplemental_reference_token'] = reference_tokens[-1]
        report['reference_top2_margins'] = reference_margins[:new_tokens]
        report['supplemental_reference_top2_margin'] = reference_margins[-1]
        if args.reference_only:
            assert fresh_software_tree() == tree, 'software changed during reference execution'
            report['status'] = 'reference-only'
            print('FULL FROZEN MODEL REFERENCE COMPLETE',flush=True)
            return 0

        image_sha = hashlib.sha256(args.xclbin.read_bytes()).hexdigest()
        assert image_sha == args.image_sha256, 'immutable image differs'
        if args.operating_clock_evidence:
            os.environ['TNCA_OPERATING_CLOCK_EVIDENCE']=str(args.operating_clock_evidence.resolve())
        from tnca.operating_clock import from_environment
        operating_clock=from_environment(image_sha)
        config = dict(xclbin=str(args.xclbin.resolve()),image_profile='core-vrf-sync-v5',
                      core_count=3,memory_ports=3,hbm_dma=True,hbm_bytes=1<<30,device_clock=True)
        os.environ.update(TNCA_PROFILE='core-vrf-sync-v5',TNCA_BACKEND='fpga-u280-multicore',
                          TNCA_BACKEND_ARGS=json.dumps(config),TNCA_TRANSFER_POLICY='resident',
                          TNCA_TORCH_DEVICE='native',TNCA_ACTIVE_CORES='3')
        os.environ.setdefault('TNCA_HOST_TIMEOUT','7200')
        os.environ.setdefault('TNCA_DEADLINE','4000000000')
        import tnca.torch_device as D
        from tnca.triton_backend.driver import launch_log
        from frozen_residency_tests import transfers
        from torch._inductor.utils import run_and_get_code
        from tnca.inductor import host_fallbacks
        from hbm_address import topology_probe_offsets
        assert os.environ.get('TNCA_LAUNCH') != 'skip', 'actual native execution is required'
        D.register()
        session = D.MEMORY.session()
        report['identity'].update(image_sha256=image_sha,backend='fpga-u280-multicore',
                                  backend_args=config,knobs=knobs(),clock_hz=100000000,
                                  clock_qualification='requires matching route/timing and board clock reports')
        report['identity'].update(operating_clock)
        report['window'] = session.dev.axi_window
        assert report['window']['size'] >= 1<<30
        topology_probe_offsets(report['window'])
        assert len(report['window']['allocation']['physical_banks']) >= 2
        assert session.dev.physical_grid_capability()[0] == 3
        assert session.dev.capabilities()['cache_ownership_handoff']
        with transfers(session) as initialization:
            base = base.eval().to('tnca')
            dcos,dsin = cos.to('tnca'),sin.to('tnca')
            dprompt = prompt.to('tnca')
        ddecode = SharedDecode(base,dcos[prefix:prefix+1],dsin[prefix:prefix+1]).eval()
        frozen_weights = tuple((p.data_ptr(),p.untyped_storage().nbytes()) for p in base.parameters())
        assert frozen_weights == tuple((p.data_ptr(),p.untyped_storage().nbytes()) for p in ddecode.parameters())
        weight_names = {D.MEMORY.locate(p.data_ptr())[0].name for p in base.parameters()}
        report['initialization_transfers'] = initialization
        report['initial_weight_upload_bytes'] = sum(e['bytes'] for e in initialization
                                                    if e['direction']=='h2d' and e['name'] in weight_names)
        assert report['initial_weight_upload_bytes'] == weight_bytes, 'weights must be uploaded exactly once'
        torch._dynamo.config.recompile_limit = new_tokens+2
        prefill = torch.compile(PrefillState(base).eval(),fullgraph=True,dynamic=False)
        decode = torch.compile(GrowState(ddecode).eval(),fullgraph=True,dynamic=False)
        device_tokens = []
        cpu_state = None
        device_state = None
        tolerance = point.tolerance
        report['tolerance'] = tolerance
        with torch.no_grad(),ieee_fp32():
            for index in range(new_tokens+1):
                if index == 0:
                    fn,inputs = prefill,(dprompt,)
                    want = reference_prefill(prompt)
                else:
                    position = prefix+index-1
                    ddecode.cos,ddecode.sin = dcos[position:position+1],dsin[position:position+1]
                    reference_decode.cos,reference_decode.sin = cos[position:position+1].float(),sin[position:position+1].float()
                    fn,inputs = decode,(device_state[3],device_state[1],device_state[2])
                    want = reference_grow(cpu_state[3],cpu_state[1].float(),cpu_state[2].float())
                with transfers(session) as cold_copies:
                    warm,codes = run_and_get_code(fn,*inputs)
                assert codes, 'generated wrapper source was not captured; fallback audit is unavailable'
                fallbacks = host_fallbacks(codes)
                assert not fallbacks, fallbacks
                assert all(t.device.type=='tnca' for t in warm)
                assert not any(e['direction']=='h2d' and e['name'] in weight_names for e in cold_copies)
                assert compare(warm[:3],want[:3],tolerance)['ok'], 'warm model output differs'
                del warm
                events = []
                original_wait = session.wait

                def capture_wait(event,*a,**kw):
                    result = original_wait(event,*a,**kw)
                    events.append(dict(state=event.state,history=list(event.history),
                                       device_timing=event.device_timing))
                    return result

                session.wait = capture_wait
                history_start = len(session.dev.history)
                try:
                    with transfers(session,stacks=True) as copies,launch_log() as launches:
                        output = fn(*inputs)
                finally:
                    session.wait = original_wait
                step = dict(index=index,kind='prefill' if index==0 else 'decode',
                            supplemental=index==new_tokens,
                            cache_length=prefix+index,launches=launches,transfers=copies,
                            cold_transfers=cold_copies,events=events,fallbacks=fallbacks,
                            completion_history=session.dev.history[history_start:],
                            wrapper_sha256=hashlib.sha256('\n'.join(codes).encode()).hexdigest())
                report['steps'].append(step)
                assert launches and all(e['upload_bytes']==e['download_bytes']==0 for e in launches)
                assert all(e['backend']=='fpga-u280-multicore' and e['device_timing'] is not None for e in launches)
                assert events and all(e['state']=='SUCCEEDED' and
                                      'DRAINING' in [s for s,t in e['history']] for e in events)
                assert any(r['op']==4 and r['status']==0 for r in step['completion_history'])
                assert all(t.device.type=='tnca' for t in output)
                assert not any(e['tensor'] or e['staging'] for e in copies), 'implicit tensor transfer'
                assert not session.unfinished
                assert frozen_weights == tuple((p.data_ptr(),p.untyped_storage().nbytes()) for p in base.parameters())
                # Validation reads are explicit and are never included in the
                # resident execution audit or converted to model throughput.
                history_start = len(session.dev.history)
                with transfers(session) as validation_copies:
                    cpu_state = tuple(t.cpu() for t in output)
                step['validation_transfers'] = validation_copies
                step['host_visibility_history'] = session.dev.history[history_start:]
                handoffs = [r for r in step['host_visibility_history'] if r['op']==18]
                assert handoffs and all(r['status']==0 for r in handoffs)
                step['visibility'] = dict(device_completion='DRAINING then quiesce then SUCCEEDED',
                    host_read='successful ownership handoff then numerical comparison')
                step['correctness'] = compare(cpu_state[:3],want[:3],tolerance)
                assert step['correctness']['ok'], step['correctness']
                assert torch.equal(cpu_state[3],cpu_state[0].float().argmax(-1))
                device_tokens.append(cpu_state[3].tolist())
                step['token'] = device_tokens[-1]
                device_state = output
                print(f'FROZEN MODEL STEP {index} {step["kind"]} KV={step["cache_length"]} correct resident',flush=True)
                (args.out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        report['tokens'] = device_tokens[:new_tokens]
        report['supplemental_token'] = device_tokens[-1]
        report['actual_decode_calls'] = sum(s['kind']=='decode' for s in report['steps'])
        assert report['actual_decode_calls'] == new_tokens
        report['tokens_match_independent_reference'] = device_tokens == reference_tokens
        report['first_token_difference'] = next((dict(index=i,device=a,reference=b,
                                                      reference_top2_margin=reference_margins[i],
                                                      supplemental=i==new_tokens)
                                                  for i,(a,b) in enumerate(zip(device_tokens,reference_tokens)) if a!=b),None)
        report['near_tie_policy'] = 'mismatch retained for explicit review; no automatic exemption'
        assert report['tokens_match_independent_reference'], 'token mismatch needs near-tie review; not automatically accepted'
        report['memory'] = dict(tensor_peak_bytes=D.MEMORY.peak,session=session.memory_stats)
        peak = max(D.MEMORY.peak,session.memory_stats['heap_peak_bytes'])+session.memory_stats['code_peak_bytes']
        report['model_window_with_25pct_headroom'] = (peak*5+3)//4
        assert report['window']['size'] >= report['model_window_with_25pct_headroom']
        assert fresh_software_tree() == tree, 'software changed during native execution'
        report['status'] = 'functional-native-model-pass'
    except Exception:
        report['status'] = 'fail'
        report['error'] = traceback.format_exc()
    finally:
        (args.out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    return 0 if report['status']=='functional-native-model-pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
