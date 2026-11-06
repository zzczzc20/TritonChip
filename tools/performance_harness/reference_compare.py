#!/usr/bin/env python3
"""Cross-validate CPU BF16/FP32 references before prospective TNCA numerics freeze."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bf16',type=Path,required=True)
    p.add_argument('--fp32',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    import torch
    from safetensors.torch import load_file
    reports=[json.loads((root/'report.json').read_text()) for root in (a.bf16,a.fp32)]
    assert all(r['weights_verified'] and r['layers']==24 and r['vocabulary']==248320 for r in reports)
    assert reports[0]['revision']==reports[1]['revision']
    rows=[]
    for name,bcase in reports[0]['workloads'].items():
        fcase=reports[1]['workloads'][name]
        assert bcase['input_ids']==fcase['input_ids']
        for b,f in zip(bcase['steps'],fcase['steps'],strict=True):
            bb,ff=load_file(str(a.bf16/b['file'])),load_file(str(a.fp32/f['file']))
            assert bb.keys()==ff.keys()
            for key in bb:
                v,w=bb[key].double(),ff[key].double()
                assert torch.isfinite(v).all() and torch.isfinite(w).all(),(name,b['step'],key)
                diff=v-w
                absolute=float(diff.abs().max()) if diff.numel() else 0
                rel_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(w).clamp_min(1e-30))
                scaled=float((diff.abs()/(w.abs()+float(w.abs().max())*0.001+1e-8)).max()) if diff.numel() else 0
                rows.append(dict(workload=name,step=b['step'],tensor=key,max_abs=absolute,rel_l2=rel_l2,scaled_max=scaled))
    maxima={kind:max(r['rel_l2'] for r in rows if (r['tensor']=='logits')==(kind=='logits')) for kind in ('logits','states')}
    # Pre-TNCA CPU-only envelope. A reference disagreement above the cap stops
    # the freeze; it never justifies an unbounded tolerance.
    thresholds={kind:min(0.1,max(0.02,value*4)) for kind,value in maxima.items()}
    passed=all(value<=0.025 for value in maxima.values())
    result=dict(schema='tnca-r41-reference-cross-validation/1',status='passed' if passed else 'failed',
                tnca_outputs_used=False,method='authentic all-layer official CPU BF16 versus FP32 widened from identical checkpoint values; every teacher-forced logits/cache state tensor',
                reference_report_sha256={kind:hashlib.sha256((root/'report.json').read_bytes()).hexdigest() for kind,root in [('bf16',a.bf16),('fp32',a.fp32)]},
                max_rel_l2=maxima,proposed_rel_l2_limits=thresholds,
                acceptance_rule='finite values, relative L2 per tensor <= locked threshold; zero-reference tensors require max_abs <= 1e-6; check all steps and states independently',rows=rows)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
