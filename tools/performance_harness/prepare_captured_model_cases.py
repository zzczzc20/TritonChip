"""Copy frozen diagnostic helpers with an explicit captured-input provider.

Compiler/runtime sources remain frozen. The generated helper diff and hashes
are evidence; a matching product hash alone never claims matching artifacts.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

from prepare_old_mapping_control import FROZEN_TREE


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.frozen_root.resolve()
    sys.path.insert(0,str(root))
    from archbench.provenance import software_tree
    if software_tree(root,refresh=True)!=FROZEN_TREE:raise ValueError('exact frozen source required')
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('captured_case_inputs.py','prepare_old_mapping_control.py'):
        shutil.copy2(Path(__file__).with_name(name),args.out/name)
    rows=[]
    for name in ('memory_service_case.py','fragment_case.py'):
        source=root/'tools/performance_harness'/name;text=source.read_text()
        changed,count=re.subn(r'^ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[2\]$',
                             "ROOT = Path(os.environ['TNCA_CASE_SOURCE_ROOT']).resolve()",text,flags=re.M)
        if count!=1:raise ValueError('unexpected source-root declaration')
        if name=='memory_service_case.py':
            old='a,b=torch.randn(args.elements),torch.randn(args.elements);reference=(a+b)*.5'
            new="from captured_case_inputs import memory_inputs; a,b=memory_inputs(args.elements);reference=(a+b)*.5"
        else:
            old='                c = torch.randn(n).to(dtype) * .1\n'
            new=old+('                from captured_case_inputs import pipeline_inputs\n'
                     "                if args.strided or args.cancel: raise ValueError('capture replay does not modify input layout/values')\n"
                     '                a,b,c = pipeline_inputs(name,m,n,k,dtype)\n')
        if changed.count(old)!=1:raise ValueError('unexpected input-generation site')
        changed=changed.replace(old,new)
        destination=args.out/name;destination.write_text(changed)
        diff=''.join(difflib.unified_diff(text.splitlines(keepends=True),changed.splitlines(keepends=True),
                                        fromfile=str(source),tofile=str(destination)))
        (args.out/(name+'.diff')).write_text(diff)
        rows.append(dict(file=name,original_sha256=sha(source),replay_sha256=sha(destination)))
    report=dict(status='captured-input-helpers-prepared',frozen_software_tree=FROZEN_TREE,
                helpers=rows,provider_sha256=sha(args.out/'captured_case_inputs.py'),
                scope='input generation and explicit source-root selection only; common compiled graph and independent numerical checks retained; postdiction, not sealed prediction')
    (args.out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
