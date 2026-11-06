"""Integrity mutations of a real completed old-mapping collection, no device access."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

import compare_old_mapping_board as audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-root','required-index','proof','old-manifest','old-collection','fixed-collection','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    def run(directory):
        return audit.compare(a.frozen_root,a.required_index,a.proof,a.old_manifest,directory,a.fixed_collection)
    result=run(a.old_collection)
    assert result['status']=='comparison-recorded' and len(result['rows'])==9
    rows=[]
    with tempfile.TemporaryDirectory(prefix='old-mapping-integrity-',dir=a.out.parent) as tmp:
        tmp=Path(tmp)
        for mutation,expected in (
            ('output-bytes','output hash mismatch'),('missing-point','exact nine-point'),
            ('wrong-image','wrong image'),('numeric-failure','status fail'),
            ('source-delta','source delta is not isolated'),('artifact-bytes','artifact missing or changed'),
            ('unfinished','complete experiment')):
            target=tmp/mutation;shutil.copytree(a.old_collection,target)
            rp=target/'out/report.json';bp=target/'out/old-mapping/batch.json';fp=target/'out/old-mapping/formal.jsonl'
            report=audit.read(rp);batch=audit.read(bp);records=[json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
            changed_formal=False;changed_batch=False
            if mutation=='output-bytes':fp.write_text(fp.read_text()+'\n')
            if mutation=='missing-point':
                records.pop();batch['configurations']-=1;changed_formal=changed_batch=True
            if mutation=='wrong-image':
                records[0]['platform']['image_sha256']='0'*64;changed_formal=True
            if mutation=='numeric-failure':records[0]['status']='fail';changed_formal=True
            if mutation=='source-delta':
                source=next(k for k in report['identity']['sources'] if k.endswith('/compiler/python/tnca_cc/frontend.py'))
                report['identity']['sources'][source]='0'*64
            if mutation=='artifact-bytes':
                rec=records[0];launch=rec['provenance']['launch_map']['launches'][0]
                directory=target/Path(rec['provenance']['bundle']['dir']).relative_to('/home/zzhouch/r41s1_old_mapping_v1')
                path=directory/'artifacts'/(launch['artifact']+'.tnca');data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            if mutation=='unfinished':report['status']='running'
            if changed_formal:fp.write_text(''.join(json.dumps(r)+'\n' for r in records))
            if changed_batch:bp.write_text(json.dumps(batch))
            # Consistent outer hashes must not make invalid inner evidence pass.
            for path,changed in ((fp,changed_formal),(bp,changed_batch)):
                if changed:
                    name=str(Path('/home/zzhouch/r41s1_old_mapping_v1')/path.relative_to(target))
                    report['steps'][0]['outputs'][name]=audit.sha(path)
            rp.write_text(json.dumps(report))
            try:run(target)
            except ValueError as error:
                assert expected in str(error),(mutation,str(error))
                rows.append(dict(mutation=mutation,rejected=str(error)))
            else:raise AssertionError('invalid collection accepted: '+mutation)
    report=dict(status='real-collection-integrity-checks-pass',positive_points=len(result['rows']),
                refusals=rows,comparator_sha256=audit.sha(Path(audit.__file__)))
    with a.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],refusals=len(rows)),indent=2))


if __name__=='__main__':main()
