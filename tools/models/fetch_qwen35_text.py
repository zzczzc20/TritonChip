#!/usr/bin/env python3
"""Fetch the authentic pinned Qwen3.5-2B snapshot; no inference or device use."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_flux2_klein import download_lfs
REPO = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'

def main():
    import httpx
    from huggingface_hub import HfApi, constants, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    rows = []
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=120),
                      limits=httpx.Limits(max_connections=args.workers*2)) as client:
        for entry in sorted(info.siblings, key=lambda s:s.size or 0):
            if not (entry.rfilename.endswith(('.json','.safetensors','.txt','.jinja'))):
                continue
            if entry.lfs:
                download_lfs(client, constants.HF_HUB_CACHE, REPO, REVISION, entry.rfilename,
                             entry.size, entry.lfs.sha256, args.workers)
                path = Path(constants.HF_HUB_CACHE)/('models--'+REPO.replace('/','--'))/'snapshots'/REVISION/entry.rfilename
            else:
                path = Path(hf_hub_download(REPO, entry.rfilename, revision=REVISION))
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20), b''):
                    h.update(chunk)
            if entry.lfs:
                assert h.hexdigest()==entry.lfs.sha256
            rows.append(dict(file=entry.rfilename,size=path.stat().st_size,sha256=h.hexdigest()))
    result = dict(repository=REPO,revision=REVISION,files=rows)
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
