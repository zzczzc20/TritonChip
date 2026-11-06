"""Fetch the pinned Qwen FP8 checkpoints into the standard Hugging Face cache layout.

    HF_HOME=~/flux-hf python fetch_qwen_fp8.py [--workers 16] [--only REPO ...]

Same method as fetch_flux2_klein.py (whose downloader this reuses): the Xet transfer stalls on the
build host, so every LFS file is fetched from the official huggingface.co resolve URL of the pinned
revision with parallel HTTP range requests and checked against the sha256 the Hub API reports for
that revision; small files go through hf_hub_download.  The result is an ordinary cache
(blobs/<sha256> + snapshots/<rev>/<path>), so `from_pretrained(<snapshot dir>)` reads it offline.
A manifest of every file (size, sha256 or git blob id) goes to $HF_HOME/qwen_fp8_manifest.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_flux2_klein import download_lfs  # noqa: E402

JOBS = {  # repo -> pinned revision (whole repo: weights, configs, tokenizer, processor, card)
    "Qwen/Qwen3-VL-2B-Instruct-FP8": "46485250d8854c0a9be4f1adbc67ca47e5bb6fa5",
    "lovedheart/Qwen3.5-4B-FP8": "752f65fd155dd34323cfc22d6143df1d48ce3c28",
}


def main():
    import httpx
    from huggingface_hub import HfApi, constants, hf_hub_download
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    api = HfApi()
    client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=120.0),
                          limits=httpx.Limits(max_connections=a.workers * 2))
    manifest = []
    for repo, rev in JOBS.items():
        if a.only and repo not in a.only:
            continue
        info = api.model_info(repo, revision=rev, files_metadata=True)
        assert info.sha == rev, (info.sha, rev)
        for s in sorted(info.siblings, key=lambda s: s.size or 0):
            if s.rfilename == ".gitattributes":
                continue
            if s.lfs:
                download_lfs(client, constants.HF_HUB_CACHE, repo, rev, s.rfilename, s.size, s.lfs.sha256, a.workers)
            else:
                hf_hub_download(repo, s.rfilename, revision=rev)
                print(f"OK {repo}/{s.rfilename} {s.size} (git blob {s.blob_id})", flush=True)
            manifest.append({"repo": repo, "revision": rev, "last_modified": str(info.last_modified),
                             "file": s.rfilename, "size": s.size,
                             "sha256": s.lfs.sha256 if s.lfs else None, "git_blob": s.blob_id})
    json.dump(manifest, open(os.path.join(constants.HF_HOME, "qwen_fp8_manifest.json"), "w"), indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
