"""Fetch the pinned FLUX.2 [klein] 4B files into the standard Hugging Face cache layout.

    HF_HOME=~/flux-hf python fetch_flux2_klein.py [--workers 16]

Why not `snapshot_download`: on the build host the Xet transfer stalls (~1 KB/s) and a single
HTTPS stream gives ~1 MB/s, while parallel HTTP range requests against the same official
huggingface.co resolve URLs scale (~11 MB/s with 16).  Every LFS file is verified against the
sha256 the Hub API reports for the pinned revision; small files go through hf_hub_download.
The result is an ordinary cache (blobs/<sha256> + snapshots/<rev>/<path> symlinks), so
`from_pretrained(<snapshot dir>)` and the runner read it offline.  A manifest of every file
(size, sha256 or git blob id) is written to $HF_HOME/flux_manifest.json.
"""
import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK = 32 << 20
JOBS = [  # (repo, pinned revision, files to keep)
    ("black-forest-labs/FLUX.2-klein-4b-fp8", "5b4408e59397a4a37ccb46afe426d8ed86379441",
     lambda p: p.endswith(".safetensors") or p in ("README.md", "LICENSE.md")),
    ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294",
     lambda p: p.split("/")[0] in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
     or p in ("model_index.json", "README.md", "LICENSE.md")),
]


def fetch_range(client, url, path, start, end):
    for attempt in range(10):
        try:
            with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                fd = os.open(path, os.O_WRONLY)
                off = start
                try:
                    for buf in r.iter_bytes(1 << 20):
                        os.pwrite(fd, buf, off)
                        off += len(buf)
                finally:
                    os.close(fd)
                if off != end + 1:
                    raise RuntimeError(f"short read {off - start} of {end - start + 1}")
            return end - start + 1
        except Exception as e:  # noqa: BLE001 - retry any transport error
            print(f"retry {os.path.basename(path)} [{start}] attempt {attempt}: {e}", flush=True)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"failed range {start}-{end} of {path}")


def download_lfs(client, hub, repo, rev, rfile, size, sha, workers):
    repo_dir = os.path.join(hub, "models--" + repo.replace("/", "--"))
    blob = os.path.join(repo_dir, "blobs", sha)
    snap = os.path.join(repo_dir, "snapshots", rev, rfile)
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    if not os.path.exists(blob):
        part, state = blob + ".pdl", blob + ".pdl.done.json"
        done = set(json.load(open(state))) if os.path.exists(state) else set()
        if not os.path.exists(part):
            with open(part, "wb") as f:
                f.truncate(size)
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{rfile}"
        ranges = [(s, min(size, s + CHUNK) - 1) for s in range(0, size, CHUNK) if s not in done]
        lock, t0, got = threading.Lock(), time.time(), 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fetch_range, client, url, part, s, e): s for s, e in ranges}
            for i, fut in enumerate(as_completed(futs)):
                got += fut.result()
                with lock:
                    done.add(futs[fut])
                    json.dump(sorted(done), open(state, "w"))
                if i % 16 == 0:
                    print(f"{rfile}: {len(done)} chunks {got / (time.time() - t0) / 1e6:.1f} MB/s", flush=True)
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for buf in iter(lambda: f.read(16 << 20), b""):
                h.update(buf)
        if h.hexdigest() != sha:
            raise RuntimeError(f"sha256 mismatch {rfile}: {h.hexdigest()} != {sha}")
        os.rename(part, blob)
        os.remove(state)
    if not os.path.lexists(snap):
        os.symlink(os.path.relpath(blob, os.path.dirname(snap)), snap)
    print(f"OK {repo}/{rfile} {size} sha256={sha}", flush=True)


def main():
    import httpx
    from huggingface_hub import HfApi, constants, hf_hub_download
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    api = HfApi()
    client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=120.0),
                          limits=httpx.Limits(max_connections=a.workers * 2))
    manifest = []
    for repo, rev, keep in JOBS:
        info = api.model_info(repo, revision=rev, files_metadata=True)
        assert info.sha == rev, (info.sha, rev)
        for s in sorted((s for s in info.siblings if keep(s.rfilename)), key=lambda s: s.size or 0):
            if s.lfs:
                download_lfs(client, constants.HF_HUB_CACHE, repo, rev, s.rfilename, s.size, s.lfs.sha256, a.workers)
            else:
                hf_hub_download(repo, s.rfilename, revision=rev)
                print(f"OK {repo}/{s.rfilename} {s.size} (git blob {s.blob_id})", flush=True)
            manifest.append({"repo": repo, "revision": rev, "file": s.rfilename, "size": s.size,
                             "sha256": s.lfs.sha256 if s.lfs else None, "git_blob": s.blob_id})
    json.dump(manifest, open(os.path.join(constants.HF_HOME, "flux_manifest.json"), "w"), indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
