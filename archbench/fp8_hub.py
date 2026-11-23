"""The Hub FP8 kernels of the `xl` tier: kernels-community/finegrained-fp8, unchanged.

FP8 points call `matmul_2d` (block-wise W8A8 matmul with fused per-K-block
activation quantisation) and `fp8_act_quant` from the Hub repository exactly as
transformers' FP8Linear does (`kernels.get_kernel("kernels-community/
finegrained-fp8", version=4)`; evidence/r41/fp8/kernels/README.md). Their Triton
source is compiled by the same standard path as every other point: Inductor
traces the `triton_op`, the TNCA Triton backend lowers it. Nothing here edits,
wraps or re-implements a kernel.

Where the kernel source comes from does not matter, what it is does: the build
is identified by the sha256 of its files, pinned below from the `metadata.json`
digests of the `torch-cuda` build of kernel commit fe2bb66fee6c (the build
validated in evidence/r41/fp8), and a source tree whose files differ is refused.

Where the files come from, in this order:
  1. a directory named by ARCHBENCH_FINEGRAINED_FP8 (a `build/torch-cuda`
     folder), imported as a package, the same import `kernels.get_local_kernel`
     performs;
  2. `kernels.get_kernel(repo, version=4)` when the `kernels` package exists and
     finds a build for the installed torch (the way transformers loads it;
     a CPU-only torch has no variant to pick);
  3. the unchanged copy in archbench/third_party/finegrained_fp8 (see its README).
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = 'kernels-community/finegrained-fp8'
KERNEL_COMMIT = 'fe2bb66fee6c2a77eccf08a0508dc57428d488ed'
HUB_VERSION = 4
# sha256 of the files of the torch-cuda build (kernel commit fe2bb66, Hub
# snapshot 0e5764aa5c27), as validated in evidence/r41/fp8/kernels.
FILES = {
    '__init__.py': 'ce17d7a851e967594fbf4854c1a3a1247c66703ab641c13d47ef5c8157cb4c8e',
    '_ops.py': '7a434eec627a0a69b54bd93a1f1ab1bf39639e99689d395392e7056fecad7634',
    'batched.py': 'd59bbd36d2b9798e09a3a86f9f3002438f6e98eb005646170fdd39a75fc175ea',
    'bayesian_autotuner.py': '8c1518bf27e9107b13060165a6dd93b397a6cfe092332be37354f151f64a56b8',
    'fused_batched.py': 'e712ca6aff1d589a9162429d546c75e2e2fd168331a9e1c21153320407672034',
    'fused_grouped.py': '55893766cd9c34a4fe9b56254f3c8cdb0de2cdb8128bb5766724c1cefa6f31d0',
    'grouped.py': '382b03b38f0d0531d4fc6d2ce5e7441576ff9b9a38f3e3200f616ac060c69206',
    'matmul.py': '0f668b5d9e0e7db0675ea4518052b9320830f64f2468f2ecbcae7d0876336807',
    'utils.py': '2cb87f98ab385e41c95fdf1e1bec777a25c8e0bb003a357a72c07a53c2b7b5d0',
}
ENV = 'ARCHBENCH_FINEGRAINED_FP8'
VENDORED = Path(__file__).resolve().parent / 'third_party' / 'finegrained_fp8'
# What a point's definition records about its kernels (hashed into the suite).
IDENTITY = dict(repo=REPO, hub_version=HUB_VERSION, kernel_commit=KERNEL_COMMIT,
                build='torch-cuda', matmul_py=FILES['matmul.py'], utils_py=FILES['utils.py'])

_module = None


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    """Refuse a source tree that is not the pinned build."""
    d = Path(directory)
    bad = [n for n, h in FILES.items() if not (d / n).is_file() or _sha256(d / n) != h]
    if bad:
        raise RuntimeError(f'{d}: not the pinned finegrained-fp8 build (kernel commit {KERNEL_COMMIT[:12]}); '
                           f'files that differ or are missing: {bad}')


def _import_dir(directory):
    meta = json.loads((Path(directory) / 'metadata.json').read_text()) \
        if (Path(directory) / 'metadata.json').is_file() else {}
    name = meta.get('id') or '_finegrained_fp8_local'
    init = Path(directory) / '__init__.py'
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(directory)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kernels():
    """The Hub module (`matmul_2d`, `fp8_act_quant`), loaded once per process."""
    global _module
    if _module is not None:
        return _module
    local = os.environ.get(ENV)
    if local:
        verify(local)
        _module = _import_dir(local)
        return _module
    try:
        import kernels as hub
        mod = hub.get_kernel(REPO, version=HUB_VERSION)
        verify(Path(mod.__file__).parent)
        _module = mod
        return mod
    except Exception:                                      # noqa: BLE001  (no package, no network, no build for this torch)
        pass
    verify(VENDORED)
    _module = _import_dir(VENDORED)
    return _module


def matmul_2d(a, w, ws, block_size, output_dtype):
    return kernels().matmul_2d(a, w, ws, block_size, output_dtype)


def fp8_act_quant(x, block_size=128):
    return kernels().fp8_act_quant(x, block_size)
