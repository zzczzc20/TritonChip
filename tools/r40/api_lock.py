#!/usr/bin/env python3
"""R40-00: the PyTorch and Triton extension APIs R40 is built on, as found.

    python tools/r40/api_lock.py [--out spec/r40/api_lock.json]
    python tools/r40/api_lock.py --check [spec/r40/api_lock.json]

doc/r40_tasks_and_acceptance.md R40-00 locks the repository to torch 2.14.0
and Triton 3.8.0 and to the extension APIs they actually have; upstream main
documentation is a reference only, and an upgrade is its own compatibility
ablation. This records, for the installed versions, whether each API R40-01
may use exists, its signature, and the sha256 of the file defining it, so a
silent change of environment is caught (`--check`) rather than discovered
as a behaviour change. Run it in the torch environment (TORCH_PYTHON).
"""
import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'spec' / 'r40' / 'api_lock.json'

VERSIONS = {'torch': '2.14.0', 'triton': '3.8.0'}

# (module, attribute, what R40 uses it for)
APIS = [
    # PrivateUse1: the device itself (doc sec 0.2, R40-01)
    ('torch.utils', 'rename_privateuse1_backend', 'name PrivateUse1 "tnca"'),
    ('torch.utils', 'generate_methods_for_privateuse1_backend', 'Tensor.tnca(), is_tnca, ...'),
    ('torch', '_register_device_module', 'torch.tnca module (device count, synchronize, streams)'),
    ('torch.utils.backend_registration', '_setup_privateuseone_for_python_backend',
     'a Python-only PrivateUse1 backend (allocator and kernels in Python)'),
    ('torch._C', '_get_privateuse1_backend_name', 'the registered name'),
    ('torch.library', 'Library', 'registering aten kernels for PrivateUse1'),
    ('torch.library', 'impl', 'registering aten kernels for PrivateUse1'),
    ('torch', 'accelerator', 'the device-generic accelerator API'),
    # Dynamo / FakeTensor
    ('torch._dynamo.device_interface', 'DeviceInterface', 'the tnca device interface'),
    ('torch._dynamo.device_interface', 'register_interface_for_device', 'registering it'),
    ('torch._subclasses.fake_tensor', 'FakeTensorMode', 'tracing on device "tnca"'),
    # Inductor
    ('torch._inductor.codegen.common', 'register_backend_for_device',
     'the tnca scheduling and wrapper (doc sec 0.2)'),
    ('torch._inductor.codegen.common', 'register_device_op_overrides', 'device op overrides'),
    ('torch._inductor.codegen.common', 'DeviceOpOverrides', 'device op overrides'),
    ('torch._inductor.codegen.common', 'get_scheduling_for_device', 'checking what is registered'),
    ('torch._inductor.codegen.common', 'get_wrapper_codegen_for_device', 'checking what is registered'),
    ('torch._inductor.codegen.triton', 'TritonScheduling', 'reused for the tnca device'),
    ('torch._inductor.codegen.wrapper', 'PythonWrapperCodegen', 'the wrapper the tnca one extends'),
    ('torch._inductor.codegen.cpu_device_op_overrides', 'CpuDeviceOpOverrides', 'the CPU path (B39)'),
    ('torch._inductor.config', 'triton', 'Inductor Triton options (tensor descriptors, max_block)'),
    # Triton
    ('triton.backends.compiler', 'BaseBackend', 'the tnca Triton backend'),
    ('triton.backends.compiler', 'GPUTarget', 'the tnca target identity'),
    ('triton.backends.driver', 'DriverBase', 'the tnca driver'),
    ('triton.backends', 'backends', 'entry-point discovery of tnca'),
    ('triton.tools.tensor_descriptor', 'TensorDescriptor', 'host-side tensor descriptors'),
]


def probe():
    out = dict(schema='tnca-r40-api-lock/1', python=sys.version.split()[0], versions={}, apis=[])
    for mod in VERSIONS:
        try:
            out['versions'][mod] = importlib.import_module(mod).__version__
        except Exception as e:                               # noqa: BLE001
            out['versions'][mod] = f'missing: {e}'
    for modname, attr, why in APIS:
        e = dict(module=modname, name=attr, used_for=why)
        try:
            mod = importlib.import_module(modname)
            obj = getattr(mod, attr)
            e['present'] = True
            try:
                e['signature'] = str(inspect.signature(obj))
            except (TypeError, ValueError):
                e['signature'] = None
            try:
                src = inspect.getsourcefile(obj if not inspect.ismodule(obj) else obj) \
                    or inspect.getsourcefile(mod)
            except TypeError:
                src = getattr(mod, '__file__', None)
            if src and Path(src).is_file():
                e['file'] = src.split('site-packages/')[-1]
                e['file_sha256'] = hashlib.sha256(Path(src).read_bytes()).hexdigest()
        except Exception as ex:                              # noqa: BLE001
            e['present'] = False
            e['error'] = f'{type(ex).__name__}: {ex}'[:200]
        out['apis'].append(e)
    try:
        from triton.backends import backends
        out['triton_backends'] = sorted(backends)
    except Exception as ex:                                  # noqa: BLE001
        out['triton_backends'] = f'error: {ex}'
    return out


def compare(frozen, now):
    bad = []
    for k, v in frozen['versions'].items():
        base = lambda s: str(s).split('+')[0]                 # noqa: E731  (2.14.0+cpu == 2.14.0+cu130)
        if base(now['versions'].get(k)) != base(v):
            bad.append(f'{k}: frozen {v}, found {now["versions"].get(k)}')
    found = {(e['module'], e['name']): e for e in now['apis']}
    for e in frozen['apis']:
        n = found.get((e['module'], e['name']))
        if n is None:
            bad.append(f"{e['module']}.{e['name']}: no longer probed")
        elif n['present'] != e['present']:
            bad.append(f"{e['module']}.{e['name']}: present {e['present']} -> {n['present']}")
        elif e.get('signature') != n.get('signature'):
            bad.append(f"{e['module']}.{e['name']}: signature {e.get('signature')} -> {n.get('signature')}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--check', nargs='?', const=str(DEFAULT))
    args = ap.parse_args(argv)
    now = probe()
    for k, v in VERSIONS.items():
        if not str(now['versions'].get(k, '')).startswith(v):
            print(f'api-lock: {k} is {now["versions"].get(k)}; R40 is locked to {v}')
    if args.check:
        bad = compare(json.loads(Path(args.check).read_text()), now)
        for b in bad:
            print(f'api-lock: {b}')
        print(f"r40-api-lock  {'PASS' if not bad else 'FAIL'}  ({len(now['apis'])} APIs, "
              f"torch {now['versions']['torch']}, triton {now['versions']['triton']})")
        return 1 if bad else 0
    text = json.dumps(now, indent=1) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    missing = [f"{e['module']}.{e['name']}" for e in now['apis'] if not e['present']]
    print(f"api-lock: {len(now['apis']) - len(missing)} of {len(now['apis'])} present"
          + (f'; absent: {missing}' if missing else ''), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
