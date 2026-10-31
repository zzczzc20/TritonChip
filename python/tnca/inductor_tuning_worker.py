"""Compile one generated Inductor configuration without opening a device.

Requests are private files created by the same-process tuning supervisor.
The supervisor owns the wall deadline and the entire process group. Pickle
is confined to these trusted compiler objects, as in Inductor's own workers.
"""
import os
from pathlib import Path
import pickle
import sys


def main(request, output):
    os.environ['TNCA_TUNE'] = '0'
    import tnca.torch_device as device
    device.register()
    from tnca.triton_backend.driver import _Sessions
    def no_device(*args, **kwargs):
        raise RuntimeError('a tuning compile worker cannot open a device Session')
    _Sessions.get = no_device
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime import triton_helpers
    source, name, config = pickle.loads(Path(request).read_bytes())
    with triton_helpers.skip_gpu_driver_setup():
        module = PyCodeCache.load(source)
        tuner = getattr(module, name)
        result = tuner._precompile_config(config)
    # Clear the JITFunction globals in the same way upstream compile workers
    # do. The parent restores its own original function before making a launcher.
    tuner.prepare_for_pickle()
    target = Path(output)
    with target.open('xb') as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:]))
