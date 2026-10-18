#!/usr/bin/env python3
"""A TNCA inventory child must import its snapshot, not an older install."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from archbench.run import _child_env


def main():
    args = SimpleNamespace(platform='tnca', backend='fpga-u280-multicore',
                           timeout=7200, backend_args=None, compile_only=False,
                           no_tensor_descriptors=False, set=[])
    with tempfile.TemporaryDirectory() as directory:
        stale = Path(directory)/'old-install'
        stale.mkdir()
        (stale/'tnca_cc').mkdir()
        (stale/'tnca_cc/__init__.py').write_text("raise RuntimeError('stale compiler imported')\n")
        with patch.dict(os.environ, {'PYTHONPATH': str(stale)}):
            env = _child_env(args)
        probe = '''import json, pathlib, tnca_cc
from tnca_cc._datapath import find
print(json.dumps(dict(compiler=str(pathlib.Path(tnca_cc.__file__).resolve()),
                     scheduler=str(pathlib.Path(find('spec','r39_sched_model.json')).resolve()))))'''
        result = subprocess.run([sys.executable, '-c', probe], cwd=directory,
                                env=env, check=True, capture_output=True, text=True)
        origins = json.loads(result.stdout)
        assert Path(origins['compiler']).is_relative_to(ROOT/'compiler/python')
        assert Path(origins['scheduler']) == ROOT/'spec/r39_sched_model.json'
        assert env['TNCA_BACKEND'] == args.backend
        assert env['PYTHONPATH'].split(os.pathsep)[-1] == str(stale)
        print('PASS: isolated inventory child imports snapshot compiler and scheduler before stale installation')


if __name__ == '__main__':
    main()
