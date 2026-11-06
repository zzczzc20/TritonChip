#!/usr/bin/env python3
"""Single-role and paired-budget controls participate in cache identity."""
import os
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm')]
import triton  # Initialize backend discovery before importing its compiler.
from tnca.aot import check_options
from tnca.triton_backend.compiler import TNCAOptions

def main():
    with patch.dict(os.environ,{},clear=True):
        base=TNCAOptions();single=TNCAOptions(single_role_pipeline=1)
        pair=TNCAOptions(rctx_roles=2,rctx_role_pairs=1)
        assert len({base.hash(),single.hash(),pair.hash()})==3
        assert check_options({'single_role_pipeline':1})['single_role_pipeline']==1
        assert check_options({'rctx_roles':2,'rctx_role_pairs':1})['rctx_role_pairs']==1
        for invalid in ({'single_role_pipeline':1,'rctx_roles':1},{'rctx_role_pairs':1},{'rctx_role_pairs':3},{'single_role_pipeline':2}):
            try:check_options(invalid)
            except ValueError:pass
            else:raise AssertionError(f'invalid options accepted: {invalid}')
        for invalid in ({'single_role_pipeline':1,'rctx_roles':1},{'rctx_role_pairs':1},{'rctx_role_pairs':3},{'single_role_pipeline':2}):
            try:TNCAOptions(**invalid)
            except ValueError:pass
            else:raise AssertionError(f'invalid JIT options accepted: {invalid}')
    print('single-role options and cache identity PASS')
if __name__=='__main__':main()
