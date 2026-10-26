#!/usr/bin/env python3
"""Link argv/refusal checks with a recording compiler; never runs Vitis."""
from pathlib import Path
import json
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix='tnca-stream-link-') as temporary:
        root = Path(temporary)
        binary = root/'bin';binary.mkdir()
        recorder = binary/'v++'
        recorder.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
            'open(os.environ["LINK_ARG_RECORD"],"w").write(json.dumps(sys.argv[1:]))\nsys.exit(7)\n')
        recorder.chmod(0o755)
        utility = binary/'xclbinutil';utility.write_text('#!/bin/sh\nexit 1\n');utility.chmod(0o755)
        vitis = root/'vitis';vitis.mkdir();(vitis/'settings64.sh').write_text('true\n')
        platforms = root/'platforms';(platforms/'xilinx_u280_gen3x16_xdma_1_202211_1').mkdir(parents=True)
        env = dict(os.environ, TNCA_BOARD='u280', XILINX_VITIS=str(vitis),
                   XILINX_PLATFORMS=str(platforms), PATH=str(binary)+os.pathsep+os.environ['PATH'])
        for key in ('LINK_RECHECK','LINK_EXTRA','TNCA_STREAM_XO','VPP_JOBS'):
            env.pop(key, None)
        stream = root/'stream';stream.mkdir();(stream/'tnca_stream.xo').write_bytes(b'fixture-stream')
        marker = stream/'tnca_variant.txt';marker.write_text('stream-hbm\n')
        for mode in ('legacy','stream','wrong-marker','same-xo','missing-xo'):
            out = root/('case-'+mode);out.mkdir();(out/'tnca_u50.xo').write_bytes(b'fixture-core')
            (out/'tnca_variant.txt').write_text('multicore-hbm\n')
            record = root/(mode+'.json');run_env = dict(env, LINK_ARG_RECORD=str(record))
            if mode != 'legacy':
                run_env['TNCA_STREAM_XO'] = str(stream/'tnca_stream.xo')
            if mode == 'wrong-marker': marker.write_text('hbm\n')
            else: marker.write_text('stream-hbm\n')
            if mode == 'same-xo': run_env['TNCA_STREAM_XO'] = str(out/'tnca_u50.xo')
            if mode == 'missing-xo': run_env['TNCA_STREAM_XO'] = str(root/'missing.xo')
            result = subprocess.run(['bash','flow/fpga/link.sh',str(out)], cwd=ROOT,
                env=run_env, capture_output=True, text=True, timeout=30)
            if mode in ('legacy','stream'):
                assert result.returncode == 7, (mode, result.stdout, result.stderr)
                args = json.loads(record.read_text())
                assert args[args.index('--kernel_frequency')+1] == '0:100'
                cfg = args[args.index('--config')+1]
                if mode == 'stream':
                    assert args[-2:] == [str(out/'tnca_u50.xo'), str(stream/'tnca_stream.xo')]
                    assert cfg == str(ROOT/'fpga/u280_multicore_stream.cfg')
                    assert any('place_r41_hbm_stream.tcl' in arg for arg in args)
                else:
                    assert args[-1] == str(out/'tnca_u50.xo') and str(stream/'tnca_stream.xo') not in args
                    assert cfg == str(ROOT/'fpga/u280_multicore_hbm.cfg')
            else:
                assert result.returncode == 2 and not record.exists(), (mode, result.stdout, result.stderr)
            print(f'STREAM LINK ARGUMENT/REFUSAL PASS {mode}', flush=True)
        bad_config = root/'bad_image.tcl'
        bad_config.write_text('set TNCA_IMAGE_PARAMS [dict create HBM_STRIPED 0 AXI_DATA_W 256 MEM_PORTS 3]\n')
        untouched = root/'refused-full-build'
        result = subprocess.run(['bash','flow/fpga/run_r41_hbm_stream.sh',str(untouched),str(bad_config),'2'],
            cwd=ROOT, capture_output=True, text=True, timeout=10)
        assert result.returncode == 2 and 'requires HBM_STRIPED=1' in result.stderr and not untouched.exists()
        print('STREAM FULL-RUN CONFIG REFUSAL PASS (no build directory or vendor tools)', flush=True)
    print('STREAM LINK FLOW PASS (recording compiler only)', flush=True)


if __name__ == '__main__':
    main()
