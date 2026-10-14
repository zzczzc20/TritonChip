#!/usr/bin/env python3
"""Random backpressure and VN/order conservation across elastic mesh links."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix='r40-mesh-pipeline-') as temp:
        for stages, seeds in ((1, (40,)), (2, (40, 41)), (4, (41,))):
            executable = str(Path(temp) / f'mesh-{stages}.vvp')
            subprocess.run(['iverilog', '-g2012', '-Irtl/common', '-Irtl/noc',
                            '-y', 'rtl/common', '-y', 'rtl/noc', '-s', 'tb_mesh',
                            f'-Ptb_mesh.LINK_STAGES={stages}', '-o', executable,
                            'tb/noc/tb_mesh.v'], cwd=ROOT, check=True, timeout=60)
            for seed in seeds:
                result = subprocess.run(['vvp', executable, '+npkt=20000', f'+seed={seed}'],
                                        cwd=ROOT, capture_output=True, text=True, timeout=180)
                print(f'LINK_STAGES={stages}\n{result.stdout}', end='', flush=True)
                if result.returncode or '\nPASS\n' not in result.stdout or '\nFAIL\n' in result.stdout:
                    raise AssertionError(f'elastic mesh failed: {result.stderr}')
    print('MESH PIPELINE PASS 4 randomized configurations', flush=True)


if __name__ == '__main__':
    main()
