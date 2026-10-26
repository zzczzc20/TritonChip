#!/usr/bin/env python3
"""NUCA line-to-AXI geometry, concurrency, ordering and fault matrix."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    cases = [(width, slots, 0, 0, 1) for width in (64, 128, 256, 512)
             for slots in (1, 3, 8)]
    cases += [(width, 3, mode, 0, 1) for width in (64, 256)
              for mode in (1, 2, 3, 4, 5)]
    cases += [(width, 3, 0, stages, 1) for width in (64, 256, 512)
              for stages in (1, 2, 4)]
    cases += [(width, 3, 0, stages, 0) for width in (64, 256)
              for stages in (0, 2)]
    with tempfile.TemporaryDirectory(prefix="tnca-line-axi-") as temp:
        for width, slots, mode, stages, write_ack in cases:
            output = str(Path(temp) / f"line-{width}-{slots}-{mode}.vvp")
            subprocess.run([
                "iverilog", "-g2012", "-s", "tb_line_axi",
                f"-Ptb_line_axi.DATA_W={width}",
                f"-Ptb_line_axi.WRITE_ACK={write_ack}", f"-Ptb_line_axi.STAGES={stages}", f"-Ptb_line_axi.OUT_N={slots}", f"-Ptb_line_axi.MODE={mode}",
                "-o", output, "rtl/fpga/tnca_line_axi.v", "tb/fpga/tb_line_axi.v", "rtl/common/tnca_line_link.v", "rtl/common/tnca_fifo.v",
            ], cwd=ROOT, check=True, timeout=60)
            subprocess.run(["vvp", output], cwd=ROOT, check=True, timeout=60)
    print(f"LINE AXI MATRIX PASS {len(cases)} configurations", flush=True)


if __name__ == "__main__":
    main()
