"""CPU unit checks for the automatic GEMM tile (R42 xfer, tnca.inductor._gemm_blocking)."""
from unittest import TestCase, main

from tnca.inductor import _gemm_blocking, GEMM_MIN_PROGRAMS, BLOCKED_BN

F16, BF16, F32, F8 = 'torch.float16', 'torch.bfloat16', 'torch.float32', 'torch.float8_e4m3fn'


class GemmBlockingTests(TestCase):
    def test_blocks_n_while_the_grid_keeps_the_cores_busy(self):
        # 64 x 128: 4 row tiles x 4 blocked column tiles = 16 programs
        self.assertEqual(_gemm_blocking('mm', 64, 128, 128, [F16, F16]), (16, BLOCKED_BN, 16))
        self.assertEqual(_gemm_blocking('addmm', 128, 256, 320, [BF16, BF16]), (16, BLOCKED_BN, 16))
        # gemv: one row tile, 14 blocked column tiles
        self.assertEqual(_gemm_blocking('mm', 1, 448, 320, [F16, F16]), (16, BLOCKED_BN, 16))

    def test_small_grids_keep_the_standard_tile(self):
        self.assertEqual(_gemm_blocking('addmm', 16, 64, 64, [F16, F16]), (16, 16, 16))      # 2 programs
        self.assertEqual(_gemm_blocking('mm', 16, 256, 64, [F16, F16]), (16, 16, 16))        # 8 programs
        one_row_short = BLOCKED_BN * (GEMM_MIN_PROGRAMS - 1)
        self.assertEqual(_gemm_blocking('mm', 16, one_row_short, 64, [F16, F16]), (16, 16, 16))
        self.assertEqual(_gemm_blocking('mm', 16, one_row_short + BLOCKED_BN, 64, [F16, F16]), (16, BLOCKED_BN, 16))

    def test_n_must_be_a_whole_number_of_blocked_tiles(self):
        self.assertEqual(_gemm_blocking('mm', 4096, 4112, 64, [F16, F16]), (16, 16, 16))     # 4112 = 32 * 128 + 16

    def test_other_kernels_and_types_are_left_alone(self):
        for op in ('bmm', 'baddbmm', 'scaled_mm'):
            self.assertEqual(_gemm_blocking(op, 4096, 4096, 4096, [F16, F16]), (16, 16, 16))
        for dtypes in ([F32, F32], [F8, F8], [F16, F32]):
            self.assertEqual(_gemm_blocking('mm', 4096, 4096, 4096, dtypes), (16, 16, 16))


if __name__ == '__main__':
    main()
