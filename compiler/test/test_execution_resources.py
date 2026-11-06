"""The execution resource plan against hand-computed counts.

Kernels go through the product compile path (tnca.aot.compile_aot), so the
plan checked is the one sealed in the artifact manifest and read by the
retile, refusal and tuning decisions. The GEMM kernel is written in Inductor's
mm-template form: wrapped row/column offsets, `offs_k < K - k * 16` masks when
K is not a tile multiple, and the result stored under `(rm < M) & (rn < N)`.

Hand-computed MACs: an MMA command is 16 x 16 x 16 physical MACs; a launch has
ceil(M/16) * ceil(N/BN) programs of ceil(K/16) K steps with BN/16 fragments
each; the valid MACs are M * N * K.
"""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'compiler/python'), str(ROOT/'tools/asm'),
                str(ROOT/'compiler/test')]
import triton_bridge

if triton_bridge.available():
    import triton
    import triton.language as tl
    from test_fragment_residency import fragment_loop

    @triton.jit
    def template_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      BN: tl.constexpr, EVEN_K: tl.constexpr):
        pid = tl.program_id(0)
        grid_n = (N + BN - 1) // BN
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        rm = pid_m * 16 + tl.arange(0, 16)
        rn = pid_n * BN + tl.arange(0, BN)
        offs_m = rm % M
        offs_n = rn % N
        rk = tl.arange(0, 16)
        acc = tl.zeros((16, BN), tl.float32)
        for k in range(0, tl.cdiv(K, 16)):
            if EVEN_K:
                a = tl.load(A + offs_m[:, None] * K + (k * 16 + rk[None, :]))
                b = tl.load(B + (k * 16 + rk[:, None]) * N + offs_n[None, :])
            else:
                a = tl.load(A + offs_m[:, None] * K + (k * 16 + rk[None, :]),
                            mask=rk[None, :] < K - k * 16, other=0.)
                b = tl.load(B + (k * 16 + rk[:, None]) * N + offs_n[None, :],
                            mask=rk[:, None] < K - k * 16, other=0.)
            acc += tl.dot(a, b)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=mask)

    @triton.jit
    def descriptor_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        BN: tl.constexpr):
        pid = tl.program_id(0)
        grid_n = (N + BN - 1) // BN
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        da = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1], block_shape=[16, 16])
        db = tl.make_tensor_descriptor(B, shape=[K, N], strides=[N, 1], block_shape=[16, BN])
        acc = tl.zeros((16, BN), tl.float32)
        for k in range(0, tl.cdiv(K, 16)):
            acc += tl.dot(da.load([pid_m * 16, k * 16]), db.load([k * 16, pid_n * BN]))
        rm = pid_m * 16 + tl.arange(0, 16)
        rn = pid_n * BN + tl.arange(0, BN)
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def runtime_k(A, B, C, K):
        rm = tl.arange(0, 16)
        rk = tl.arange(0, 16)
        acc = tl.zeros((16, 16), tl.float32)
        for k in range(0, K, 16):
            a = tl.load(A + rm[:, None] * K + (k + rk[None, :]))
            b = tl.load(B + (k + rk[:, None]) * 16 + rm[None, :])
            acc += tl.dot(a, b)
        tl.store(C + rm[:, None] * 16 + rm[None, :], acc)

    @triton.jit
    def f32_dot(A, B, C):
        rm = tl.arange(0, 16)
        a = tl.load(A + rm[:, None] * 16 + rm[None, :])
        b = tl.load(B + rm[:, None] * 16 + rm[None, :])
        tl.store(C + rm[:, None] * 16 + rm[None, :], tl.dot(a, b, input_precision='ieee'))


GEMM_SIG = dict(A='*fp16', B='*fp16', C='*fp16', M='constexpr', N='constexpr',
                K='constexpr', BN='constexpr', EVEN_K='constexpr')


def _cdiv(a, b):
    return -(-a // b)


@unittest.skipUnless(triton_bridge.available(), 'Triton is required')
class ExecutionResourcePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def compile(self, fn, signature, constexprs, **options):
        from tnca import aot
        out = Path(self.tmp.name)/f'{fn.__name__}-{len(list(Path(self.tmp.name).iterdir()))}.tnca'
        result = aot.compile_aot(fn, signature=signature, constexprs=constexprs,
                                 profile='core-vrf-sync-v5', output=str(out),
                                 options=dict(dict(index_opt=1, opt_level=2), **options))
        plan = aot.plan_of(result.manifest)
        self.assertIsNotNone(plan, 'the sealed manifest carries a finalized plan')
        return result, plan

    def gemm(self, m, n, k, bn, even_k):
        return self.compile(template_gemm, GEMM_SIG, dict(M=m, N=n, K=k, BN=bn, EVEN_K=even_k))

    def test_fragments_and_macs_for_full_wide_tail_and_masked_shapes(self):
        # (M, N, K, BN, EVEN_K): 16x16 and 16x32 full tiles, the 17x65x33
        # tail of the fragment case, and a masked 20x24x40 problem.
        for m, n, k, bn, even in ((32, 32, 32, 16, True), (16, 32, 32, 32, True),
                                  (17, 65, 33, 32, False), (20, 24, 40, 16, False)):
            with self.subTest(shape=(m, n, k), tile=bn):
                _result, plan = self.gemm(m, n, k, bn, even)
                site, = plan['gemm']['sites']
                fragments, trips = bn // 16, _cdiv(k, 16)
                programs = _cdiv(m, 16) * _cdiv(n, bn)
                physical = programs * trips * fragments * 16 * 16 * 16
                self.assertEqual(site['logical_tile'], [16, bn, 16])
                self.assertEqual(site['physical_fragments'], fragments)
                self.assertEqual(site['k_loop_trips'], trips)
                self.assertEqual(site['per_k_step'], dict(
                    mma_commands=fragments, physical_macs=fragments*4096,
                    logical_macs=16*bn*16, fragment_padding_macs=0))
                self.assertEqual((site['problem']['M'], site['problem']['N'],
                                  site['problem']['K']), (m, n, k))
                self.assertEqual(site['macs'], dict(
                    status='static', scope='per launch', programs=programs,
                    assumes_grid=site['macs']['assumes_grid'],
                    mma_commands=programs*trips*fragments, physical=physical,
                    valid=m*n*k, padding=physical - m*n*k,
                    tile_padding=physical - m*n*k, fragment_padding=0))
                self.assertEqual(plan['gemm']['static_mma_commands'],
                                 dict(expected_from_sites=fragments, emitted=fragments, match=True))
                self.assertEqual(plan['commands']['mma'], fragments)
                self.assertEqual(plan['commands']['mma_in_k_loops'], fragments)
        # The hand values for the tail: 6 programs x 3 K steps x 2 fragments.
        _result, plan = self.gemm(17, 65, 33, 32, False)
        self.assertEqual(plan['gemm']['sites'][0]['macs']['physical'], 147456)
        self.assertEqual(plan['gemm']['sites'][0]['macs']['valid'], 36465)
        self.assertEqual(plan['gemm']['sites'][0]['macs']['padding'], 110991)

    def test_resident_k_loop_has_no_accumulator_round_trips(self):
        for bn in (16, 32):
            with self.subTest(tile=bn):
                _result, plan = self.gemm(17, 65, 33, bn, False)
                trips = plan['accumulator_round_trips']
                self.assertEqual((trips['k_loop_reads'], trips['k_loop_writes'],
                                  trips['k_loop_zeros']), (0, 0, 0))
                # The epilogue reads every row of every fragment once.
                self.assertEqual(trips['outside_k_loop_reads'], 16 * bn // 16)

    def test_live_old_accumulator_round_trips_are_counted(self):
        sig = dict(A='*fp16', B='*fp16', C='*fp32', K='constexpr', N='constexpr', OLD_USE='constexpr')
        _r, resident = self.compile(fragment_loop, sig, dict(K=64, N=32, OLD_USE=False), window_reuse=1)
        _r, live = self.compile(fragment_loop, sig, dict(K=64, N=32, OLD_USE=True), window_reuse=1)
        self.assertEqual(resident['accumulator_round_trips']['k_loop_reads'], 0)
        trips = live['accumulator_round_trips']
        # Two fragments, each storing its old 16 rows before the update.
        self.assertGreaterEqual(trips['k_loop_reads'], 2 * 16)
        self.assertGreater(trips['k_loop_writes'], 0)

    def test_descriptor_and_pointer_transfers(self):
        sig = dict(A='*fp16', B='*fp16', C='*fp32', M='constexpr', N='constexpr',
                   K='constexpr', BN='constexpr')
        _r, plan = self.compile(descriptor_gemm, sig, dict(M=17, N=65, K=33, BN=16))
        # One A and one B box per K step through the transfer engine; no
        # LSU access in the K loop; the epilogue stores 16 rows by pointer.
        self.assertEqual(plan['transfers']['descriptor_in_k_loops'], 2)
        self.assertEqual(plan['transfers']['pointer_in_k_loops'], 0)
        self.assertEqual(plan['commands']['transfer_in_k_loops'], 2)
        site, = plan['gemm']['sites']
        self.assertEqual((site['problem']['M'], site['problem']['N'], site['problem']['K']),
                         (17, 65, 33))
        self.assertEqual(site['macs']['programs'], 2 * 5)
        self.assertEqual(site['macs']['physical'], 10 * 3 * 4096)
        # A 16x32 B box is two compact fragments: three engine commands.
        _r, wide = self.compile(descriptor_gemm, sig, dict(M=17, N=65, K=33, BN=32))
        self.assertEqual(wide['transfers']['descriptor_in_k_loops'], 3)
        self.assertEqual(wide['gemm']['sites'][0]['physical_fragments'], 2)
        # The pointer GEMM's tiles go through the LSU (and scratch).
        _r, pointer = self.gemm(17, 65, 33, 16, False)
        self.assertGreater(pointer['transfers']['pointer_in_k_loops'], 0)

    def test_unstated_extents_stay_unknown(self):
        _r, plan = self.compile(runtime_k, dict(A='*fp16', B='*fp16', C='*fp32', K='i32'), {})
        site, = plan['gemm']['sites']
        self.assertIsNone(site['k_loop_trips'])
        self.assertEqual(site['macs']['status'], 'unknown')
        self.assertEqual(site['per_k_step']['physical_macs'], 4096)

    def test_binary32_dot_is_not_counted_as_mma(self):
        _r, plan = self.compile(f32_dot, dict(A='*fp32', B='*fp32', C='*fp32'), {})
        site, = plan['gemm']['sites']
        self.assertEqual(site['engine'], 'vector-fma')
        self.assertEqual(plan['commands']['mma'], 0)
        self.assertTrue(plan['gemm']['static_mma_commands']['match'])

    def test_resources_admission_and_tuning_read_the_plan(self):
        from tnca_cc.execution_resources import admission, tuning_resources
        result, plan = self.gemm(17, 65, 33, 32, False)
        res = plan['resources']
        measured = result.manifest['optional_metadata']['report']['resources']
        self.assertEqual(res['code_bytes'], measured['code_bytes'])
        self.assertEqual(res['vrf_peak_registers'], measured['vector_regs_used'])
        self.assertEqual(res['accumulators'], 2)
        self.assertEqual(res['window_registers'], plan['window_registers'])
        self.assertEqual(admission(plan, code_limit=res['code_bytes']), [])
        self.assertEqual(admission(plan, code_limit=res['code_bytes'] - 1),
                         [('code_bytes', res['code_bytes'], res['code_bytes'] - 1)])
        used, limits = tuning_resources(plan, code_limit=0x1c000)
        self.assertEqual(used['code_bytes'], res['code_bytes'])
        self.assertEqual(limits, dict(code_bytes=0x1c000, vector_regs=res['vrf_limit'],
                                      accumulators=res['accumulator_limit'],
                                      barriers=res['barrier_limit'],
                                      descriptors=res['descriptor_limit']))

    def test_refusals_carry_their_reason(self):
        _r, plan = self.compile(f32_dot, dict(A='*fp32', B='*fp32', C='*fp32'), {}, rctx_roles=1)
        refused = [d for d in plan['decisions'] if d['decision'] == 'rctx_roles']
        self.assertEqual(len(refused), 1)
        self.assertEqual(refused[0]['status'], 'refused')
        self.assertIn('MMA', refused[0]['reason'])


if __name__ == '__main__':
    unittest.main()
