"""Ordering negatives for debug barriers, aliases, role handoff and async IO."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler/python'), str(ROOT/'compiler/test'), str(ROOT/'tools/model')]
import triton_bridge
import rtl_harness as H
from tnca_cc.codegen import CodeGen
from tnca_cc.diagnostics import Diagnostic
from tnca_cc.frontend import TTIRImporter
from tnca_cc.ir import Builder, Op, make_kernel
from tnca_cc.pipeline import build_pipeline, plan_pipeline
from tnca_cc.schedule import schedule_kernel
from tnca_cc.target import TargetInfo

if triton_bridge.available():
    import triton
    import triton.language as tl

    @triton.jit
    def store_barrier_reload(X, Y, Z):
        i = tl.arange(0, 16)
        value = tl.load(X+i)+1.
        tl.store(Y+i, value)
        tl.debug_barrier()
        shuffled = tl.load(Y+(i ^ 1))
        tl.store(Z+i, shuffled)


class BarrierOrderingTests(unittest.TestCase):
    def record_rtl(self, program, *, expect_timeout=False, **kwargs):
        ok, log = H.run(program, **kwargs)
        destination = os.environ.get('R41_ORDERING_OUT')
        if destination:
            out = Path(destination)/program.name
            out.mkdir(parents=True, exist_ok=False)
            bench = Path(H.bench_path(kwargs.get('bench', 'fast')))
            files = {}
            for suffix in ('.s', '.hex'):
                source = Path(H.BUILD)/f'cc_{program.name}{suffix}'
                shutil.copy2(source, out/source.name)
                files[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            (out/'stdout.log').write_text(log)
            (out/'record.json').write_text(json.dumps(dict(
                actual_pass=ok, expected_timeout=expect_timeout,
                bench_sha256=hashlib.sha256(bench.read_bytes()).hexdigest(),
                arguments=kwargs, generated_files_sha256=files), indent=2)+'\n')
        return ok, log

    def test_fence_does_not_complete_async_reads_or_writes(self):
        for store in (False, True):
            for wait in (False, True):
                with self.subTest(store=store, wait=wait):
                    k = make_kernel('async_fence_boundary')
                    b = Builder(k, k.role(0).region)
                    w = b.window(1)
                    d = b.descriptor(b.const(0x2000))
                    token = b.barrier_arm(b.barrier(), arrivals=0, ops=1)
                    operation = b.dma_store if store else b.dma_load
                    operation(w, d, [b.const(0)], token, dtype='f32')
                    b.fence()
                    if wait:
                        b.barrier_wait(token)
                    value = b.window_reg(w, 0)
                    if store:
                        b._emit('assign', [], [value, b.vmovi(1)])
                    else:
                        b.store(b.const(0x8000), value)
                    cg = CodeGen(k, TargetInfo.default(), check_protocol=False)
                    cg.assign_shared()
                    if wait:
                        cg.run()
                    else:
                        with self.assertRaisesRegex(Diagnostic, 'still (reading|writing)'):
                            cg.run()

    @unittest.skipUnless(triton_bridge.available(), 'real Triton frontend required')
    def test_store_reload_alias_and_scheduled_memory_order_on_rtl(self):
        self.assertTrue(H.available(), H.why_unavailable())
        for alias in (False, True):
            for bench in ('fast', 'slow', 'ooo'):
                with self.subTest(alias=alias, bench=bench):
                    target = TargetInfo.default()
                    module, _context = triton_bridge.make_ttir(store_barrier_reload,
                        dict(X='*fp32', Y='*fp32', Z='*fp32'), {})
                    y = 0x4000 if alias else 0x5000
                    kernel, _ = TTIRImporter(target, 'store_barrier_reload', index_opt=True).run(
                        module, 'store_barrier_reload', dict(X=0x4000, Y=y, Z=0x6000),
                        param_names=['X', 'Y', 'Z'])
                    schedule_kernel(kernel, target)
                    ops = kernel.role(0).region.ops
                    fences = [i for i, op in enumerate(ops) if op.kind == 'fence']
                    self.assertEqual(len(fences), 1, 'debug_barrier must not disappear or become a no-op')
                    fence = fences[0]
                    self.assertEqual(ops[fence].attrs, dict(order='acq_rel', scope='device'))
                    self.assertTrue(any(op.kind in ('store', 'scatter') for op in ops[:fence]))
                    self.assertTrue(any(op.kind in ('load', 'gather') for op in ops[fence+1:]))
                    cg = CodeGen(kernel, target, check_protocol=False)
                    cg.assign_shared(); cg.run()
                    p = H.Program(f'barrier_alias_{int(alias)}_{bench}').kernel(cg).check_base(0x6000)
                    for lane in range(16):
                        p.check_word(0x6000, lane*4, H.f32_bits((lane ^ 1)*7.+4.))
                    p.words(0x4000, H.pack32([H.f32_bits(i*7.+3.) for i in range(16)]), 'input')
                    if not alias:
                        p.words(0x5000, H.pack32([0xDEADBEEF]*16), 'intermediate')
                    p.words(0x6000, H.pack32([0xDEADBEEF]*16), 'output')
                    ok, log = self.record_rtl(p, bench=bench, max_cycles=20000)
                    self.assertTrue(ok, log)

    def test_plain_fence_cannot_replace_cross_role_release(self):
        self.assertTrue(H.available(), H.why_unavailable())
        from test_pipeline import tensormap
        for remove_release in (False, True):
            with self.subTest(remove_release=remove_release):
                target = TargetInfo.default()
                plan = plan_pipeline(target, stages=1, iterations=2)
                def consume(b, i, window):
                    b.store(b.const(0xE100+i*64), b.window_reg(window, 0))
                k, _ = build_pipeline(target, plan, desc_addr=0xC000,
                                      coords_for=lambda i: (0, i*16), consume=consume)
                if remove_release:
                    consumer = k.role(1)
                    consumer.region.ops = [Op('fence', [], [], dict(order='acq_rel', scope='device'))
                        if op.kind == 'barrier_arrive' else op for op in consumer.region.ops]
                p, _ = H.multirole_program(f'cross_role_fence_{int(remove_release)}', k, target)
                p.words(0x8000, H.pack16(list(range(16*32))), 'input')
                p.words(0xC000, tensormap(0x8000, 16, 32, 64, 2, 16, 16), 'descriptor')
                p.words(0xE100, H.pack32([0xDEADBEEF]*32), 'output')
                p.check_base(0xE100)
                for iteration in range(2):
                    for lane in range(16):
                        first = (lane//8)*32+iteration*16+2*(lane%8)
                        p.check_word(0xE100, iteration*64+lane*4, first | ((first+1) << 16))
                ok, log = self.record_rtl(p, expect_timeout=remove_release,
                                          extra_args=('+allroles', '+aliasvrf'), max_cycles=5000)
                if remove_release:
                    self.assertFalse(ok, 'a plain fence incorrectly supplied another role\'s release')
                    self.assertRegex(log.lower(), 'timeout|timed out|max.?cycle')
                else:
                    self.assertTrue(ok, log)


if __name__ == '__main__':
    unittest.main()
