"""Run from root with PYTHONPATH=compiler/python; should reject short windows."""
from tnca_cc import TargetInfo
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.codegen import CodeGen

kernel = make_kernel('audit_k64_short_windows')
b = Builder(kernel, kernel.role(0).region)
a, other = b.window(8), b.window(8)
acc, barrier = b.accumulator(), b.barrier(init=True)
token = b.barrier_arm(barrier, ops=1)
b.mma_async(acc, a, other, token, k='k64')
b.barrier_wait(token)
b.exit()
cg = CodeGen(kernel, TargetInfo.default())
cg.assign_shared()
cg.run()
print('DEFECT REPRODUCED: K64 FP16 accepted with 8-register windows; needs 32 each')
print(cg.b.text())
