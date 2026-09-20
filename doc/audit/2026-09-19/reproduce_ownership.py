"""Audit reproducer: succeeds today, but compilation should reject the overwrite.
Run from the repository root: PYTHONPATH=compiler/python python3 doc/audit/2026-09-19/reproduce_ownership.py
No RTL execution is performed; this tests the compiler safety boundary.
"""
from tnca_cc import TargetInfo
from tnca_cc.ir import Builder, make_kernel
from tnca_cc.codegen import CodeGen

kernel = make_kernel('audit_overwrite')
b = Builder(kernel, kernel.role(0).region)
a, other = b.window(8), b.window(8)
acc, barrier = b.accumulator(), b.barrier(init=True)
value = b.vmovi(7)
token = b.barrier_arm(barrier, ops=1)
b.mma_async(acc, a, other, token)
b.window_put(a, 0, value)
b.barrier_wait(token)
b.exit()
cg = CodeGen(kernel, TargetInfo.default())
cg.assign_shared()
cg.run()
print('DEFECT REPRODUCED: compiler accepted overwrite before MMA completion')
print(cg.b.text())
