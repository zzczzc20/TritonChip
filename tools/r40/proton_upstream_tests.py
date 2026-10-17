#!/usr/bin/env python3
"""Pinned upstream Proton dialect -> importer -> TNCA timestamp machine code."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/'python', ROOT/'compiler/python', ROOT/'tools/asm'):
    sys.path.insert(0, str(p))
from triton._C.libtriton import ir, proton
from tnca_cc.frontend import TTIRImporter
from tnca_cc.target import TargetInfo
from tnca_cc.proton import lower_records
from tnca_cc.codegen import CodeGen
from tnca_asm import Assembler


def main():
    ctx = ir.context()
    ir.load_dialects(ctx)
    proton.load_dialects(ctx)
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp)/'record.mlir'
        source.write_text('''module {
  tt.func public @record_only() {
    proton.record start "kernel"
    proton.record start "compute"
    proton.record end "compute"
    proton.record end "kernel"
    tt.return
  }
}
''')
        mod = ir.parse_mlir_module(str(source), ctx)
        target = TargetInfo.default()
        kernel, _ = TTIRImporter(target, 'record_only').run(mod, 'record_only', {})
        records = [o for o in kernel.role(0).region.ops if o.kind == 'proton.record']
        assert [o.attrs for o in records] == [
            dict(name='kernel', isStart=True), dict(name='compute', isStart=True),
            dict(name='compute', isStart=False), dict(name='kernel', isStart=False)]
        lowered, layout = lower_records(kernel, buffer_base=0x20000, buffer_bytes=64)
        cg = CodeGen(lowered, target)
        cg.assign_shared()
        assembly = cg.run().text()
        Assembler().assemble(assembly)
        assert assembly.count(', clock') == 4, assembly
        assert layout['required_bytes'] == 48
        print('PASS: actual pinned Proton dialect, nested record effects, importer and assembled timestamps')


if __name__ == '__main__':
    main()
