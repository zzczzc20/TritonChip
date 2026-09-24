#!/usr/bin/env python3
"""Compile original TileBench JIT functions without rewriting their bodies.

The host benchmark harness depends on CUDA/PyTorch. We load only JIT
function definitions, their JIT helpers and literal module constants from
its AST. Each definition retains its original filename and line numbers;
Triton reads the original source via inspect. GPU autotuning/launch wrappers
are not executed. Block sizes are explicit target configuration, not source
edits. This gate is separate from the representative regression suite.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for path in (ROOT / 'compiler/python', ROOT / 'compiler/test', HERE):
    sys.path.insert(0, str(path))


def load(path):
    import triton
    import triton.language as tl
    text = path.read_text()
    parsed = ast.parse(text, filename=str(path))
    namespace = {'triton': triton, 'tl': tl, '__name__': 'tilebench_' + path.parent.name}
    nodes = []
    names = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and any(
                ast.unparse(d) == 'triton.jit' for d in node.decorator_list):
            # A heuristic decorator supplies launch metadata, not kernel code.
            # The caller binds those constexprs explicitly in this AOT flow.
            node.decorator_list = [d for d in node.decorator_list
                                   if ast.unparse(d) == 'triton.jit']
            nodes.append(node)
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                if not (isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == 'tl.constexpr'):
                    continue
            nodes.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and 'libdevice' in ast.unparse(node):
            nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


POINTERS = set('X Y Z S Out A B Q cos sin input output hist src dst sums chunk_offsets PhiK B_Seqlen Mid_O Mid_O_LogExpSum layout_csr_row_indices layout_csr_col_indices'.split())
INT_POINTERS = set('targets_ptr topk_idx_ptr dest_ptr B_Seqlen layout_csr_row_indices layout_csr_col_indices hist hist_ptr partial_ptr sums chunk_offsets'.split())


def configuration(oid, fn):
    """A small, explicit configuration for source-compatibility discovery.

    Numerical validation is a separate step. A successful compile here must
    not be described as an operator returning correct results.
    """
    import triton.language as tl
    ce, sig, args = {}, {}, {}
    for i, name in enumerate(fn.arg_names):
        if i in fn.constexprs:
            value = 8 if oid == 'interleave' else 16
            if name in ('N', 'N_SIZE', 'E', 'BLOCK_CLASSES', 'BLOCK_N_SIZE'):
                value = 16
            elif name in ('M', 'C', 'C_P2', 'ROWS', 'BLOCK_M', 'BLOCK_R', 'BLOCK_SIZE_R') and oid not in (
                    'matmul_fp32_fp16_fp8', 'matmul_int8', 'batched_matmul',
                    'flash_attention', 'block_sparse_attention', 'streamk_matmul',
                    'linear_self_attention'):
                value = 1
            elif name in ('K', 'K2', 'BLOCK_SIZE_K') and oid in ('moe_topk_gating', 'top_k_selection'):
                value = 4
            elif name in ('kernel_size', 'kernel_rows', 'kernel_cols'):
                value = 3
            elif name in ('GROUP_SIZE_M', 'GROUPSIZE', 'GROUP_M', 'NUM_SMS',
                          'num_heads', 'num_kv_heads', 'num_layout', 'NUM_D_BLOCKS',
                          'ROPE_GROUP_SIZE', 'n_heads', 'stride', 'S_COLS'):
                value = 1
            elif name in ('BACKWARD_PASS', 'IS_CAUSAL', 'DT_ID', 'padding'):
                value = 0
            elif name in ('EVEN_M', 'EVEN_N'):
                value = True
            elif name in ('softmax_scale', 'eps'):
                value = 0.125
            elif name == 'ACC_TYPE':
                value = tl.float32
            elif name == 'K' and oid == 'matmul_int8':
                value = 64
            elif name == 'RADIX':
                value = 4
            elif name == 'FIELD_BITS':
                value = 16
            elif name == 'FIELD_MASK':
                value = 65535
            ce[name], sig[name] = value, 'constexpr'
        elif 'desc' in name:
            dims = ([1, 16, 16] if oid == 'batched_matmul' else
                    [1, 1, 16, 16] if oid in ('flash_attention', 'block_sparse_attention')
                    else [16, 16])
            dtype = ('i32' if name == 'c_desc' else 'i8') if oid == 'matmul_int8' else 'fp16'
            sig[name] = f'tensordesc<{dtype}[{",".join(map(str, dims))}]>'
            strides = [1]
            for d in reversed(dims[1:]):
                strides.insert(0, strides[0] * d)
            args[name] = {'address': 0x18000 + i * 256,
                          'shape': dims, 'strides': strides}
        elif name.endswith('_ptr') or name in POINTERS or (oid == 'streamk_matmul' and name == 'C'):
            elem = 'i32' if name in INT_POINTERS else 'fp32'
            if oid == 'histogramming' and name == 'input_ptr':
                elem = 'i32'
            if oid == 'argmax' and name == 'Out':
                elem = 'i64'
            if oid == 'quantize_global' and name == 'out_ptr':
                elem = 'fp16'
            sig[name], args[name] = '*' + elem, 0xA000 + i * 1024
        elif name in ('eps', 'p', 'sm_scale'):
            sig[name], args[name] = 'fp32', 0.125
        else:
            sig[name], args[name] = 'i32', 16
    return sig, ce, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=ROOT.parent / 'Tilebench')
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--json', type=Path, default=ROOT / 'sim/build/r37_upstream.json')
    args = ap.parse_args()
    source = args.source.resolve()
    operators = source / 'tilebench/benchmarks/operators'
    if not operators.exists():
        operators = source / 'benchmarks/operators'
    spec = json.loads((ROOT / 'spec/r37_tilebench.json').read_text())
    from tnca_cc import TargetInfo
    from tnca_cc.frontend import TTIRImporter
    from tnca_cc.codegen import CodeGen
    import triton_bridge
    rows = []
    for entry in spec['operators']:
        oid = entry['id']
        if args.only and oid not in args.only:
            continue
        path = operators / oid / 'impl_triton.py'
        row = {'id': oid, 'path': str(path), 'stages': [], 'ok': False}
        rows.append(row)
        try:
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            funcs = load(path)
            for name, fn in funcs.items():
                if name in ('_phi', '_swizzle_tile'):
                    continue  # JIT helpers, inlined into the entry points
                stage = {'name': name, 'ok': False}
                row['stages'].append(stage)
                sig, ce, vals = configuration(oid, fn)
                stage['signature'] = sig
                stage['constexprs'] = {k: str(v) for k, v in ce.items()}
                try:
                    mod, ctx = triton_bridge.make_ttir(fn, sig, ce)
                    stage['ttir'] = str(mod)
                    imp = TTIRImporter(TargetInfo.default(), name, scratch_addr=0x18000)
                    kernel, _ = imp.run(mod, fn.__name__, vals,
                                        param_names=triton_bridge.param_names(fn))
                    cg = CodeGen(kernel, TargetInfo.default(), check_protocol=False)
                    cg.assign_shared()
                    cg.run()
                    stage['ok'] = True
                except Exception as e:
                    stage['detail'] = f'{type(e).__name__}: {e}'
                print(f"{'ok' if stage['ok'] else 'NO'} {oid}/{name}: "
                      + stage.get('detail', 'compiles').splitlines()[0], flush=True)
            row['ok'] = bool(row['stages']) and all(s['ok'] for s in row['stages'])
        except Exception as e:
            row['detail'] = f'{type(e).__name__}: {e}'
            print(f'NO {oid}: {e}', flush=True)
    passed = sum(r['ok'] for r in rows)
    print(f'Original TileBench source: {passed}/{len(rows)} operators compile all stages')
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({'source': str(source), 'operators': rows,
                                    'passed': passed, 'total': len(rows)}, indent=2) + '\n')
    return 0 if rows and passed == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
