"""Prepare an isolated, single-change negative control from frozen Stage 1.

Only the projected SSA-user lookup introduced by the fragment repair is
removed. The standard compiler, numerical references and runtime remain intact.
The result is a named diagnostic source variant, never the delivered product.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

FROZEN_COMMIT = '25094197f6e6'
FROZEN_TREE = 'files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a'
REGISTER_BLOCKS = 'compiler/python/tnca_cc/register_blocks.py'
REGISTER_BLOCKS_SHA = 'afe678aaa3e7a935fb0019f8be17286f51ee218c21c04fcee3aeed660b29cdf8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remove_projected_users(text):
    module = ast.parse(text)
    classes = [n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='_OpView']
    if len(classes)!=1:
        raise ValueError('expected one original _OpView class')
    methods = [n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='value_users']
    if len(methods)!=1 or methods[0].decorator_list:
        raise ValueError('expected one undecorated projected-user method')
    method=methods[0];lines=text.splitlines(keepends=True)
    removed=''.join(lines[method.lineno-1:method.end_lineno])
    result=''.join(lines[:method.lineno-1]+lines[method.end_lineno:])
    ast.parse(result)
    return result,removed


def prepare(source, destination, report_path):
    source,destination=source.resolve(),destination.resolve()
    report_path=report_path.resolve()
    if destination==source or destination.is_relative_to(source):
        raise ValueError('diagnostic destination must be outside the frozen source tree')
    if report_path.is_relative_to(source) or report_path.is_relative_to(destination):
        raise ValueError('preparation report must be outside both product source trees')
    if report_path.exists():
        raise ValueError('preparation report already exists; preserve the previous evidence')
    if destination.exists():
        raise ValueError('diagnostic destination already exists; inspect it instead of overwriting')
    sys.path.insert(0,str(source))
    from archbench.provenance import software_tree, _tree_files
    if software_tree(source,refresh=True)!=FROZEN_TREE or sha(source/REGISTER_BLOCKS)!=REGISTER_BLOCKS_SHA:
        raise ValueError('source is not the exact frozen Stage 1 product')
    text=(source/REGISTER_BLOCKS).read_text()
    changed,removed=remove_projected_users(text)
    # These are outside the product hash and unnecessary for execution.
    def ignore(directory,names):
        skip={'.git','__pycache__','.pytest_cache'}
        if Path(directory)==source:
            skip|={'sim','evidence'}
        return set(names)&skip
    shutil.copytree(source,destination,ignore=ignore)
    if software_tree(destination,refresh=True)!=FROZEN_TREE:
        raise ValueError('copied source identity differs before the ablation')
    (destination/REGISTER_BLOCKS).write_text(changed)
    a={str(p.relative_to(source)):sha(p) for p in _tree_files(source)}
    b={str(p.relative_to(destination)):sha(p) for p in _tree_files(destination)}
    differences=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
    if differences!=[REGISTER_BLOCKS]:
        raise ValueError(f'control must change exactly one product file: {differences}')
    if software_tree(source,refresh=True)!=FROZEN_TREE:
        raise ValueError('frozen source changed during preparation')
    report=dict(schema='tnca-old-mapping-control/1',status='isolated-control-prepared',
                frozen_commit=FROZEN_COMMIT,frozen_software_tree=FROZEN_TREE,
                control_software_tree=software_tree(destination,refresh=True),
                source=str(source),destination=str(destination),changed_product_files=differences,
                source_file_sha256=REGISTER_BLOCKS_SHA,control_file_sha256=sha(destination/REGISTER_BLOCKS),
                removed_method=removed,removed_method_sha256=hashlib.sha256(removed.encode()).hexdigest(),
                preparation_script_sha256=sha(__file__),
                scope='negative control removes only virtual-column SSA-user projection; numerical and performance checks still required',
                unchanged='all other product files, including runtime, Inductor templates, references, ISA and RTL')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(prepare(args.source,args.destination,args.report),indent=2))


if __name__=='__main__':main()
