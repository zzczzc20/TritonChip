"""Replay saved diagnostic CPU inputs exactly; performs no TNCA computation."""
import hashlib
import json
import os
from pathlib import Path

from prepare_old_mapping_control import FROZEN_TREE


def context():
    directory=Path(os.environ['TNCA_CAPTURED_INPUTS']).resolve()
    report=json.loads((directory/'report.json').read_text())
    if (report['status']!='functional-pass' or report['software_tree']!=FROZEN_TREE or
        report['backend']!='fpga-u280-multicore'):
        raise ValueError('complete frozen-source physical diagnostic input capture required')
    return directory,report


def tensor(directory,entry,shape,dtype):
    import torch
    path=directory/(entry.get('path') or entry['logical_contiguous_bytes'])
    if not path.resolve().is_relative_to(directory):raise ValueError('input escapes captured directory')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('captured input bytes changed')
    if entry['shape']!=list(shape) or entry['dtype']!=str(dtype):raise ValueError('captured input shape/dtype mismatch')
    value=torch.frombuffer(bytearray(data),dtype=dtype).clone().reshape(shape)
    if list(value.stride())!=entry['stride']:raise ValueError('capture replay requires its original contiguous layout')
    return value


def memory_inputs(elements):
    import torch
    directory,report=context();entries={i['name']:i for i in report['inputs']}
    if report['elements']!=elements or set(entries)!={'a','b'}:raise ValueError('memory input population differs')
    return tuple(tensor(directory,entries[name],(elements,),torch.float32) for name in ('a','b'))


def pipeline_inputs(graph,m,n,k,dtype):
    directory,report=context()
    if report['strided'] or report.get('cancel'):raise ValueError('only unmodified contiguous capture replay supported')
    rows=[r for r in report['rows'] if r['graph']==graph and r['shape']==[m,n,k]]
    if len(rows)!=1 or rows[0]['status']!='pass':raise ValueError('one successful captured workload required')
    entries={i['argument']:i for i in rows[0]['inputs']}
    if set(entries)!={'a','b','c'}:raise ValueError('pipeline input population differs')
    return tuple(tensor(directory,entries[name],shape,dtype) for name,shape in
                 (('a',(m,k)),('b',(k,n)),('c',(n,))))
