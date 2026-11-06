"""Analyze the pinned official language model without allocating checkpoint data.

This is an ATen shape/storage analysis, not numerical execution, a TNCA
compilation, or a device capacity qualification.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import operator
from pathlib import Path
import resource
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--prefill', type=int, default=128)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.prefill <= 0:
        ap.error('positive prefill length required')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='tnca-official-language-meta-analysis/1', status='running',
                  scope='all official language layers, prefill, no cache, all logits; meta tensors only',
                  limitations=['no numerical validation', 'no TNCA code generation',
                               'ATen storage liveness is not an Inductor allocator peak',
                               'persistent decode caches are not part of this no-cache graph'])
    def save():
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    started = time.monotonic()
    try:
        import torch
        import transformers
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.utils._pytree import tree_leaves
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        sources = json.loads(args.sources.read_text())
        audited = {}
        for name, entry in sources['files'].items():
            module_name = name.removeprefix('src/').removesuffix('.py').replace('/', '.')
            module = importlib.import_module(module_name)
            actual = digest(module.__file__)
            if actual != entry['sha256']:
                raise ValueError(f'official source identity mismatch: {name}')
            audited[name] = actual
        config = Qwen3_5TextConfig(**json.loads(args.config.read_text())['text_config'])
        config._attn_implementation = 'eager'
        with torch.device('meta'):
            model = modeling.Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        inventory = json.loads(args.inventory.read_text())
        selected = inventory['selected_tensors']
        matched = {}
        for name, parameter in model.named_parameters():
            source_name = name.replace('model.', 'model.language_model.', 1)
            entry = selected[source_name]
            if list(parameter.shape) != entry['shape']:
                raise ValueError(f'checkpoint shape mismatch: {name}')
            dtype = {'BF16': torch.bfloat16, 'F32': torch.float32}[entry['dtype']]
            parameter.data = parameter.data.to(dtype=dtype)
            matched[source_name] = parameter.numel()*parameter.element_size()
        if set(matched) != set(selected) or sum(matched.values()) != inventory['selected_distinct_bytes']:
            raise ValueError('official language parameters do not exactly cover audited checkpoint inventory')
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise ValueError('official tied embedding alias was lost')
        report.update(tools=dict(torch=str(torch.__version__), transformers=transformers.__version__),
                      official_source_revision=sources['revision'], official_source_sha256=audited,
                      checkpoint_revision=inventory['revision'], config_sha256=digest(args.config),
                      inventory_sha256=digest(args.inventory), parameter_count=len(matched),
                      parameter_bytes=sum(matched.values()), tied_embedding=True,
                      layers=len(model.model.layers), prefill=args.prefill)
        save()
        ids = torch.empty((1,args.prefill), dtype=torch.int64, device='meta')
        def forward(input_ids):
            # Public mask-mapping argument: one fixed, unpadded causal bucket.
            # Avoid value-dependent packed-sequence discovery on meta tensors.
            mask = torch.full((args.prefill,args.prefill), float('-inf'),
                              dtype=torch.bfloat16, device=input_ids.device).triu(1)
            masks = {'full_attention':mask[None,None,:,:], 'linear_attention':None}
            return model(input_ids=input_ids, attention_mask=masks, use_cache=False).logits
        report['mask_adapter'] = 'public mask mapping for unpadded fixed causal prefill; triangular mask construction included in graph'
        trace_started = time.monotonic()
        with torch.no_grad():
            graph = make_fx(forward, tracing_mode='real')(ids)
        report['trace_seconds'] = time.monotonic()-trace_started
        code = graph.code
        (args.out/'aten_graph.py').write_text(code)
        parameters = {p.untyped_storage()._cdata for p in graph.parameters()}
        parameters.update(b.untyped_storage()._cdata for b in graph.buffers())
        # make_fx metadata can wrap each view in a fresh FakeTensor storage.
        # Recover aliases from the ATen schema, not metadata object identity.
        parent = {}
        def root(key):
            while key in parent:
                key = parent[key]
            return key
        def merge(a,b):
            a,b = root(a),root(b)
            if a != b:
                parent[a] = b
        def tensors(value):
            return [v for v in tree_leaves(value) if isinstance(v,torch.Tensor)]
        def values(argument):
            if isinstance(argument,torch.fx.Node):
                return tensors(argument.meta.get('val'))
            return [v for node in tree_leaves(argument) if isinstance(node,torch.fx.Node)
                    for v in tensors(node.meta.get('val'))]
        for node in graph.graph.nodes:
            outputs = tensors(node.meta.get('val'))
            if node.op == 'get_attr':
                attr = graph
                for part in node.target.split('.'):
                    attr = getattr(attr,part)
                if isinstance(attr,torch.Tensor):
                    for value in outputs:
                        merge(value.untyped_storage()._cdata,attr.untyped_storage()._cdata)
            elif node.op == 'call_function' and node.target is operator.getitem:
                source,index = node.args
                if isinstance(source,torch.fx.Node):
                    source_value = source.meta.get('val')
                    if isinstance(source_value,(tuple,list)):
                        for result,original in zip(outputs,tensors(source_value[index]),strict=True):
                            merge(result.untyped_storage()._cdata,original.untyped_storage()._cdata)
            elif node.op == 'call_function' and node.target is torch.ops.aten._unsafe_view.default:
                # _unsafe_view deliberately omits alias annotations for
                # autograd, but its result uses the input allocation.
                original = values(node.args[0])
                if len(original) != 1 or len(outputs) != 1:
                    raise ValueError('unexpected _unsafe_view tensor arity')
                merge(outputs[0].untyped_storage()._cdata,original[0].untyped_storage()._cdata)
            elif node.op == 'call_function' and hasattr(node.target,'_schema'):
                schema = node.target._schema
                for returned in schema.returns:
                    if returned.alias_info is None:
                        continue
                    aliases = returned.alias_info.before_set | returned.alias_info.after_set
                    # PyTorch's schema API omits the element alias set of
                    # Tensor(a)[] for these view-list operators.
                    if not aliases and schema.name in ('aten::split_with_sizes','aten::split','aten::unbind'):
                        aliases = schema.arguments[0].alias_info.before_set
                    candidates = []
                    for i,arg in enumerate(schema.arguments):
                        if arg.alias_info is None:
                            continue
                        if aliases & (arg.alias_info.before_set | arg.alias_info.after_set):
                            actual = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
                            candidates.extend(values(actual))
                    roots = {root(v.untyped_storage()._cdata) for v in candidates}
                    if len(roots) != 1:
                        raise ValueError(f'cannot resolve storage alias for {node.name}: {schema}')
                    if len(schema.returns) != 1:
                        raise ValueError(f'multiple aliased returns need explicit handling: {schema}')
                    for result in outputs:
                        merge(result.untyped_storage()._cdata,next(iter(roots)))
        parameters = {root(key) for key in parameters}
        intervals = {}
        rows = []
        nodes = list(graph.graph.nodes)
        indexes = {n:i for i,n in enumerate(nodes)}
        histogram = Counter()
        for i,node in enumerate(nodes):
            if node.op == 'call_function':
                histogram[str(node.target)] += 1
            tensors = []
            for value in tree_leaves(node.meta.get('val')):
                if not isinstance(value, torch.Tensor):
                    continue
                storage = value.untyped_storage()
                key = root(storage._cdata)
                size = storage.nbytes()
                last = max((indexes[user] for user in node.users), default=i)
                if key not in parameters:
                    entry = intervals.setdefault(key, dict(start=i, last=last, bytes=size, producers=[]))
                    entry['start'] = min(entry['start'], i)
                    entry['last'] = max(entry['last'], last)
                    entry['bytes'] = max(entry['bytes'], size)
                    entry['producers'].append(node.name)
                tensors.append(dict(shape=list(value.shape), stride=list(value.stride()),
                                    dtype=str(value.dtype), storage_bytes=size,
                                    persistent_parameter_or_buffer=key in parameters))
            rows.append(dict(name=node.name, op=node.op, target=str(node.target), tensors=tensors))
        live = peak = peak_at = 0
        starts, ends = Counter(), Counter()
        for entry in intervals.values():
            starts[entry['start']] += entry['bytes']
            ends[entry['last']] += entry['bytes']
        for i in range(len(nodes)):
            live += starts[i]
            if live > peak:
                peak,peak_at = live,i
            live -= ends[i]
        if live:
            raise ValueError('storage liveness sweep did not drain')
        (args.out/'aten_nodes.json').write_text(json.dumps(rows, indent=2)+'\n')
        (args.out/'activation_storage_intervals.json').write_text(json.dumps(list(intervals.values()),indent=2)+'\n')
        report.update(status='meta-analysis-complete', aten_nodes=len(nodes),
                      aten_operator_histogram=dict(sorted(histogram.items())),
                      aten_graph_source_bytes=len(code.encode()), aten_graph_sha256=digest(args.out/'aten_graph.py'),
                      activation_storage_peak_bytes=peak, activation_storage_peak_node=nodes[peak_at].name,
                      activation_storage_scope='straight-line ATen graph, allocate before last-use release, aliases share storage; excludes operator internal workspaces',
                      storage_alias_method='ATen schema aliases and get_attr/getitem provenance; metadata storage IDs alone are insufficient',
                      analysis_script_sha256=digest(__file__),
                      parameter_plus_activation_storage_peak_bytes=sum(matched.values())+peak,
                      peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      total_seconds=time.monotonic()-started)
    except Exception:
        report.update(status='failed-preserved', error=traceback.format_exc(), total_seconds=time.monotonic()-started)
        raise
    finally:
        save()
    print(json.dumps({k:report[k] for k in ('status','layers','aten_nodes','activation_storage_peak_bytes','total_seconds')},indent=2))


if __name__ == '__main__':
    main()
