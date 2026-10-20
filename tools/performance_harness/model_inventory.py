#!/usr/bin/env python3
"""Inventory checkpoint storage from pinned safetensors headers, without weights.

The include prefix selects a branch for feasibility analysis; it never alters
the checkpoint or an execution graph. Runtime scratch/state/allocator peaks
must be measured separately before claiming capacity sufficiency.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from urllib.parse import quote
from urllib.request import Request, urlopen


DTYPE_BYTES = dict(BOOL=1, U8=1, I8=1, F8_E4M3=1, F8_E5M2=1,
                   I16=2, U16=2, F16=2, BF16=2, I32=4, U32=4,
                   F32=4, I64=8, U64=8, F64=8)


def validate_header(header, *, payload_bytes=None):
    """Check byte counts and disjoint ranges before counting storage.

    Exact aliases are counted once; partial overlaps and differently typed
    aliases cannot substantiate a distinct-storage capacity estimate.
    """
    ranges = {}
    for name, value in header.items():
        if name == '__metadata__':
            continue
        shape, dtype = value['shape'], value['dtype']
        lo, hi = value['data_offsets']
        if not (type(lo) is int and type(hi) is int and 0 <= lo <= hi):
            raise ValueError(f'invalid tensor storage offsets: {name}')
        if (not isinstance(shape, list) or
                any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError(f'invalid tensor shape: {name}')
        if dtype not in DTYPE_BYTES:
            raise ValueError(f'unsupported tensor dtype: {dtype}')
        if hi-lo != math.prod(shape)*DTYPE_BYTES[dtype]:
            raise ValueError(f'tensor dtype/shape/storage byte mismatch: {name}')
        if payload_bytes is not None and hi > payload_bytes:
            raise ValueError(f'tensor extends past shard payload: {name}')
        if hi == lo:
            continue
        if (lo, hi) in ranges and ranges[lo, hi] != dtype:
            raise ValueError(f'differently typed storage aliases: {name}')
        ranges[lo, hi] = dtype
    end = 0
    for lo, hi in sorted(ranges):
        if lo < end:
            raise ValueError('partially overlapping tensor storage ranges')
        end = hi
    if payload_bytes is not None and end != payload_bytes:
        raise ValueError('header does not cover the end of the shard payload')
    return dict(tensors=sum(name != '__metadata__' for name in header),
                distinct_storages=len(ranges), distinct_bytes=sum(hi-lo for lo, hi in ranges))


def audit_snapshot(out):
    """Independently audit a saved header-only inventory without networking."""
    report = json.loads((out/'inventory.json').read_text())
    for name, file in (('repository', 'repository.json'), ('config', 'config.json'),
                       ('index', 'index.json')):
        if hashlib.sha256((out/file).read_bytes()).hexdigest() != report['source_sha256'][name]:
            raise ValueError(f'saved {name} source identity changed')
    index = json.loads((out/'index.json').read_text())['weight_map']
    found, selected, shards = {}, {}, {}
    for shard, expected in report['source_sha256']['headers'].items():
        raw = (out/(shard+'.header.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'saved header identity changed: {shard}')
        header = json.loads(raw)
        size = report['shard_lfs'][shard].get('size')
        payload = size-8-len(raw) if size is not None else None
        shards[shard] = validate_header(header, payload_bytes=payload)
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if index.get(name) != shard or name in found:
                raise ValueError('saved index/header coverage mismatch')
            found[name] = (shard, *value['data_offsets'])
            if name.startswith(report['include_prefix']):
                selected[name] = found[name]
                if report['selected_tensors'][name] != dict(shard=shard, dtype=value['dtype'],
                        shape=value['shape'], offsets=value['data_offsets'],
                        bytes=value['data_offsets'][1]-value['data_offsets'][0]):
                    raise ValueError(f'selected tensor accounting changed: {name}')
    if set(found) != set(index) or set(selected) != set(report['selected_tensors']):
        raise ValueError('incomplete saved inventory coverage')
    all_bytes = sum(hi-lo for _, lo, hi in set(found.values()))
    selected_bytes = sum(hi-lo for _, lo, hi in set(selected.values()))
    if (all_bytes != report['checkpoint_distinct_bytes'] or
            selected_bytes != report['selected_distinct_bytes'] or
            len(selected) != report['selected_tensor_count'] or
            len(set(selected.values())) != report['selected_distinct_storages']):
        raise ValueError('saved inventory storage total mismatch')
    return dict(schema='tnca-storage-header-audit/1', status='pass',
                revision=report['revision'], shards=shards,
                selected_distinct_bytes=selected_bytes, checkpoint_distinct_bytes=all_bytes)


def read_bounded(url, limit, *, byte_range=None):
    headers = {'User-Agent': 'TritonChip-feasibility-inventory/1'}
    if byte_range is not None:
        headers['Range'] = f'bytes={byte_range[0]}-{byte_range[1]}'
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        status = response.status
        content_range = response.headers.get('Content-Range')
        data = response.read(limit)
    return data, status, content_range


def json_file(url, path):
    raw, _, _ = read_bounded(url, 16 << 20)
    value = json.loads(raw)
    path.write_bytes(raw)
    return value, hashlib.sha256(raw).hexdigest()


def tensor_header(url):
    first, status, content_range = read_bounded(url, 8, byte_range=(0, 7))
    if len(first) != 8:
        raise ValueError('truncated safetensors header length')
    length, = struct.unpack('<Q', first)
    if not 0 < length <= 16 << 20:
        raise ValueError('invalid or unbounded safetensors header length')
    raw, status, content_range = read_bounded(url, length + 8, byte_range=(8, length + 7))
    if status == 200:
        raw = raw[8:]
    elif status != 206 or not (content_range or '').startswith('bytes 8-'):
        raise ValueError('server returned an unexpected byte range')
    if len(raw) != length:
        raise ValueError('truncated safetensors header')
    return json.loads(raw), raw


def inventory(repo, revision, prefix, out):
    out.mkdir(parents=True, exist_ok=False)
    base = f'https://huggingface.co/{quote(repo, safe="/")}'
    info, api_sha = json_file(f'https://huggingface.co/api/models/{quote(repo, safe="/")}/revision/{quote(revision, safe="")}?blobs=true', out/'repository.json')
    commit = info['sha']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('the repository revision did not resolve to an immutable commit')
    resolve = f'{base}/resolve/{commit}'
    cfg, config_sha = json_file(resolve+'/config.json', out/'config.json')
    index, index_sha = json_file(resolve+'/model.safetensors.index.json', out/'index.json')
    tensors, headers, files = {}, {}, {}
    siblings = {item['rfilename']: item for item in info.get('siblings', [])}
    for shard in sorted(set(index['weight_map'].values())):
        header, raw = tensor_header(resolve+'/'+quote(shard, safe='/'))
        target = out/(shard+'.header.json')
        target.write_bytes(raw)
        headers[shard] = hashlib.sha256(raw).hexdigest()
        files[shard] = siblings.get(shard, {}).get('lfs', {})
        size = files[shard].get('size')
        validate_header(header, payload_bytes=size-8-len(raw) if size is not None else None)
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if index['weight_map'].get(name) != shard:
                raise ValueError(f'checkpoint index/header mismatch: {name}')
            lo, hi = value['data_offsets']
            if not (type(lo) is int and type(hi) is int and 0 <= lo <= hi):
                raise ValueError('invalid tensor storage offsets')
            tensors[name] = dict(shard=shard, dtype=value['dtype'], shape=value['shape'],
                                 bytes=hi-lo, offsets=[lo, hi])
    if set(tensors) != set(index['weight_map']):
        raise ValueError('incomplete checkpoint tensor coverage')
    selected = {name: value for name, value in tensors.items() if name.startswith(prefix)}
    if not selected:
        raise ValueError('the requested language/branch prefix selected no tensors')
    storages = {(v['shard'], *v['offsets']): v['bytes'] for v in selected.values()}
    all_storages = {(v['shard'], *v['offsets']): v['bytes'] for v in tensors.values()}
    sizes = Counter()
    for key, n in storages.items():
        dtype = next(v['dtype'] for v in selected.values() if (v['shard'], *v['offsets']) == key)
        sizes[dtype] += n
    text = cfg.get('text_config', cfg)
    layers = Counter(text.get('layer_types', []))
    report = dict(schema='tnca-checkpoint-storage-inventory/1', repository=repo, revision=commit,
                  repository_url=f'{base}/tree/{commit}', include_prefix=prefix,
                  source_sha256=dict(repository=api_sha, config=config_sha, index=index_sha, headers=headers),
                  shard_lfs=files, checkpoint_distinct_bytes=sum(all_storages.values()),
                  selected_tensor_count=len(selected), selected_distinct_storages=len(storages),
                  selected_distinct_bytes=sum(storages.values()), selected_bytes_by_dtype=dict(sizes),
                  selected_elements=sum(math.prod(v['shape']) for v in selected.values()),
                  selected_tensors=selected, excluded_tensor_count=len(tensors)-len(selected),
                  text_configuration=text, layer_types=dict(layers),
                  tied_embeddings_declared=text.get('tie_word_embeddings', cfg.get('tie_word_embeddings', False)),
                  runtime_capacity_verdict='not-measured: requires state/scratch/pack/code/allocator peak',
                  execution_verdict='not-measured: headers are not functional execution')
    (out/'inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-existing', type=Path)
    ap.add_argument('--repo')
    ap.add_argument('--revision', default='main')
    ap.add_argument('--include-prefix')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.audit_existing:
        report = audit_snapshot(args.audit_existing)
        (args.audit_existing/'header-audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        raise SystemExit(0)
    if not (args.repo and args.include_prefix and args.out):
        ap.error('--repo, --include-prefix and --out are required for a new inventory')
    report = inventory(args.repo, args.revision, args.include_prefix, args.out)
    print(json.dumps({k: report[k] for k in ('repository', 'revision', 'selected_tensor_count',
                                             'selected_distinct_bytes', 'checkpoint_distinct_bytes',
                                             'layer_types', 'runtime_capacity_verdict')}, indent=2))
