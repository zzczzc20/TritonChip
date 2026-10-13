"""Opt-in standard SDPA inference template, initially for 16-wide heads.

One query per program streams 16 keys at a time through descriptors. The
stable FP32 softmax state stays in registers. Other shapes retain the
upstream math decomposition. This checkpoint requires FX graph caching off.
"""
_library = None


def enable():
    global _library
    if _library is not None:
        return
    import torch
    import os
    from torch._decomp import decomposition_table
    from torch._inductor import config, ir
    from torch._inductor.decomposition import decompositions
    from torch._inductor.lowering import register_lowering
    from torch._inductor.select_algorithm import TritonTemplate, realize_inputs
    if config.fx_graph_cache:
        raise RuntimeError('experimental online SDPA requires fx_graph_cache=False')
    if os.environ.get('TNCA_PACKED_WINDOW_UNPACK', '0') != '1':
        raise RuntimeError('online SDPA requires TNCA_PACKED_WINDOW_UNPACK=1 for real DMA loads')
    original_op = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default
    original = decomposition_table[original_op]
    library = torch.library.Library('tnca', 'FRAGMENT')
    library.define('sdpa_online_v1(Tensor q, Tensor k, Tensor v, Tensor? mask, bool causal, float scale) -> Tensor')

    def refuse(*args, **kwargs):
        raise RuntimeError('tnca online SDPA requires torch.compile(fullgraph=True)')
    library.impl('sdpa_online_v1', refuse, 'PrivateUse1')

    @torch.library.register_fake('tnca::sdpa_online_v1')
    def fake(q, k, v, mask, causal, scale):
        b, h, length, width = q.shape
        return torch.empty((length, b, h, width), dtype=q.dtype,
                           device=q.device).permute(1, 2, 0, 3)

    def grid(b, h, length, width, meta):
        return (b * h * length, 1, 1)

    common = '''
    B = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    L = {{size("Q", 2)}}
    S = {{size("K", 2)}}
    HK = {{size("K", 1)}}
    pid = tl.program_id(0)
    row = pid % L
    head = (pid // L) % H
    batch = pid // (L * H)
    kh = head // (H // HK)
    d = tl.arange(0, 16)
    r = tl.arange(0, 16)
    q = tl.load(Q + batch * {{stride("Q", 0)}} + head * {{stride("Q", 1)}} + row * {{stride("Q", 2)}} + d).to(tl.float32)
    kd = tl.make_tensor_descriptor(K + batch * {{stride("K", 0)}} + kh * {{stride("K", 1)}}, shape=[S, 16], strides=[{{stride("K", 2)}}, 1], block_shape=[16, 16])
    vd = tl.make_tensor_descriptor(V + batch * {{stride("V", 0)}} + kh * {{stride("V", 1)}}, shape=[S, 16], strides=[{{stride("V", 2)}}, 1], block_shape=[16, 16])
    m = tl.full((), -float('inf'), tl.float32)
    z = tl.full((), 0, tl.float32)
    o = tl.full((16,), 0, tl.float32)
    for block in range(tl.cdiv(S, 16)):
        pos = block * 16 + r
        k = kd.load([block * 16, 0]).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * SCALE
        valid = pos < S
        if CAUSAL:
            valid = valid & (pos <= row)
        MASK_CODE
        score = tl.where(valid, score, -float('inf'))
        new_m = tl.maximum(m, tl.max(score, 0))
        safe_m = tl.where(new_m == -float('inf'), 0., new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(score - safe_m)
        v = vd.load([block * 16, 0]).to(tl.float32)
        o = o * alpha + tl.sum(p[:, None] * v, 0)
        z = z * alpha + tl.sum(p, 0)
        m = new_m
    result = o / tl.where(z > 0., z, 1.)
    {{store_output(("batch", "head", "row", "d"), "result", "d < 16", val_shape=(16,))}}
'''
    mask_code = '''mask = tl.load(Mask + batch * MS0 + head * MS1 + row * MS2 + pos * MS3, pos < S, other=0)
        if BOOL_MASK:
            valid = valid & mask
        else:
            score = score + mask.to(tl.float32)'''
    templates = {}
    for has_mask in (False, True):
        header = '{{def_kernel("Q", "K", "V"' + (', "Mask"' if has_mask else '') + ')}}'
        templates[has_mask] = TritonTemplate(
            name=f'tnca_sdpa_online_v1_{int(has_mask)}', grid=grid,
            source=header + common.replace('MASK_CODE', mask_code if has_mask else ''))

    @register_lowering(torch.ops.tnca.sdpa_online_v1.default, type_promotion_kind=None)
    def lower(q, k, v, mask, causal, scale):
        nodes = list(realize_inputs(q, k, v))
        strides = [0, 0, 0, 0]
        if mask is not None:
            mask = realize_inputs(mask)
            nodes.append(mask)
            sizes, actual = list(mask.get_size()), list(mask.get_stride())
            strides = [0] * (4 - len(sizes)) + [0 if size == 1 else stride
                                                         for size, stride in zip(sizes, actual)]
        b, h, length, width = q.get_size()
        layout = ir.FixedLayout(q.get_device(), q.get_dtype(), [b, h, length, width],
                                [h * width, width, b * h * width, 1])
        choice = templates[mask is not None].generate(
            input_nodes=tuple(nodes), layout=layout, num_stages=1, num_warps=1,
            call_sizes=[b, h, length, width], CAUSAL=causal, SCALE=scale,
            BOOL_MASK=mask is not None and mask.get_dtype() == torch.bool,
            MS0=strides[0], MS1=strides[1], MS2=strides[2], MS3=strides[3])
        return choice.output_node()

    def decompose(q, k, v, dropout_p=0., is_causal=False, *, attn_mask=None, scale=None):
        eligible = (q.device.type == 'tnca' and q.ndim == k.ndim == v.ndim == 4
                    and q.dtype in (torch.float16, torch.bfloat16) and k.dtype == v.dtype == q.dtype
                    and q.shape[-1] == k.shape[-1] == v.shape[-1] == 16
                    and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
                    and q.shape[0] == k.shape[0] == v.shape[0]
                    and k.shape[1:3] == v.shape[1:3] and q.shape[1] % k.shape[1] == 0
                    and q.shape[-2] > 0 and k.shape[-2] > 0 and dropout_p == 0.
                    and not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))))
        if attn_mask is not None:
            shape = (1,) * (4 - attn_mask.ndim) + tuple(attn_mask.shape)
            eligible = eligible and attn_mask.ndim <= 4 and all(
                dim in (1, want) for dim, want in zip(shape, (q.shape[0], q.shape[1], q.shape[2], k.shape[2])))
        baseline = original(q, k, v, dropout_p, is_causal, attn_mask=attn_mask, scale=scale)
        if not eligible:
            return baseline
        result = torch.ops.tnca.sdpa_online_v1(q, k, v, attn_mask, is_causal,
                                              scale if scale is not None else 16 ** -.5)
        # Keep the upstream auxiliary-output semantics. F.sdpa consumes only
        # output 0; AOT/Inductor DCE removes the unused math graph entirely.
        return result, baseline[1]

    decompositions[original_op] = decompose
    _library = library
