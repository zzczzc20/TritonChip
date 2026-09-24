"""Adversarial inputs for the same, unmodified representative kernels.

Small dimensions deliberately leave inactive lanes in a physical vector.
Output tails and every input are checked by execute.py, not just the
active result. References here describe the source kernel's semantics
(e.g. radix_sort is one pass and top_k_selection retains tied values).
"""
import copy
import math

import cases as C


def build(Case):
    baseline = C.build(Case)
    out = {}

    def add(oid, label):
        case = copy.deepcopy(baseline[oid])
        out[f'{oid}/{label}'] = case
        return case

    # Ordinary masked elementwise operations: a single lane, an odd tail,
    # and no active lanes. Leave the rest of each output allocation intact.
    pointwise = ('vector_add', 'mul2', 'relu', 'leaky_relu', 'sigmoid',
                 'fused_activation', 'swiglu', 'dropout', 'quantize_global',
                 'dequantize_rowwise', 'weight_dequant',
                 'batch_normalization', 'matrix_copy')
    for oid in pointwise:
        for n in (0, 1, 7, 15):
            case = add(oid, f'tail{n}')
            case.scalars['n'] = n
            case.expect = case.expect[:n]

    for n in (1, 7, 15):
        for oid in ('reverse_array', 'interleave', 'radix_sort',
                    'argmax', 'mean_reduction', 'l2_norm', 'softmax',
                    'rmsnorm', 'layernorm', 'cross_entropy',
                    'kl_divergence', 'moe_topk_gating', 'top_k_selection',
                    '1d_conv', 'rope', 'destindex'):
            case = add(oid, f'tail{n}')
            case.scalars['n'] = n
            vals = next(iter(case.buffers.values()))[1][:n]
            if oid == 'reverse_array':
                case.expect = list(reversed(vals))
            elif oid == 'interleave':
                case.expect = case.expect[:2 * n]
            elif oid == 'radix_sort':
                case.expect = ([v for v in vals if not v & 1]
                               + [v for v in vals if v & 1])
            elif oid == 'argmax':
                case.expect = [vals.index(max(vals))]
            elif oid == 'mean_reduction':
                case.expect = [sum(vals) / n]
            elif oid == 'l2_norm':
                case.expect = [math.sqrt(sum(v * v for v in vals))]
            elif oid in ('softmax', 'cross_entropy', 'moe_topk_gating'):
                peak = max(vals)
                es = [math.exp(v - peak) for v in vals]
                if oid == 'softmax':
                    case.expect = [v / sum(es) for v in es]
                elif oid == 'cross_entropy':
                    case.scalars['target'] = n - 1
                    case.expect = [peak + math.log(sum(es)) - vals[-1]]
                else:
                    case.expect = [1.0 / sum(es)]
                    case.extra_outputs = [('idx_ptr', 0,
                                           [max(i for i, v in enumerate(vals)
                                                if v == peak)])]
            elif oid in ('rmsnorm', 'layernorm'):
                eps = 0.125
                case.scalars['eps'] = eps
                ws = case.buffers['w_ptr'][1][:n]
                if oid == 'rmsnorm':
                    scale = math.sqrt(sum(v * v for v in vals) / n + eps)
                    case.expect = [v / scale * w for v, w in zip(vals, ws)]
                else:
                    ds = [v - sum(vals) / n for v in vals]
                    # The kernel masks the variance reduction as well.
                    scale = math.sqrt(sum(v * v for v in ds) / n + eps)
                    bs = case.buffers['b_ptr'][1][:n]
                    case.expect = [v / scale * w + b
                                   for v, w, b in zip(ds, ws, bs)]
            elif oid == 'kl_divergence':
                qs = case.buffers['q_ptr'][1][:n]
                case.expect = [sum(p * math.log(p / q) for p, q in zip(vals, qs))]
            elif oid == 'top_k_selection':
                case.expect = [v if sum(w > v for w in vals) < 4 else -math.inf
                               for v in vals]
            elif oid == '1d_conv':
                ws = case.buffers['w_ptr'][1]
                case.expect = [sum(ws[r] * vals[i + r]
                                   for r in range(3) if i + r < n)
                               for i in range(n)]
            elif oid == 'rope':
                xs = case.buffers['x_ptr'][1]
                cs, sn = case.buffers['cos_ptr'][1], case.buffers['sin_ptr'][1]
                case.expect = [xs[i] * cs[i] - xs[n+i] * sn[i] for i in range(n)]
                case.expect += [xs[i] * sn[i] + xs[n+i] * cs[i] for i in range(n)]
            elif oid == 'destindex':
                perm = list(reversed(range(n))) + [0] * (16 - n)
                case.buffers['idx_ptr'] = ('i32', perm)
                case.expect = list(reversed(vals))

    for oid in ('argmax', 'moe_topk_gating', 'bitonic_sort', 'top_k_selection'):
        case = add(oid, 'ties')
        vals = [float(i % 3 - 1) for i in range(16)]
        name = 'logit_ptr' if oid == 'moe_topk_gating' else 'x_ptr'
        case.buffers[name] = ('fp32', vals)
        if oid == 'argmax':
            case.expect = [2]  # tl.argmax breaks ties to the left
        elif oid == 'moe_topk_gating':
            case.expect = [1 / sum(math.exp(v - 1) for v in vals)]
            case.extra_outputs = [('idx_ptr', 0, [14])]  # kernel uses max index
        elif oid == 'bitonic_sort':
            case.expect = sorted(vals)
        else:
            case.expect = [v if v == 1 else -math.inf for v in vals]

    for oid in ('l2_norm', 'rmsnorm', 'layernorm'):
        case = add(oid, 'zeros')
        case.buffers['x_ptr'] = ('fp32', [0.0] * 16)
        if oid == 'l2_norm':
            case.expect = [0.0]
        else:
            case.scalars['eps'] = 0.125
            case.expect = ([0.0] * 16 if oid == 'rmsnorm'
                           else case.buffers['b_ptr'][1][:])

    case = add('sigmoid', 'large')
    vals = [-80.0 + 10 * i for i in range(16)]
    case.buffers['x_ptr'] = ('fp32', vals)
    case.expect = [1 / (1 + math.exp(-v)) for v in vals]

    # Check every histogram bin, including nonzero, unaligned output offsets.
    for b in range(1, 4):
        case = add('histogramming', f'bin{b}')
        case.program_id = (b, 0, 0)
        case.out_offset = b
        case.expect = [case.buffers['x_ptr'][1].count(b)]

    # Run the second block with actual hardware program IDs. Prefix results
    # remain untouched; arithmetic still uses the original source kernel.
    for oid in pointwise:
        if oid in ('dequantize_rowwise', 'weight_dequant'):
            continue  # these use program_id as a row, not a vector block
        case = add(oid, 'block1')
        for name, (dtype, vals) in list(case.buffers.items()):
            if len(vals) == 16:
                case.buffers[name] = (dtype, vals + vals)
        case.scalars['n'] = 23
        case.program_id = (1, 0, 0)
        case.out_offset = 16
        case.expect = case.expect[:7]

    case = add('streamk_matmul', 'slice1_accumulate')
    case.program_id = (1, 0, 0)
    a, b = case.buffers['a_ptr'][1], case.buffers['b_ptr'][1]
    initial = [float(i % 7 - 3) for i in range(256)]
    case.buffers['c_ptr'] = ('fp32', initial)
    case.expect = [initial[m * 16 + n] + sum(a[m * 32 + 16 + k] * b[(16+k) * 16 + n]
                                           for k in range(16))
                   for m in range(16) for n in range(16)]

    # INT8 extremes exercise sign extension and the exact MMA substitution.
    case = add('matmul_int8', 'extremes')
    a = [(-128, 127, -1, 0)[i % 4] for i in range(256)]
    b = [(127, -128, 1, 0)[(i // 16 + i % 16) % 4] for i in range(256)]
    case.buffers['a_ptr'], case.buffers['b_ptr'] = ('i8', a), ('i8', b)
    case.expect = [sum(a[m*16+k] * b[k*16+n] for k in range(16)) * 0.25
                   for m in range(16) for n in range(16)]
    # Values beyond FP16 range and between adjacent FP16 numbers catch a
    # "fallback" that narrows the inputs. Dyadic products here are exact.
    case = add('matmul_fp32_fp16_fp8', 'fp32')
    a = [65536.0 + (i % 16) * 0.125 for i in range(256)]
    b = [((i // 16 + i % 16) % 7 - 3) * 0.25 for i in range(256)]
    case.buffers['a_ptr'], case.buffers['b_ptr'] = ('fp32', a), ('fp32', b)
    case.signature = {'a_ptr': '*fp32', 'b_ptr': '*fp32'}
    case.expect = [sum(a[m*16+k] * b[k*16+n] for k in range(16))
                   for m in range(16) for n in range(16)]
    for dtype, signature in (('e4m3', '*fp8e4nv'), ('e5m2', '*fp8e5')):
        case = add('matmul_fp32_fp16_fp8', dtype)
        # All values are exactly representable in either FP8 format.
        a = [float(i % 7 - 3) * 0.5 for i in range(256)]
        b = [float((i // 16 + i % 16) % 5 - 2) * 0.25 for i in range(256)]
        case.buffers['a_ptr'], case.buffers['b_ptr'] = (dtype, a), (dtype, b)
        case.signature = {'a_ptr': signature, 'b_ptr': signature}
        case.expect = [sum(a[m*16+k] * b[k*16+n] for k in range(16))
                       for m in range(16) for n in range(16)]
    return out
