"""Reduced-grid proxies of the heavy points of the 145 (R42 dse study, domain: SystemC/TLM model).

Loaded with `kfuse_speed_tlm_run.py --points-module`. A proxy keeps every dimension that defines a kernel and its per-program cost
(reduction depth, row length, head dimension, dtype, epilogue) and reduces only what adds independent programs (rows, tokens, heads,
batch, vocabulary), so that the model finishes in about a minute. It stands for ONE heavy point; the model cycles of the proxy are
scaled to that point by the ratio of the ArchBench demand named in PROXY_OF ('macs' or 'bytes') or by an explicit factor. They are
measurement proxies of this study: not registered in any suite, no result outside evidence/r42/dse/. The GEMM-shaped proxies of the
kfuse-speed study (tools/r42/kfuse_speed_proxies.py) are reused unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kfuse_speed_proxies as KP   # noqa: E402,F401  (registers its proxies)
import kfuse_speed_xl_points       # noqa: E402,F401  (registers the Hub FP8 proxies of the kfuse-speed study)
import archbench.ops as O          # noqa: E402
import archbench.models as M       # noqa: E402
from archbench.registry import Demand   # noqa: E402

PROXY_FOR = {stands: (pid, 'macs') for pid, stands in KP.PROXY_OF.items()}   # stands-for id -> (proxy id, demand kind of the scale)


def reg(pid, stands_for, kind):
    PROXY_FOR[stands_for] = (pid, kind)


def conv(nb, cin, h, w, cout, stands_for):
    def mk(seed, nb=nb, cin=cin, h=h, w=w, cout=cout):
        g = O.gen(seed)
        return O.Conv(cin, cout, g), (O.randn(g, (nb, cin, h, w), 'fp32'),)
    O.op('conv2d', 'large', 'fp32', dict(n=nb, cin=cin, h=h, w=w, cout=cout), mk,
         Demand(macs=nb * cout * cin * (3 * h - 2) * (3 * w - 2), read_bytes=(nb * cin * h * w + cout * cin * 9) * 4,
                write_bytes=nb * cout * h * w * 4), track=('general',), note='proxy of ' + stands_for)
    reg(f'op.conv2d/large/fp32-n{nb}-cin{cin}-h{h}-w{w}-cout{cout}', stands_for, 'macs')


def resnet(nb, ch, hw, stands_for):
    taps = (3 * hw - 2) ** 2

    def mk(seed, nb=nb, ch=ch, hw=hw):
        g = O.gen(seed)
        return M.ResNetBlock(ch, g).eval(), (O.randn(g, (nb, ch, hw, hw), 'fp32'),)
    M.model('resnet_block', 'large', 'fp32', dict(n=nb, c=ch, hw=hw), mk,
            Demand(macs=2 * nb * ch * ch * taps, read_bytes=4 * (2 * ch * ch * 9 + 8 * ch + nb * ch * hw * hw),
                   write_bytes=4 * nb * ch * hw * hw), track=('general',), tol=1e-3, note='proxy of ' + stands_for)
    reg(f'model.resnet_block/large/fp32-n{nb}-c{ch}-hw{hw}', stands_for, 'macs')


def mm32(m, k, n, stands_for):
    O.op('matmul', 'large', 'fp32', dict(m=m, k=k, n=n),
         lambda seed, m=m, k=k, n=n: (O.MatMul(), (O.randn(O.gen(seed), (m, k), 'fp32'), O.randn(O.gen(seed + 1), (k, n), 'fp32'))),
         Demand(macs=m * k * n, read_bytes=(m * k + k * n) * 4, write_bytes=m * n * 4), track=('general',), note='proxy of ' + stands_for)
    reg(f'op.matmul/large/fp32-m{m}-k{k}-n{n}', stands_for, 'macs')


def transpose(r, c, stands_for):
    O.op('transpose', 'large', 'fp32', dict(r=r, c=c), lambda seed, r=r, c=c: (O.Transpose(), (O.randn(O.gen(seed), (r, c), 'fp32'),)),
         Demand(read_bytes=r * c * 4, write_bytes=r * c * 4), tol=0.0, track=('general',), note='proxy of ' + stands_for)
    reg(f'op.transpose/large/fp32-r{r}-c{c}', stands_for, 'bytes')


def embedding(v, d, t, stands_for):
    def mk(seed, v=v, d=d, t=t):
        g = O.gen(seed)
        import torch
        return O.Embedding(v, d, 'fp16', g), (torch.randperm(v, generator=g)[:t],)
    O.op('embedding', 'large', 'fp16', dict(v=v, d=d, t=t), mk, Demand(read_bytes=t * 8 + t * d * 2, write_bytes=t * d * 2), tol=0.0,
         note='proxy of ' + stands_for)
    reg(f'op.embedding/large/fp16-v{v}-d{d}-t{t}', stands_for, 'bytes')


def eltwise(family, module, n_inputs, fp, sfu, n, stands_for):
    def make(seed, n=n):
        g = O.gen(seed)
        return module(), tuple(O.randn(g, (n,), 'fp16') for _ in range(n_inputs))
    O.op(family, 'large', 'fp16', dict(n=n), make, Demand(fp_ops=fp * n, sfu_ops=sfu * n, read_bytes=n_inputs * n * 2, write_bytes=n * 2),
         note='proxy of ' + stands_for)
    reg(f'op.{family}/large/fp16-n{n}', stands_for, 'bytes')


def softmax(r, c, stands_for):
    n = r * c
    O.op('softmax', 'large', 'fp16', dict(r=r, c=c), lambda seed, r=r, c=c: (O.Softmax(), (O.randn(O.gen(seed), (r, c), 'fp16', 2.0),)),
         Demand(fp_ops=3 * n, sfu_ops=n, read_bytes=n * 2, write_bytes=n * 2), note='proxy of ' + stands_for)
    reg(f'op.softmax/large/fp16-r{r}-c{c}', stands_for, 'bytes')


def attn_prefill(h, s, d, stands_for):
    b = 1
    tri = s * (s + 1) // 2
    O.op('attention_prefill', 'large', 'fp16', dict(b=b, h=h, s=s, d=d),
         lambda seed, b=b, h=h, s=s, d=d: (O.CausalAttention(), tuple(O.randn(O.gen(seed + i), (b, h, s, d), 'fp16') for i in range(3))),
         Demand(macs=2 * b * h * d * tri, fp_ops=3 * b * h * tri, sfu_ops=b * h * tri, read_bytes=3 * b * h * s * d * 2,
                write_bytes=b * h * s * d * 2), note='proxy of ' + stands_for)
    reg(f'op.attention_prefill/large/fp16-b{b}-h{h}-s{s}-d{d}', stands_for, 'macs')


def gemv16(k, n, stands_for):
    def mk(seed, k=k, n=n):
        g = O.gen(seed)
        return O._linear(k, n, 'fp16', g, False), (O.randn(g, (1, k), 'fp16'),)
    O.op('gemv', 'large', 'fp16', dict(k=k, n=n), mk, Demand(macs=k * n, read_bytes=(k + k * n) * 2, write_bytes=n * 2),
         note='proxy of ' + stands_for)
    reg(f'op.gemv/large/fp16-k{k}-n{n}', stands_for, 'macs')


# ---- the 113-point heavy tiers (large, hbm) that the model cannot run at full size -------------------------------
conv(1, 64, 28, 28, 64, 'op.conv2d/large/fp32-n8-cin64-h56-w56-cout64')
resnet(1, 64, 28, 'model.resnet_block/large/fp32-n8-c64-hw56')
mm32(48, 2048, 2048, 'op.matmul/large/fp32-m2048-k2048-n2048')
transpose(1024, 4096, 'op.transpose/hbm/fp32-r8192-c8192')
reg('op.transpose/large/fp32-r1024-c4096', 'op.transpose/large/fp32-r4096-c4096', 'bytes')
embedding(8192, 2048, 512, 'op.embedding/hbm/fp16-v65536-d2048-t8192')
eltwise('silu', O.SiLU, 1, 2, 1, 1 << 22, 'op.silu/hbm/fp16-n67108864')
softmax(256, 8192, 'op.softmax/hbm/fp16-r4096-c8192')
attn_prefill(4, 512, 64, 'op.attention_prefill/large/fp16-b1-h32-s512-d64')
# ---- xl tier fp16 points ---------------------------------------------------------------------------------------------
gemv16(2048, 16384, 'op.gemv/xl/fp16-k2048-n151936')
embedding(16384, 2048, 2048, 'op.embedding/xl/fp16-v151936-d2048-t2048')
O.op('linear', 'large', 'fp16', dict(m=96, k=2048, n=6144, bias=0),
     lambda seed: (O._linear(2048, 6144, 'fp16', O.gen(seed), False), (O.randn(O.gen(seed), (96, 2048), 'fp16'),)),
     Demand(macs=96 * 2048 * 6144, read_bytes=(96 * 2048 + 2048 * 6144) * 2, write_bytes=96 * 6144 * 2), note='proxy of xl linear')
reg('op.linear/large/fp16-m96-k2048-n6144-bias0', 'op.linear/xl/fp16-m2048-k2048-n6144-bias0', 'macs')


def gqa_prefill(h, kv, s, stands_for_list):
    import archbench.xl as X
    tri = s * (s + 1) // 2

    def make(seed, h=h, kv=kv, s=s):
        g = X.gen(seed)
        return X.GQA(True), (X._randn(g, (1, h, s, X.HD), 1.0, X.FP16), X._randn(g, (1, kv, s, X.HD), 1.0, X.FP16),
                             X._randn(g, (1, kv, s, X.HD), 1.0, X.FP16))
    O.op('attention_prefill', 'large', 'fp16', dict(b=1, h=h, kv=kv, s=s, d=X.HD), make,
         Demand(macs=2 * h * X.HD * tri, fp_ops=3 * h * tri, sfu_ops=h * tri, read_bytes=(h + 2 * kv) * s * X.HD * 2, write_bytes=h * s * X.HD * 2),
         note='proxy of ' + ', '.join(stands_for_list))
    for st in stands_for_list:
        reg(f'op.attention_prefill/large/fp16-b1-h{h}-kv{kv}-s{s}-d{X.HD}', st, 'macs')


gqa_prefill(4, 2, 512, ['op.attention_prefill/xl/fp16-b1-h16-kv8-s1024-d128', 'op.attention_prefill/xl/fp16-b1-h16-kv8-s2048-d128'])


if __name__ == '__main__':      # python tools/r42/dse_proxies.py OUT.json  (needs the archbench imports: run where torch exists)
    import json
    out = {stands: dict(proxy=pid, kind=kind) for stands, (pid, kind) in PROXY_FOR.items()}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(len(out), 'proxies ->', sys.argv[1])
