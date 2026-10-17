"""Standard Torch decode with a fixed KV capacity and tensor position.

This is a correctness path using decomposed SDPA. It does not claim online
attention or reduced cache traffic: masked cache updates touch the full bucket.
The frozen ArchBench model and point definitions remain unchanged.
"""
import torch
import torch.nn.functional as F
from archbench.ops import rotate_half


class BucketDecodeStep(torch.nn.Module):
    def __init__(self, model, cos, sin, capacity):
        super().__init__()
        self.model = model
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        # Explicit one-time initialization upload, before steady transfer audit.
        self.register_buffer('slots', torch.arange(capacity).to(cos.device))

    def forward(self, token, keys, values, position):
        c = self.model.c
        cos = self.cos.index_select(0, position)
        sin = self.sin.index_select(0, position)
        write = (self.slots == position).view(1, 1, -1, 1)
        visible = (self.slots <= position).view(1, 1, 1, -1)
        x = F.embedding(token, self.model.tok)
        nks, nvs = [], []
        for i, layer in enumerate(self.model.layers):
            y = F.rms_norm(x, (c.dim,), layer.attn_norm, c.eps)
            attn = layer.attn
            b, s, _ = y.shape
            q = attn.wq(y).view(b, s, c.n_heads, c.hd).transpose(1, 2)
            k = attn.wk(y).view(b, s, c.n_kv_heads, c.hd).transpose(1, 2)
            v = attn.wv(y).view(b, s, c.n_kv_heads, c.hd).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            full_k = torch.where(write, k, keys[i])
            full_v = torch.where(write, v, values[i])
            keys[i].copy_(full_k)
            values[i].copy_(full_v)
            rep = c.n_heads // c.n_kv_heads
            if rep > 1:
                full_k = full_k[:, :, None].expand(b, c.n_kv_heads, rep, -1, c.hd).reshape(b, c.n_heads, -1, c.hd)
                full_v = full_v[:, :, None].expand(b, c.n_kv_heads, rep, -1, c.hd).reshape(b, c.n_heads, -1, c.hd)
            o = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=visible)
            x = x + attn.wo(o.transpose(1, 2).reshape(b, s, c.n_heads * c.hd))
            y = F.rms_norm(x, (c.dim,), layer.mlp_norm, c.eps)
            x = x + layer.w2(F.silu(layer.w1(y)) * layer.w3(y))
            nks.append(k)
            nvs.append(v)
        logits = self.model._head(x)
        return logits, torch.stack(nks), torch.stack(nvs), logits.argmax(dim=-1), position + 1
