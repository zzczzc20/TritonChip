"""Standard Torch decode with a fixed KV capacity and tensor position.

This is a correctness path using decomposed SDPA. It does not claim online
attention or reduced cache traffic: masked cache updates touch the full bucket.
The frozen ArchBench model and point definitions remain unchanged.
"""
import torch
import torch.nn.functional as F
from archbench.ops import rotate_half


class GenerationPrefillState(torch.nn.Module):
    """Original prefill outputs, optionally padded to one resident KV bucket."""
    def __init__(self, model, capacity=None):
        super().__init__()
        self.model = model
        self.capacity = capacity

    def forward(self, tokens):
        m = self.model
        x = F.embedding(tokens,m.tok)
        keys,values = [],[]
        for layer in m.layers:
            x,k,v = layer(x,m.cos,m.sin)
            keys.append(k);values.append(v)
        logits = m._head(x[:,-1:,:])
        keys,values = torch.stack(keys),torch.stack(values)
        if self.capacity is not None:
            padding = self.capacity - keys.shape[-2]
            if padding < 0:
                raise ValueError('KV bucket is smaller than the complete original prompt')
            keys,values = F.pad(keys,(0,0,0,padding)),F.pad(values,(0,0,0,padding))
            position = torch.full((1,),tokens.shape[1],dtype=torch.int64,device=tokens.device)
            return logits,keys,values,logits.float().argmax(-1),position
        return logits,keys,values,logits.float().argmax(-1)


class BucketDecodeStep(torch.nn.Module):
    def __init__(self, model, cos, sin, capacity):
        super().__init__()
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('KV capacity must be a positive integer')
        if cos.shape != sin.shape or cos.ndim != 2 or cos.shape[0] < capacity or cos.shape[1] != model.c.hd:
            raise ValueError('resident RoPE tables must cover the complete KV bucket')
        self.model = model
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        # Explicit one-time initialization upload, before steady transfer audit.
        self.register_buffer('slots', torch.arange(capacity).to(cos.device))

    def forward(self, token, keys, values, position):
        c = self.model.c
        if keys.ndim != 5 or keys.shape != values.shape or keys.shape[0] != c.n_layers or keys.shape[1] != token.shape[0] or keys.shape[2] != c.n_kv_heads or keys.shape[3] != self.slots.shape[0] or keys.shape[4] != c.hd:
            raise ValueError('cache shape differs from this fixed bucket specialization')
        if position.shape != (1,) or position.dtype != torch.int64:
            raise ValueError('decode position must be one resident int64 tensor value')
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


class BucketGenerateStep(torch.nn.Module):
    """Expose the resident full cache for the complete generation audit.

    Cache objects stay the same across all positions; only explicit validation
    downloads leave the device. The tensor position advances on the device.
    """
    def __init__(self, model, cos, sin, capacity):
        super().__init__()
        self.step = BucketDecodeStep(model, cos, sin, capacity)

    def forward(self, token, keys, values, position):
        logits, _new_k, _new_v, next_token, next_position = self.step(token, keys, values, position)
        return logits, keys, values, next_token, next_position
