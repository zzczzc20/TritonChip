# Online SDPA checkpoint — not accepted

`make r40-online-attention` uses real native TNCA tensors and standard
`torch.nn.functional.scaled_dot_product_attention` under fullgraph Inductor.
The opt-in `tnca.online_attention.enable()` installs a device-specific
decomposition and Inductor Triton template. It requires FX graph caching
off and `TNCA_PACKED_WINDOW_UNPACK=1`. The audit also enables window reuse.
Default SDPA behavior remains the upstream mathematical decomposition.

Initial scope: inference, FP16/BF16, 16-wide heads with contiguous last
dimension. A program owns one query, streams 16 keys at a time, and retains
FP32 running max, normalization sum and output numerator. K/V use actual
descriptor DMA, followed by software unpack of the window snapshot.
Other shapes/dtypes/gradient paths retain the upstream decomposition.
The unused upstream auxiliary result is preserved in the decomposition;
standard F.sdpa uses output 0, and AOT/Inductor eliminate its unused math
graph. Explicit auxiliary-output consumers are not a fused checkpoint.

Native RTL 6/6: decode tail, causal prefill tail, bool/float all-mask rows,
permuted batch/head strides, BF16. Each online artifact contains two actual
t.ld.async instructions, two descriptors/barriers, 33 vector registers per
reported role and 512 B private scratch. Steady tensor/staging transfers
are zero. Unmasked graphs execute one kernel and allocate only the output
tensor (32/96/384 B); no full score/probability tensor is allocated. Bool
mask normalization additionally allocates 120 B and runs a second kernel.
All-mask rows return zeros and every output matches the FP32 reference.

`online_attention6_final.json` records allocation sizes, resource ledgers
and executed artifact hashes. Its full_inventory flag refers only to the
six checkpoint cases. A previous probe had descriptor metadata but LSU
data reads; the machine-code audit rejected it, prompting real packed DMA
unpack. It is not part of the positive evidence.

This is a correctness/allocation checkpoint, not frozen performance
acceptance. Long prefill, fused GQA/MQA, broader head dimensions, measured
intermediate HBM traffic, the 1.5 speed threshold and full attention/model
inventory remain pending. Fixed-bucket/64-step KV evidence is separate.
