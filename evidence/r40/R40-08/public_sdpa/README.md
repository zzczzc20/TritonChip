# R40-08 public SDPA/GQA/MQA checkpoint — not accepted

The opt-in online template now intercepts eligible standard public SDPA
before its CPU-style backend selector chooses a decomposed GQA/MQA path.
Unsupported dtype/layout/shape, training/dropout and incompatible head
counts retain upstream decomposition. Positive heads are required before
head division. The existing flash auxiliary-output semantics are retained.

Native RTL passes GQA Hq=4/Hkv=2 with odd S=19 and causal BF16 MQA
Hq=4/Hkv=1 with S=17 (2/2). Each executes one fused kernel, allocates only
the final 384-byte output, loads both K/V through real descriptor DMA, and
records zero steady tensor transfers. State remains FP32; no probability
FP16 conversion or relaxed numerical threshold is introduced.

All original six cases pass with the public interception: causal/tail,
batch/permuted strides, FP16/BF16 and boolean/float all-mask rows. The
boolean mask normalization retains its separate 120-byte allocation and
kernel, fully recorded. The longer L=17/S=129 check is running separately.
These are correctness, provenance and allocation smokes; full frozen
attention/model inventory, larger heads, long-prefill speed and intermediate
traffic thresholds remain required. The fixed-bucket 64-step board audit is
independent; no R40-08 acceptance is claimed.
