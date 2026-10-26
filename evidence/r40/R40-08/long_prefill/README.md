# Online SDPA long-prefill tail checkpoint

Jump-host native TNCA HBM RTL simulation, public standard Torch SDPA test
`long_prefill_tail`: batch/head 1, query length 17, key/value length 129,
head dimension 16, FP16 inputs and FP32 online state. The existing tolerance
checks passed on warm and resident audited execution; one kernel, two descriptor
loads, 512 bytes scratch, 33 VRF registers per role, no steady tensor transfers.

The JSON explicitly declares performance unmeasured and inventory partial.
This is not the full R40-08 long-attention performance or traffic acceptance.
Source: `tools/r40/online_attention_tests.py` at 970afbc; source tree
`~/tritonchip-r40-online2`, profile `core-vrf-sync-v5/hbm`.
