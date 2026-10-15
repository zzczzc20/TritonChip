# R40-08 wide-head functional checkpoints

These are functional diagnostics, not performance or full R40-08 signoff.

* `automatic_activation.json`: automatic TNCA_SDPA=fused activation, actual RTL execution, all three selected D32/D64 cases pass (replaces the earlier one-case partial capture).
* `board_hybrid.json`: three selected cases pass independently on the hybrid single-core U280 image.
* `board_inventory_initial_partial.json` / failure log: the default local-prefix full sequence passes nine cases and then fails D32 decode numerics. This failure remains open.
* `board_hbm_inventory.json` / log / `run_board_hbm.sh`: all fourteen cases pass on the same board image when heap_base=524288 puts tensor/argument storage beyond its local prefix. This does not certify the failing default policy, HBM traffic reduction or performance.
* `board_failure_probe.py` / JSON: diagnostic inspection after the sequence failure; input and code bytes agree with their sources, and the actual argument/descriptor bytes are preserved. Private runtime APIs and manual replays are diagnostic only.
* `board_await_squash_negative.*`: a temporary codegen probe adding a taken branch after mb.wait still fails. It is negative evidence, not a committed implementation change. The original codegen was restored.

`board_sources.json` identifies source overlay, image hash and loaded UUID. `board_dynamic_regions.json` independently records the loaded board UUID. The image is a single-core hybrid image, not the pending three-core ownership build. `sources.json` retains the historical RTL checkpoint source identity; it must not be relabelled as the current implementation. SHA256SUMS protects all files in this directory except itself.

The historical default-policy sequence failure above is resolved by the SPMD TensorMap publication fix; the complete same-image 14/14 regression and causal snapshot are in ../descriptor_publication/. Historical raw failures and their source metadata remain unchanged.
