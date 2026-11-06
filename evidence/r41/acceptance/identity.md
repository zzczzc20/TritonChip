# Run-1 identity reconciliation

Evidence cutoff: 2026-10-26 UTC. This binds measured campaigns to the run-1
candidate, not to an unselected final B41-F release.

| Identity | Reconciled value and evidence |
|---|---|
| Image | SHA256 `30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de`; UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3` |
| Hardware source | `c9f25787494a8414f9ab19e03bea6fb3fa794d20`; source archive SHA256 `4a2dcaf62b7b3c9d6ff94144dc51793003f8e49e51c9817a19361738abfadd48` |
| Source verification | 128 registered files, digest `14abdc9d8f302f164336c5cb6176b4051fe6e1afb8640f720b3fd551eb2c7660`; measured `s2-identity/source.json` reports no missing, changed or extra files |
| Configuration | `core-vrf-sync-v5-x8`, three cores, three 256-bit outlets; `HAS_TENSOR_XPOSE=2`, `HAS_SFU=1`; complete parameters in `evidence/r41/stage2/integration/image.json` and `image.tcl` |
| Build | Xilinx Vivado/Vitis/v++ 2023.2; U280 multicore-HBM flow, two link jobs, 100 MHz requested; source/flow and launch details in `evidence/r41/stage2/run1/README.md` and its report archive |
| Checkpoint | SHA256 `311bd49cdfe5870056c9496804112edc263e6adc2ae9caeecd7a409967045215`; host `~/tnca_images/r41s2_run1_c9f25787494a/level0_wrapper_routed.dcp` |
| Qualified clock | `100000000` Hz; qualified-clock file SHA256 `49868f45b5ced2f662e4f76bbb1bb4a2be166625915d869f20812afcee42f0d7`; 1 MHz counter resolution; UUID and ICAP agree |
| Physical audit | `physical-audit.tgz`: routed timing, route legality, placement, CDC, after-load capture and qualification. Core 0/1/2 on SLR0/1/2; 56 platform critical CDC findings match the reviewed baseline |
| Main board software | Board commit `86a3d929cf3d`, product `files:c560dca3f1158106d11437948d09979d4d2b9ae181f3bb5ebaacc5fa6079290e`; Torch 2.14.0+cpu, Triton 3.8.0, Python 3.10.12 |
| Endurance software | Corrected run commit `9bc8990`, product `files:a86af4ae352dd47d9910b35fbe411fcaef920c483bd6dcc7db330972584e1303`; same image and clock; independent evidence archive retains the harness fixes' result |
| Memory | Original suite and endurance expose 4 GiB; separate FP8 and high-address campaigns expose 8 GiB. These are window configurations of the same image; do not combine them into an 8 GiB endurance claim |

`board_index.json` records raw member hashes and host provenance.
`closeout.py` verifies the qualification file's hash against the formal records,
checks each physical evidence-file hash, validates original point identities,
and checks the measured source digest against `image_source.json`.
The physical reports are available locally; the large xclbin and DCP remain
at their documented host paths, with their recorded hashes.

Historical notes in the run-1 and preparation READMEs saying the image was
not loaded describe their creation time. The later measured after-load
qualification and campaign records supersede that state. They do not erase it.
The current merged software has known differences from the registered source;
its source-check test is not evidence that the measured tree was wrong, and
historical campaign passes do not automatically transfer to today's tree.

FP8 raw reports retain the stale `rtl-simulation` label. Their FPGA backend,
image environment, measured launches and queue probes establish board execution;
this audit records the discrepancy rather than editing raw evidence. The six
NaN-sign-only comparisons are accepted under the recorded `nan_ok` rule;
“100 bitwise cases” must not be read as identical NaN encodings in every case.
