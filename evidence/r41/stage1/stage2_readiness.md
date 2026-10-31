# Stage 2: ready and blocked (from Stage 1 evidence)

This answers S1-05 item 8. Each row names the Stage 2 package, what Stage 1 established, and what still blocks it.
"Ready" means the inputs Stage 2 needs to start exist. It does not mean the Stage 2 requirement passes.

| Package | Ready (evidence) | Blocking or open (evidence) |
|---|---|---|
| **S2-00 freeze** | Official Qwen3.5-2B revision pinned (15852e8c), with 320 language storages totalling 3,763,655,360 B (`kickoff/qwen-storage/`); 24-layer prefill 128/129 graphs (`kickoff/language-meta-v1`); full-model compile, code and memory estimates (`model_estimates.md`); budget fields from the plan | Numeric tolerances need an independent reference cross-check before the freeze. The XL-core manifest is not written. The R41 holdout (`spec/r41/holdout.json`, 4 sealed points) has no registry entry or runner yet |
| **S2-01 100 MHz, three cores** | M1–M4 merged and RTL-regressed (`kickoff/merged-boundaries`); residual census of 753 families / 12,799 endpoints with F01–F10 named per structure (`frequency_preflight.md`); F07/C1 unit screen (`f07_ooc/`); per-SLR delta preflight (`regional_preflight.md`) | F01–F06 and F08–F10 have no implementation or functional screen. No routed census of the merged RTL. SLR0's 95.24 % CLB packing is a standing risk. Budget: one integration implementation plus one corrective implementation (S2-00) |
| **S2-02 memory service / MMA** | R40 ports attribution, where the home-slice engine is the limit (`evidence/r40/v3/model_attribution`); memory-service case on RTL with three domains (`memory_service.md`); fast and slow DRAM paths on the image's real HBM address map, with cycle-exact replay (`hbm_controller/`) | The slow path ends `inconclusive-fixed-point` on every GEMM configuration (`kickoff/predictions-v2`): request sets change between passes. It cannot rank memory-service candidates until the fixed point converges or is replaced. MMA candidates not yet modelled |
| **S2-03 locality / pipeline** | Consumer design, storage/alias/lifetime rules, TMU controls and board plan (`locality_design.md`); pipeline-admission case, where a third ring stage is slower on the tail (`pipeline_admission.md`) | No runtime consumer implemented; Mode A stays default-off |
| **S2-04 Qwen text inference** | D256 attention, depthwise and FP32 recurrence representative graphs pass (`kickoff/depthwise-regression-v2`, `primitive-ir-v2`); full compile estimate (`model_estimates.md`) | The standard path has six blockers:<br>B1. Decompose-K skinny decode GEMMs (`NoValidChoicesError`).<br>B2. Missing libdevice `log1p`.<br>B3. 256-wide `cumsum` split.<br>B4. Host `linalg_solve_triangular`.<br>B5. int64 `position_ids` to float.<br>B6. RoPE `cos`/`sin` lowering.<br>Capacity: device peak 3.558 GiB, so the 4 GiB window misses the 1.25× margin by 481 MB. A ≥6 GiB window means a new image exposing more PCs, plus >4 GiB addressing. Code working set: 603 KB, 4.6× the code store, which means about 6.2 MB of code uploads per prefill and 28 reloads per decode step |
| **S2-05 XL / selector** | Public TNCA_TUNE consumer with budgets, isolation, identity cache and conservative fallback (`tuning.md`); execution resource plan consumed by retile, code-store and tuning decisions | Physical tuning qualification is part of the Stage 1 board batch. No GEMM-tile candidate in the tuner: the model predicts 16×32 slower than 16×16 on 8 of 9 GEMM-reuse points (`kickoff/gemm-subset-predictions-v2.json`), pending board confirmation |
| **SystemC profile for the new image** | Model at M2 structure with dual DRAM paths and the real controller map | Still uncalibrated: R40 holdout median 21.5 % / p95 34.7 %. Board residuals of the Stage 1 predictions come from the board batch |

The Stage 1 board batch (`launches/board_queue.sh`) fills the rows that depend on board measurement. This list is
updated with those results in the Stage 1 report.

## Board closeout update, 2026-10-27 UTC

The frozen Stage 1 product is `25094197f6e6`, software
`files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`.
The [Stage 1 report](../../../../doc/r41_stage1_report.md) and
[board evidence](board/README.md) supersede the pending-board statements above.
The earlier table records Stage 1 preparation; it is not a current Stage 2
completion claim. Later [Stage 2 board results](../stage2/board/results.md)
are a separate 100 MHz / 8 GiB product and cannot substitute for the Stage 1 suite.

| Package | Stage 1 board input now available | Remaining boundary for Stage 2 |
|---|---|---|
| S2-01 clock/anchors | 9/9 numerical sessions on 90 MHz B40; graph cycle ranges 0.035%, 1.54%, 0.614% | Use the independent Stage 2 identity and timing evidence for 100 MHz |
| S2-02 memory service | 9 functional cases; depth 1→8 median cycles 17,000→15,870; three outlets 15,836 | Additional outlet gain unconfirmed; matched postdiction residuals are +12.15% to +29.48%; keep defaults |
| S2-03 pipeline/locality | 48 numerical rows; tail-mm stage 3 slower than stage 2 in all repetitions (median +3.60%) | Physical overlap not directly measured; 14 matched postdictions range from −7.22% to +62.57%; two stage-3 addmm rows are not comparable because bindings differ. Mode A default-off |
| S2-05 GEMM/selector | 18/18 GEMM configurations numerical; isolated old/fixed comparison completes all nine points; wide candidate slower than conservative on 8/9 points. Three tuning selections and three cache hits; deadline recovery passes | General wide-tile no-slower hypothesis fails; no default promotion or subset ≥1.15× target pass. Full-suite device-time nonregression passes against 113 B40-OBS points |
| Model profile | Prospective fast residual +1.94%/−3.13%; ratio error +5.23%. GEMM fast median absolute 6.44%, p95 21.95%. Matched postdiction covers 17 comparable rows and refuses two mismatched rows | Model remains uncalibrated; postdiction is not sealed prediction or unseen holdout. Slow predictions inconclusive/not-run; no refit performed |
| S2-04 full model | No new Stage 1 full-model board evidence | Earlier compile estimates and six blockers do not become capability passes; consult Stage 2 evidence for later progress |
| Stage 1 sign-off | Seven completed manifests, matched old-mapping comparison and matched postdiction archived and judged on the frozen identity | **Closed:** original suite 113/113, device-time nonregression and all 61 Required acceptance checks pass. GEMM no-slower remains a failed performance hypothesis |

The one interrupted large attention-decode point was requalified (timed 1) and
measured in a separate timed-3 process on October 26 09:04 UTC; the other 112
points were reused. The interrupted record and pre-retry state are archived.
The old-mapping comparison's 752 archive members pass hash/size checks and all
seven integrity-mutation refusals pass. Frozen acceptance reports 61 pass,
0 fail, 0 not-measured and 0 blocked, with exit status zero. Stage 1 is
accepted on its frozen identity; Stage 2 and overall R41 remain separate.

See the linked Stage 2 results and acceptance ledger for the independent
100 MHz product, its later measurements and its remaining Required items.
Stage 1 acceptance does not close Stage 2 full-model, locality, lifecycle,
XL/selector or final-reproduction requirements. Window functionality does not
measure bandwidth, and model postdiction never substitutes for a board run.
