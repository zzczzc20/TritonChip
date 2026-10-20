# Actual frozen-graph candidate compilation preflight

Raw reports cover the deduplicated union of the full frozen HBM14 inventory and frozen holdout14. Candidate block widths 16 and 32 were fixed before execution; no strategy, threshold or winner was selected using holdout results. All points retain original sizes, seed and definition hashes. These are real compiler invocations from immutable software bbb9321, with 600 seconds total compile preflight budget per graph across both candidates. They provide no board measurement, tuning result, numerical pass or gate pass.

The compile-only CPU migration harness retains its legacy 393216-byte staging limit; records saying all kernels compiled and then capacity-exceeded reflect that harness limit, not a measured capacity result for the pending multi-PC image. Successful candidate compilation is distinguished from execution in summary.json. Rejected wide candidates retain the original Unsupported diagnostic; they are not silently accepted or removed from evidence.

The raw archive also preserves an initial misconfigured attempt: invoking the internal --child entry omitted the environment normally supplied by the parent, so --compile-only/--set were not applied. It was stopped, retained and superseded by the explicitly configured cpu_skip runs. It is excluded from summary.json. Neither attempt is performance evidence.

Final hardware/software signoff still requires the three-core checkpoint, actual candidate measurements, full native77, HBM14 and the frozen model on the delivered image. Runtime identity fixes after bbb9321 are tested separately and cannot be retroactively attributed to these raw reports.
