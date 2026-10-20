# CI backend-boundary failure and correction

The complete failed GitHub job log is retained compressed, from run 38679960975 / job 116242336108. The sole failure was R36-BE-01, reporting Popen in python/tnca/autotune.py. This is the generic external candidate measurement supervisor introduced to enforce compilation/measurement deadlines, not simulator binding logic.

The corrected checker permits that one call shape inside run_measurement (child_argv(command), DEVNULL stdin, owned process group). The module remains scanned for simulator/elaborator names; extra process launches and relocation remain refused. Mutations exercise both simulator strings beside the allowed call, a simulator argument replacing the allowed argv, an extra launcher, an unowned group and runtime relocation.

Raw local and jump-host results are retained. run.sh executes the corrected checker as an explicit external validation harness against immutable source 8abf527, without editing the archived software. All three backend tests passed, including actual execution of one artifact through both RTL bindings and refusal of unavailable backends. This is CI regression evidence, not a new R40 gate pass. Current-head GitHub CI still needs its own successful terminal result.
