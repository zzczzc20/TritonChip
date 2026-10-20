# R40-03.contract: legacy profile compatibility

Actual immutable-source run at 85fa7d8 reproduces the complete frozen B39 reference: 149 kernels at each of O0/O1/O2, all 447 artifacts byte-identical. The retained script disables bank allocation, window reuse and packed unpack, while the default block cap remains 16. The reference schema, baseline software commit and complete artifact membership are asserted by contract_tests.py before comparison. The reference SHA is bound in gate.json.

Source review of shared register-block/tile lowering and Inductor configuration confirms dispatch by shape, dtype, IR operation and target capability, without U280/SLR or benchmark-family dispatch. Compiler sources and Inductor configuration are unchanged between the descriptor fix 2b670c4 and this archived software; the current experimental packed-row implementation is stashed and excluded. Runtime changes are verified separately.

This gate certifies the disabled-option legacy software artifact contract. It certifies no routed hardware, wide tile performance, actual memory capacity, full suites or final-image acceptance. Image is not applicable, clock and performance are not-measured. The frozen v2 contract is unchanged.
