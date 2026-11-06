# Stage 1 freeze-check test inputs

`inputs_25094197f6e6.tgz` contains the exact files named by the Stage 1 lock
and the lock itself, extracted from the accepted Stage 1 source commit
`25094197f6e6`. Its SHA-256 is
`0233c2dc2b311633e5cc05b08f68469e2c166538b366db4acaa0e55bf156dcfe`.
The framework's freeze-record unit test uses these historical inputs because
the current tree legitimately adds the later memory-bound registry and the
acceptance contract has evolved. The test checks that changing a locked input
is still rejected. This fixture is not a full product archive and does not
claim that the current tree satisfies the original Stage 1 lock.

On 2026-10-30, the first framework test attempt hit `/tmp` user quota. A
retry with `TMPDIR` on the workspace filesystem reached the freeze-record
test and failed because it incorrectly required the current tree to match
the old lock: [`harness_before.log`](harness_before.log), SHA-256
`7753ace487be46e4f480905dd9021c6da9b0d1971da80b37e2731a4928d112ce`.
After using the frozen fixture, `make r41-harness-tests PYTHON=python3`
passed all five test groups: [`harness_after.log`](harness_after.log), SHA-256
`7ee246542c1ce7a4df12d7b41710cba018b229f3157151852730ce8a2071b659`.
The apparent `FAIL` lines within the board-tool group are intentional
negative fixtures; its unittest result is OK.
