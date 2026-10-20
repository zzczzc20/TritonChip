# Native role, physical register and barrier checkpoint

No required R40 gate is promoted. Complete-chip timing and qualification remain
not-measured. All construction, execution and mutation work ran on the GB10.

[report.json](report.json) and [raw.tar.gz](raw.tar.gz) retain source archive
`420a64faac6b6221a5b71f4bf16d48ac36b3e7145678f7b9cc2f97c4b3e8b93d`,
normal/mutant binaries and model-source identities, the actual compiler outputs,
input/reference buffers, traces and full logs. The raw archive SHA is
`e40d6cab47adacab5833e00dd6df31784bdc02c046a2a16ffbe2dc54af2c098d`.

The executor now starts active roles with private scalar and predicate state,
private scoreboards and an explicit physical vector-register mapping. Register
values commit at their modeled writeback time with lane enables. Shared role
aliases do not share scoreboard readiness: a consumer cannot obtain an
implicit wait for its producer from the simulator.

The release/acquire case runs two aliased roles on all three physical-core
workers over successive programs. The consumer sleeps on generation 1 before
the producer arms it. The producer issues an SFU result and arrives; arrival
waits for its register writes before releasing the consumer. The consumer then
reads the shared physical value and stores the independent expected result.

A separately built source clone removes only the arrive register-writeback
wait. Its source hash is
`9d491045da43c247c5f1f214ed4929f18daa9eafbebc24e62bd8aca454a7d459`.
The consumer reads the old value and numerical validation fails. The mutant's
own harness/source identity is used; it is not attributed to the normal model.
This demonstrates why eager shared writes or shared readiness would hide the
missing release obligation. It does not claim deterministic values for
uninitialized hardware registers.

Additional cases check disjoint scalar/predicate/vector role state, a sparse
role-2 launch, 70000 barrier generations, initialization under a sleeping
waiter, invalidation wakeup, arrival underflow, illegal token IDs and quiescent
deadlock detection. Failures return no completed prediction and do not time out
or become a zero-cycle result. The pre-existing native numeric/compiler/grid
campaign also passes with the new physical writeback model.

Limitations remain explicit. Only the tested role mappings are qualified at
this development level. RSLICE, grouped program distribution, arbitrary vector
bases and complete mask combinations are not established. Operation-completion
accounting has an API but native DMA/MMA engines do not yet drive it. RF port
arbitration and cycle accuracy remain incomplete. This checkpoint neither
replaces the frozen native77/HBM14/model inventories nor authorizes an FPGA run.
