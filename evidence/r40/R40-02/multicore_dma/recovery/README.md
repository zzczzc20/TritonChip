# Recovery with DMA ownership and forced address reuse

Source: `025cf62` runtime/RTL plus the recovery-test/guard extensions committed
with this evidence. Jump host root `~/tritonchip-r40-dma-fast`, locked Torch/
Triton, Icarus 12. Three cores, three outlets, two-stage physical platform
stand-in, cache maintenance and DMA enabled.

`run.log`: the actual standard native Torch graph and storage lifetime checks
pass three recovery epochs. Each epoch reuses the original physical address;
old tensor/view/copy/launch/resize and event handles are refused; destroying
old storage preserves the new allocation and numerical result.

`guards.log`: 8/8. Failed/refused/AXI-faulted ownership and host observation
timeouts perform no BO read/write. A host timeout is not a successful grant
or a cancellation. The previous six-case guard result remains retained at
the parent checkpoint.

These are functional checks on isolated RTL bindings. They do not establish
board reload/recovery behavior, independent copy/compute overlap, the frozen
1000-call stress inventory or any performance gate.
