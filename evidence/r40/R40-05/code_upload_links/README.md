# Private code upload links

Each private RAM receives a snapshotted write/read command through its own
elastic pipeline. Replies return independently; a central acknowledgement mask
waits for every RAM, and retains core zero's read data. Host data access, grid
admission and idle reporting wait until the code transaction drains. The
zero-stage default retains the legacy one-cycle response. The U280 platform
binding and generated image configuration use two stages.

`matrix.log`: 1/3/8 physical core geometries, zero/1/2/4 stages; every private RAM
is checked at the last code word. Byte masks merge correctly; changing host
address/data after acceptance cannot change the request; responses occur once;
launch/idle stay blocked until acknowledgement and return after drain.

`board_protocol.log`: full AXI-Lite platform binding, two 455-program 3D grids,
private code upload/readback, coherent shared-line stores, timestamps, port
traffic, zero launch, stale/invalid requests and deadline drain pass with the
code pipeline enabled. These are functional results, not routed timing evidence.
The legacy FPGA regression also passes 8/8 (`legacy_fpga.log`). The new native
Torch run is pending separately.

The earlier three-master linked build continues from its immutable pre-code-link
snapshot. Its route results must not be attributed to this code revision.
