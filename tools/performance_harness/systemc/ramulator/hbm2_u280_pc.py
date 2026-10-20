"""Ramulator2 configuration of one U280 HBM2 pseudo channel (the slow DRAM path).

    PYTHONPATH=RAMULATOR2/python python3 -m ramulator export hbm2_u280_pc.py -o hbm2_u280_pc.yaml

One memory system with one controller = one pseudo channel; the bridge
(tnca_dram_bridge.cpp) instantiates one per pseudo channel a launch touches,
renaming the controller and memory-system implementations to its own
subclasses (same behaviour, observable state). Assumptions, all documented in
the M3 README:

- HBM2 (JESD235) as on the Alveo U280: 2 stacks x 8 channels x 2 pseudo
  channels of 256 MiB (2 Gb), 64-bit pseudo channel, BL4 = 32 bytes, 4 bank
  groups x 4 banks, 16 K rows of 1 KiB, Ramulator2's "HBM2_2Gb" organization
  with pseudochannel=1 so that each controller is one independent pseudo
  channel (the two pseudo channels of a channel share the command bus in the
  device; that coupling is not modelled).
- 1.8 Gb/s (tCK 1111 ps, 900 MHz), the U280 HBM clock. Ramulator2's HBM2
  presets stop at 1.6/2.0/2.4 Gb/s and are marked "Ramulator guesstimate";
  the 1.8 Gb/s preset below keeps the 2.0 Gb/s nanosecond values (tCL = tRCD
  = tRP = 14 ns, tRAS 34 ns, tWR 16 ns, tWTR 6/8 ns, RL/WL as in the preset)
  rounded up to whole 1111 ps cycles; the derived values (tRRD, tFAW, tRFC
  for a 4 Gb channel = 260 ns, tREFI 3.9 us, tCCD) come from Ramulator2's own
  JESD235D rules at that tCK; tRTW from JESD235D Note 23 with tDQSQ = 95 ps
  (interpolated between 2.0 and 1.6 Gb/s).
- Controller: Ramulator2's HBM12 controller, FR-FCFS, open-page row policy,
  all-bank refresh every tREFI on (no per-bank refresh), 32-entry read and
  write queues with the default write watermarks (0.8 / 0.2). The Xilinx
  controller's reordering depth and page policy are not public in detail; this
  is the generic choice the M3 design asked for.
- Addresses are mapped by the bridge (pass-through mapper): see
  tnca_dram_bridge.cpp map_address() for the assumed PG276 ROW_COLUMN_BANK
  layout.
"""
import ramulator
from ramulator.dram.hbm2 import HBM2

HBM2.timing_presets["HBM2_1800Mbps_TNCA"] = {
    "rate": 1800, "nBL": 2,
    "nCL": 13, "nRCDRD": 13, "nRCDWR": 11,
    "nRP": 13, "nRAS": 31, "nWR": 15,
    "nRTPL": 5, "nCWL": 5,
    "nWTRS": 6, "nWTRL": 8,
    "tCK_ps": 1111,
}

frontend = ramulator.frontend.External(clock_ratio=1)
dram = HBM2(org_preset="HBM2_2Gb", timing_preset="HBM2_1800Mbps_TNCA",
            pseudochannel=1, channel_width=64, nRTW=14, verbose=True)
ctrl = ramulator.controller.HBM12(
    dram=dram,
    scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
)
mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=1,
    controllers=[ctrl],
    channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
)
sim = ramulator.Simulation(frontend, mem)
