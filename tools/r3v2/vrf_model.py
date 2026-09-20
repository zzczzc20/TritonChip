"""
An independent model of the candidate VRF -- R3V2-RF-02.

Written from spec/r3v2_contract.json and doc/phase3_tasks_and_acceptance.md,
not from the Verilog. A model derived from the design agrees with it about
a shared mistake, which is the thing "独立模型逐项比较" is guarding against.

The rules, stated so a disagreement can be argued about:

  phys  = vbase[rctx] + idx, truncated to PHYS_W bits
  bank  = phys % 4          row = phys // 4

  Each bank takes at most one read and one write per cycle. The
  requesters are scanned round robin from a pointer that advances past
  whoever was served last, so a requester that keeps asking waits at most
  (C-1)*B+A cycles. Lowest-index-first would let port 0 hold a bank for as
  long as it wanted and starve the matrix behind it.

  A bank holding a response the consumer has not taken accepts no read.

  A write updates, and validates, only the lanes its mask names.

  A read of a lane whose valid bit is clear returns zero. The valid bits
  are cleared by reset; the data is not. A read of the address being
  written sees the new data in the lanes the write names -- and their new
  validity too, or a write followed immediately by a read would return
  zero for a lane that had just been set.

  A reset drops a response in flight rather than delivering it into the
  next epoch.
"""

LANES = 16
ELEN = 32
MASK32 = (1 << ELEN) - 1


class CandidateVRF:
    def __init__(self, phys_vregs=128, num_banks=4, num_ports=4,
                 phys_w=7, vbase=(0, 64), zero_invalid=True):
        self.phys_vregs = phys_vregs
        self.num_banks = num_banks
        self.num_ports = num_ports
        self.phys_w = phys_w
        self.vbase = list(vbase)
        self.zero_invalid = zero_invalid
        self.rows = phys_vregs // num_banks
        # The array keeps whatever it had; None is "never written", which
        # the RTL shows as X and the architecture shows as zero.
        self.mem = [[None] * LANES for _ in range(phys_vregs)]
        self.valid = [[False] * LANES for _ in range(phys_vregs)]
        # One held response per bank: (port, value) or None
        self.held = [None] * num_banks
        # The round-robin pointer, advanced past whoever was served last.
        self.rr = 0

    # ---- addressing -------------------------------------------------------
    def phys(self, rctx, idx):
        return (self.vbase[rctx] + idx) & ((1 << self.phys_w) - 1)

    def bank_row(self, phys):
        return phys % self.num_banks, phys // self.num_banks

    def unmap(self, bank, row):
        return row * self.num_banks + bank

    # ---- one cycle --------------------------------------------------------
    def cycle(self, write, reads, ready):
        """Apply one cycle and return what is visible after the edge.

        Structural rather than behavioural, because the first version was
        not and the difference cost an afternoon. It modelled a response as
        arriving one cycle after acceptance, which is true of the data and
        false of the observation: the bench samples after the clock edge,
        so a one-cycle read is already in the response register on the same
        line that requested it.

        The order below is the hardware's:

          1. a bank is free if its response register is empty or the
             consumer is taking what is in it
          2. arbitration accepts at most one read per free bank
          3. the value is sampled from the array as it stands BEFORE this
             cycle's write, plus the write-first bypass
          4. the response registers update
          5. the write lands
          6. what is visible is the response registers after the update

        Returns (accepted_ports, visible) where visible is
        [(port, [lane values])].
        """
        w_phys = None
        if write is not None:
            wr, wi, wmask, wword = write
            w_phys = self.phys(wr, wi)

        # 1. which banks can take a read
        free = []
        for b in range(self.num_banks):
            held = self.held[b]
            free.append(held is None or ready.get(held[0], True))

        # 2. arbitration, lowest port first.
        #
        # A port already holding a response the consumer has not taken is
        # not given another. Both this model and the RTL allowed it at
        # first, and then two banks held responses for one port and the
        # observer saw whichever the wiring happened to prefer.
        busy = set()
        for b in range(self.num_banks):
            held = self.held[b]
            if held is not None and not ready.get(held[0], True):
                busy.add(held[0])

        accepted, taken = [], {}
        by_port = {port: (rctx, idx) for port, rctx, idx in reads}
        last = self.rr
        for k in range(self.num_ports):
            port = (self.rr + k) % self.num_ports
            if port not in by_port:
                continue
            rctx, idx = by_port[port]
            p = self.phys(rctx, idx)
            b, _r = self.bank_row(p)
            if b in taken or not free[b] or port in busy:
                continue
            taken[b] = (port, p)
            accepted.append(port)
            last = port
        if accepted:
            self.rr = (last + 1) % self.num_ports

        # 3. the value each accepted read carries
        sampled = {}
        for b, (port, p) in taken.items():
            value = []
            for lane in range(LANES):
                if (write is not None and p == w_phys
                        and (wmask >> lane) & 1):
                    value.append((wword + lane) & MASK32)
                elif self.zero_invalid and not self.valid[p][lane]:
                    value.append(0)
                else:
                    value.append(self.mem[p][lane])
            sampled[b] = (port, value)

        # 4. the response registers
        for b in range(self.num_banks):
            if free[b]:
                self.held[b] = sampled.get(b)

        # 5. the write
        if write is not None:
            for lane in range(LANES):
                if (wmask >> lane) & 1:
                    self.mem[w_phys][lane] = (wword + lane) & MASK32
                    self.valid[w_phys][lane] = True

        # 6. what an observer sees after the edge
        visible = [h for h in self.held if h is not None]
        return accepted, visible

    def reset(self):
        """Clear the valid bits and the responses in flight, not the array.

        This is the whole of the initialisation contract in three lines:
        the architecture reads zero afterwards because nothing is valid,
        the storage still holds what it held because a memory does not
        forget, and an epoch's response does not arrive in the next one.
        """
        self.valid = [[False] * LANES for _ in range(self.phys_vregs)]
        self.held = [None] * self.num_banks
        self.rr = 0
