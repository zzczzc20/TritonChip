"""
An independent model of the banked register file -- R3-03 RF-02.

Written from doc/phase3_tasks_and_acceptance.md and the interface, not from
tnca_vrf_banked.v. That is the whole point: a model derived from the RTL
would agree with it about a shared mistake, which is the failure mode the
contract is guarding against when it asks for "独立模型逐请求比较".

The rules this model implements, stated here so a disagreement can be
argued about rather than just observed:

  phys  = vbase[rctx] + idx          (truncated to PHYS_W bits)
  bank  = phys % NUM_BANKS
  row   = phys // NUM_BANKS

  One read and one write per bank per cycle. Among the read requests that
  want the same bank in the same cycle, the lowest port index is accepted
  and the rest are not -- they are refused, not queued, and the layer above
  is expected to retry.

  A write updates only the lanes its mask names.

  A read accepted in cycle c is OBSERVED at sample point c+LATENCY-1. The
  bench samples after the clock edge that follows the request, so a
  one-cycle read is already there at the first sample; the offset is a
  convention about when the observer looks, not about how long the memory
  takes, and it is written here so the two ends agree about it. And
  a read of the address being written in cycle c sees the new data in the
  lanes the write names (write-first), and the old data elsewhere.

  A register never written reads as unknown. The model represents that as
  None and the comparison treats a response against it as "no claim",
  because the RTL will produce X there and X is not a value to compare.
"""

LANES = 16
ELEN = 32
MASK32 = (1 << ELEN) - 1


class BankedRF:
    def __init__(self, phys_vregs=128, num_banks=4, num_ports=4,
                 phys_w=7, latency=1, vbase=(0, 64)):
        self.phys_vregs = phys_vregs
        self.num_banks = num_banks
        self.num_ports = num_ports
        self.phys_w = phys_w
        self.latency = latency
        self.vbase = list(vbase)
        self.bank_depth = phys_vregs // num_banks
        # None means never written: unknown, not zero.
        self.mem = [[None] * LANES for _ in range(phys_vregs)]
        # responses in flight: list of (due_cycle, port, value)
        self.pipe = []

    # ---- the address mapping, and its inverse -----------------------------
    def phys(self, rctx, idx):
        return (self.vbase[rctx] + idx) & ((1 << self.phys_w) - 1)

    def bank_row(self, phys):
        return phys % self.num_banks, phys // self.num_banks

    def unmap(self, bank, row):
        """RF-01 asks that the mapping be invertible. Here is the inverse."""
        return row * self.num_banks + bank

    # ---- one cycle --------------------------------------------------------
    def cycle(self, n, write, reads):
        """Apply one cycle.

        write: None, or (rctx, idx, mask, word). The data word is splatted
               with the lane index added, the same convention the bench uses.
        reads: list of (port, rctx, idx) requests presented this cycle.

        Returns (accepted_ports, responses_due_now) where a response is
        (port, [lane values or None]).
        """
        w_phys = w_bank = None
        if write is not None:
            wr, wi, wmask, wword = write
            w_phys = self.phys(wr, wi)
            w_bank, _w_row = self.bank_row(w_phys)

        # arbitration: one read per bank, lowest port first
        taken = {}
        accepted = []
        for port, rctx, idx in sorted(reads):
            p = self.phys(rctx, idx)
            b, _r = self.bank_row(p)
            if b in taken:
                continue
            taken[b] = (port, p)
            accepted.append(port)

        # the value each accepted read will return, sampled NOW: write-first
        # in the lanes the concurrent write names.
        for b, (port, p) in taken.items():
            value = list(self.mem[p])
            if write is not None and p == w_phys:
                for lane in range(LANES):
                    if (wmask >> lane) & 1:
                        value[lane] = (wword + lane) & MASK32
            self.pipe.append((n + self.latency - 1, port, value))

        # the write itself lands at the end of the cycle
        if write is not None:
            for lane in range(LANES):
                if (wmask >> lane) & 1:
                    self.mem[w_phys][lane] = (wword + lane) & MASK32

        due = [(port, value) for (c, port, value) in self.pipe if c == n]
        self.pipe = [e for e in self.pipe if e[0] != n]
        return accepted, due

    def reset(self):
        """RF-03: the response pipeline is cleared; the array is not.

        A memory does not forget its contents on reset. What must not happen
        is a response arriving that belongs to a request from before it.
        """
        self.pipe = []
