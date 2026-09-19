"""
Implicit register footprints.

compiler_overview.md sec 13.4 is a short section with a large consequence:

    大范围隐式寄存器读必须进入编译器模型
      - MMA 从 base 开始读取整个 fragment;
      - Tensor DMA 从 Sb 开始读取 rank 个 scalar coordinates;
      - 异步 store 延迟读取源窗口.
    这些都不能只靠 opcode 显式字段中的单个寄存器 index 建 liveness.

Every one of these reads many registers while naming one. A liveness analysis
built from the operand fields sees `m.mma.async a0, v32, v40` touching two
registers when the matrix unit will read sixteen, over many cycles, after the
instruction has retired. Overwriting v33 in between is not a stall -- it is a
different answer, and ISA sec 9C.3 puts that squarely on the compiler:

    A m.mma.async that has not completed leaves three things undefined, and
    the compiler owns all three:
      - its source registers may not be overwritten;
      - its destination accumulator may not be read by the vector unit;
      - that accumulator may not be reassigned to another program.

So the footprints are computed here, from the ISA's own layout rules, and the
ownership model uses them rather than the operand list.
"""
from dataclasses import dataclass

from .diagnostics import Diagnostic

# ISA sec 3: S0 reads as zero and discards writes.
SREG_ZERO = 0
SREG_COUNT = 64
VREG_BYTES = 64          # VLEN(16) lanes x 32 bits


@dataclass(frozen=True)
class RegRange:
    """A half-open range of registers in one file."""
    file: str            # 'v' | 's' | 'p' | 'a'
    first: int
    count: int

    @property
    def last(self):
        return self.first + self.count - 1

    def overlaps(self, other):
        return (self.file == other.file
                and self.first <= other.last
                and other.first <= self.last)

    def __contains__(self, index):
        return self.first <= index <= self.last

    def __str__(self):
        if self.count == 1:
            return f"{self.file}{self.first}"
        return f"{self.file}{self.first}..{self.file}{self.last}"


def fragment_registers(rows, k, dtype_bits=16):
    """How many vector registers a `rows` x `k` fragment occupies.

    ISA sec 9C.2: a fragment is a window's registers read as a flat row-major
    byte array -- exactly what the transfer engine writes, so nothing repacks
    between a t.ld.async and the m.mma.async that consumes it.
    """
    total_bytes = rows * k * (dtype_bits // 8)
    return (total_bytes + VREG_BYTES - 1) // VREG_BYTES


def mma_source_footprint(base, *, rows=16, k=16, dtype_bits=16):
    """The registers an m.mma.async reads, given the register it names.

    It names the base. It reads the fragment.
    """
    return RegRange('v', base, fragment_registers(rows, k, dtype_bits))


def mma_footprints(a_base, b_base, *, m=16, n=16, k=16, dtype_bits=16):
    """Both source fragments of one m.mma.async step."""
    return [mma_source_footprint(a_base, rows=m, k=k, dtype_bits=dtype_bits),
            mma_source_footprint(b_base, rows=k, k=n, dtype_bits=dtype_bits)]


def dma_coordinate_footprint(sb, rank):
    """The scalar registers a transfer reads for its coordinates.

    ISA sec 9E: consecutive scalar registers starting at Sb, one per
    descriptor rank. The instruction names only Sb.

    Refuses a base whose run would leave the file, rather than returning a
    range that wraps. At Sb = S63 the next index is S0, which reads as zero and
    discards writes, so the coordinate could never be set and the transfer
    would run at 0 on that axis while the program believed otherwise. The
    hardware raises ILLEGAL_INSTRUCTION for exactly this; the compiler should
    never emit it in the first place.
    """
    if rank < 1:
        raise Diagnostic(f"a transfer needs at least one coordinate, got rank {rank}")
    last = sb + rank - 1
    if last >= SREG_COUNT:
        raise Diagnostic(
            f"coordinate base s{sb} with rank {rank} would read s{last}",
            because=(f"the scalar file ends at s{SREG_COUNT - 1}; the index "
                     f"wraps to s{last % SREG_COUNT}, and s{SREG_ZERO} reads "
                     "as zero with writes discarded"),
            remedy=f"allocate the coordinate run at s{SREG_COUNT - rank} or below",
            source='ISA sec 9E; compiler_overview.md sec 13.4')
    if sb == SREG_ZERO:
        raise Diagnostic(
            "coordinate base s0 reads as zero",
            because='ISA sec 3: S0 reads as zero and discards writes, so the '
                    'first coordinate could never be set',
            remedy='allocate the coordinate run at s1 or above',
            source='ISA sec 9E; compiler_overview.md sec 13.4')
    return RegRange('s', sb, rank)


def dma_store_footprint(window_base, window_len):
    """The registers an asynchronous store reads -- late.

    sec 13.4's third case. The read happens after the issuing instruction has
    retired, so the range stays live past the point a naive liveness analysis
    ends it, and the window cannot be reused until the store's completion.
    """
    return RegRange('v', window_base, window_len)


class OwnershipChecker:
    """Reject a write that lands inside an outstanding command's read set.

    This is the check sec 5.4 says needs a dedicated verifier, because
    MemoryEffectsOpInterface is not an address-range mechanism: marking an op
    `MemRead` does not say *which* sixteen registers, and that is the only
    thing that matters here.
    """

    def __init__(self):
        self._live = []     # (cmd_id, RegRange, description)

    def hold(self, cmd_id, ranges, description):
        for r in ranges:
            self._live.append((cmd_id, r, description))

    def release(self, cmd_id):
        self._live = [e for e in self._live if e[0] != cmd_id]

    def check_write(self, file, index, *, by='an instruction'):
        for cmd_id, r, desc in self._live:
            if r.file == file and index in r:
                raise Diagnostic(
                    f"{by} writes {file}{index}, inside {r} which command "
                    f"#{cmd_id} is still reading",
                    because=f"{desc} reads the whole range, not just the "
                            "register its encoding names",
                    remedy='wait for the command, or place the write outside '
                           'the range',
                    source='ISA sec 9C.3; compiler_overview.md sec 13.4')
        return True

    def check_write_range(self, rng, *, by='an instruction'):
        for index in range(rng.first, rng.first + rng.count):
            self.check_write(rng.file, index, by=by)
        return True

    @property
    def live_ranges(self):
        return [(c, r, d) for c, r, d in self._live]
