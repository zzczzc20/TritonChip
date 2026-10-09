"""What a register allocation costs, in bank conflicts.

doc/phase3_5_tasks_and_acceptance.md R35-ALLOC-01: "cost 使用最终
physical register→bank 映射".

The mapping is not this file's invention. R3 v2 froze it in
spec/r3v2_contract.json: 128 physical registers over 4 banks, selected
by the low two bits of the physical register number, one accepted read
and one accepted write per bank per cycle. Those numbers are read from
that file rather than repeated here, so a hardware change that moved
them would move this too instead of silently disagreeing.

Section 3.1 is the rule this file exists to respect. A bank is a
property of the PHYSICAL register that allocation chose -- not of a
virtual layout -- so this runs after allocation and reads the register
numbers that were actually emitted. Reading bank behaviour off a layout
is exactly what section 3.1 forbids, and the way to be sure it is not
happening is for the cost to have no access to a layout at all.
"""
import json
import re

from ._datapath import find

# The shipped copy in an installed wheel, the checkout's otherwise -- the same
# lookup the ISA and the targets use (see _datapath).
_CONTRACT = find('spec', 'r3v2_contract.json')


def hardware():
    """The frozen VRF geometry, from R3 v2's contract."""
    with open(_CONTRACT) as fh:
        d = json.load(fh)
    return {'banks': int(d['vrf']['banks']),
            'physical_registers': int(d['vrf']['physical_registers']),
            'reads_per_bank_per_cycle':
                int(d['ports']['accepted_reads_per_bank_per_cycle']),
            'bank_select': d['vrf']['bank_select']}


def bank_of(phys, banks=None):
    """Which bank a physical register lives in."""
    if banks is None:
        banks = hardware()['banks']
    return int(phys) % banks


# Instructions that read more than one vector register in one go, and
# therefore can collide in a bank. A store reads its data register; a
# three-source FMA reads three.
_READS = re.compile(r'^\s*(?:@\w+\s+)?([\w.]+)\s+(.*)$')


def reads_of(line):
    """The vector registers an instruction READS.

    The destination is excluded: a write goes to the write port, which
    the contract counts separately, and counting it as a read would
    report a conflict that the hardware does not have.
    """
    m = _READS.match(line)
    if not m:
        return []
    mnem, rest = m.group(1), m.group(2)
    if mnem.startswith('//') or mnem.startswith('.'):
        return []
    regs = re.findall(r'\bv(\d+)\b', rest)
    if not regs:
        return []
    # A store names its data register and no destination; everything else
    # writes its first operand.
    if mnem.startswith(('v.st', 'vf.st', 't.st')):
        return [int(r) for r in regs]
    return [int(r) for r in regs[1:]]


def conflicts(asm_text, banks=None):
    """Same-bank reads, instruction by instruction.

    One accepted read per bank per cycle, so an instruction that reads
    two DIFFERENT registers in the same bank needs two cycles to collect
    them. The number reported is the number of EXTRA cycles that costs,
    which is what an allocator would be trying to reduce.

    Distinct registers, not distinct operands: `vf.add v1, v0, v0` names
    v0 twice and the file reads it once. Counting that as a conflict
    would have an allocator working to separate a register from itself.
    """
    if banks is None:
        banks = hardware()['banks']
    total, detail = 0, []
    for line in asm_text.splitlines():
        regs = reads_of(line)
        distinct = sorted(set(regs))
        if len(distinct) < 2:
            continue
        seen = {}
        for r in distinct:
            b = bank_of(r, banks)
            seen[b] = seen.get(b, 0) + 1
        extra = sum(n - 1 for n in seen.values() if n > 1)
        if extra:
            total += extra
            detail.append({'line': line.strip(), 'registers': distinct,
                           'banks': [bank_of(r, banks) for r in distinct],
                           'extra_cycles': extra})
    return {'extra_cycles': total, 'sites': len(detail), 'detail': detail}


def moves(asm_text):
    """Data movement the compiler introduced, counted and named.

    R35-ALLOC-01: "纯 rename 无额外 move，实际 move 全部计费". A rename
    that emitted a move would not be a rename, and a real conversion
    that emitted moves nobody counted would be free on paper.
    """
    kinds = {'v.mov': 'register copy', 'v.permute': 'lane permutation',
             'v.shuffle.xor': 'lane shuffle', 'v.shuffle.up': 'lane shuffle',
             'v.shuffle.down': 'lane shuffle', 'v.rotate': 'lane rotate'}
    found = []
    for line in asm_text.splitlines():
        m = _READS.match(line)
        if not m:
            continue
        mnem = m.group(1)
        if mnem in kinds:
            found.append({'instruction': mnem, 'what': kinds[mnem],
                          'line': line.strip()})
    return {'count': len(found), 'moves': found}
