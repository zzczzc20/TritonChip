"""Explicit logical layout for the 32-PC, 4 KiB-striped HBM image."""
import operator

STRIPED_CAP = 64
STRIPED_REGISTER = 0x120
STRIPED_SIGNATURE = 0x5234315300010C05
STRIPED_BYTES = 1 << 33
DEVICE_RESERVED_RANGES = ((0x7fff0000, 0x7fff1000),)


class StripedHBMLayout:
    """A bijection over a complete, base-zero 8 GiB allocation.

    Logical pages go round-robin to the 32 PCs; pages within a PC remain
    consecutive. Offsets passed here are BO-relative, not TNCA addresses.
    """
    name = 'pc32-page4k-v1'

    def __init__(self, base, size):
        if operator.index(base) != 0 or operator.index(size) != STRIPED_BYTES:
            raise ValueError('striped HBM requires a base-zero 8 GiB allocation')

    @staticmethod
    def physical(offset):
        offset = operator.index(offset)
        if not 0 <= offset < STRIPED_BYTES:
            raise ValueError('logical HBM offset outside the striped allocation')
        return ((offset & 0x1F000) << 16) | ((offset >> 17) << 12) | (offset & 0xFFF)

    @staticmethod
    def logical(offset):
        offset = operator.index(offset)
        if not 0 <= offset < STRIPED_BYTES:
            raise ValueError('physical HBM offset outside the striped allocation')
        return ((offset & 0x0FFFF000) << 5) | ((offset >> 28) << 12) | (offset & 0xFFF)

    @classmethod
    def spans(cls, offset, nbytes):
        """Yield (input-relative offset, physical BO offset, byte count)."""
        offset, nbytes = operator.index(offset), operator.index(nbytes)
        if offset < 0 or nbytes < 0 or offset > STRIPED_BYTES or nbytes > STRIPED_BYTES-offset:
            raise ValueError('HBM transfer outside the striped allocation')
        done = 0
        while done < nbytes:
            size = min(nbytes-done, 4096-((offset+done) & 4095))
            yield done, cls.physical(offset+done), size
            done += size

    @classmethod
    def pc_ranges(cls, offset, nbytes):
        """Return contiguous per-PC ranges in O(32), without a page list."""
        offset, nbytes = operator.index(offset), operator.index(nbytes)
        if offset < 0 or nbytes < 0 or offset > STRIPED_BYTES or nbytes > STRIPED_BYTES-offset:
            raise ValueError('HBM transfer outside the striped allocation')
        if not nbytes:
            return []
        first, first_byte = divmod(offset, 4096)
        last, last_byte = divmod(offset+nbytes-1, 4096)
        result = []
        for pc in range(32):
            lo = first+(pc-first)%32
            hi = last-(last-pc)%32
            if lo > hi:continue
            begin = first_byte if lo==first else 0
            end = last_byte+1 if hi==last else 4096
            size = ((hi-lo)//32)*4096+end-begin
            result.append((pc,(pc<<28)+(lo//32)*4096+begin,size))
        return result

    @classmethod
    def groups(cls, offset, nbytes):
        """Coalesce each PC's physical transfer while retaining gather order."""
        groups = {}
        for logical, physical, size in cls.spans(offset, nbytes):
            pc = physical >> 28
            if pc not in groups:
                groups[pc] = [physical, 0, []]
            group = groups[pc]
            if physical != group[0]+group[1]:
                raise ValueError('non-contiguous physical PC transfer')
            group[1] += size
            group[2].append((logical, size))
        return groups.values()

    def write(self, accessor, offset, data):
        data = memoryview(data).cast('B')
        for physical, _, pieces in self.groups(offset, len(data)):
            accessor.bo_write(physical, b''.join(data[start:start+size] for start, size in pieces))

    def read(self, accessor, offset, nbytes):
        nbytes = operator.index(nbytes)
        groups = self.groups(offset, nbytes)
        result = bytearray(nbytes)
        for physical, size, pieces in groups:
            data = accessor.bo_read(physical, size)
            if len(data) != size:
                raise ValueError('short striped HBM DMA read')
            consumed = 0
            for logical, length in pieces:
                result[logical:logical+length] = data[consumed:consumed+length]
                consumed += length
        return bytes(result)
