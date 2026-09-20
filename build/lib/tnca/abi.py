"""
Kernel argument ABI v1 -- doc/phase2_tasks_and_acceptance.md R2-06.

The numbers live in `spec/r2_abi_v1.json` and are read from it, not repeated
here: a second copy of a slot size is a second thing to get wrong, and the
contract is the file the tests check against.

What this module is careful about, in the order the contract raises it:

  - an out-of-range integer is an ERROR, never a truncation. A wrapped index
    is a wrong answer wearing the shape of a right one;
  - an f32 is a bit pattern. Going through a decimal string loses -0.0 and a
    NaN payload, and this is the layer where that would be invisible;
  - the high half of a 32-bit slot must be ZERO, and is checked on decode as
    well as written on encode. A kernel that reads the whole slot must see a
    number, not whatever the last launch left there;
  - a pointer argument is a handle plus an offset until the runtime resolves
    it. This module never invents an address.

Encoding is deliberately separate from validation: `encode_block` takes values
that have already been checked, and `check_value` says why one is not
acceptable. The runtime needs the second without the first when it validates
a submission before any resource is committed.
"""
import json
import os
import struct

from . import _data
from .errors import ArgumentError, SignatureError, VersionError

_SPEC_PATH = _data.path('spec', 'r2_abi_v1.json', env='TNCA_ABI_SPEC')

with open(_SPEC_PATH) as _fh:
    SPEC = json.load(_fh)

ABI_VERSION = tuple(SPEC['abi_version'])
SLOT_BYTES = SPEC['slot_bytes']
BLOCK_ALIGNMENT = SPEC['block_alignment']
TYPES = SPEC['types']
REJECTED = SPEC['rejected_scalar_types']

# A pointer argument is (handle, byte_offset) at the API and an address on the
# device. Kept as a plain tuple rather than a class so a caller can pass one
# without importing anything.
POINTER_TYPES = {'device_ptr64'}


def supports(version):
    """Can this implementation read a block declared for `version`?

    Same major and a minor no greater than ours. A higher minor may contain
    slots we would place at the wrong offset, and guessing is the one thing a
    binary interface may not do.
    """
    major, minor = version
    return major == ABI_VERSION[0] and minor <= ABI_VERSION[1]


def require_version(version):
    if not supports(tuple(version)):
        raise VersionError(
            f"argument ABI {version[0]}.{version[1]} is not supported",
            detail={'supported': f"{ABI_VERSION[0]}.x up to "
                                 f"{ABI_VERSION[0]}.{ABI_VERSION[1]}"},
            remedy='recompile the artifact against this runtime')


def block_size(n_args):
    """Bytes the argument block occupies. Zero arguments is zero bytes."""
    if n_args < 0:
        raise ArgumentError(f"negative argument count {n_args}")
    return n_args * SLOT_BYTES


def offset_of(index):
    if index < 0:
        raise ArgumentError(f"negative argument index {index}")
    return index * SLOT_BYTES


def check_type(name, type_name):
    """Is this a type the ABI carries, and if not, why not?"""
    if type_name in TYPES:
        return
    if type_name in REJECTED:
        raise SignatureError(
            f"parameter {name!r} has type {type_name!r}, which the v1 "
            f"argument ABI does not carry",
            detail={'reason': REJECTED[type_name]},
            remedy='pass it through a buffer element, or widen it to a '
                   'supported scalar type explicitly')
    raise SignatureError(
        f"parameter {name!r} has unknown type {type_name!r}",
        detail={'supported': ', '.join(sorted(TYPES))})


def check_value(name, type_name, value):
    """Reject a value the ABI cannot represent, with the reason.

    Called before anything is allocated or written, so a refusal costs
    nothing and leaves no side effect (R2-08: admission errors happen before
    any role starts).
    """
    check_type(name, type_name)
    spec = TYPES[type_name]

    if type_name in POINTER_TYPES:
        # Resolved by the runtime, which owns the handle table. What can be
        # said here is the shape.
        if not (isinstance(value, tuple) and len(value) == 2):
            raise ArgumentError(
                f"parameter {name!r} is a device pointer and needs "
                f"(buffer_handle, byte_offset), got {type(value).__name__}",
                remedy='pass a handle from this session, never a host address')
        _handle, offset = value
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ArgumentError(
                f"parameter {name!r}: byte offset must be an integer, got "
                f"{type(offset).__name__}")
        if offset < 0:
            raise ArgumentError(
                f"parameter {name!r}: negative byte offset {offset}")
        return

    if type_name == 'f32':
        # bool is an int in Python and would silently become 1.0.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArgumentError(
                f"parameter {name!r} is f32 and got "
                f"{type(value).__name__} {value!r}")
        return

    # Integers. bool passes isinstance(int) and is refused by name, because a
    # caller who passes True almost certainly meant something wider.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArgumentError(
            f"parameter {name!r} is {type_name} and got "
            f"{type(value).__name__} {value!r}",
            remedy='pass a Python int; a float is not silently truncated')
    if not (spec['min'] <= value <= spec['max']):
        raise ArgumentError(
            f"parameter {name!r} = {value} does not fit {type_name}",
            detail={'range': f"{spec['min']} .. {spec['max']}"},
            remedy='widen the parameter type; the value is not truncated')


def encode_slot(type_name, value):
    """One 8-byte slot, little-endian, high half zero where it must be."""
    spec = TYPES[type_name]
    if type_name == 'f32':
        bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
        return struct.pack('<I', bits) + b'\x00\x00\x00\x00'
    if type_name in ('i32', 'u32'):
        raw = value & 0xFFFFFFFF
        return struct.pack('<I', raw) + b'\x00\x00\x00\x00'
    if type_name in ('i64', 'u64', 'device_ptr64'):
        return struct.pack('<Q', value & 0xFFFFFFFFFFFFFFFF)
    raise SignatureError(f"no encoding for {type_name!r}")


def decode_slot(type_name, raw):
    """The inverse, and it checks what encode promised.

    A decoder that ignores the high half cannot notice a producer that failed
    to zero it, which is exactly the residue-from-the-last-launch bug the
    zero rule exists to prevent.
    """
    if len(raw) != SLOT_BYTES:
        raise ArgumentError(
            f"a slot is {SLOT_BYTES} bytes, got {len(raw)}")
    spec = TYPES[type_name]
    if spec.get('high_half') == 'must be zero' and raw[4:] != b'\x00' * 4:
        raise ArgumentError(
            f"{type_name} slot has a non-zero high half: {raw[4:].hex()}",
            detail={'rule': 'spec/r2_abi_v1.json: high_half must be zero'})
    if type_name == 'f32':
        return struct.unpack('<f', raw[:4])[0]
    if type_name == 'i32':
        return struct.unpack('<i', raw[:4])[0]
    if type_name == 'u32':
        return struct.unpack('<I', raw[:4])[0]
    if type_name == 'i64':
        return struct.unpack('<q', raw)[0]
    if type_name in ('u64', 'device_ptr64'):
        return struct.unpack('<Q', raw)[0]
    raise SignatureError(f"no decoding for {type_name!r}")


def f32_bits(value):
    """The bit pattern an f32 argument will carry.

    Exposed because "did -0.0 survive?" is a question about bits, and every
    test that asks it should ask the same way the encoder answers it.
    """
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def encode_block(params, values):
    """The argument block.

    `params` is the frozen signature: a list of (name, type_name) in order.
    `values` maps name -> value, already resolved (a pointer is an integer
    device address by this point, not a handle).

    Order comes from `params`, never from the mapping. Binding by the order a
    caller happened to build a dict in is the audit's F07, one phase earlier
    and one layer down.
    """
    names = [n for n, _t in params]
    missing = [n for n in names if n not in values]
    extra = [k for k in values if k not in names]
    if missing or extra:
        raise ArgumentError(
            f"argument mismatch: missing {missing}, unexpected {extra}",
            detail={'signature': ', '.join(f"{n}: {t}" for n, t in params)})

    out = bytearray()
    for name, type_name in params:
        value = values[name]
        if type_name in POINTER_TYPES:
            if isinstance(value, tuple):
                raise ArgumentError(
                    f"parameter {name!r}: the runtime must resolve the "
                    "handle to a device address before encoding")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ArgumentError(
                    f"parameter {name!r}: resolved pointer must be an "
                    f"integer address, got {type(value).__name__}")
            if not (0 <= value <= TYPES['u64']['max']):
                raise ArgumentError(
                    f"parameter {name!r}: address {value} is outside 64 bits")
        else:
            check_value(name, type_name, value)
        out += encode_slot(type_name, value)

    assert len(out) == block_size(len(params))
    return bytes(out)


def decode_block(params, blob):
    """Read a block back. Used by the tests and by `inspect`."""
    want = block_size(len(params))
    if len(blob) != want:
        raise ArgumentError(
            f"argument block is {len(blob)} bytes, the signature needs {want}")
    out = {}
    for i, (name, type_name) in enumerate(params):
        raw = blob[offset_of(i):offset_of(i) + SLOT_BYTES]
        try:
            out[name] = decode_slot(type_name, raw)
        except ArgumentError as e:
            raise ArgumentError(
                f"parameter {name!r} at offset {offset_of(i)}: {e.message}",
                detail=e.detail)
    return out


def describe(params):
    """The per-parameter table the manifest carries."""
    rows = []
    for i, (name, type_name) in enumerate(params):
        check_type(name, type_name)
        spec = TYPES[type_name]
        rows.append(dict(name=name, type=type_name, index=i,
                         offset=offset_of(i), slot_bytes=SLOT_BYTES,
                         value_bytes=spec['value_bytes'],
                         alignment=SLOT_BYTES))
    return rows
