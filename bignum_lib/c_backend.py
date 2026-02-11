"""Optional native backend wrappers for 256-bit operations."""

from __future__ import annotations

from typing import Tuple

try:
    from ot_dsim import _cops as _native
except Exception:
    _native = None


XLEN_BITS = 256
XLEN_BYTES = XLEN_BITS // 8
LIMB_BITS = 32
LIMBS = XLEN_BITS // LIMB_BITS
HALF_LIMB_BITS = LIMB_BITS // 2
HALF_WORD_BITS = XLEN_BITS // 2

XLEN_MASK = (1 << XLEN_BITS) - 1
LIMB_MASK = (1 << LIMB_BITS) - 1
HALF_LIMB_MASK = (1 << HALF_LIMB_BITS) - 1
HALF_WORD_MASK = (1 << HALF_WORD_BITS) - 1


def is_available() -> bool:
    return _native is not None


def _require_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")


def _check_u256(name: str, value: int) -> None:
    _require_int(name, value)
    if value < 0 or value > XLEN_MASK:
        raise OverflowError(f"{name} must fit in {XLEN_BITS} bits")


def _to_u256_bytes(name: str, value: int) -> bytes:
    _check_u256(name, value)
    return value.to_bytes(XLEN_BYTES, byteorder="little", signed=False)


def _from_u256_bytes(raw: bytes) -> int:
    return int.from_bytes(raw, byteorder="little", signed=False)


def _mask_for_bits(bits: int) -> int:
    if bits <= 0:
        raise ValueError("bit width must be positive")
    return (1 << bits) - 1


def add_u256(lhs: int, rhs: int, carry: bool = False) -> Tuple[int, int]:
    _check_u256("lhs", lhs)
    _check_u256("rhs", rhs)
    carry_in = 1 if carry else 0

    if _native is None:
        total = lhs + rhs + carry_in
        return total & XLEN_MASK, int(total >> XLEN_BITS)

    out_raw, carry_out = _native.u256_add(
        _to_u256_bytes("lhs", lhs),
        _to_u256_bytes("rhs", rhs),
        carry_in,
    )
    return _from_u256_bytes(out_raw), int(carry_out)


def sub_u256(lhs: int, rhs: int, borrow: bool = False) -> Tuple[int, int]:
    _check_u256("lhs", lhs)
    _check_u256("rhs", rhs)
    borrow_in = 1 if borrow else 0

    if _native is None:
        total = lhs - rhs - borrow_in
        borrow_out = 1 if total < 0 else 0
        if borrow_out:
            total += 1 << XLEN_BITS
        return total & XLEN_MASK, borrow_out

    out_raw, borrow_out = _native.u256_sub(
        _to_u256_bytes("lhs", lhs),
        _to_u256_bytes("rhs", rhs),
        borrow_in,
    )
    return _from_u256_bytes(out_raw), int(borrow_out)


def cmp_u256(lhs: int, rhs: int) -> int:
    _check_u256("lhs", lhs)
    _check_u256("rhs", rhs)

    if _native is None:
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return 0

    return int(_native.u256_cmp(_to_u256_bytes("lhs", lhs), _to_u256_bytes("rhs", rhs)))


def and_u256(lhs: int, rhs: int) -> int:
    _check_u256("lhs", lhs)
    _check_u256("rhs", rhs)

    if _native is None:
        return lhs & rhs

    return _from_u256_bytes(
        _native.u256_and(_to_u256_bytes("lhs", lhs), _to_u256_bytes("rhs", rhs))
    )


def or_u256(lhs: int, rhs: int) -> int:
    _check_u256("lhs", lhs)
    _check_u256("rhs", rhs)

    if _native is None:
        return lhs | rhs

    return _from_u256_bytes(
        _native.u256_or(_to_u256_bytes("lhs", lhs), _to_u256_bytes("rhs", rhs))
    )


def xor_u256(lhs: int, rhs: int) -> int:
    _check_u256("lhs", lhs)
    _check_u256("rhs", rhs)

    if _native is None:
        return lhs ^ rhs

    return _from_u256_bytes(
        _native.u256_xor(_to_u256_bytes("lhs", lhs), _to_u256_bytes("rhs", rhs))
    )


def not_u256(value: int) -> int:
    _check_u256("value", value)

    if _native is None:
        return (~value) & XLEN_MASK

    return _from_u256_bytes(_native.u256_not(_to_u256_bytes("value", value)))


def shl_u256(value: int, shift_bits: int) -> int:
    _check_u256("value", value)
    _require_int("shift_bits", shift_bits)
    if shift_bits < 0:
        raise ValueError("shift_bits must be non-negative")

    if _native is None:
        if shift_bits >= XLEN_BITS:
            return 0
        return (value << shift_bits) & XLEN_MASK

    return _from_u256_bytes(
        _native.u256_shl(_to_u256_bytes("value", value), shift_bits)
    )


def shr_u256(value: int, shift_bits: int) -> int:
    _check_u256("value", value)
    _require_int("shift_bits", shift_bits)
    if shift_bits < 0:
        raise ValueError("shift_bits must be non-negative")

    if _native is None:
        if shift_bits >= XLEN_BITS:
            return 0
        return value >> shift_bits

    return _from_u256_bytes(
        _native.u256_shr(_to_u256_bytes("value", value), shift_bits)
    )


def get_limb(
    value: int, idx: int, limb_bits: int = LIMB_BITS, xlen_bits: int = XLEN_BITS
) -> int:
    _require_int("idx", idx)
    if idx < 0:
        raise IndexError("limb index out of range")
    if xlen_bits % limb_bits:
        raise ValueError("xlen_bits must be divisible by limb_bits")

    total_limbs = xlen_bits // limb_bits
    if idx >= total_limbs:
        raise IndexError("limb index out of range")

    _require_int("value", value)
    value_mask = _mask_for_bits(xlen_bits)
    if value < 0 or value > value_mask:
        raise OverflowError(f"value must fit in {xlen_bits} bits")

    if _native is not None and limb_bits == LIMB_BITS and xlen_bits == XLEN_BITS:
        return int(_native.u256_get_limb(_to_u256_bytes("value", value), idx))

    limb_mask = _mask_for_bits(limb_bits)
    return (value >> (idx * limb_bits)) & limb_mask


def set_limb(
    value: int,
    idx: int,
    limb_value: int,
    limb_bits: int = LIMB_BITS,
    xlen_bits: int = XLEN_BITS,
) -> int:
    _require_int("idx", idx)
    if idx < 0:
        raise IndexError("limb index out of range")
    if xlen_bits % limb_bits:
        raise ValueError("xlen_bits must be divisible by limb_bits")

    total_limbs = xlen_bits // limb_bits
    if idx >= total_limbs:
        raise IndexError("limb index out of range")

    _require_int("value", value)
    value_mask = _mask_for_bits(xlen_bits)
    if value < 0 or value > value_mask:
        raise OverflowError(f"value must fit in {xlen_bits} bits")

    _require_int("limb_value", limb_value)
    limb_mask = _mask_for_bits(limb_bits)
    if limb_value < 0 or limb_value > limb_mask:
        raise OverflowError(f"limb_value must fit in {limb_bits} bits")

    if _native is not None and limb_bits == LIMB_BITS and xlen_bits == XLEN_BITS:
        updated = _native.u256_set_limb(_to_u256_bytes("value", value), idx, limb_value)
        return _from_u256_bytes(updated)

    shift = idx * limb_bits
    clear_mask = ~(limb_mask << shift) & value_mask
    return (value & clear_mask) | (limb_value << shift)


def set_half_limb(
    value: int,
    idx: int,
    half_limb_value: int,
    upper: bool,
    limb_bits: int = LIMB_BITS,
    xlen_bits: int = XLEN_BITS,
) -> int:
    if limb_bits % 2:
        raise ValueError("limb_bits must be even")

    half_limb_bits = limb_bits // 2
    half_limb_mask = _mask_for_bits(half_limb_bits)

    _require_int("half_limb_value", half_limb_value)
    if half_limb_value < 0 or half_limb_value > half_limb_mask:
        raise OverflowError(f"half_limb_value must fit in {half_limb_bits} bits")

    if _native is not None and limb_bits == LIMB_BITS and xlen_bits == XLEN_BITS:
        updated = _native.u256_set_half_limb(
            _to_u256_bytes("value", value),
            idx,
            bool(upper),
            half_limb_value,
        )
        return _from_u256_bytes(updated)

    base = set_limb(
        value, idx, get_limb(value, idx, limb_bits, xlen_bits), limb_bits, xlen_bits
    )
    shift = idx * limb_bits + (half_limb_bits if upper else 0)
    value_mask = _mask_for_bits(xlen_bits)
    clear_mask = ~(half_limb_mask << shift) & value_mask
    return (base & clear_mask) | (half_limb_value << shift)


def set_half_word(
    value: int, idx: int, half_word_value: int, xlen_bits: int = XLEN_BITS
) -> int:
    _require_int("idx", idx)
    if idx < 0 or idx >= 2:
        raise IndexError("half-word index out of range")

    half_word_bits = xlen_bits // 2
    half_word_mask = _mask_for_bits(half_word_bits)

    _require_int("half_word_value", half_word_value)
    if half_word_value < 0 or half_word_value > half_word_mask:
        raise OverflowError(f"half_word_value must fit in {half_word_bits} bits")

    _require_int("value", value)
    value_mask = _mask_for_bits(xlen_bits)
    if value < 0 or value > value_mask:
        raise OverflowError(f"value must fit in {xlen_bits} bits")

    if _native is not None and xlen_bits == XLEN_BITS:
        updated = _native.u256_set_half_word(
            _to_u256_bytes("value", value),
            idx,
            half_word_value.to_bytes(
                HALF_WORD_BITS // 8, byteorder="little", signed=False
            ),
        )
        return _from_u256_bytes(updated)

    shift = idx * half_word_bits
    clear_mask = ~(half_word_mask << shift) & value_mask
    return (value & clear_mask) | (half_word_value << shift)
