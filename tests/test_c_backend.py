import random
import unittest
from contextlib import contextmanager

from ot_dsim.bignum_lib import c_backend
from ot_dsim.bignum_lib.machine import Machine


class CBackendTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0xC0FFEE)

    def rand_u256(self):
        return self.rng.getrandbits(c_backend.XLEN_BITS)

    @contextmanager
    def force_python_backend(self):
        native = c_backend._native
        c_backend._native = None
        try:
            yield
        finally:
            c_backend._native = native

    def test_add_sub_matches_python_math(self):
        for _ in range(200):
            lhs = self.rand_u256()
            rhs = self.rand_u256()
            carry_in = bool(self.rng.getrandbits(1))

            add_val, add_carry = c_backend.add_u256(lhs, rhs, carry_in)
            add_expected = lhs + rhs + int(carry_in)
            self.assertEqual(add_val, add_expected & c_backend.XLEN_MASK)
            self.assertEqual(add_carry, int(add_expected >> c_backend.XLEN_BITS))

            borrow_in = bool(self.rng.getrandbits(1))
            sub_val, sub_borrow = c_backend.sub_u256(lhs, rhs, borrow_in)
            sub_expected = lhs - rhs - int(borrow_in)
            expected_borrow = 1 if sub_expected < 0 else 0
            if expected_borrow:
                sub_expected += 1 << c_backend.XLEN_BITS
            self.assertEqual(sub_val, sub_expected & c_backend.XLEN_MASK)
            self.assertEqual(sub_borrow, expected_borrow)

    def test_compare_bitwise_and_shift(self):
        for _ in range(200):
            lhs = self.rand_u256()
            rhs = self.rand_u256()
            shift = self.rng.randrange(0, c_backend.XLEN_BITS + 64)

            expected_cmp = 0
            if lhs < rhs:
                expected_cmp = -1
            elif lhs > rhs:
                expected_cmp = 1
            self.assertEqual(c_backend.cmp_u256(lhs, rhs), expected_cmp)

            self.assertEqual(c_backend.and_u256(lhs, rhs), lhs & rhs)
            self.assertEqual(c_backend.or_u256(lhs, rhs), lhs | rhs)
            self.assertEqual(c_backend.xor_u256(lhs, rhs), lhs ^ rhs)
            self.assertEqual(c_backend.not_u256(lhs), (~lhs) & c_backend.XLEN_MASK)

            expected_shl = (lhs << shift) & c_backend.XLEN_MASK
            expected_shr = lhs >> shift if shift < c_backend.XLEN_BITS else 0
            self.assertEqual(c_backend.shl_u256(lhs, shift), expected_shl)
            self.assertEqual(c_backend.shr_u256(lhs, shift), expected_shr)

    def test_limb_updates_are_exact(self):
        for _ in range(200):
            base = self.rand_u256()
            idx = self.rng.randrange(0, c_backend.LIMBS)
            new_limb = self.rng.getrandbits(c_backend.LIMB_BITS)

            updated = c_backend.set_limb(base, idx, new_limb)
            self.assertEqual(c_backend.get_limb(updated, idx), new_limb)

            shift = idx * c_backend.LIMB_BITS
            expected = (base & ~(c_backend.LIMB_MASK << shift)) | (new_limb << shift)
            self.assertEqual(updated, expected)

    def test_half_limb_and_half_word_updates_are_exact(self):
        for _ in range(100):
            base = self.rand_u256()
            idx = self.rng.randrange(0, c_backend.LIMBS)
            upper = bool(self.rng.getrandbits(1))
            half_limb_val = self.rng.getrandbits(c_backend.HALF_LIMB_BITS)

            updated_half_limb = c_backend.set_half_limb(base, idx, half_limb_val, upper)
            half_shift = idx * c_backend.LIMB_BITS + (
                c_backend.HALF_LIMB_BITS if upper else 0
            )
            expected_half_limb = (base & ~(c_backend.HALF_LIMB_MASK << half_shift)) | (
                half_limb_val << half_shift
            )
            self.assertEqual(updated_half_limb, expected_half_limb)

            half_word_idx = self.rng.randrange(0, 2)
            half_word_val = self.rng.getrandbits(c_backend.HALF_WORD_BITS)
            updated_half_word = c_backend.set_half_word(
                base, half_word_idx, half_word_val
            )
            hw_shift = half_word_idx * c_backend.HALF_WORD_BITS
            expected_half_word = (base & ~(c_backend.HALF_WORD_MASK << hw_shift)) | (
                half_word_val << hw_shift
            )
            self.assertEqual(updated_half_word, expected_half_word)

    def test_machine_set_reg_limb_overwrites_full_limb(self):
        machine = Machine([], [None])
        machine.set_reg(
            3, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        )
        machine.set_reg_limb(3, 0, 0)
        self.assertEqual(machine.get_reg_limb(3, 0), 0)

    def test_backend_flag(self):
        self.assertIsInstance(c_backend.is_available(), bool)

    def test_python_fallback_path(self):
        lhs = self.rand_u256()
        rhs = self.rand_u256()
        with self.force_python_backend():
            add_val, add_carry = c_backend.add_u256(lhs, rhs)
            self.assertEqual(add_val, (lhs + rhs) & c_backend.XLEN_MASK)
            self.assertEqual(add_carry, int((lhs + rhs) >> c_backend.XLEN_BITS))

            idx = self.rng.randrange(0, c_backend.LIMBS)
            limb = self.rng.getrandbits(c_backend.LIMB_BITS)
            updated = c_backend.set_limb(lhs, idx, limb)
            self.assertEqual(c_backend.get_limb(updated, idx), limb)


if __name__ == "__main__":
    unittest.main()
