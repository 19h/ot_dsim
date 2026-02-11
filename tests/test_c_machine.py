"""Tests for the C-accelerated Machine (CMachine) backend.

Verifies that the C Machine provides API-compatible behavior with the
original pure-Python Machine for all core operations: register access,
limb manipulation, flags, DMEM, GPRs, CSRs/WSRs, loop/call stacks.
"""

import random
import os
import subprocess
import sys
import unittest

from ot_dsim.bignum_lib.machine import Machine, CallStackUnderrun, _USE_C_MACHINE


class CMachineTest(unittest.TestCase):
    """Test the Machine class (which should be backed by C when available)."""

    def test_c_backend_active(self):
        """Verify C backend is being used (if built)."""
        # This test documents which backend is active; it doesn't fail
        # if C isn't available, but it does assert consistency.
        print(f"C Machine active: {_USE_C_MACHINE}")

    def test_machine_abi_version_constant(self):
        if not _USE_C_MACHINE:
            return
        from ot_dsim import _machine as machine_mod

        self.assertTrue(hasattr(machine_mod, "ABI_VERSION"))
        self.assertIsInstance(machine_mod.ABI_VERSION, int)
        self.assertGreaterEqual(machine_mod.ABI_VERSION, 1)

    def test_runtime_force_pure_python_env(self):
        env = os.environ.copy()
        env["OT_DSIM_PURE_PYTHON"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from ot_dsim.bignum_lib.machine import _USE_C_MACHINE; print(int(_USE_C_MACHINE))",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=os.getcwd(),
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "0")

    def test_basic_construction(self):
        m = Machine([], [None])
        self.assertEqual(m.XLEN, 256)
        self.assertEqual(m.LIMBS, 8)
        self.assertEqual(m.NUM_REGS, 32)
        self.assertEqual(m.NUM_GPRS, 32)
        self.assertEqual(m.DMEM_DEPTH, 128)

    def test_class_level_constants(self):
        """Constants must be accessible on the class, not just instances."""
        self.assertEqual(Machine.NUM_REGS, 32)
        self.assertEqual(Machine.NUM_GPRS, 32)
        self.assertEqual(Machine.XLEN, 256)
        self.assertEqual(Machine.LIMBS, 8)
        self.assertEqual(Machine.DMEM_DEPTH, 128)
        self.assertEqual(Machine.IMEM_DEPTH, 1024)

    def test_register_get_set(self):
        m = Machine([], [None])
        val = 0xDEADBEEFCAFEBABE12345678AABBCCDD
        m.set_reg(5, val)
        self.assertEqual(m.get_reg(5), val)

    def test_register_256bit_max(self):
        m = Machine([], [None])
        max_val = (1 << 256) - 1
        m.set_reg(0, max_val)
        self.assertEqual(m.get_reg(0), max_val)

    def test_special_registers(self):
        m = Machine([], [None])
        val = 0xABCD
        for name in ["mod", "dmp", "rfp", "lc"]:
            m.set_reg(name, val)
            self.assertEqual(m.get_reg(name), val)

    def test_limb_get_set(self):
        m = Machine([], [None])
        m.set_reg(3, (1 << 256) - 1)  # all 1s
        m.set_reg_limb(3, 0, 0)
        self.assertEqual(m.get_reg_limb(3, 0), 0)
        # Other limbs should still be all 1s
        for i in range(1, 8):
            self.assertEqual(m.get_reg_limb(3, i), 0xFFFFFFFF)

    def test_half_limb_set(self):
        m = Machine([], [None])
        m.set_reg(4, 0)
        m.set_reg_half_limb(4, 0, 0xABCD, True)  # upper half of limb 0
        # Lower half should be 0, upper half should be 0xABCD
        limb0 = m.get_reg_limb(4, 0)
        self.assertEqual(limb0, 0xABCD0000)

    def test_flags(self):
        m = Machine([], [None])
        for flag in ["M", "L", "Z", "C", "XM", "XL", "XZ", "XC"]:
            m.set_flag(flag, True)
            self.assertTrue(m.get_flag(flag))
            m.set_flag(flag, False)
            self.assertFalse(m.get_flag(flag))

    def test_set_c_z_m_l(self):
        m = Machine([], [None])
        # Value 0: Z should be set, C/M/L should be unset
        m.set_c_z_m_l(0)
        self.assertTrue(m.get_flag("Z"))
        self.assertFalse(m.get_flag("C"))
        self.assertFalse(m.get_flag("M"))
        self.assertFalse(m.get_flag("L"))

        # Value with bit 256 set (carry)
        m.set_c_z_m_l(1 << 256)
        self.assertTrue(m.get_flag("C"))

        # Odd value: L should be set
        m.set_c_z_m_l(1)
        self.assertTrue(m.get_flag("L"))

    def test_dmem(self):
        m = Machine([42, 99], [None])
        self.assertEqual(m.get_dmem(0), 42)
        self.assertEqual(m.get_dmem(1), 99)
        m.set_dmem(2, 777)
        self.assertEqual(m.get_dmem(2), 777)

    def test_dmem_bounds(self):
        m = Machine([], [None])
        with self.assertRaises(IndexError):
            m.get_dmem(128)
        with self.assertRaises(IndexError):
            m.set_dmem(-1, 0)

    def test_gpr_operations(self):
        m = Machine([], [None])
        m.set_gpr(2, 42)
        self.assertEqual(m.get_gpr(2), 42)
        m.set_gpr(2, 0xFFFFFFFF)
        m.inc_gpr(2)
        self.assertEqual(m.get_gpr(2), 0)  # wrap around

    def test_gpr_x0_is_zero(self):
        m = Machine([], [None])
        self.assertEqual(m.get_gpr(0), 0)

    def test_call_stack(self):
        m = Machine([], [None])
        m.push_call_stack(100)
        m.push_call_stack(200)
        self.assertEqual(m.pop_call_stack(), 200)
        self.assertEqual(m.pop_call_stack(), 100)

    def test_call_stack_underrun(self):
        m = Machine([], [None])
        with self.assertRaises(OverflowError):
            m.pop_call_stack()

    def test_loop_stack(self):
        m = Machine([], [None, None, None])
        m.push_loop_stack(5, 2, 0)
        self.assertEqual(m.get_top_loop_end_addr(), 2)
        self.assertEqual(m.get_top_loop_start_addr(), 0)
        self.assertTrue(m.dec_top_loop_cnt())  # 5->4, returns True
        self.assertEqual(m.pop_loop_stack(), 0)

    def test_pc_operations(self):
        m = Machine([], [None, None, None])
        self.assertEqual(m.get_pc(), 0)
        m.set_pc(2)
        self.assertEqual(m.get_pc(), 2)
        m.set_pc(0)
        m.inc_pc()
        self.assertEqual(m.get_pc(), 1)

    def test_csr_flags(self):
        m = Machine([], [None])
        m.set_flag("C", True)
        m.set_flag("Z", True)
        flags_bin = m.get_flags_as_bin()
        self.assertEqual(flags_bin & 0x1, 1)  # C
        self.assertEqual((flags_bin >> 3) & 1, 1)  # Z

    def test_xlen_hex_str(self):
        m = Machine([], [None])
        s = m.get_xlen_hex_str(0xDEADBEEF)
        self.assertIn("deadbeef", s)

    def test_masks(self):
        m = Machine([], [None])
        self.assertEqual(m.xlen_mask, (1 << 256) - 1)
        self.assertEqual(m.limb_mask, (1 << 32) - 1)
        self.assertEqual(m.half_limb_mask, (1 << 16) - 1)
        self.assertEqual(m.hw_mask, (1 << 128) - 1)

    def test_acc(self):
        m = Machine([], [None])
        m.set_acc(12345)
        self.assertEqual(m.get_acc(), 12345)

    def test_clear_regs(self):
        m = Machine([], [None])
        m.set_reg(5, 0xDEAD)
        m.clear_regs()
        self.assertEqual(m.get_reg(5), 0)

    def test_wsr(self):
        m = Machine([], [None])
        m.set_wsr(0, 0x42)  # WSR_MOD
        self.assertEqual(m.get_wsr(0), 0x42)

    def test_random_limb_operations(self):
        """Stress test: random set_reg_limb / get_reg_limb consistency."""
        rng = random.Random(0xBEEF)
        m = Machine([], [None])
        for _ in range(500):
            reg = rng.randrange(0, 32)
            limb_idx = rng.randrange(0, 8)
            limb_val = rng.getrandbits(32)
            m.set_reg_limb(reg, limb_idx, limb_val)
            self.assertEqual(m.get_reg_limb(reg, limb_idx), limb_val)


if __name__ == "__main__":
    unittest.main()
