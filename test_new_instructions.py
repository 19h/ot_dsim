#!/usr/bin/env python3
"""Test the newly added Cr50 instructions: sigchk, bm, notx"""

from ot_dsim.bignum_lib.instructions import InstructionFactory, InsContext

# Test cases from the actual Cr50 P256 firmware
# Extracted from dcrypto_p256.hex
test_cases = [
    # sigini instructions from the firmware
    ('sigini', 0xf8efb341),  # @0x174: sigini #7320385
    ('sigini', 0xf8123456),  # @0x17e: sigini #1193046
    ('sigini', 0xf8895efc),  # @0x198: sigini #9002748
    ('sigini', 0xf883423f),  # @0x1ad: sigini #8602175
    ('sigini', 0xf8fa73bc),  # @0x1eb: sigini #16413628
    ('sigini', 0xf8123456),  # @0x235: sigini #1193046 (repeated)
    
    # sigchk instructions from the firmware
    ('sigchk', 0xf91fda16),  # @0x178: sigchk #2087446
    ('sigchk', 0xf97d2764),  # @0x188: sigchk #8202084
    ('sigchk', 0xf9856147),  # @0x192: sigchk #8741191
    ('sigchk', 0xf98e4420),  # @0x1c4: sigchk #9323552
    ('sigchk', 0xf9fc9dbf),  # @0x1f3: sigchk #16555455
    ('sigchk', 0xf92598c3),  # @0x1f5: sigchk #2463939
    ('sigchk', 0xf9328977),  # @0x24c: sigchk #3311991
    ('sigchk', 0xf92598c3),  # @0x24e: sigchk #2463939 (repeated)
    ('sigchk', 0xf987c02a),  # @0x24c: sigchk #8896554
    ('sigchk', 0xf94d6eff),  # @0x2b3: sigchk #5074687
    ('sigchk', 0xf94f6304),  # @0x2bc: sigchk #5202692
    ('sigchk', 0xf903f90e),  # @0x2d4: sigchk #260366
    ('sigchk', 0xf9e02e49),  # @0x2d6: sigchk #14691913
    ('sigchk', 0xf9901a43),  # @0x233: sigchk #9443907
    
    # bm (branch if MSB) instructions
    ('bm', 0x10002194),      # @0x188: bm inv_b2b
    ('bm', 0x100021a8),      # @0x194: bm inv_012
    
    # notx (NOT extended) instructions  
    ('notx', 0x4a630000),    # @0x31a: notx r24, r24
    ('not', 0x48674000),     # @0x1b9: not r25, r26 (encoded with fun=0 per a.inc)
]

print("Testing instruction decoding from Cr50 P256 firmware:")
print("=" * 70)

factory = InstructionFactory()
ctx = InsContext()

success_count = 0
total_count = len(test_cases)

for expected_name, ins_word in test_cases:
    try:
        ins = factory.factory_bin(ins_word, ctx)
        _, asm_str, malformed = ins.get_asm_str()
        
        # Extract just the mnemonic
        actual_mnem = asm_str.split()[0]
        
        if actual_mnem == expected_name and not malformed:
            status = "✓"
            success_count += 1
        else:
            status = "✗"
            
        print(f"{status} 0x{ins_word:08x} -> {asm_str:35s} (expected: {expected_name})")
        
        if malformed:
            print(f"  WARNING: Instruction marked as malformed")
        if actual_mnem != expected_name:
            print(f"  ERROR: Expected '{expected_name}' but got '{actual_mnem}'")
            
    except Exception as e:
        print(f"✗ 0x{ins_word:08x} ({expected_name}) -> ERROR: {e}")

print("=" * 70)
print(f"Results: {success_count}/{total_count} tests passed")
print("=" * 70)

print("=" * 70)
print("\nTesting instruction encoding (assembly -> binary):")
print("=" * 70)

# Use the same context
encode_tests = [
    ('sigini #7320385', 0xf8efb341),   # Actual immediate is 23 bits, not 24
    ('sigchk #2087446', 0xf91fda16),
]

for asm_str, expected_word in encode_tests:
    try:
        ins = factory.factory_asm(0, asm_str, ctx)
        actual_word = ins.ins
        
        status = "✓" if actual_word == expected_word else "✗"
        print(f"{status} '{asm_str:30s}' -> 0x{actual_word:08x} (expected: 0x{expected_word:08x})")
        
        if actual_word != expected_word:
            print(f"  MISMATCH: Got 0x{actual_word:08x}, expected 0x{expected_word:08x}")
            
    except Exception as e:
        print(f"✗ '{asm_str}' -> ERROR: {e}")

print("=" * 70)
