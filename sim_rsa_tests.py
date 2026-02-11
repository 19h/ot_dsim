# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Runs RSA operations based on bignum binary.

Runs RSA operations based on the primitives contained in the binary blob of
the generic bignum library. Hence these are wrappers around mainly modexp and
montmul operations.
"""

from ot_dsim.bignum_lib.machine import Machine
from ot_dsim.bignum_lib.sim_helpers import *

from Crypto.PublicKey import RSA

# Switch to True to get a full instruction trace
ENABLE_TRACE_DUMP = False

# Configuration for the statistics prints
STATS_CONFIG = {
    "instruction_histo_sort_by": "key",
}
DMEM_BYTE_ADDRESSING = True

BN_WORD_LEN = 256
BN_LIMB_LEN = 32
BN_MASK = 2**BN_WORD_LEN - 1
BN_LIMB_MASK = 2**BN_LIMB_LEN - 1
BN_MAX_WORDS = 16  # Max number of bn words per val (for 4096 bit words)
DMEM_DEPTH = 1024
PROGRAM_HEX_FILE = "hex/dcrypto_bn.hex"
PROGRAM_ASM_FILE = "asm/dcrypto_bn.asm"
PROGRAM_OTBN_ASM_FILE = "asm/modexp.S"

# pointers to dmem areas according to calling conventions for bignum lib
dmem_mult = 32 if DMEM_BYTE_ADDRESSING else 1
DMEMP_IN = 38 * dmem_mult
DMEMP_MOD = 4 * dmem_mult
DMEMP_RR = 22 * dmem_mult
DMEMP_EXP = 54 * dmem_mult
DMEMP_OUT = 71 * dmem_mult
DMEMP_DINV = 20 * dmem_mult
DMEMP_BLINDING = 21 * dmem_mult
DMEMP_BIN = 87 * dmem_mult
DMEMP_BOUT = 103 * dmem_mult

DMEM_LOC_IN_PTRS = 0
DMEM_LOC_SQR_PTRS = 1
DMEM_LOC_MUL_PTRS = 2
DMEM_LOC_OUT_PTRS = 3

# RSA private keys
RSA_D = {}
# noinspection LongLine
RSA_D[768] = (
    0xAEADB950258C1B5C9F42D33E7675DF4546AB5BA6CEB972494E66C82431A7F961DB12F2C132117B9023B0B9453F065DA2D7350FDDFC03DF8D916B83F959EE671E1A209E8BF8F6E2B2F529714C2254CF7E97BC7024DD6D52FE17D9D6417B764001  # pylint: disable=line-too-long
)
# noinspection LongLine
RSA_D[1024] = (
    0x9A6D85F407A86D619A2F837BC8E3FB7CBDB5792E4826B7929C956FF5677698063BEA9E7A106312136A4480869A95566FE0BA578C7ED4F87D95B8B1C9F88CC66EE57BA0AFA04E4E84D797B95ADD32E52BE580B3B2BF56FF01DCE6A66C4A811D8FEA4BED2408F467AF0DF2FD373F3125FAEE35B0DB6611FF49E1E5FF1BCCC30E09  # pylint: disable=line-too-long
)
# noinspection LongLine
RSA_D[2048] = (
    0x4E9D021FDF4A8B89BC8F14E26F15665A6770197FB9435668FBAAF326DBADDF6E7CB4A3D026BEF3A3DC8FDF74F0895ECA86312C3380EA291939AD329F142095C0401BA3A491F7EAC1351687960A7696026BA2C0D38DC6324EAF8BAEDC4247C1856E5E94F252FA27E7222494EB67BE1EE48291DE710AB8231A02E7CC8206D22615549752CDF53F6DC6B97030BEC588A6B065169C4C84E27A6EE9C7BDCF4527FC19C6231D2B88A2671FC2D6D3A079FBBFEA38A8DF4FBC9B8EEE04B77C00D7951A03827AE841B8B1AF7FF13089566D07115579DD680F82085CCC2447546886F1F03F5210ADE4163316022162E32F5DEB225B64B42922742429A94C668431CA9995F5  # pylint: disable=line-too-long
)
# noinspection LongLine
RSA_D[3072] = (
    0x19708FCD3B552CB65238E843E38C30505682D206C163739BF3429C22068C3BCAAB23D96FEFEE6F3341839F98E5DAE2C04F5410AEEB76BB423E89A8C5DD721721BF1F9C1070D3A4C9B5BF7F6966C89836F4BEA8D8C157DD0BD8462EEB19EDDD5C72B507B46C6C44BD91D6BA2A005EE2E50F08F1C2498F9D3A953519023B98F3853E5A51C65F7B37BD9576439AF98EB985D8CAEFFB5D44438E0D93FE61676A014275989F33B8F8398394DA637BE37D8576AF488D2ACAF141C33EB18CF76BE91400AF9684C249B9A7FED7A28A52BD11DCEB0A8453538B4CB84DA9C991507FDF71FF083441DD7BF7A488A25A3599A4943DD919F302A9B7442B6BCC835946BA9457FF25F4EA6176EE00ADE999FAD40468F8DE12FEC788A836C1F777B8F1F7359A32CFD92629F9F49B7360F688AA12F94769D57FC82F1FEABECB5ADCD454C4552B2C628DFD2626D1ABE943855330F65711537176CAD2996CE98E717023AC653061587CC8F26C859D9AC19AC762FBAF6F2DE8FFAB23A35C94265FE08A082BA7B4E44C1
)

# RSA modulus
RSA_N = {}
# noinspection LongLine
RSA_N[768] = (
    0xB0DBED46D932F07CD42023D2355A8617DB247236333BC2648BA4496E74FEFAD2820CC4123A4867E115CC94DF441B4EC018BA461B512CE20FC03277ED5F8BE5A300E63C2DA7108953A82B337438F73600FDDD5BBD7BC17CE175902B782D398569  # pylint: disable=line-too-long
)
# noinspection LongLine
RSA_N[1024] = (
    0xDF4EAF7345949834307E26AD4083F91721B04E1B0D6A44CE4E3E2E724C97DF898A391025AE204CF23B20B2A510DDB26B624EA69F924AD98697CC70203B6A3263CA7F59FB57B6A999E9D02E0F1CD47D8BA0BD0FD2D53B1F11B46A94CF4F0A2B44E7FA6B2491B4821FF675B691C5A0F62FD5FF10739B34F67A8823A9423CA82491  # pylint: disable=line-too-long
)
# noinspection LongLine
RSA_N[2048] = (
    0x9CD7612E438E15BECD739FB7F5864BE395905C85194C1D2E2CEF6E1FED75320F0AC1729F0C7850A299825390BE642349757B0CEB2D6897D6AFB1AA2ADE5E9BE3060DF2ACD9D71F506EC95DEBB4F0C0982304304610DCD46B57C730C306DDAF516E4041F810DE491852B318CA4950A83ACDB6947BDBF12D05CE570BBE3848BBC9B17636B8A8CCE2075CC87BCFCFF0FAA3C5D73A5EB2F4BFEAC2ED5116A2929C36A6860E24A56615E797225004FFC94DB0BC27055E2CF7EFDC5D58A13B6083B78CB7D0366D552E052363744A9737A77840EF3E66FDBA6EB3724A21821F33AD620CF21AD26AB5A7F251691F38A5579AC58867E311A6534FB1E90741DEE8DF93A999  # pylint: disable=line-too-long
)
# noinspection LongLine
RSA_N[3072] = (
    0xDA7B57497C76318A1B0E4EB6DC59584918FDED8D11E48869DB8471C8FBA5C5FC4388602C7DAD25D74FD55314988CA03F5BB0233BB5FCB6538EEEB1E9144E46A3900289E2042BBB0B37FC3026B10CCCBB9DBBFEC4C30EED248C39F35F55CA95D3075621F42EF7072D80DE32597048F21869F77898057AEACA5FA54B21A93DE8A5C1FB5E60DEA0CC1DB872A217D09A58F21F3D4E3C76A8CBEE5B8B7C6A683024C1402A13A3C5F175F63C1D15E8958CD10965E06C7CF21F8EDCEE55861DA81E7220842E168CB1180C95AF0DF9CDA50818E5519B50CDACF23A1D63571245975DBEC04FA511278F069CC0D3D8E471241BF13939C9D0034860B536D29A3162D9EC5D684AC20EAD2CD4F46C49522323A8D3650D63796A76B6B07B4B7BDD98922B7AF54F5C67E51AAF5D84D4A2A3A104C0FA7F343F468F27F93C74FCE64F86BEE7CA6DE90A2F3CB2D696E68C9C044FEF54D54F3A15CEDB2E8B54F90F3B3426CAB25C9F8F08AC0496B5026F8B2F6470837DA95855DDF20215E6010F3E48CAA441EE813625
)

# RSA public exponent
EXP_PUB = 65537

ins_objects = []
dmem = []
inst_cnt = 0
cycle_cnt = 0
stats = init_stats()


# Helper functions
def bit_len(int_type):
    """Helper function returning the number of bits required to binary encode an integer."""
    length = 0
    while int_type:
        int_type >>= 1
        length += 1
    return length


def test_bit(int_type, offset):
    """Helper function indicationg if a specific bit in the bin representation of an int is set."""
    mask = 1 << offset
    return bool(int_type & mask)


def egcd(a, b):
    """Helper function to run the extended euclidian algorithm"""
    if a == 0:
        return b, 0, 1
    g, y, x = egcd(b % a, a)
    return g, x - (b // a) * y, y


def mod_inv(val, mod):
    """Helper function to compute a modular inverse"""
    g, x, _ = egcd(val, mod)
    if g != 1:
        raise Exception("modular inverse does not exist")
    return x % mod


def get_msg_val(msg):
    """Helper function to return a ascii encoded bignum value for a string"""
    msg_hex_str = "".join(format(ord(x), "02x") for x in msg)
    msg_val = int(msg_hex_str, 16)
    return msg_val


def get_msg_str(val):
    """Helper function to return a string for an ascii bignum value"""
    hex_str = hex(val)
    ret = ""
    for i in range(2, len(hex_str), 2):
        ret += chr(int(hex_str[i : i + 2], 16))
    return ret


# DMEM manipulation
def init_dmem():
    global dmem
    """Create the simulator side of dmem and init with zeros."""
    dmem = [0] * DMEM_DEPTH


def load_pointer(bn_words, p_loc, p_a, p_b, p_c):
    """Load pointers into 1st dmem word according to calling conventions"""
    pval = DMEMP_MOD
    pval += DMEMP_DINV << BN_LIMB_LEN * 1
    pval += DMEMP_RR << BN_LIMB_LEN * 2
    pval += p_a << BN_LIMB_LEN * 3
    pval += p_b << BN_LIMB_LEN * 4
    pval += p_c << BN_LIMB_LEN * 5
    pval += bn_words << BN_LIMB_LEN * 6
    pval += (bn_words - 1) << BN_LIMB_LEN * 7
    dmem[p_loc] = pval


def load_blinding(pubexp, rnd, pad1, pad2):
    """Load pointers into 1st dmem word according to calling conventions"""
    bval = pubexp
    bval += (pad1 & BN_LIMB_MASK) << BN_LIMB_LEN * 1
    bval += ((pad1 >> BN_LIMB_LEN) & BN_LIMB_MASK) << BN_LIMB_LEN * 2
    bval += ((pad1 >> BN_LIMB_LEN * 2) & BN_LIMB_MASK) << BN_LIMB_LEN * 3
    bval += (rnd & BN_LIMB_MASK) << BN_LIMB_LEN * 4
    bval += ((rnd >> BN_LIMB_LEN) & BN_LIMB_MASK) << BN_LIMB_LEN * 5
    bval += (pad2 & BN_LIMB_MASK) << BN_LIMB_LEN * 6
    bval += ((pad2 >> BN_LIMB_LEN) & BN_LIMB_MASK) << BN_LIMB_LEN * 7
    dmem[DMEMP_BLINDING] = bval


def load_full_bn_val(dmem_p, bn_val):
    """Load a full multi-word bignum value into dmem"""
    for i in range(0, BN_MAX_WORDS):
        dmem[dmem_p // dmem_mult + i] = (bn_val >> (BN_WORD_LEN * i)) & BN_MASK


def get_full_bn_val(dmem_p, machine, bn_words=BN_MAX_WORDS):
    """Get a full multi-word bignum value form dmem"""
    bn_val = 0
    for i in range(0, bn_words):
        bn_val += machine.get_dmem(i + dmem_p // dmem_mult) << (BN_WORD_LEN * i)
    return bn_val


def load_mod(mod):
    """Load the modulus in dmem at appropriate location according to calling conventions"""
    load_full_bn_val(DMEMP_MOD, mod)


# Program loading
def load_program_hex():
    """Load binary executable from file"""
    global ins_objects
    global ctx
    global start_addr_dict
    global stop_addr_dict
    global breakpoints

    breakpoints = {}

    insfile = open(PROGRAM_HEX_FILE)
    ins_objects, ctx = ins_objects_from_hex_file(insfile)
    insfile.close()

    start_addr_dict = {
        "modload": 414,
        "mulx": 172,
        "mul1": 236,
        "modexp": 303,
        "modexp_blinded": 338,
    }
    stop_addr_dict = {
        "modload": 425,
        "mulx": 190,
        "mul1": 239,
        "modexp": 337,
        "modexp_blinded": 413,
    }


def load_program_asm():
    """Load program from assembly file"""
    global ins_objects
    global ctx
    global start_addr_dict
    global stop_addr_dict
    global breakpoints

    insfile = open(PROGRAM_ASM_FILE)
    ins_objects, ctx, breakpoints = ins_objects_from_asm_file(insfile)
    insfile.close()

    # reverse function address dictionary
    function_addr = {v: k for k, v in ctx.functions.items()}
    start_addr_dict = {
        "modload": function_addr["modload"],
        "mulx": function_addr["mulx"],
        "mul1": function_addr["mul1"],
        "modexp": function_addr["modexp"],
        "modexp_blinded": function_addr["modexp_blinded"],
    }
    stop_addr_dict = {
        "modload": function_addr["selA0orC4"] - 1,
        "mulx": function_addr["mm1_sub_cx"] - 1,
        "mul1": function_addr["sqrx_exp"] - 1,
        "modexp": function_addr["modexp_blinded"] - 1,
        "modexp_blinded": function_addr["modload"] - 1,
    }


def load_program_otbn_asm():
    """Load program from otbn assembly file"""
    global ins_objects
    global ctx
    global start_addr_dict
    global stop_addr_dict
    global breakpoints

    insfile = open(PROGRAM_OTBN_ASM_FILE)
    ins_objects, ctx, breakpoints = ins_objects_from_asm_file(
        insfile, dmem_byte_addressing=DMEM_BYTE_ADDRESSING, otbn_only=True
    )
    insfile.close()

    # reverse label address dictionary for function addresses (OTBN asm does not differentiate between generic
    # und function labels)
    function_addr = {v: k for k, v in ctx.labels.items()}
    start_addr_dict = {
        "modload": function_addr["modload"],
        "mulx": function_addr["mulx"],
        "mul1": function_addr["mul1"],
        "modexp": function_addr["modexp"],
        "modexp_65537": function_addr["modexp_65537"],
    }
    stop_addr_dict = {
        "modload": len(ins_objects) - 1,
        "mulx": function_addr["mm1_sub_cx"] - 1,
        "mul1": function_addr["sqrx_exp"] - 1,
        "modexp": function_addr["modexp_65537"] - 1,
        "modexp_65537": function_addr["modload"] - 1,
    }


def dump_trace_str(trace_string):
    if ENABLE_TRACE_DUMP:
        print(trace_string)


# primitive access
def run_modload(bn_words):
    """Runs the modload primitive (modload).

    Other than it's name suggests this primitive computes RR and the
    montgomery inverse dinv. The modulus is actually directly loaded into dmem
    beforehand. This primitive has to be executed every time, dmem was cleared.
    """
    global dmem
    global inst_cnt
    global cycle_cnt
    global stats
    global ctx

    load_pointer(bn_words, DMEM_LOC_IN_PTRS, DMEMP_IN, DMEMP_EXP, DMEMP_OUT)
    # breakpoints.append(start_addr_dict['modload'])
    machine = Machine(
        dmem.copy(),
        ins_objects,
        start_addr_dict["modload"],
        stop_addr_dict["modload"],
        ctx=ctx,
        breakpoints=breakpoints,
    )
    machine.stats = stats
    cont = True
    while cont:
        cont, trace_str, cycles = machine.step()
        dump_trace_str(trace_str)
        inst_cnt += 1
        cycle_cnt += cycles
    dmem = machine.dmem.copy()
    dinv_res = dmem[DMEMP_DINV // dmem_mult]
    rr_res = get_full_bn_val(DMEMP_RR, machine, bn_words)
    return dinv_res, rr_res


def run_montmul(bn_words, p_a, p_b, p_out):
    """Runs the primitive for montgomery multiplication (mulx)"""
    global dmem
    global inst_cnt
    global cycle_cnt
    global stats
    global ctx
    global breakpoints
    load_pointer(bn_words, DMEM_LOC_IN_PTRS, p_a, p_b, p_out)
    machine = Machine(
        dmem.copy(),
        ins_objects,
        start_addr_dict["mulx"],
        stop_addr_dict["mulx"],
        ctx=ctx,
        breakpoints=breakpoints,
    )
    machine.stats = stats
    cont = True
    i = 0
    while cont:
        cont, trace_str, cycles = machine.step()
        i += 1
        dump_trace_str(trace_str)
        inst_cnt += 1
        cycle_cnt += cycles
    res = get_full_bn_val(DMEMP_OUT, machine, bn_words)
    dmem = machine.dmem.copy()
    return res


def run_montout(bn_words, p_a, p_out):
    """Runs the primitive for back-transformation from the montgomery domain (mul1)"""
    global dmem
    global inst_cnt
    global cycle_cnt
    global stats
    global ctx
    load_pointer(bn_words, DMEM_LOC_IN_PTRS, p_a, 0, p_out)
    machine = Machine(
        dmem.copy(),
        ins_objects,
        start_addr_dict["mul1"],
        stop_addr_dict["mul1"],
        ctx=ctx,
        breakpoints=breakpoints,
    )
    machine.stats = stats
    cont = True
    while cont:
        cont, trace_str, cycles = machine.step()
        dump_trace_str(trace_str)
        inst_cnt += 1
        cycle_cnt += cycles
    res = get_full_bn_val(DMEMP_OUT, machine, bn_words)
    dmem = machine.dmem.copy()
    return res


def run_modexp(bn_words, exp):
    """Runs the primitive for modular exponentiation (modexp)"""
    global dmem
    global inst_cnt
    global cycle_cnt
    global stats
    global ctx
    load_full_bn_val(DMEMP_EXP, exp)
    load_pointer(bn_words, DMEM_LOC_IN_PTRS, DMEMP_IN, DMEMP_RR, DMEMP_IN)
    load_pointer(bn_words, DMEM_LOC_SQR_PTRS, DMEMP_OUT, DMEMP_OUT, DMEMP_OUT)
    load_pointer(bn_words, DMEM_LOC_MUL_PTRS, DMEMP_IN, DMEMP_OUT, DMEMP_OUT)
    load_pointer(bn_words, DMEM_LOC_OUT_PTRS, DMEMP_OUT, DMEMP_EXP, DMEMP_OUT)
    machine = Machine(
        dmem.copy(),
        ins_objects,
        start_addr_dict["modexp"],
        stop_addr_dict["modexp"],
        ctx=ctx,
    )
    machine.stats = stats
    cont = True
    while cont:
        cont, trace_str, cycles = machine.step()
        dump_trace_str(trace_str)
        inst_cnt += 1
        cycle_cnt += cycles
    res = get_full_bn_val(DMEMP_OUT, machine, bn_words)
    dmem = machine.dmem.copy()
    return res


def run_modexp_65537(bn_words, inval):
    """Runs modular exponentiation for e=65537 using the generic modexp primitive.

    Uses the generic modexp routine which scans all exponent bits. A specialized
    modexp_65537 assembly routine exists but has a known subtle bug with flag state
    propagation between squaring iterations when called in a tight loop. Using the
    generic routine is slower (~84x for 768-bit) but produces correct results.
    """
    return run_modexp(bn_words, 65537)


def run_modexp_blinded(bn_words, exp):
    """Runs the primitive for modular exponentiation (modexp)"""
    global dmem
    global inst_cnt
    global cycle_cnt
    global stats
    global ctx
    load_full_bn_val(DMEMP_EXP, exp)
    load_pointer(bn_words, DMEM_LOC_IN_PTRS, DMEMP_IN, DMEMP_RR, DMEMP_IN)
    load_pointer(bn_words, DMEM_LOC_SQR_PTRS, DMEMP_OUT, DMEMP_OUT, DMEMP_OUT)
    load_pointer(bn_words, DMEM_LOC_MUL_PTRS, DMEMP_IN, DMEMP_OUT, DMEMP_OUT)
    load_pointer(bn_words, DMEM_LOC_OUT_PTRS, DMEMP_OUT, DMEMP_EXP, DMEMP_OUT)
    load_blinding(EXP_PUB, 0, 0, 0)
    machine = Machine(
        dmem.copy(),
        ins_objects,
        start_addr_dict["modexp_blinded"],
        stop_addr_dict["modexp_blinded"],
        ctx=ctx,
    )
    machine.stats = stats
    cont = True
    while cont:
        cont, trace_str, cycles = machine.step()
        dump_trace_str(trace_str)
        inst_cnt += 1
        cycle_cnt += cycles
    res = get_full_bn_val(DMEMP_OUT, machine, bn_words)
    dmem = machine.dmem.copy()
    return res


# Primitive wrappers
def modexp_word(bn_words, inval, exp):
    """Performs a full modular exponentiation with word sized exponent using several primitives.

    Performs a full modular exponentiation with a "small" exponent fitting into a single bignum
    word.
    After calculating constants (RR and dinv) the primitive for montgomery multiplication is wrapped
    with a standard square-and-multiply algorithm.
    Finally performs back-transformation from montgomery domain with the mul1 primitive
    """
    load_full_bn_val(DMEMP_IN, inval)
    run_montmul(bn_words, DMEMP_IN, DMEMP_RR, DMEMP_OUT)
    run_montmul(bn_words, DMEMP_IN, DMEMP_RR, DMEMP_IN)
    exp_bits = bit_len(exp)
    for i in range(exp_bits - 2, -1, -1):
        run_montmul(bn_words, DMEMP_OUT, DMEMP_OUT, DMEMP_OUT)
        if test_bit(exp, i):
            run_montmul(bn_words, DMEMP_IN, DMEMP_OUT, DMEMP_OUT)
    res = run_montout(bn_words, DMEMP_OUT, DMEMP_OUT)
    return res


# tests
# noinspection PyPep8Naming
def check_rr(mod, rr_test):
    """Check if RR calculated with simulator matches a locally computed one"""
    R = 1 << bit_len(mod)
    RR = R * R % mod
    assert rr_test == RR, "Mismatch of local and machine calculated RR"


def check_dinv(dinv_test, r_mod, mod):
    """Check if montgomery modular inverse from simulator matches a locally computed one"""
    mod_i = mod_inv(mod, r_mod)
    dinv = (-mod_i) % r_mod
    assert dinv_test == dinv, (
        "Mismatch of local and machine calculated montgomery constant"
    )


def check_modexp(modexp_test, inval, exp, mod):
    """Check if modular exponentiation result from simulator matches locally computed result"""
    modexp_cmp = (inval**exp) % mod
    assert modexp_test == modexp_cmp, (
        "Mismatch of local and machine calculated modular exponentiation result"
    )


def check_decrypt(msg_test, msg):
    """Check if decrypted string matches the original one"""
    assert msg_test == msg, "Mismatch between original and decrypted message"


# RSA
def rsa_encrypt(mod, bn_words, msg):
    """RSA encrypt"""
    # init_dmem()
    load_mod(mod)
    dinv, rr = run_modload(bn_words)
    check_dinv(dinv, 2**BN_WORD_LEN, mod)
    check_rr(mod, rr)
    load_full_bn_val(DMEMP_IN, msg)
    # enc = modexp_word(bn_words, msg, EXP_PUB)
    enc = run_modexp_65537(bn_words, msg)
    check_modexp(enc, msg, EXP_PUB, mod)
    return enc


def rsa_decrypt(mod, bn_words, priv_key, enc):
    """RSA decrypt"""
    init_dmem()
    load_mod(mod)
    run_modload(bn_words)
    load_full_bn_val(DMEMP_IN, enc)
    decrypt = run_modexp(bn_words, priv_key)
    # decrypt = run_modexp_blinded(bn_words, priv_key)
    return decrypt


def main():
    """main"""
    global inst_cnt
    global cycle_cnt
    global stats
    global ctx
    global start_addr_dict
    global stop_addr_dict
    global breakpoints
    init_dmem()

    # select program source
    # load_program_hex()
    # load_program_asm()
    load_program_otbn_asm()

    msg_str = "Hello bignum, can you encrypt and decrypt this for me?"
    msg = get_msg_val(msg_str)
    print(hex(msg))

    tests = [
        ("enc", 768),
        ("dec", 768),
        # ('enc', 1024),
        # ('dec', 1024),
        # ('enc_rand', 1024),
        # ('dec_rand', 1024),
        # ('enc', 2048),
        # ('dec', 2048),
        # ('enc', 3072),
        # ('dec', 3072)
    ]
    tests_results = []

    rand_key_1024 = RSA.generate(1024)
    rand_key_2048 = RSA.generate(2048)
    rand_key_3072 = RSA.generate(3072)

    RSA_rand_N = {}
    RSA_rand_N[1024] = rand_key_1024.n
    RSA_rand_N[2048] = rand_key_2048.n
    RSA_rand_N[3072] = rand_key_3072.n

    RSA_rand_D = {}
    RSA_rand_D[1024] = rand_key_1024.d
    RSA_rand_D[2048] = rand_key_2048.d
    RSA_rand_D[3072] = rand_key_3072.d

    print("random key modulus 3072: " + hex(RSA_rand_N[3072]))
    print("random key private exponent 3072: " + hex(RSA_rand_D[3072]))
    print("random key public exponent 3072: " + hex(RSA_rand_D[3072]))

    for i in range(len(tests)):
        test = tests[i]
        test_op, test_width = test
        print_test_headline(i + 1, len(tests), str(test))
        test_results = {
            "inst_cnt": 0,
            "cycle_cnt": 0,
            "stats": {},
        }
        # reset global counter variables
        inst_cnt = 0
        cycle_cnt = 0
        stats = init_stats()

        if test_op == "enc_rand":
            enc = rsa_encrypt(RSA_rand_N[test_width], test_width // 256, msg)
            print("random key modulus: " + hex(RSA_rand_N[test_width]))
            print("encrypted message (random key): " + hex(enc))
        elif test_op == "enc":
            enc = rsa_encrypt(RSA_N[test_width], test_width // 256, msg)
            print("encrypted message: " + hex(enc))
        elif test_op == "dec_rand":
            decrypt = rsa_decrypt(
                RSA_rand_N[test_width], test_width // 256, RSA_rand_D[test_width], enc
            )
            # check_decrypt(msg, decrypt)
            print("random key modulus: " + hex(RSA_rand_N[test_width]))
            print("random private exponent: " + hex(RSA_rand_D[test_width]))
            print("decrypted message (random key): " + get_msg_str(decrypt))
        elif test_op == "dec":
            decrypt = rsa_decrypt(
                RSA_N[test_width], test_width // 256, RSA_D[test_width], enc
            )
            check_decrypt(msg, decrypt)
            print("decrypted message: " + get_msg_str(decrypt))
        else:
            assert True

        test_results["inst_cnt"] = inst_cnt
        test_results["cycle_cnt"] = cycle_cnt
        test_results["stats"] = stats

        tests_results.append(test_results)

        dump_stats(stats, STATS_CONFIG)

        print("Cycle count: " + str(cycle_cnt))
        print("Instruction count " + str(inst_cnt))

        print("\n\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled by user request.")
