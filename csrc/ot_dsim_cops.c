#include <Python.h>

#include <stdint.h>
#include <string.h>

#define U256_BYTES 32
#define U256_LIMBS 8

static int parse_fixed_buffer(PyObject *obj,
                              const char *name,
                              Py_ssize_t expected_len,
                              Py_buffer *view) {
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO) != 0) {
        return -1;
    }
    if (view->len != expected_len) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be exactly %zd bytes",
                     name,
                     expected_len);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

static PyObject *bytes_from_u256(const uint8_t value[U256_BYTES]) {
    return PyBytes_FromStringAndSize((const char *)value, U256_BYTES);
}

static PyObject *py_u256_add(PyObject *self, PyObject *args) {
    PyObject *lhs_obj;
    PyObject *rhs_obj;
    int carry_in = 0;
    Py_buffer lhs;
    Py_buffer rhs;
    uint8_t out[U256_BYTES];
    const uint8_t *lhs_bytes;
    const uint8_t *rhs_bytes;
    unsigned int carry;
    PyObject *out_bytes;
    PyObject *ret;
    Py_ssize_t idx;

    (void)self;

    if (!PyArg_ParseTuple(args, "OO|p:u256_add", &lhs_obj, &rhs_obj, &carry_in)) {
        return NULL;
    }
    if (parse_fixed_buffer(lhs_obj, "lhs", U256_BYTES, &lhs) != 0) {
        return NULL;
    }
    if (parse_fixed_buffer(rhs_obj, "rhs", U256_BYTES, &rhs) != 0) {
        PyBuffer_Release(&lhs);
        return NULL;
    }

    lhs_bytes = (const uint8_t *)lhs.buf;
    rhs_bytes = (const uint8_t *)rhs.buf;
    carry = carry_in ? 1U : 0U;

    for (idx = 0; idx < U256_BYTES; ++idx) {
        unsigned int sum = (unsigned int)lhs_bytes[idx] +
                           (unsigned int)rhs_bytes[idx] + carry;
        out[idx] = (uint8_t)(sum & 0xFFU);
        carry = sum >> 8;
    }

    PyBuffer_Release(&lhs);
    PyBuffer_Release(&rhs);

    out_bytes = bytes_from_u256(out);
    if (out_bytes == NULL) {
        return NULL;
    }

    ret = Py_BuildValue("Ni", out_bytes, (int)carry);
    return ret;
}

static PyObject *py_u256_sub(PyObject *self, PyObject *args) {
    PyObject *lhs_obj;
    PyObject *rhs_obj;
    int borrow_in = 0;
    Py_buffer lhs;
    Py_buffer rhs;
    uint8_t out[U256_BYTES];
    const uint8_t *lhs_bytes;
    const uint8_t *rhs_bytes;
    int borrow;
    PyObject *out_bytes;
    PyObject *ret;
    Py_ssize_t idx;

    (void)self;

    if (!PyArg_ParseTuple(args, "OO|p:u256_sub", &lhs_obj, &rhs_obj, &borrow_in)) {
        return NULL;
    }
    if (parse_fixed_buffer(lhs_obj, "lhs", U256_BYTES, &lhs) != 0) {
        return NULL;
    }
    if (parse_fixed_buffer(rhs_obj, "rhs", U256_BYTES, &rhs) != 0) {
        PyBuffer_Release(&lhs);
        return NULL;
    }

    lhs_bytes = (const uint8_t *)lhs.buf;
    rhs_bytes = (const uint8_t *)rhs.buf;
    borrow = borrow_in ? 1 : 0;

    for (idx = 0; idx < U256_BYTES; ++idx) {
        int diff = (int)lhs_bytes[idx] - (int)rhs_bytes[idx] - borrow;
        if (diff < 0) {
            diff += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out[idx] = (uint8_t)diff;
    }

    PyBuffer_Release(&lhs);
    PyBuffer_Release(&rhs);

    out_bytes = bytes_from_u256(out);
    if (out_bytes == NULL) {
        return NULL;
    }

    ret = Py_BuildValue("Ni", out_bytes, borrow);
    return ret;
}

static PyObject *py_u256_cmp(PyObject *self, PyObject *args) {
    PyObject *lhs_obj;
    PyObject *rhs_obj;
    Py_buffer lhs;
    Py_buffer rhs;
    const uint8_t *lhs_bytes;
    const uint8_t *rhs_bytes;
    Py_ssize_t idx;
    int cmp = 0;

    (void)self;

    if (!PyArg_ParseTuple(args, "OO:u256_cmp", &lhs_obj, &rhs_obj)) {
        return NULL;
    }
    if (parse_fixed_buffer(lhs_obj, "lhs", U256_BYTES, &lhs) != 0) {
        return NULL;
    }
    if (parse_fixed_buffer(rhs_obj, "rhs", U256_BYTES, &rhs) != 0) {
        PyBuffer_Release(&lhs);
        return NULL;
    }

    lhs_bytes = (const uint8_t *)lhs.buf;
    rhs_bytes = (const uint8_t *)rhs.buf;
    for (idx = U256_BYTES - 1; idx >= 0; --idx) {
        if (lhs_bytes[idx] < rhs_bytes[idx]) {
            cmp = -1;
            break;
        }
        if (lhs_bytes[idx] > rhs_bytes[idx]) {
            cmp = 1;
            break;
        }
    }

    PyBuffer_Release(&lhs);
    PyBuffer_Release(&rhs);

    return PyLong_FromLong((long)cmp);
}

static PyObject *py_u256_and(PyObject *self, PyObject *args) {
    PyObject *lhs_obj;
    PyObject *rhs_obj;
    Py_buffer lhs;
    Py_buffer rhs;
    uint8_t out[U256_BYTES];
    const uint8_t *lhs_bytes;
    const uint8_t *rhs_bytes;
    Py_ssize_t idx;

    (void)self;

    if (!PyArg_ParseTuple(args, "OO:u256_and", &lhs_obj, &rhs_obj)) {
        return NULL;
    }
    if (parse_fixed_buffer(lhs_obj, "lhs", U256_BYTES, &lhs) != 0) {
        return NULL;
    }
    if (parse_fixed_buffer(rhs_obj, "rhs", U256_BYTES, &rhs) != 0) {
        PyBuffer_Release(&lhs);
        return NULL;
    }

    lhs_bytes = (const uint8_t *)lhs.buf;
    rhs_bytes = (const uint8_t *)rhs.buf;
    for (idx = 0; idx < U256_BYTES; ++idx) {
        out[idx] = (uint8_t)(lhs_bytes[idx] & rhs_bytes[idx]);
    }

    PyBuffer_Release(&lhs);
    PyBuffer_Release(&rhs);

    return bytes_from_u256(out);
}

static PyObject *py_u256_or(PyObject *self, PyObject *args) {
    PyObject *lhs_obj;
    PyObject *rhs_obj;
    Py_buffer lhs;
    Py_buffer rhs;
    uint8_t out[U256_BYTES];
    const uint8_t *lhs_bytes;
    const uint8_t *rhs_bytes;
    Py_ssize_t idx;

    (void)self;

    if (!PyArg_ParseTuple(args, "OO:u256_or", &lhs_obj, &rhs_obj)) {
        return NULL;
    }
    if (parse_fixed_buffer(lhs_obj, "lhs", U256_BYTES, &lhs) != 0) {
        return NULL;
    }
    if (parse_fixed_buffer(rhs_obj, "rhs", U256_BYTES, &rhs) != 0) {
        PyBuffer_Release(&lhs);
        return NULL;
    }

    lhs_bytes = (const uint8_t *)lhs.buf;
    rhs_bytes = (const uint8_t *)rhs.buf;
    for (idx = 0; idx < U256_BYTES; ++idx) {
        out[idx] = (uint8_t)(lhs_bytes[idx] | rhs_bytes[idx]);
    }

    PyBuffer_Release(&lhs);
    PyBuffer_Release(&rhs);

    return bytes_from_u256(out);
}

static PyObject *py_u256_xor(PyObject *self, PyObject *args) {
    PyObject *lhs_obj;
    PyObject *rhs_obj;
    Py_buffer lhs;
    Py_buffer rhs;
    uint8_t out[U256_BYTES];
    const uint8_t *lhs_bytes;
    const uint8_t *rhs_bytes;
    Py_ssize_t idx;

    (void)self;

    if (!PyArg_ParseTuple(args, "OO:u256_xor", &lhs_obj, &rhs_obj)) {
        return NULL;
    }
    if (parse_fixed_buffer(lhs_obj, "lhs", U256_BYTES, &lhs) != 0) {
        return NULL;
    }
    if (parse_fixed_buffer(rhs_obj, "rhs", U256_BYTES, &rhs) != 0) {
        PyBuffer_Release(&lhs);
        return NULL;
    }

    lhs_bytes = (const uint8_t *)lhs.buf;
    rhs_bytes = (const uint8_t *)rhs.buf;
    for (idx = 0; idx < U256_BYTES; ++idx) {
        out[idx] = (uint8_t)(lhs_bytes[idx] ^ rhs_bytes[idx]);
    }

    PyBuffer_Release(&lhs);
    PyBuffer_Release(&rhs);

    return bytes_from_u256(out);
}

static PyObject *py_u256_not(PyObject *self, PyObject *args) {
    PyObject *word_obj;
    Py_buffer word;
    uint8_t out[U256_BYTES];
    const uint8_t *word_bytes;
    Py_ssize_t idx;

    (void)self;

    if (!PyArg_ParseTuple(args, "O:u256_not", &word_obj)) {
        return NULL;
    }
    if (parse_fixed_buffer(word_obj, "word", U256_BYTES, &word) != 0) {
        return NULL;
    }

    word_bytes = (const uint8_t *)word.buf;
    for (idx = 0; idx < U256_BYTES; ++idx) {
        out[idx] = (uint8_t)(~word_bytes[idx]);
    }

    PyBuffer_Release(&word);

    return bytes_from_u256(out);
}

static PyObject *py_u256_shl(PyObject *self, PyObject *args) {
    PyObject *word_obj;
    Py_ssize_t shift;
    Py_buffer word;
    uint8_t out[U256_BYTES];
    const uint8_t *word_bytes;

    (void)self;

    if (!PyArg_ParseTuple(args, "On:u256_shl", &word_obj, &shift)) {
        return NULL;
    }
    if (shift < 0) {
        PyErr_SetString(PyExc_ValueError, "shift must be non-negative");
        return NULL;
    }
    if (parse_fixed_buffer(word_obj, "word", U256_BYTES, &word) != 0) {
        return NULL;
    }

    word_bytes = (const uint8_t *)word.buf;
    memcpy(out, word_bytes, U256_BYTES);

    if (shift >= 256) {
        memset(out, 0, U256_BYTES);
    } else if (shift > 0) {
        Py_ssize_t byte_shift = shift / 8;
        int bit_shift = (int)(shift % 8);
        Py_ssize_t i;
        /* Shift by whole bytes first (little-endian: shifting left means
         * moving bytes toward higher indices). */
        if (byte_shift > 0) {
            for (i = U256_BYTES - 1; i >= byte_shift; --i)
                out[i] = out[i - byte_shift];
            for (i = 0; i < byte_shift; ++i)
                out[i] = 0;
        }
        /* Then shift remaining bits within bytes. */
        if (bit_shift > 0) {
            uint8_t carry = 0;
            for (i = byte_shift; i < U256_BYTES; ++i) {
                uint8_t next_carry = (uint8_t)(out[i] >> (8 - bit_shift));
                out[i] = (uint8_t)((out[i] << bit_shift) | carry);
                carry = next_carry;
            }
        }
    }

    PyBuffer_Release(&word);

    return bytes_from_u256(out);
}

static PyObject *py_u256_shr(PyObject *self, PyObject *args) {
    PyObject *word_obj;
    Py_ssize_t shift;
    Py_buffer word;
    uint8_t out[U256_BYTES];
    const uint8_t *word_bytes;

    (void)self;

    if (!PyArg_ParseTuple(args, "On:u256_shr", &word_obj, &shift)) {
        return NULL;
    }
    if (shift < 0) {
        PyErr_SetString(PyExc_ValueError, "shift must be non-negative");
        return NULL;
    }
    if (parse_fixed_buffer(word_obj, "word", U256_BYTES, &word) != 0) {
        return NULL;
    }

    word_bytes = (const uint8_t *)word.buf;
    memcpy(out, word_bytes, U256_BYTES);

    if (shift >= 256) {
        memset(out, 0, U256_BYTES);
    } else if (shift > 0) {
        Py_ssize_t byte_shift = shift / 8;
        int bit_shift = (int)(shift % 8);
        Py_ssize_t i;
        /* Shift by whole bytes first (little-endian: shifting right means
         * moving bytes toward lower indices). */
        if (byte_shift > 0) {
            for (i = 0; i < U256_BYTES - byte_shift; ++i)
                out[i] = out[i + byte_shift];
            for (i = U256_BYTES - byte_shift; i < U256_BYTES; ++i)
                out[i] = 0;
        }
        /* Then shift remaining bits within bytes. */
        if (bit_shift > 0) {
            Py_ssize_t top = U256_BYTES - byte_shift;
            uint8_t carry = 0;
            for (i = top - 1; i >= 0; --i) {
                uint8_t next_carry = (uint8_t)(out[i] << (8 - bit_shift));
                out[i] = (uint8_t)((out[i] >> bit_shift) | carry);
                carry = next_carry;
            }
        }
    }

    PyBuffer_Release(&word);

    return bytes_from_u256(out);
}

static PyObject *py_u256_get_limb(PyObject *self, PyObject *args) {
    PyObject *word_obj;
    Py_ssize_t limb_idx;
    Py_buffer word;
    const uint8_t *word_bytes;
    Py_ssize_t offset;
    uint32_t limb;

    (void)self;

    if (!PyArg_ParseTuple(args, "On:u256_get_limb", &word_obj, &limb_idx)) {
        return NULL;
    }
    if (limb_idx < 0 || limb_idx >= U256_LIMBS) {
        PyErr_SetString(PyExc_IndexError, "limb index out of range");
        return NULL;
    }
    if (parse_fixed_buffer(word_obj, "word", U256_BYTES, &word) != 0) {
        return NULL;
    }

    word_bytes = (const uint8_t *)word.buf;
    offset = limb_idx * 4;
    limb = (uint32_t)word_bytes[offset] |
           ((uint32_t)word_bytes[offset + 1] << 8) |
           ((uint32_t)word_bytes[offset + 2] << 16) |
           ((uint32_t)word_bytes[offset + 3] << 24);

    PyBuffer_Release(&word);

    return PyLong_FromUnsignedLong((unsigned long)limb);
}

static PyObject *py_u256_set_limb(PyObject *self, PyObject *args) {
    PyObject *word_obj;
    Py_ssize_t limb_idx;
    unsigned long limb_val;
    Py_buffer word;
    uint8_t out[U256_BYTES];
    const uint8_t *word_bytes;
    Py_ssize_t offset;

    (void)self;

    if (!PyArg_ParseTuple(args, "Onk:u256_set_limb", &word_obj, &limb_idx, &limb_val)) {
        return NULL;
    }
    if (limb_idx < 0 || limb_idx >= U256_LIMBS) {
        PyErr_SetString(PyExc_IndexError, "limb index out of range");
        return NULL;
    }
    if (limb_val > 0xFFFFFFFFUL) {
        PyErr_SetString(PyExc_OverflowError, "limb value out of range");
        return NULL;
    }
    if (parse_fixed_buffer(word_obj, "word", U256_BYTES, &word) != 0) {
        return NULL;
    }

    word_bytes = (const uint8_t *)word.buf;
    memcpy(out, word_bytes, U256_BYTES);
    offset = limb_idx * 4;
    out[offset] = (uint8_t)(limb_val & 0xFFUL);
    out[offset + 1] = (uint8_t)((limb_val >> 8) & 0xFFUL);
    out[offset + 2] = (uint8_t)((limb_val >> 16) & 0xFFUL);
    out[offset + 3] = (uint8_t)((limb_val >> 24) & 0xFFUL);

    PyBuffer_Release(&word);

    return bytes_from_u256(out);
}

static PyObject *py_u256_set_half_limb(PyObject *self, PyObject *args) {
    PyObject *word_obj;
    Py_ssize_t limb_idx;
    int upper;
    unsigned long half_limb_val;
    Py_buffer word;
    uint8_t out[U256_BYTES];
    const uint8_t *word_bytes;
    Py_ssize_t offset;

    (void)self;

    if (!PyArg_ParseTuple(args,
                          "Onpk:u256_set_half_limb",
                          &word_obj,
                          &limb_idx,
                          &upper,
                          &half_limb_val)) {
        return NULL;
    }
    if (limb_idx < 0 || limb_idx >= U256_LIMBS) {
        PyErr_SetString(PyExc_IndexError, "limb index out of range");
        return NULL;
    }
    if (half_limb_val > 0xFFFFUL) {
        PyErr_SetString(PyExc_OverflowError, "half-limb value out of range");
        return NULL;
    }
    if (parse_fixed_buffer(word_obj, "word", U256_BYTES, &word) != 0) {
        return NULL;
    }

    word_bytes = (const uint8_t *)word.buf;
    memcpy(out, word_bytes, U256_BYTES);
    offset = limb_idx * 4 + (upper ? 2 : 0);
    out[offset] = (uint8_t)(half_limb_val & 0xFFUL);
    out[offset + 1] = (uint8_t)((half_limb_val >> 8) & 0xFFUL);

    PyBuffer_Release(&word);

    return bytes_from_u256(out);
}

static PyObject *py_u256_set_half_word(PyObject *self, PyObject *args) {
    PyObject *word_obj;
    Py_ssize_t half_word_idx;
    PyObject *half_word_obj;
    Py_buffer word;
    Py_buffer half_word;
    uint8_t out[U256_BYTES];
    const uint8_t *word_bytes;
    const uint8_t *half_word_bytes;
    Py_ssize_t offset;

    (void)self;

    if (!PyArg_ParseTuple(args,
                          "OnO:u256_set_half_word",
                          &word_obj,
                          &half_word_idx,
                          &half_word_obj)) {
        return NULL;
    }
    if (half_word_idx < 0 || half_word_idx >= 2) {
        PyErr_SetString(PyExc_IndexError, "half-word index out of range");
        return NULL;
    }
    if (parse_fixed_buffer(word_obj, "word", U256_BYTES, &word) != 0) {
        return NULL;
    }
    if (parse_fixed_buffer(half_word_obj, "half_word", 16, &half_word) != 0) {
        PyBuffer_Release(&word);
        return NULL;
    }

    word_bytes = (const uint8_t *)word.buf;
    half_word_bytes = (const uint8_t *)half_word.buf;
    memcpy(out, word_bytes, U256_BYTES);
    offset = half_word_idx * 16;
    memcpy(out + offset, half_word_bytes, 16);

    PyBuffer_Release(&word);
    PyBuffer_Release(&half_word);

    return bytes_from_u256(out);
}

static PyMethodDef module_methods[] = {
    {"u256_add", py_u256_add, METH_VARARGS, "Add two little-endian 256-bit values."},
    {"u256_sub", py_u256_sub, METH_VARARGS, "Subtract two little-endian 256-bit values."},
    {"u256_cmp", py_u256_cmp, METH_VARARGS, "Compare two little-endian 256-bit values."},
    {"u256_and", py_u256_and, METH_VARARGS, "Bitwise and for little-endian 256-bit values."},
    {"u256_or", py_u256_or, METH_VARARGS, "Bitwise or for little-endian 256-bit values."},
    {"u256_xor", py_u256_xor, METH_VARARGS, "Bitwise xor for little-endian 256-bit values."},
    {"u256_not", py_u256_not, METH_VARARGS, "Bitwise not for a little-endian 256-bit value."},
    {"u256_shl", py_u256_shl, METH_VARARGS, "Shift left a little-endian 256-bit value."},
    {"u256_shr", py_u256_shr, METH_VARARGS, "Shift right a little-endian 256-bit value."},
    {"u256_get_limb", py_u256_get_limb, METH_VARARGS, "Read a 32-bit limb from a 256-bit value."},
    {"u256_set_limb", py_u256_set_limb, METH_VARARGS, "Write a 32-bit limb into a 256-bit value."},
    {"u256_set_half_limb", py_u256_set_half_limb, METH_VARARGS, "Write a 16-bit half-limb into a 256-bit value."},
    {"u256_set_half_word", py_u256_set_half_word, METH_VARARGS, "Write a 128-bit half-word into a 256-bit value."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_cops",
    "Native 256-bit operations for ot_dsim.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__cops(void) {
    return PyModule_Create(&module_def);
}
