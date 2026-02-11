/*
 * ot_dsim C Machine core.
 *
 * Implements the Machine state (registers, flags, DMEM, IMEM references,
 * loop/call stacks) entirely in C and exposes it as a CPython extension type.
 *
 * The instruction decode/execute layer stays in Python but calls into this
 * fast C machine for all state access (the hot path).
 *
 * Copyright lowRISC contributors.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Python.h>
#include <structmember.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Constants matching machine.py                                       */
/* ------------------------------------------------------------------ */

#define NUM_REGS       32
#define NUM_GPRS       32
#define XLEN           256
#define GPR_WIDTH      32
#define LIMBS          8
#define DMEM_DEPTH     128
#define IMEM_DEPTH     1024
#define LOOP_STACK_SZ  16
#define CALL_STACK_SZ  16

#define XLEN_BYTES     (XLEN / 8)       /* 32 */
#define LIMB_BITS      (XLEN / LIMBS)   /* 32 */
#define HALF_LIMB_BITS (LIMB_BITS / 2)  /* 16 */
#define QW_BITS        (XLEN / 4)       /* 64 */
#define HW_BITS        (XLEN / 2)       /* 128 */

/* Python int masks are computed in __init__ and cached as PyObject* */

#define CSR_FLAG     0x7C0
#define CSR_MOD_BASE 0x7D0
#define CSR_RNG      0xFC0
#define WSR_MOD      0
#define WSR_RND      1
#define OT_DSIM_MACHINE_ABI_VERSION 1

/* ------------------------------------------------------------------ */
/* Loop-stack entry                                                    */
/* ------------------------------------------------------------------ */
typedef struct {
    long cnt;
    long end_addr;
    long start_addr;
} LoopEntry;

/* ------------------------------------------------------------------ */
/* CMachine type                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    PyObject_HEAD

    /* Wide data registers (WDRs): stored as Python ints for 256-bit width */
    PyObject *r[NUM_REGS];          /* r0..r31 */
    PyObject *mod;
    PyObject *dmp;
    PyObject *rfp;
    PyObject *lc;
    PyObject *rnd;
    PyObject *acc;

    /* GPRs (32-bit) */
    long gpr[NUM_GPRS];

    /* Flags */
    int M, L, Z, C;
    int XM, XL, XZ, XC;

    /* Program counter */
    long pc;
    long stop_addr;
    int finishFlag;

    /* DMEM: stored as Python list (elements are Python ints for 256-bit) */
    PyObject *dmem;       /* Python list */
    PyObject *init_dmem;  /* Python list of bools */

    /* IMEM: Python list (instruction objects) */
    PyObject *imem;

    /* Loop stack */
    LoopEntry loop_stack[LOOP_STACK_SZ];
    int loop_sp;  /* number of entries */

    /* Call stack */
    long call_stack[CALL_STACK_SZ];
    int call_sp;

    /* Valid half-limb tracking per register */
    /* r_valid_half_limbs[reg][half_limb] */
    int r_valid_half_limbs[NUM_REGS][LIMBS * 2];

    /* Precomputed masks (Python ints) */
    PyObject *xlen_mask;     /* (1<<256)-1 */
    PyObject *limb_mask;     /* (1<<32)-1  */
    PyObject *half_limb_mask;/* (1<<16)-1  */
    PyObject *hw_mask;       /* (1<<128)-1 */
    PyObject *qw_mask;       /* (1<<64)-1  */
    PyObject *gpr_mask;      /* (1<<32)-1  */

    /* Breakpoints dict: addr -> (passes, counter) */
    PyObject *breakpoints;

    /* Force-break state */
    int fb_active;
    int fb_consider_callstack;
    long fb_callstack;
    int fb_consider_loopstack;
    long fb_loopstack;

    /* Context (assembler context, may be None) */
    PyObject *ctx;

    /* Stats dict */
    PyObject *stats;

    /* Limb/half/qw widths (as C ints for fast access) */
    int limb_width;
    int half_limb_width;
    int qw_width;
    int hw_width;
} CMachine;

/* Forward declarations */
static PyTypeObject CMachineType;
static PyObject *CallStackUnderrun;

/* ------------------------------------------------------------------ */
/* Helper: create Python int mask for N bits                           */
/* ------------------------------------------------------------------ */
static PyObject *make_mask(int bits) {
    PyObject *one = PyLong_FromLong(1);
    PyObject *shift = PyLong_FromLong(bits);
    PyObject *shifted = PyNumber_Lshift(one, shift);
    PyObject *result = PyNumber_Subtract(shifted, one);
    Py_DECREF(one);
    Py_DECREF(shift);
    Py_DECREF(shifted);
    return result;
}

/* Helper: Python int zero */
static PyObject *py_zero(void) {
    return PyLong_FromLong(0);
}

/* Helper: extract 32-bit limb from Python int value */
static long extract_limb(PyObject *val, int lidx) {
    /* (val >> (lidx * 32)) & 0xFFFFFFFF */
    PyObject *shift = PyLong_FromLong(lidx * LIMB_BITS);
    PyObject *shifted = PyNumber_Rshift(val, shift);
    PyObject *mask = PyLong_FromUnsignedLong(0xFFFFFFFFUL);
    PyObject *result = PyNumber_And(shifted, mask);
    long ret = PyLong_AsLong(result);
    Py_DECREF(shift);
    Py_DECREF(shifted);
    Py_DECREF(mask);
    Py_DECREF(result);
    return ret;
}

/* Helper: modify a limb in a Python int value, return new Python int (new ref) */
static PyObject *modify_limb(PyObject *val, int lidx, long limbval, PyObject *xlen_mask_obj) {
    int shift_amt = lidx * LIMB_BITS;
    PyObject *shift = PyLong_FromLong(shift_amt);
    PyObject *limb_mask_py = PyLong_FromUnsignedLong(0xFFFFFFFFUL);

    /* clear_mask = ~(limb_mask << shift) & xlen_mask */
    PyObject *shifted_mask = PyNumber_Lshift(limb_mask_py, shift);
    PyObject *inv = PyNumber_Invert(shifted_mask);
    PyObject *clear_mask = PyNumber_And(inv, xlen_mask_obj);

    /* cleared = val & clear_mask */
    PyObject *cleared = PyNumber_And(val, clear_mask);

    /* new_bits = limbval << shift */
    PyObject *limb_py = PyLong_FromLong(limbval);
    PyObject *new_bits = PyNumber_Lshift(limb_py, shift);

    /* result = cleared | new_bits */
    PyObject *result = PyNumber_Or(cleared, new_bits);

    Py_DECREF(shift);
    Py_DECREF(limb_mask_py);
    Py_DECREF(shifted_mask);
    Py_DECREF(inv);
    Py_DECREF(clear_mask);
    Py_DECREF(cleared);
    Py_DECREF(limb_py);
    Py_DECREF(new_bits);

    return result;
}

/* Helper: modify half-limb in a Python int */
static PyObject *modify_half_limb(PyObject *val, int lidx, long half_val, int upper, PyObject *xlen_mask_obj) {
    int shift_amt = (lidx * 2 + (upper ? 1 : 0)) * HALF_LIMB_BITS;
    PyObject *shift = PyLong_FromLong(shift_amt);
    PyObject *hl_mask = PyLong_FromUnsignedLong(0xFFFFUL);

    PyObject *shifted_mask = PyNumber_Lshift(hl_mask, shift);
    PyObject *inv = PyNumber_Invert(shifted_mask);
    PyObject *clear_mask = PyNumber_And(inv, xlen_mask_obj);
    PyObject *cleared = PyNumber_And(val, clear_mask);

    PyObject *hl_py = PyLong_FromLong(half_val);
    PyObject *new_bits = PyNumber_Lshift(hl_py, shift);
    PyObject *result = PyNumber_Or(cleared, new_bits);

    Py_DECREF(shift);
    Py_DECREF(hl_mask);
    Py_DECREF(shifted_mask);
    Py_DECREF(inv);
    Py_DECREF(clear_mask);
    Py_DECREF(cleared);
    Py_DECREF(hl_py);
    Py_DECREF(new_bits);

    return result;
}

/* Helper: modify half-word in a Python int */
static PyObject *modify_half_word(PyObject *val, int hw_idx, PyObject *hw_val, PyObject *hw_mask_obj, PyObject *xlen_mask_obj) {
    int shift_amt = hw_idx * HW_BITS;
    PyObject *shift = PyLong_FromLong(shift_amt);

    PyObject *shifted_mask = PyNumber_Lshift(hw_mask_obj, shift);
    PyObject *inv = PyNumber_Invert(shifted_mask);
    PyObject *clear_mask = PyNumber_And(inv, xlen_mask_obj);
    PyObject *cleared = PyNumber_And(val, clear_mask);

    PyObject *new_bits = PyNumber_Lshift(hw_val, shift);
    PyObject *result = PyNumber_Or(cleared, new_bits);

    Py_DECREF(shift);
    Py_DECREF(shifted_mask);
    Py_DECREF(inv);
    Py_DECREF(clear_mask);
    Py_DECREF(cleared);
    Py_DECREF(new_bits);

    return result;
}

/* Helper: extract a quarter-word from a Python int */
static PyObject *extract_qw(PyObject *val, int qwidx) {
    PyObject *shift = PyLong_FromLong(qwidx * QW_BITS);
    PyObject *shifted = PyNumber_Rshift(val, shift);
    PyObject *mask = make_mask(QW_BITS);
    PyObject *result = PyNumber_And(shifted, mask);
    Py_DECREF(shift);
    Py_DECREF(shifted);
    Py_DECREF(mask);
    return result;
}

/* Helper: test bit at position */
static int test_bit(PyObject *val, int pos) {
    PyObject *shift = PyLong_FromLong(pos);
    PyObject *shifted = PyNumber_Rshift(val, shift);
    PyObject *one = PyLong_FromLong(1);
    PyObject *result = PyNumber_And(shifted, one);
    int ret = PyObject_IsTrue(result);
    Py_DECREF(shift);
    Py_DECREF(shifted);
    Py_DECREF(one);
    Py_DECREF(result);
    return ret;
}

/* Helper: check Python int is in [0, mask] */
static int check_val_range(PyObject *val, PyObject *mask, const char *msg) {
    PyObject *zero = PyLong_FromLong(0);
    int lt_zero = PyObject_RichCompareBool(val, zero, Py_LT);
    if (lt_zero) {
        Py_DECREF(zero);
        PyErr_SetString(PyExc_OverflowError, msg);
        return -1;
    }
    int gt_mask = PyObject_RichCompareBool(val, mask, Py_GT);
    Py_DECREF(zero);
    if (gt_mask) {
        PyErr_SetString(PyExc_OverflowError, msg);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* CMachine.__init__                                                   */
/* ------------------------------------------------------------------ */
static int
CMachine_init(CMachine *self, PyObject *args, PyObject *kwds) {
    PyObject *dmem_list;
    PyObject *imem_list;
    long s_addr = 0;
    PyObject *stop_addr_obj = Py_None;
    PyObject *ctx_obj = Py_None;
    PyObject *breakpoints_obj = Py_None;

    static char *kwlist[] = {"dmem", "imem", "s_addr", "stop_addr", "ctx", "breakpoints", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|lOOO", kwlist,
                                      &dmem_list, &imem_list,
                                      &s_addr, &stop_addr_obj, &ctx_obj, &breakpoints_obj))
        return -1;

    /* Precompute masks */
    self->xlen_mask = make_mask(XLEN);
    self->limb_mask = make_mask(LIMB_BITS);
    self->half_limb_mask = make_mask(HALF_LIMB_BITS);
    self->hw_mask = make_mask(HW_BITS);
    self->qw_mask = make_mask(QW_BITS);
    self->gpr_mask = make_mask(GPR_WIDTH);

    self->limb_width = LIMB_BITS;
    self->half_limb_width = HALF_LIMB_BITS;
    self->qw_width = QW_BITS;
    self->hw_width = HW_BITS;

    /* Initialize all wide registers to 0 */
    for (int i = 0; i < NUM_REGS; i++) {
        self->r[i] = py_zero();
    }
    self->mod = py_zero();
    self->dmp = py_zero();
    self->rfp = py_zero();
    self->lc = py_zero();
    /* rnd has a special default */
    self->rnd = PyLong_FromString(
        "9999999999999999999999999999999999999999999999999999999999999999", NULL, 16);
    self->acc = py_zero();

    /* GPRs */
    memset(self->gpr, 0, sizeof(self->gpr));

    /* Flags */
    self->M = self->L = self->Z = self->C = 0;
    self->XM = self->XL = self->XZ = self->XC = 0;

    /* Valid half-limb tracking */
    for (int i = 0; i < NUM_REGS; i++) {
        for (int j = 0; j < LIMBS * 2; j++) {
            self->r_valid_half_limbs[i][j] = 0;
        }
    }

    /* PC */
    self->pc = s_addr;
    self->finishFlag = 0;

    /* IMEM - store reference */
    Py_INCREF(imem_list);
    self->imem = imem_list;

    /* stop_addr */
    if (stop_addr_obj == Py_None || stop_addr_obj == NULL) {
        self->stop_addr = PyList_Size(imem_list) - 1;
    } else {
        self->stop_addr = PyLong_AsLong(stop_addr_obj);
    }

    /* DMEM */
    Py_ssize_t dmem_len = PyList_Size(dmem_list);
    self->dmem = PyList_New(DMEM_DEPTH);
    self->init_dmem = PyList_New(DMEM_DEPTH);
    for (Py_ssize_t i = 0; i < DMEM_DEPTH; i++) {
        if (i < dmem_len) {
            PyObject *item = PyList_GetItem(dmem_list, i);
            Py_INCREF(item);
            PyList_SET_ITEM(self->dmem, i, item);
            Py_INCREF(Py_True);
            PyList_SET_ITEM(self->init_dmem, i, Py_True);
        } else {
            PyObject *z = py_zero();
            PyList_SET_ITEM(self->dmem, i, z);
            Py_INCREF(Py_False);
            PyList_SET_ITEM(self->init_dmem, i, Py_False);
        }
    }

    /* Loop/call stacks */
    self->loop_sp = 0;
    self->call_sp = 0;

    /* Force-break */
    self->fb_active = 0;
    self->fb_consider_callstack = 0;
    self->fb_callstack = 0;
    self->fb_consider_loopstack = 0;
    self->fb_loopstack = 0;

    /* Context */
    Py_INCREF(ctx_obj);
    self->ctx = ctx_obj;

    /* Breakpoints */
    self->breakpoints = PyDict_New();
    if (breakpoints_obj != Py_None && breakpoints_obj != NULL) {
        /* Iterable of breakpoint addresses */
        PyObject *iter = PyObject_GetIter(breakpoints_obj);
        if (iter) {
            PyObject *item;
            while ((item = PyIter_Next(iter)) != NULL) {
                PyObject *val = Py_BuildValue("(li)", 1, 1);
                PyDict_SetItem(self->breakpoints, item, val);
                Py_DECREF(val);
                Py_DECREF(item);
            }
            Py_DECREF(iter);
        }
        PyErr_Clear();
    }

    /* Stats */
    self->stats = PyDict_New();

    return 0;
}

static void
CMachine_dealloc(CMachine *self) {
    for (int i = 0; i < NUM_REGS; i++) {
        Py_XDECREF(self->r[i]);
    }
    Py_XDECREF(self->mod);
    Py_XDECREF(self->dmp);
    Py_XDECREF(self->rfp);
    Py_XDECREF(self->lc);
    Py_XDECREF(self->rnd);
    Py_XDECREF(self->acc);
    Py_XDECREF(self->dmem);
    Py_XDECREF(self->init_dmem);
    Py_XDECREF(self->imem);
    Py_XDECREF(self->xlen_mask);
    Py_XDECREF(self->limb_mask);
    Py_XDECREF(self->half_limb_mask);
    Py_XDECREF(self->hw_mask);
    Py_XDECREF(self->qw_mask);
    Py_XDECREF(self->gpr_mask);
    Py_XDECREF(self->breakpoints);
    Py_XDECREF(self->ctx);
    Py_XDECREF(self->stats);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ------------------------------------------------------------------ */
/* get_reg / set_reg                                                   */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_reg(CMachine *self, PyObject *args) {
    PyObject *ridx_obj;
    if (!PyArg_ParseTuple(args, "O", &ridx_obj))
        return NULL;

    if (PyLong_Check(ridx_obj)) {
        long idx = PyLong_AsLong(ridx_obj);
        if (idx < 0 || idx >= NUM_REGS) {
            PyErr_SetString(PyExc_IndexError, "register index out of range");
            return NULL;
        }
        Py_INCREF(self->r[idx]);
        return self->r[idx];
    }
    if (PyUnicode_Check(ridx_obj)) {
        const char *name = PyUnicode_AsUTF8(ridx_obj);
        if (strcmp(name, "mod") == 0) { Py_INCREF(self->mod); return self->mod; }
        if (strcmp(name, "dmp") == 0) { Py_INCREF(self->dmp); return self->dmp; }
        if (strcmp(name, "rfp") == 0) { Py_INCREF(self->rfp); return self->rfp; }
        if (strcmp(name, "lc") == 0)  { Py_INCREF(self->lc);  return self->lc; }
        if (strcmp(name, "rnd") == 0) { Py_INCREF(self->rnd); return self->rnd; }
        PyErr_SetString(PyExc_ValueError, "Invalid special register");
        return NULL;
    }
    PyErr_SetString(PyExc_TypeError, "register index must be int or str");
    return NULL;
}

static PyObject *
CMachine_set_reg(CMachine *self, PyObject *args) {
    PyObject *ridx_obj;
    PyObject *value;
    PyObject *valid_limb_obj = Py_None;
    PyObject *valid_half_limb_obj = Py_None;

    if (!PyArg_ParseTuple(args, "OO|OO", &ridx_obj, &value, &valid_limb_obj, &valid_half_limb_obj))
        return NULL;

    /* Range check */
    if (check_val_range(value, self->xlen_mask, "register value out of range") < 0)
        return NULL;

    if (PyLong_Check(ridx_obj)) {
        long idx = PyLong_AsLong(ridx_obj);
        if (idx < 0 || idx >= NUM_REGS) {
            PyErr_SetString(PyExc_IndexError, "register index out of range");
            return NULL;
        }

        /* Update valid half-limb tracking */
        if (valid_limb_obj != Py_None) {
            long vl = PyLong_AsLong(valid_limb_obj);
            self->r_valid_half_limbs[idx][vl * 2] = 1;
            self->r_valid_half_limbs[idx][vl * 2 + 1] = 1;
        } else if (valid_half_limb_obj != Py_None) {
            long vhl = PyLong_AsLong(valid_half_limb_obj);
            self->r_valid_half_limbs[idx][vhl] = 1;
        } else {
            for (int j = 0; j < LIMBS * 2; j++)
                self->r_valid_half_limbs[idx][j] = 1;
        }

        Py_INCREF(value);
        Py_DECREF(self->r[idx]);
        self->r[idx] = value;
        Py_RETURN_NONE;
    }

    if (PyUnicode_Check(ridx_obj)) {
        const char *name = PyUnicode_AsUTF8(ridx_obj);
        PyObject **target = NULL;
        if (strcmp(name, "mod") == 0) target = &self->mod;
        else if (strcmp(name, "dmp") == 0) target = &self->dmp;
        else if (strcmp(name, "rfp") == 0) target = &self->rfp;
        else if (strcmp(name, "lc") == 0)  target = &self->lc;
        else if (strcmp(name, "rnd") == 0) target = &self->rnd;
        else {
            PyErr_SetString(PyExc_ValueError, "Invalid special register");
            return NULL;
        }
        Py_INCREF(value);
        Py_DECREF(*target);
        *target = value;
        Py_RETURN_NONE;
    }

    PyErr_SetString(PyExc_TypeError, "register index must be int or str");
    return NULL;
}

/* ------------------------------------------------------------------ */
/* get_reg_limb / set_reg_limb                                         */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_reg_limb(CMachine *self, PyObject *args) {
    PyObject *ridx_obj;
    int lidx;
    if (!PyArg_ParseTuple(args, "Oi", &ridx_obj, &lidx))
        return NULL;
    if (lidx < 0 || lidx >= LIMBS) {
        PyErr_SetString(PyExc_IndexError, "limb index out of range");
        return NULL;
    }

    /* Get the register value */
    PyObject *reg_args = Py_BuildValue("(O)", ridx_obj);
    PyObject *regval = CMachine_get_reg(self, reg_args);
    Py_DECREF(reg_args);
    if (!regval) return NULL;

    long limb = extract_limb(regval, lidx);
    Py_DECREF(regval);
    return PyLong_FromLong(limb);
}

static PyObject *
CMachine_set_reg_limb(CMachine *self, PyObject *args) {
    PyObject *ridx_obj;
    int lidx;
    long value;
    if (!PyArg_ParseTuple(args, "Oil", &ridx_obj, &lidx, &value))
        return NULL;
    if (lidx < 0 || lidx >= LIMBS) {
        PyErr_SetString(PyExc_IndexError, "limb index out of range");
        return NULL;
    }

    /* Get current register value */
    PyObject *reg_args = Py_BuildValue("(O)", ridx_obj);
    PyObject *regval = CMachine_get_reg(self, reg_args);
    Py_DECREF(reg_args);
    if (!regval) return NULL;

    PyObject *new_val = modify_limb(regval, lidx, value, self->xlen_mask);
    Py_DECREF(regval);

    /* set_reg with valid_limb=lidx */
    PyObject *lidx_py = PyLong_FromLong(lidx);
    PyObject *set_args = Py_BuildValue("(OOOO)", ridx_obj, new_val, lidx_py, Py_None);
    PyObject *result = CMachine_set_reg(self, set_args);
    Py_DECREF(set_args);
    Py_DECREF(new_val);
    Py_DECREF(lidx_py);
    return result;
}

/* ------------------------------------------------------------------ */
/* set_reg_half_limb                                                   */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_set_reg_half_limb(CMachine *self, PyObject *args) {
    PyObject *ridx_obj;
    int lidx;
    long value;
    int upper;
    if (!PyArg_ParseTuple(args, "Oilp", &ridx_obj, &lidx, &value, &upper))
        return NULL;
    if (lidx < 0 || lidx >= LIMBS) {
        PyErr_SetString(PyExc_IndexError, "limb index out of range");
        return NULL;
    }

    PyObject *reg_args = Py_BuildValue("(O)", ridx_obj);
    PyObject *regval = CMachine_get_reg(self, reg_args);
    Py_DECREF(reg_args);
    if (!regval) return NULL;

    PyObject *new_val = modify_half_limb(regval, lidx, value, upper, self->xlen_mask);
    Py_DECREF(regval);

    PyObject *set_args = Py_BuildValue("(OO)", ridx_obj, new_val);
    PyObject *result = CMachine_set_reg(self, set_args);
    Py_DECREF(set_args);
    Py_DECREF(new_val);
    return result;
}

/* ------------------------------------------------------------------ */
/* get_reg_qw                                                          */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_reg_qw(CMachine *self, PyObject *args) {
    PyObject *ridx_obj;
    int qwidx;
    if (!PyArg_ParseTuple(args, "Oi", &ridx_obj, &qwidx))
        return NULL;
    if (qwidx < 0 || qwidx >= 4) {
        PyErr_SetString(PyExc_IndexError, "quarter-word index out of range");
        return NULL;
    }

    PyObject *reg_args = Py_BuildValue("(O)", ridx_obj);
    PyObject *regval = CMachine_get_reg(self, reg_args);
    Py_DECREF(reg_args);
    if (!regval) return NULL;

    PyObject *qw = extract_qw(regval, qwidx);
    Py_DECREF(regval);
    return qw;
}

/* ------------------------------------------------------------------ */
/* set_reg_half_word                                                   */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_set_reg_half_word(CMachine *self, PyObject *args) {
    PyObject *ridx_obj;
    int hw_idx;
    PyObject *hw_value;
    if (!PyArg_ParseTuple(args, "OiO", &ridx_obj, &hw_idx, &hw_value))
        return NULL;
    if (hw_idx < 0 || hw_idx >= 2) {
        PyErr_SetString(PyExc_IndexError, "half-word index out of range");
        return NULL;
    }

    PyObject *reg_args = Py_BuildValue("(O)", ridx_obj);
    PyObject *regval = CMachine_get_reg(self, reg_args);
    Py_DECREF(reg_args);
    if (!regval) return NULL;

    PyObject *new_val = modify_half_word(regval, hw_idx, hw_value, self->hw_mask, self->xlen_mask);
    Py_DECREF(regval);

    PyObject *set_args = Py_BuildValue("(OO)", ridx_obj, new_val);
    PyObject *result = CMachine_set_reg(self, set_args);
    Py_DECREF(set_args);
    Py_DECREF(new_val);
    return result;
}

/* ------------------------------------------------------------------ */
/* get_reg_valid_half_limbs                                            */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_reg_valid_half_limbs(CMachine *self, PyObject *args) {
    int ridx;
    if (!PyArg_ParseTuple(args, "i", &ridx))
        return NULL;
    if (ridx < 0 || ridx >= NUM_REGS) {
        PyErr_SetString(PyExc_IndexError, "register index out of range");
        return NULL;
    }
    PyObject *lst = PyList_New(LIMBS * 2);
    for (int j = 0; j < LIMBS * 2; j++) {
        PyList_SET_ITEM(lst, j, PyBool_FromLong(self->r_valid_half_limbs[ridx][j]));
    }
    return lst;
}

/* ------------------------------------------------------------------ */
/* GPR operations                                                      */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_set_gpr(CMachine *self, PyObject *args) {
    int gpr;
    long value;
    if (!PyArg_ParseTuple(args, "il", &gpr, &value))
        return NULL;
    if (gpr < 0 || gpr >= NUM_GPRS) {
        PyErr_SetString(PyExc_ValueError, "Invalid GPR referenced");
        return NULL;
    }

    /* Writing to x1 pushes to call stack */
    if (gpr == 1) {
        if (self->call_sp >= CALL_STACK_SZ) {
            PyErr_SetString(PyExc_OverflowError, "Call stack overflow");
            return NULL;
        }
        self->call_stack[self->call_sp++] = value;
    }
    if (gpr >= 2) {
        self->gpr[gpr] = value;
    }

    /* Mirror to special wide registers */
    if (gpr >= 8 && gpr < 16) {
        PyObject *new_rfp = modify_limb(self->rfp, gpr - 8, value, self->xlen_mask);
        Py_DECREF(self->rfp);
        self->rfp = new_rfp;
    }
    if (gpr >= 16 && gpr < 24) {
        PyObject *new_dmp = modify_limb(self->dmp, gpr - 16, value, self->xlen_mask);
        Py_DECREF(self->dmp);
        self->dmp = new_dmp;
    }
    if (gpr >= 24) {
        PyObject *new_lc = modify_limb(self->lc, gpr - 24, value, self->xlen_mask);
        Py_DECREF(self->lc);
        self->lc = new_lc;
    }

    Py_RETURN_NONE;
}

static PyObject *
CMachine_get_gpr(CMachine *self, PyObject *args) {
    int gpr;
    if (!PyArg_ParseTuple(args, "i", &gpr))
        return NULL;
    if (gpr < 0 || gpr >= NUM_GPRS) {
        PyErr_SetString(PyExc_ValueError, "Invalid GPR referenced");
        return NULL;
    }

    if (gpr == 0) return PyLong_FromLong(0);
    if (gpr == 1) {
        /* Pop from call stack */
        if (self->call_sp <= 0) {
            PyErr_SetString(PyExc_OverflowError, "Call stack underrun");
            return NULL;
        }
        return PyLong_FromLong(self->call_stack[--self->call_sp]);
    }
    if (gpr >= 2 && gpr < 8) return PyLong_FromLong(self->gpr[gpr]);
    if (gpr >= 8 && gpr < 16) return PyLong_FromLong(extract_limb(self->rfp, gpr - 8));
    if (gpr >= 16 && gpr < 24) return PyLong_FromLong(extract_limb(self->dmp, gpr - 16));
    if (gpr >= 24) return PyLong_FromLong(extract_limb(self->lc, gpr - 24));

    Py_RETURN_NONE;  /* unreachable */
}

static PyObject *
CMachine_inc_gpr(CMachine *self, PyObject *args) {
    int gpr;
    if (!PyArg_ParseTuple(args, "i", &gpr))
        return NULL;

    PyObject *gpr_val_obj = CMachine_get_gpr(self, args);
    if (!gpr_val_obj) return NULL;
    long val = PyLong_AsLong(gpr_val_obj);
    Py_DECREF(gpr_val_obj);

    long new_val = (val + 1) & 0xFFFFFFFF;
    PyObject *set_args = Py_BuildValue("(il)", gpr, new_val);
    PyObject *result = CMachine_set_gpr(self, set_args);
    Py_DECREF(set_args);
    return result;
}

static PyObject *
CMachine_inc_gpr_wlen_bytes(CMachine *self, PyObject *args) {
    int gpr;
    if (!PyArg_ParseTuple(args, "i", &gpr))
        return NULL;

    PyObject *get_args = Py_BuildValue("(i)", gpr);
    PyObject *gpr_val_obj = CMachine_get_gpr(self, get_args);
    Py_DECREF(get_args);
    if (!gpr_val_obj) return NULL;
    long val = PyLong_AsLong(gpr_val_obj);
    Py_DECREF(gpr_val_obj);

    long new_val = (val + XLEN / 8) & 0xFFFFFFFF;
    PyObject *set_args = Py_BuildValue("(il)", gpr, new_val);
    PyObject *result = CMachine_set_gpr(self, set_args);
    Py_DECREF(set_args);
    return result;
}

/* ------------------------------------------------------------------ */
/* CSR / WSR                                                           */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_csr(CMachine *self, PyObject *args) {
    int csr;
    if (!PyArg_ParseTuple(args, "i", &csr))
        return NULL;

    if (csr == CSR_FLAG) {
        /* Return flags as binary */
        int val = self->C | (self->L << 1) | (self->M << 2) | (self->Z << 3)
                  | (self->XC << 4) | (self->XL << 5) | (self->XM << 6) | (self->XZ << 7);
        return PyLong_FromLong(val);
    }
    if ((csr & 0xFF8) == CSR_MOD_BASE) {
        int limb_idx = csr & 0x7;
        return PyLong_FromLong(extract_limb(self->mod, limb_idx));
    }
    if (csr == CSR_RNG) {
        return PyLong_FromLong(extract_limb(self->rnd, 0));
    }
    PyErr_SetString(PyExc_ValueError, "Invalid CSR");
    return NULL;
}

static PyObject *
CMachine_set_csr(CMachine *self, PyObject *args) {
    int csr;
    long val;
    if (!PyArg_ParseTuple(args, "il", &csr, &val))
        return NULL;

    if (csr == CSR_FLAG) {
        self->C  = (val >> 0) & 1;
        self->L  = (val >> 1) & 1;
        self->M  = (val >> 2) & 1;
        self->Z  = (val >> 3) & 1;
        self->XC = (val >> 4) & 1;
        self->XL = (val >> 5) & 1;
        self->XM = (val >> 6) & 1;
        self->XZ = (val >> 7) & 1;
        Py_RETURN_NONE;
    }
    if ((csr & 0xFF8) == CSR_MOD_BASE) {
        int limb_idx = csr & 0x7;
        PyObject *new_mod = modify_limb(self->mod, limb_idx, val, self->xlen_mask);
        Py_DECREF(self->mod);
        self->mod = new_mod;
        Py_RETURN_NONE;
    }
    if (csr == CSR_RNG) {
        PyObject *new_rnd = modify_limb(self->rnd, 0, val, self->xlen_mask);
        Py_DECREF(self->rnd);
        self->rnd = new_rnd;
        Py_RETURN_NONE;
    }
    PyErr_SetString(PyExc_ValueError, "Invalid CSR");
    return NULL;
}

static PyObject *
CMachine_get_wsr(CMachine *self, PyObject *args) {
    int wsr;
    if (!PyArg_ParseTuple(args, "i", &wsr))
        return NULL;
    if (wsr == WSR_MOD) { Py_INCREF(self->mod); return self->mod; }
    if (wsr == WSR_RND) { Py_INCREF(self->rnd); return self->rnd; }
    PyErr_Format(PyExc_ValueError, "Invalid WSR: %d", wsr);
    return NULL;
}

static PyObject *
CMachine_set_wsr(CMachine *self, PyObject *args) {
    int wsr;
    PyObject *val;
    if (!PyArg_ParseTuple(args, "iO", &wsr, &val))
        return NULL;
    if (wsr == WSR_MOD) {
        Py_INCREF(val);
        Py_DECREF(self->mod);
        self->mod = val;
        Py_RETURN_NONE;
    }
    if (wsr == WSR_RND) {
        /* RND WSR is not writable per spec */
        Py_RETURN_NONE;
    }
    PyErr_SetString(PyExc_ValueError, "Invalid WSR");
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Flag operations                                                     */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_flag(CMachine *self, PyObject *args) {
    const char *flag;
    if (!PyArg_ParseTuple(args, "s", &flag))
        return NULL;

    int val = 0;
    if (strcmp(flag, "M") == 0) val = self->M;
    else if (strcmp(flag, "L") == 0) val = self->L;
    else if (strcmp(flag, "Z") == 0) val = self->Z;
    else if (strcmp(flag, "C") == 0) val = self->C;
    else if (strcmp(flag, "XM") == 0) val = self->XM;
    else if (strcmp(flag, "XL") == 0) val = self->XL;
    else if (strcmp(flag, "XZ") == 0) val = self->XZ;
    else if (strcmp(flag, "XC") == 0) val = self->XC;
    else {
        PyErr_SetString(PyExc_ValueError, "Invalid flag identifier");
        return NULL;
    }
    return PyBool_FromLong(val);
}

static PyObject *
CMachine_set_flag(CMachine *self, PyObject *args) {
    const char *flag;
    int val;
    if (!PyArg_ParseTuple(args, "si", &flag, &val))
        return NULL;

    val = val ? 1 : 0;
    if (strcmp(flag, "M") == 0) self->M = val;
    else if (strcmp(flag, "L") == 0) self->L = val;
    else if (strcmp(flag, "Z") == 0) self->Z = val;
    else if (strcmp(flag, "C") == 0) self->C = val;
    else if (strcmp(flag, "XM") == 0) self->XM = val;
    else if (strcmp(flag, "XL") == 0) self->XL = val;
    else if (strcmp(flag, "XZ") == 0) self->XZ = val;
    else if (strcmp(flag, "XC") == 0) self->XC = val;
    else {
        PyErr_SetString(PyExc_ValueError, "Invalid flag identifier");
        return NULL;
    }
    Py_RETURN_NONE;
}

/* set_c_z_m_l(val) - set C, Z, M, L from 257-bit value */
static PyObject *
CMachine_set_c_z_m_l(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;

    /* C = bit 256, M = bit 255, L = bit 0, Z = (val & xlen_mask) == 0 */
    self->C = test_bit(val, XLEN);
    self->M = test_bit(val, XLEN - 1);
    self->L = test_bit(val, 0);
    PyObject *masked = PyNumber_And(val, self->xlen_mask);
    PyObject *zero = PyLong_FromLong(0);
    self->Z = PyObject_RichCompareBool(masked, zero, Py_EQ);
    Py_DECREF(masked);
    Py_DECREF(zero);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_setx_c_z_m_l(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;

    self->XC = test_bit(val, XLEN);
    self->XM = test_bit(val, XLEN - 1);
    self->XL = test_bit(val, 0);
    PyObject *masked = PyNumber_And(val, self->xlen_mask);
    PyObject *zero = PyLong_FromLong(0);
    self->XZ = PyObject_RichCompareBool(masked, zero, Py_EQ);
    Py_DECREF(masked);
    Py_DECREF(zero);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_set_z_m_l(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;

    PyObject *masked = PyNumber_And(val, self->xlen_mask);
    PyObject *zero = PyLong_FromLong(0);
    self->Z = PyObject_RichCompareBool(masked, zero, Py_EQ);
    Py_DECREF(masked);
    Py_DECREF(zero);
    self->M = test_bit(val, XLEN - 1);
    self->L = test_bit(val, 0);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_setx_z_m_l(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;

    PyObject *masked = PyNumber_And(val, self->xlen_mask);
    PyObject *zero = PyLong_FromLong(0);
    self->XZ = PyObject_RichCompareBool(masked, zero, Py_EQ);
    Py_DECREF(masked);
    Py_DECREF(zero);
    self->XM = test_bit(val, XLEN - 1);
    self->XL = test_bit(val, 0);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_set_c_m(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;
    self->C = test_bit(val, XLEN);
    self->M = test_bit(val, XLEN - 1);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_setx_c_m(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;
    self->XC = test_bit(val, XLEN);
    self->XM = test_bit(val, XLEN - 1);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_set_l(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;
    self->L = test_bit(val, 0);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_setx_l(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;
    self->XL = test_bit(val, 0);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_get_flags_as_bin(CMachine *self, PyObject *Py_UNUSED(args)) {
    int val = self->C | (self->L << 1) | (self->M << 2) | (self->Z << 3)
              | (self->XC << 4) | (self->XL << 5) | (self->XM << 6) | (self->XZ << 7);
    return PyLong_FromLong(val);
}

static PyObject *
CMachine_set_flags_as_bin(CMachine *self, PyObject *args) {
    int flags;
    if (!PyArg_ParseTuple(args, "i", &flags))
        return NULL;
    self->C  = (flags >> 0) & 1;
    self->L  = (flags >> 1) & 1;
    self->M  = (flags >> 2) & 1;
    self->Z  = (flags >> 3) & 1;
    self->XC = (flags >> 4) & 1;
    self->XL = (flags >> 5) & 1;
    self->XM = (flags >> 6) & 1;
    self->XZ = (flags >> 7) & 1;
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Accumulator                                                         */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_acc(CMachine *self, PyObject *Py_UNUSED(args)) {
    Py_INCREF(self->acc);
    return self->acc;
}

static PyObject *
CMachine_set_acc(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;
    Py_INCREF(val);
    Py_DECREF(self->acc);
    self->acc = val;
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* PC operations                                                       */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_pc(CMachine *self, PyObject *Py_UNUSED(args)) {
    return PyLong_FromLong(self->pc);
}

static PyObject *
CMachine_set_pc(CMachine *self, PyObject *args) {
    long pc;
    int clearfinish = 0;
    if (!PyArg_ParseTuple(args, "l|p", &pc, &clearfinish))
        return NULL;
    if (pc < 0 || pc >= PyList_Size(self->imem)) {
        PyErr_Format(PyExc_IndexError, "Address %ld out of range (0 to %zd)", pc, PyList_Size(self->imem));
        return NULL;
    }
    self->pc = pc;
    if (clearfinish)
        self->finishFlag = 0;
    Py_RETURN_NONE;
}

static PyObject *
CMachine_inc_pc(CMachine *self, PyObject *Py_UNUSED(args)) {
    long new_pc = self->pc + 1;
    if (new_pc < 0 || new_pc >= PyList_Size(self->imem)) {
        PyErr_Format(PyExc_IndexError, "PC increment out of range");
        return NULL;
    }
    self->pc = new_pc;
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* DMEM operations                                                     */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_dmem(CMachine *self, PyObject *args) {
    long address;
    if (!PyArg_ParseTuple(args, "l", &address))
        return NULL;
    if (address < 0 || address >= DMEM_DEPTH) {
        PyErr_SetString(PyExc_IndexError, "DMEM address out of range");
        return NULL;
    }

    PyObject *init = PyList_GetItem(self->init_dmem, address);
    if (init == Py_False) {
        PySys_WriteStderr("Warning: reading from uninitialized dmem memory address: 0x%lx\n", address);
    }

    PyObject *val = PyList_GetItem(self->dmem, address);
    Py_INCREF(val);
    return val;
}

static PyObject *
CMachine_set_dmem(CMachine *self, PyObject *args) {
    long address;
    PyObject *value;
    if (!PyArg_ParseTuple(args, "lO", &address, &value))
        return NULL;
    if (address < 0 || address >= DMEM_DEPTH) {
        PyErr_SetString(PyExc_IndexError, "DMEM address out of range");
        return NULL;
    }
    if (check_val_range(value, self->xlen_mask, "DMEM value out of range") < 0)
        return NULL;

    Py_INCREF(value);
    PyList_SetItem(self->dmem, address, value);  /* steals ref */
    Py_INCREF(Py_True);
    PyList_SetItem(self->init_dmem, address, Py_True);  /* steals ref */
    Py_RETURN_NONE;
}

static PyObject *
CMachine_get_dmem_otbn(CMachine *self, PyObject *args) {
    long address;
    if (!PyArg_ParseTuple(args, "l", &address))
        return NULL;
    long dmem_addr = address / 32;
    int limb = (int)((address % 32) / 4);
    if (dmem_addr < 0 || dmem_addr >= DMEM_DEPTH) {
        PyErr_SetString(PyExc_IndexError, "DMEM address out of range");
        return NULL;
    }
    PyObject *cell = PyList_GetItem(self->dmem, dmem_addr);
    long val = extract_limb(cell, limb);
    return PyLong_FromLong(val);
}

static PyObject *
CMachine_set_dmem_otbn(CMachine *self, PyObject *args) {
    long address;
    long value;
    if (!PyArg_ParseTuple(args, "ll", &address, &value))
        return NULL;
    long dmem_addr = address / 32;
    int limb = (int)((address % 32) / 4);
    if (dmem_addr < 0 || dmem_addr >= DMEM_DEPTH) {
        PyErr_SetString(PyExc_IndexError, "DMEM address out of range");
        return NULL;
    }
    PyObject *cell = PyList_GetItem(self->dmem, dmem_addr);
    PyObject *new_val = modify_limb(cell, limb, value, self->xlen_mask);
    PyList_SetItem(self->dmem, dmem_addr, new_val); /* steals ref */
    Py_INCREF(Py_True);
    PyList_SetItem(self->init_dmem, dmem_addr, Py_True);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Loop stack                                                          */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_push_loop_stack(CMachine *self, PyObject *args) {
    long cnt, end_addr, start_addr;
    if (!PyArg_ParseTuple(args, "lll", &cnt, &end_addr, &start_addr))
        return NULL;
    if (self->loop_sp >= LOOP_STACK_SZ) {
        PyErr_SetString(PyExc_OverflowError, "Loop stack overflow");
        return NULL;
    }
    self->loop_stack[self->loop_sp].cnt = cnt;
    self->loop_stack[self->loop_sp].end_addr = end_addr;
    self->loop_stack[self->loop_sp].start_addr = start_addr;
    self->loop_sp++;
    Py_RETURN_NONE;
}

static PyObject *
CMachine_dec_top_loop_cnt(CMachine *self, PyObject *Py_UNUSED(args)) {
    if (self->loop_sp <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Nothing on loop stack to decrement");
        return NULL;
    }
    if (self->loop_stack[self->loop_sp - 1].cnt > 0) {
        self->loop_stack[self->loop_sp - 1].cnt--;
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
CMachine_get_top_loop_end_addr(CMachine *self, PyObject *Py_UNUSED(args)) {
    if (self->loop_sp <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Nothing on loop stack");
        return NULL;
    }
    return PyLong_FromLong(self->loop_stack[self->loop_sp - 1].end_addr);
}

static PyObject *
CMachine_get_top_loop_start_addr(CMachine *self, PyObject *Py_UNUSED(args)) {
    if (self->loop_sp <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Nothing on loop stack");
        return NULL;
    }
    return PyLong_FromLong(self->loop_stack[self->loop_sp - 1].start_addr);
}

static PyObject *
CMachine_pop_loop_stack(CMachine *self, PyObject *Py_UNUSED(args)) {
    if (self->loop_sp <= 0) {
        PyErr_SetString(PyExc_OverflowError, "Loop stack underrun");
        return NULL;
    }
    self->loop_sp--;
    return PyLong_FromLong(self->loop_stack[self->loop_sp].start_addr);
}

/* ------------------------------------------------------------------ */
/* Call stack                                                           */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_push_call_stack(CMachine *self, PyObject *args) {
    long address;
    if (!PyArg_ParseTuple(args, "l", &address))
        return NULL;
    if (self->call_sp >= CALL_STACK_SZ) {
        PyErr_SetString(PyExc_OverflowError, "Call stack overflow");
        return NULL;
    }
    self->call_stack[self->call_sp++] = address;
    Py_RETURN_NONE;
}

static PyObject *
CMachine_pop_call_stack(CMachine *self, PyObject *Py_UNUSED(args)) {
    if (self->call_sp <= 0) {
        PyErr_SetString(CallStackUnderrun, "Call stack underrun");
        return NULL;
    }
    return PyLong_FromLong(self->call_stack[--self->call_sp]);
}

/* ------------------------------------------------------------------ */
/* get_instruction                                                     */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_instruction(CMachine *self, PyObject *args) {
    long address;
    if (!PyArg_ParseTuple(args, "l", &address))
        return NULL;
    if (address < 0 || address >= PyList_Size(self->imem)) {
        PyErr_Format(PyExc_IndexError, "Address %ld out of range (0 to %zd)", address, PyList_Size(self->imem));
        return NULL;
    }
    PyObject *instr = PyList_GetItem(self->imem, address);
    Py_INCREF(instr);
    return instr;
}

/* ------------------------------------------------------------------ */
/* finish                                                              */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_finish(CMachine *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"breakpoint", NULL};
    int breakpoint = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", kwlist, &breakpoint))
        return NULL;
    self->finishFlag = 1;
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* clear_regs                                                          */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_clear_regs(CMachine *self, PyObject *Py_UNUSED(args)) {
    for (int i = 0; i < NUM_REGS; i++) {
        Py_DECREF(self->r[i]);
        self->r[i] = py_zero();
    }
    Py_DECREF(self->mod); self->mod = py_zero();
    Py_DECREF(self->dmp); self->dmp = py_zero();
    Py_DECREF(self->rfp); self->rfp = py_zero();
    Py_DECREF(self->lc);  self->lc = py_zero();
    Py_DECREF(self->rnd);
    self->rnd = PyLong_FromString(
        "9999999999999999999999999999999999999999999999999999999999999999", NULL, 16);
    Py_DECREF(self->acc); self->acc = py_zero();
    self->pc = 0;
    memset(self->gpr, 0, sizeof(self->gpr));
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* reset                                                               */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_reset(CMachine *self, PyObject *args, PyObject *kwds) {
    PyObject *dmem_list;
    PyObject *imem_list;
    long s_addr = 0;
    PyObject *stop_addr_obj = Py_None;
    int clear_regs = 0;

    static char *kwlist[] = {"dmem", "imem", "s_addr", "stop_addr", "clear_regs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|lOp", kwlist,
                                      &dmem_list, &imem_list, &s_addr, &stop_addr_obj, &clear_regs))
        return NULL;

    /* Flags */
    self->M = self->L = self->Z = self->C = 0;
    self->XM = self->XL = self->XZ = self->XC = 0;

    if (clear_regs) {
        CMachine_clear_regs(self, NULL);
    }

    /* Valid half limbs */
    for (int i = 0; i < NUM_REGS; i++)
        for (int j = 0; j < LIMBS * 2; j++)
            self->r_valid_half_limbs[i][j] = 0;

    /* DMEM */
    Py_ssize_t dmem_len = PyList_Size(dmem_list);
    Py_DECREF(self->dmem);
    Py_DECREF(self->init_dmem);
    self->dmem = PyList_New(DMEM_DEPTH);
    self->init_dmem = PyList_New(DMEM_DEPTH);
    for (Py_ssize_t i = 0; i < DMEM_DEPTH; i++) {
        if (i < dmem_len) {
            PyObject *item = PyList_GetItem(dmem_list, i);
            Py_INCREF(item);
            PyList_SET_ITEM(self->dmem, i, item);
            Py_INCREF(Py_True);
            PyList_SET_ITEM(self->init_dmem, i, Py_True);
        } else {
            PyList_SET_ITEM(self->dmem, i, py_zero());
            Py_INCREF(Py_False);
            PyList_SET_ITEM(self->init_dmem, i, Py_False);
        }
    }

    /* IMEM */
    Py_DECREF(self->imem);
    Py_INCREF(imem_list);
    self->imem = imem_list;

    /* Loop/call stacks */
    self->loop_sp = 0;
    self->call_sp = 0;

    /* PC */
    self->pc = s_addr;
    if (stop_addr_obj == Py_None || stop_addr_obj == NULL) {
        self->stop_addr = PyList_Size(imem_list) - 1;
    } else {
        self->stop_addr = PyLong_AsLong(stop_addr_obj);
    }

    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Hex formatting (matching Python Machine)                            */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_limb_hex_str(CMachine *self, PyObject *args) {
    PyObject *val;
    int idx;
    if (!PyArg_ParseTuple(args, "Oi", &val, &idx))
        return NULL;
    long limb = extract_limb(val, idx);
    return PyUnicode_FromFormat("0x%08lx", limb);
}

static PyObject *
CMachine_get_xlen_hex_str(CMachine *self, PyObject *args) {
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O", &val))
        return NULL;

    char buf[80];
    int pos = 0;
    for (int i = LIMBS - 1; i >= 0; i--) {
        long limb = extract_limb(val, i);
        pos += snprintf(buf + pos, sizeof(buf) - pos, "%08lx", limb);
        if (i > 0)
            buf[pos++] = ' ';
    }
    buf[pos] = '\0';
    return PyUnicode_FromString(buf);
}

/* ------------------------------------------------------------------ */
/* get_full_dmem / dump_dmem                                           */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_full_dmem(CMachine *self, PyObject *Py_UNUSED(args)) {
    Py_INCREF(self->dmem);
    return self->dmem;
}

/* ------------------------------------------------------------------ */
/* stat_record_instr (delegated to Python stats dict)                  */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_stat_record_instr(CMachine *self, PyObject *args) {
    PyObject *instr;
    if (!PyArg_ParseTuple(args, "O", &instr))
        return NULL;

    /* instr.get_asm_str() -> (encoding, asm_str) */
    PyObject *asm_result = PyObject_CallMethod(instr, "get_asm_str", NULL);
    if (!asm_result) return NULL;

    PyObject *asm_str = PyTuple_GetItem(asm_result, 1);
    /* Split on space, take first token, strip */
    PyObject *parts = PyUnicode_Split(asm_str, NULL, 1);
    PyObject *opcode = PyList_GetItem(parts, 0);
    PyObject *stripped = PyObject_CallMethod(opcode, "strip", NULL);

    /* stats['instruction_histo'] counter */
    PyObject *key = PyUnicode_FromString("instruction_histo");
    PyObject *histo = PyDict_GetItem(self->stats, key);
    if (!histo) {
        /* import collections.Counter */
        PyObject *collections = PyImport_ImportModule("collections");
        PyObject *counter_cls = PyObject_GetAttrString(collections, "Counter");
        histo = PyObject_CallNoArgs(counter_cls);
        PyDict_SetItem(self->stats, key, histo);
        Py_DECREF(counter_cls);
        Py_DECREF(collections);
        /* histo is now borrowed from dict + our extra ref */
    } else {
        Py_INCREF(histo);
    }

    /* histo[stripped] += 1 */
    PyObject *cur = PyObject_GetItem(histo, stripped);
    if (!cur) {
        PyErr_Clear();
        cur = PyLong_FromLong(0);
    }
    PyObject *one = PyLong_FromLong(1);
    PyObject *new_cnt = PyNumber_Add(cur, one);
    PyObject_SetItem(histo, stripped, new_cnt);

    Py_DECREF(new_cnt);
    Py_DECREF(one);
    Py_DECREF(cur);
    Py_DECREF(histo);
    Py_DECREF(key);
    Py_DECREF(stripped);
    Py_DECREF(parts);
    Py_DECREF(asm_result);

    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Breakpoint operations (minimal for now)                             */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_breakpoints(CMachine *self, PyObject *Py_UNUSED(args)) {
    Py_INCREF(self->breakpoints);
    return self->breakpoints;
}

static PyObject *
CMachine_toggle_breakpoint(CMachine *self, PyObject *args) {
    PyObject *bp;
    int passes = 1;
    int msg = 0;
    if (!PyArg_ParseTuple(args, "O|ip", &bp, &passes, &msg))
        return NULL;

    long addr;
    if (PyLong_Check(bp)) {
        addr = PyLong_AsLong(bp);
    } else if (PyUnicode_Check(bp)) {
        /* Try numeric string first */
        const char *s = PyUnicode_AsUTF8(bp);
        char *endp;
        if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            addr = strtol(s + 2, &endp, 16);
        } else {
            addr = strtol(s, &endp, 10);
            if (*endp != '\0') {
                /* Label lookup via ctx */
                if (self->ctx == Py_None) {
                    PyErr_SetString(PyExc_ValueError, "Label breakpoints only possible with assembly context");
                    return NULL;
                }
                /* ctx.functions / ctx.labels */
                PyObject *functions = PyObject_GetAttrString(self->ctx, "functions");
                PyObject *labels = PyObject_GetAttrString(self->ctx, "labels");
                /* Reverse lookup */
                PyObject *items;
                int found = 0;
                if (functions) {
                    items = PyDict_Values(functions);
                    /* Actually need reverse: name -> addr */
                    PyObject *fkeys = PyDict_Keys(functions);
                    PyObject *fvals = PyDict_Values(functions);
                    Py_ssize_t flen = PyList_Size(fkeys);
                    for (Py_ssize_t i = 0; i < flen; i++) {
                        PyObject *name = PyList_GetItem(fvals, i);
                        if (PyUnicode_Compare(name, bp) == 0) {
                            addr = PyLong_AsLong(PyList_GetItem(fkeys, i));
                            found = 1;
                            break;
                        }
                    }
                    Py_XDECREF(fkeys);
                    Py_XDECREF(fvals);
                    Py_XDECREF(items);
                }
                if (!found && labels) {
                    PyObject *lkeys = PyDict_Keys(labels);
                    PyObject *lvals = PyDict_Values(labels);
                    Py_ssize_t llen = PyList_Size(lkeys);
                    for (Py_ssize_t i = 0; i < llen; i++) {
                        PyObject *name = PyList_GetItem(lvals, i);
                        if (PyUnicode_Compare(name, bp) == 0) {
                            addr = PyLong_AsLong(PyList_GetItem(lkeys, i));
                            found = 1;
                            break;
                        }
                    }
                    Py_XDECREF(lkeys);
                    Py_XDECREF(lvals);
                }
                Py_XDECREF(functions);
                Py_XDECREF(labels);
                if (!found) {
                    PyErr_Format(PyExc_ValueError, "function or label '%s' not found", s);
                    return NULL;
                }
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "breakpoint must be int or str");
        return NULL;
    }

    PyObject *addr_key = PyLong_FromLong(addr);
    if (PyDict_Contains(self->breakpoints, addr_key)) {
        PyDict_DelItem(self->breakpoints, addr_key);
    } else {
        if (addr >= 0 && addr < IMEM_DEPTH) {
            PyObject *val = Py_BuildValue("(ii)", passes, 1);
            PyDict_SetItem(self->breakpoints, addr_key, val);
            Py_DECREF(val);
        }
    }
    Py_DECREF(addr_key);
    Py_RETURN_NONE;
}

static PyObject *
CMachine_set_breakpoint(CMachine *self, PyObject *args) {
    PyObject *bp;
    int passes = 1;
    int msg = 0;
    if (!PyArg_ParseTuple(args, "O|ip", &bp, &passes, &msg))
        return NULL;

    /* Simplified: just resolve to addr and set */
    long addr;
    if (PyLong_Check(bp)) {
        addr = PyLong_AsLong(bp);
    } else {
        const char *s = PyUnicode_AsUTF8(bp);
        char *endp;
        addr = strtol(s, &endp, 0);
        if (*endp != '\0') {
            /* Label resolution - delegate to toggle logic concepts */
            /* For simplicity, just set numeric for now */
            PyErr_SetString(PyExc_ValueError, "Label breakpoints: use toggle_breakpoint");
            return NULL;
        }
    }

    if (addr >= 0 && addr < IMEM_DEPTH) {
        PyObject *addr_key = PyLong_FromLong(addr);
        PyObject *val = Py_BuildValue("(ii)", passes, 1);
        PyDict_SetItem(self->breakpoints, addr_key, val);
        Py_DECREF(val);
        Py_DECREF(addr_key);
    }
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* step() - core simulation step                                       */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_step(CMachine *self, PyObject *Py_UNUSED(args)) {
    int halt = 0;

    if (self->pc == self->stop_addr)
        halt = 1;
    if (self->finishFlag)
        halt = 1;

    /* Check breakpoints */
    int is_break = 0;
    long passes = 0;

    /* Force break check */
    if (self->fb_active) {
        if (self->fb_consider_loopstack && self->loop_sp == self->fb_loopstack) {
            is_break = 1;
            self->fb_active = 0;
        } else if (self->fb_consider_callstack && self->call_sp == self->fb_callstack) {
            is_break = 1;
            self->fb_active = 0;
        } else if (!self->fb_consider_callstack && !self->fb_consider_loopstack) {
            is_break = 1;
            self->fb_active = 0;
        }
    }

    /* Regular breakpoint check */
    if (!is_break && PyDict_Size(self->breakpoints) > 0) {
        PyObject *pc_key = PyLong_FromLong(self->pc);
        PyObject *bp_val = PyDict_GetItem(self->breakpoints, pc_key);
        if (bp_val) {
            long bp_passes = PyLong_AsLong(PyTuple_GetItem(bp_val, 0));
            long bp_cnt = PyLong_AsLong(PyTuple_GetItem(bp_val, 1));
            if (bp_cnt == bp_passes) {
                is_break = 1;
                passes = bp_passes;
                PyObject *new_val = Py_BuildValue("(ll)", bp_passes, (long)1);
                PyDict_SetItem(self->breakpoints, pc_key, new_val);
                Py_DECREF(new_val);
            } else {
                PyObject *new_val = Py_BuildValue("(ll)", bp_passes, bp_cnt + 1);
                PyDict_SetItem(self->breakpoints, pc_key, new_val);
                Py_DECREF(new_val);
            }
        }
        Py_DECREF(pc_key);
    }

    /* Handle breakpoint (call back into Python debugger) */
    if (is_break) {
        /* Use Python-level break handling - call print_asm and input loop */
        /* We'll emit a simple message and let the Python wrapper handle interactive debug */
        if (passes) {
            PySys_WriteStdout("Breakpoint hit at address %ld at pass %ld.\n", self->pc, passes);
        } else {
            PySys_WriteStdout("Breakpoint hit at address %ld.\n", self->pc);
        }
        /* For non-interactive: just continue */
    }

    /* Execute instruction */
    PyObject *instr = PyList_GetItem(self->imem, self->pc);
    if (!instr) return NULL;

    /* stat_record_instr */
    PyObject *sr_args = Py_BuildValue("(O)", instr);
    PyObject *sr_result = CMachine_stat_record_instr(self, sr_args);
    Py_DECREF(sr_args);
    if (!sr_result) {
        PyErr_Clear();  /* Non-fatal */
    } else {
        Py_DECREF(sr_result);
    }

    /* Get cycles */
    PyObject *cycles = PyObject_CallMethod(instr, "get_cycles", NULL);
    if (!cycles) return NULL;

    /* Execute: trace_str, jump_addr = instr.execute(self) */
    PyObject *self_obj = (PyObject *)self;
    PyObject *exec_result = PyObject_CallMethod(instr, "execute", "O", self_obj);
    if (!exec_result) return NULL;

    PyObject *trace_str = PyTuple_GetItem(exec_result, 0);
    PyObject *jump_addr_obj = PyTuple_GetItem(exec_result, 1);

    /* Loop stack handling */
    if (self->loop_sp > 0 && self->pc == self->loop_stack[self->loop_sp - 1].end_addr) {
        if (self->loop_stack[self->loop_sp - 1].cnt > 0) {
            self->loop_stack[self->loop_sp - 1].cnt--;
            /* jump to loop start */
            jump_addr_obj = PyLong_FromLong(self->loop_stack[self->loop_sp - 1].start_addr);
        } else {
            self->loop_sp--;
            /* continue without jump - jump_addr_obj stays None */
        }
    }

    int cont = 1;
    if (jump_addr_obj != Py_None && jump_addr_obj != NULL) {
        long jump_addr = PyLong_AsLong(jump_addr_obj);
        if (jump_addr < 0 || jump_addr >= PyList_Size(self->imem)) {
            Py_DECREF(exec_result);
            Py_DECREF(cycles);
            PyErr_SetString(PyExc_RuntimeError, "Invalid jump address");
            return NULL;
        }
        self->pc = jump_addr;
    } else {
        if (self->pc + 1 >= PyList_Size(self->imem)) {
            cont = 0;
        } else {
            self->pc++;
        }
    }

    if (halt)
        cont = 0;

    Py_INCREF(trace_str);
    Py_INCREF(cycles);
    Py_DECREF(exec_result);

    PyObject *result = Py_BuildValue("(NNN)",
                                      PyBool_FromLong(cont),
                                      trace_str,
                                      cycles);
    return result;
}

/* ------------------------------------------------------------------ */
/* Properties exposed to Python                                        */
/* ------------------------------------------------------------------ */
static PyObject *
CMachine_get_finishFlag(CMachine *self, void *closure) {
    (void)closure;
    return PyBool_FromLong(self->finishFlag);
}

static int
CMachine_set_finishFlag(CMachine *self, PyObject *value, void *closure) {
    (void)closure;
    self->finishFlag = PyObject_IsTrue(value);
    return 0;
}

static PyObject *
CMachine_get_loop_stack_py(CMachine *self, void *closure) {
    (void)closure;
    PyObject *lst = PyList_New(self->loop_sp);
    for (int i = 0; i < self->loop_sp; i++) {
        PyObject *entry = Py_BuildValue("(lll)",
            self->loop_stack[i].cnt,
            self->loop_stack[i].end_addr,
            self->loop_stack[i].start_addr);
        PyList_SET_ITEM(lst, i, entry);
    }
    return lst;
}

static PyObject *
CMachine_get_call_stack_py(CMachine *self, void *closure) {
    (void)closure;
    PyObject *lst = PyList_New(self->call_sp);
    for (int i = 0; i < self->call_sp; i++) {
        PyList_SET_ITEM(lst, i, PyLong_FromLong(self->call_stack[i]));
    }
    return lst;
}

static PyObject *
CMachine_get_ctx(CMachine *self, void *closure) {
    (void)closure;
    Py_INCREF(self->ctx);
    return self->ctx;
}

static int
CMachine_set_ctx(CMachine *self, PyObject *value, void *closure) {
    (void)closure;
    Py_INCREF(value);
    Py_XDECREF(self->ctx);
    self->ctx = value;
    return 0;
}

static PyObject *
CMachine_get_stats_py(CMachine *self, void *closure) {
    (void)closure;
    Py_INCREF(self->stats);
    return self->stats;
}

static int
CMachine_set_stats_py(CMachine *self, PyObject *value, void *closure) {
    (void)closure;
    Py_INCREF(value);
    Py_XDECREF(self->stats);
    self->stats = value;
    return 0;
}

static PyObject *
CMachine_get_pc_prop(CMachine *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->pc);
}

static int
CMachine_set_pc_prop(CMachine *self, PyObject *value, void *closure) {
    (void)closure;
    self->pc = PyLong_AsLong(value);
    return 0;
}

static PyObject *
CMachine_get_stop_addr_prop(CMachine *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->stop_addr);
}

static int
CMachine_set_stop_addr_prop(CMachine *self, PyObject *value, void *closure) {
    (void)closure;
    self->stop_addr = PyLong_AsLong(value);
    return 0;
}

/* Properties for mod/dmp/rfp/lc/rnd/acc as direct Python attribute access */
static PyObject *CMachine_get_mod(CMachine *self, void *c) { (void)c; Py_INCREF(self->mod); return self->mod; }
static int CMachine_set_mod(CMachine *self, PyObject *v, void *c) { (void)c; Py_INCREF(v); Py_DECREF(self->mod); self->mod = v; return 0; }
static PyObject *CMachine_get_dmp_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->dmp); return self->dmp; }
static int CMachine_set_dmp_prop(CMachine *self, PyObject *v, void *c) { (void)c; Py_INCREF(v); Py_DECREF(self->dmp); self->dmp = v; return 0; }
static PyObject *CMachine_get_rfp_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->rfp); return self->rfp; }
static int CMachine_set_rfp_prop(CMachine *self, PyObject *v, void *c) { (void)c; Py_INCREF(v); Py_DECREF(self->rfp); self->rfp = v; return 0; }
static PyObject *CMachine_get_lc_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->lc); return self->lc; }
static int CMachine_set_lc_prop(CMachine *self, PyObject *v, void *c) { (void)c; Py_INCREF(v); Py_DECREF(self->lc); self->lc = v; return 0; }
static PyObject *CMachine_get_rnd_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->rnd); return self->rnd; }
static int CMachine_set_rnd_prop(CMachine *self, PyObject *v, void *c) { (void)c; Py_INCREF(v); Py_DECREF(self->rnd); self->rnd = v; return 0; }
static PyObject *CMachine_get_acc_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->acc); return self->acc; }
static int CMachine_set_acc_prop(CMachine *self, PyObject *v, void *c) { (void)c; Py_INCREF(v); Py_DECREF(self->acc); self->acc = v; return 0; }

/* r[] access */
static PyObject *CMachine_get_r(CMachine *self, void *c) {
    (void)c;
    /* Return list of all wide registers */
    PyObject *lst = PyList_New(NUM_REGS);
    for (int i = 0; i < NUM_REGS; i++) {
        Py_INCREF(self->r[i]);
        PyList_SET_ITEM(lst, i, self->r[i]);
    }
    return lst;
}

/* gpr[] access */
static PyObject *CMachine_get_gpr_arr(CMachine *self, void *c) {
    (void)c;
    PyObject *lst = PyList_New(NUM_GPRS);
    for (int i = 0; i < NUM_GPRS; i++) {
        PyList_SET_ITEM(lst, i, PyLong_FromLong(self->gpr[i]));
    }
    return lst;
}

/* dmem access */
static PyObject *CMachine_get_dmem_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->dmem); return self->dmem; }
static int CMachine_set_dmem_prop(CMachine *self, PyObject *value, void *c) {
    (void)c;
    if (!PyList_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "dmem must be a list");
        return -1;
    }
    Py_INCREF(value);
    Py_DECREF(self->dmem);
    self->dmem = value;
    /* Also rebuild init_dmem to match the new size, marking all as True
     * since the caller is providing pre-initialized data. */
    Py_ssize_t sz = PyList_Size(value);
    PyObject *new_init = PyList_New(sz);
    Py_ssize_t i;
    for (i = 0; i < sz; ++i) {
        Py_INCREF(Py_True);
        PyList_SET_ITEM(new_init, i, Py_True);
    }
    Py_DECREF(self->init_dmem);
    self->init_dmem = new_init;
    return 0;
}
static PyObject *CMachine_get_imem_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->imem); return self->imem; }
static PyObject *CMachine_get_init_dmem_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->init_dmem); return self->init_dmem; }

/* Constant class attrs */
static PyObject *CMachine_get_XLEN(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(XLEN); }
static PyObject *CMachine_get_LIMBS(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(LIMBS); }
static PyObject *CMachine_get_NUM_REGS(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(NUM_REGS); }
static PyObject *CMachine_get_NUM_GPRS(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(NUM_GPRS); }
static PyObject *CMachine_get_GPR_WIDTH(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(GPR_WIDTH); }
static PyObject *CMachine_get_DMEM_DEPTH(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(DMEM_DEPTH); }
static PyObject *CMachine_get_IMEM_DEPTH(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(IMEM_DEPTH); }
static PyObject *CMachine_get_I_TYPE_IMM_WIDTH(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(12); }
static PyObject *CMachine_get_LOOP_STACK_SIZE(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(LOOP_STACK_SZ); }
static PyObject *CMachine_get_CALL_STACK_SIZE(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(CALL_STACK_SZ); }
static PyObject *CMachine_get_CSR_FLAG(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(CSR_FLAG); }
static PyObject *CMachine_get_CSR_MOD_BASE(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(CSR_MOD_BASE); }
static PyObject *CMachine_get_CSR_RNG(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(CSR_RNG); }
static PyObject *CMachine_get_WSR_MOD(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(WSR_MOD); }
static PyObject *CMachine_get_WSR_RND(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(WSR_RND); }
static PyObject *CMachine_get_DEFAULT_DUMP_FILENAME(CMachine *self, void *c) { (void)self; (void)c; return PyUnicode_FromString("dmem_dump.hex"); }
static PyObject *CMachine_get_xlen_mask_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->xlen_mask); return self->xlen_mask; }
static PyObject *CMachine_get_limb_mask_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->limb_mask); return self->limb_mask; }
static PyObject *CMachine_get_half_limb_mask_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->half_limb_mask); return self->half_limb_mask; }
static PyObject *CMachine_get_hw_mask_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->hw_mask); return self->hw_mask; }
static PyObject *CMachine_get_qw_mask_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->qw_mask); return self->qw_mask; }
static PyObject *CMachine_get_gpr_mask_prop(CMachine *self, void *c) { (void)c; Py_INCREF(self->gpr_mask); return self->gpr_mask; }
static PyObject *CMachine_get_limb_width_prop(CMachine *self, void *c) { (void)c; return PyLong_FromLong(self->limb_width); }
static PyObject *CMachine_get_half_limb_width_prop(CMachine *self, void *c) { (void)c; return PyLong_FromLong(self->half_limb_width); }
static PyObject *CMachine_get_qw_width_prop(CMachine *self, void *c) { (void)c; return PyLong_FromLong(self->qw_width); }
static PyObject *CMachine_get_hw_width_prop(CMachine *self, void *c) { (void)c; return PyLong_FromLong(self->hw_width); }
static PyObject *CMachine_get_half_xlen_mask(CMachine *self, void *c) { (void)c; Py_INCREF(self->hw_mask); return self->hw_mask; }
static PyObject *CMachine_get_reg_idx_width(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(5); }
static PyObject *CMachine_get_reg_idx_mask(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(31); }
static PyObject *CMachine_get_dmem_idx_width(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(7); }
static PyObject *CMachine_get_dmem_idx_mask(CMachine *self, void *c) { (void)self; (void)c; return PyLong_FromLong(127); }

/* Flag direct properties (for direct .M, .L, .Z, .C, .XM, .XL, .XZ, .XC access) */
static PyObject *CMachine_get_M(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->M); }
static int CMachine_set_M(CMachine *s, PyObject *v, void *c) { (void)c; s->M = PyObject_IsTrue(v); return 0; }
static PyObject *CMachine_get_L(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->L); }
static int CMachine_set_L(CMachine *s, PyObject *v, void *c) { (void)c; s->L = PyObject_IsTrue(v); return 0; }
static PyObject *CMachine_get_Z(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->Z); }
static int CMachine_set_Z(CMachine *s, PyObject *v, void *c) { (void)c; s->Z = PyObject_IsTrue(v); return 0; }
static PyObject *CMachine_get_C_flag(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->C); }
static int CMachine_set_C_flag(CMachine *s, PyObject *v, void *c) { (void)c; s->C = PyObject_IsTrue(v); return 0; }
static PyObject *CMachine_get_XM(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->XM); }
static int CMachine_set_XM(CMachine *s, PyObject *v, void *c) { (void)c; s->XM = PyObject_IsTrue(v); return 0; }
static PyObject *CMachine_get_XL(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->XL); }
static int CMachine_set_XL(CMachine *s, PyObject *v, void *c) { (void)c; s->XL = PyObject_IsTrue(v); return 0; }
static PyObject *CMachine_get_XZ(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->XZ); }
static int CMachine_set_XZ(CMachine *s, PyObject *v, void *c) { (void)c; s->XZ = PyObject_IsTrue(v); return 0; }
static PyObject *CMachine_get_XC(CMachine *s, void *c) { (void)c; return PyBool_FromLong(s->XC); }
static int CMachine_set_XC(CMachine *s, PyObject *v, void *c) { (void)c; s->XC = PyObject_IsTrue(v); return 0; }

/* force_break as tuple property */
static PyObject *CMachine_get_force_break(CMachine *s, void *c) {
    (void)c;
    return Py_BuildValue("(pplpl)",
        s->fb_active, s->fb_consider_callstack, s->fb_callstack,
        s->fb_consider_loopstack, s->fb_loopstack);
}

static int CMachine_set_force_break(CMachine *s, PyObject *v, void *c) {
    (void)c;
    if (!PyTuple_Check(v) || PyTuple_Size(v) != 5) {
        PyErr_SetString(PyExc_TypeError, "force_break must be a 5-tuple");
        return -1;
    }
    s->fb_active = PyObject_IsTrue(PyTuple_GetItem(v, 0));
    s->fb_consider_callstack = PyObject_IsTrue(PyTuple_GetItem(v, 1));
    s->fb_callstack = PyLong_AsLong(PyTuple_GetItem(v, 2));
    s->fb_consider_loopstack = PyObject_IsTrue(PyTuple_GetItem(v, 3));
    s->fb_loopstack = PyLong_AsLong(PyTuple_GetItem(v, 4));
    return 0;
}

/* ------------------------------------------------------------------ */
/* Method table                                                        */
/* ------------------------------------------------------------------ */
static PyMethodDef CMachine_methods[] = {
    {"get_reg", (PyCFunction)CMachine_get_reg, METH_VARARGS, NULL},
    {"set_reg", (PyCFunction)CMachine_set_reg, METH_VARARGS, NULL},
    {"get_reg_limb", (PyCFunction)CMachine_get_reg_limb, METH_VARARGS, NULL},
    {"set_reg_limb", (PyCFunction)CMachine_set_reg_limb, METH_VARARGS, NULL},
    {"set_reg_half_limb", (PyCFunction)CMachine_set_reg_half_limb, METH_VARARGS, NULL},
    {"get_reg_qw", (PyCFunction)CMachine_get_reg_qw, METH_VARARGS, NULL},
    {"set_reg_half_word", (PyCFunction)CMachine_set_reg_half_word, METH_VARARGS, NULL},
    {"get_reg_valid_half_limbs", (PyCFunction)CMachine_get_reg_valid_half_limbs, METH_VARARGS, NULL},
    {"set_gpr", (PyCFunction)CMachine_set_gpr, METH_VARARGS, NULL},
    {"get_gpr", (PyCFunction)CMachine_get_gpr, METH_VARARGS, NULL},
    {"inc_gpr", (PyCFunction)CMachine_inc_gpr, METH_VARARGS, NULL},
    {"inc_gpr_wlen_bytes", (PyCFunction)CMachine_inc_gpr_wlen_bytes, METH_VARARGS, NULL},
    {"get_csr", (PyCFunction)CMachine_get_csr, METH_VARARGS, NULL},
    {"set_csr", (PyCFunction)CMachine_set_csr, METH_VARARGS, NULL},
    {"get_wsr", (PyCFunction)CMachine_get_wsr, METH_VARARGS, NULL},
    {"set_wsr", (PyCFunction)CMachine_set_wsr, METH_VARARGS, NULL},
    {"get_flag", (PyCFunction)CMachine_get_flag, METH_VARARGS, NULL},
    {"set_flag", (PyCFunction)CMachine_set_flag, METH_VARARGS, NULL},
    {"set_c_z_m_l", (PyCFunction)CMachine_set_c_z_m_l, METH_VARARGS, NULL},
    {"setx_c_z_m_l", (PyCFunction)CMachine_setx_c_z_m_l, METH_VARARGS, NULL},
    {"set_z_m_l", (PyCFunction)CMachine_set_z_m_l, METH_VARARGS, NULL},
    {"setx_z_m_l", (PyCFunction)CMachine_setx_z_m_l, METH_VARARGS, NULL},
    {"set_c_m", (PyCFunction)CMachine_set_c_m, METH_VARARGS, NULL},
    {"setx_c_m", (PyCFunction)CMachine_setx_c_m, METH_VARARGS, NULL},
    {"set_l", (PyCFunction)CMachine_set_l, METH_VARARGS, NULL},
    {"setx_l", (PyCFunction)CMachine_setx_l, METH_VARARGS, NULL},
    {"get_flags_as_bin", (PyCFunction)CMachine_get_flags_as_bin, METH_NOARGS, NULL},
    {"set_flags_as_bin", (PyCFunction)CMachine_set_flags_as_bin, METH_VARARGS, NULL},
    {"get_acc", (PyCFunction)CMachine_get_acc, METH_NOARGS, NULL},
    {"set_acc", (PyCFunction)CMachine_set_acc, METH_VARARGS, NULL},
    {"get_pc", (PyCFunction)CMachine_get_pc, METH_NOARGS, NULL},
    {"set_pc", (PyCFunction)CMachine_set_pc, METH_VARARGS, NULL},
    {"inc_pc", (PyCFunction)CMachine_inc_pc, METH_NOARGS, NULL},
    {"get_dmem", (PyCFunction)CMachine_get_dmem, METH_VARARGS, NULL},
    {"set_dmem", (PyCFunction)CMachine_set_dmem, METH_VARARGS, NULL},
    {"get_dmem_otbn", (PyCFunction)CMachine_get_dmem_otbn, METH_VARARGS, NULL},
    {"set_dmem_otbn", (PyCFunction)CMachine_set_dmem_otbn, METH_VARARGS, NULL},
    {"push_loop_stack", (PyCFunction)CMachine_push_loop_stack, METH_VARARGS, NULL},
    {"dec_top_loop_cnt", (PyCFunction)CMachine_dec_top_loop_cnt, METH_NOARGS, NULL},
    {"get_top_loop_end_addr", (PyCFunction)CMachine_get_top_loop_end_addr, METH_NOARGS, NULL},
    {"get_top_loop_start_addr", (PyCFunction)CMachine_get_top_loop_start_addr, METH_NOARGS, NULL},
    {"pop_loop_stack", (PyCFunction)CMachine_pop_loop_stack, METH_NOARGS, NULL},
    {"push_call_stack", (PyCFunction)CMachine_push_call_stack, METH_VARARGS, NULL},
    {"pop_call_stack", (PyCFunction)CMachine_pop_call_stack, METH_NOARGS, NULL},
    {"get_instruction", (PyCFunction)CMachine_get_instruction, METH_VARARGS, NULL},
    {"finish", (PyCFunction)CMachine_finish, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clear_regs", (PyCFunction)CMachine_clear_regs, METH_NOARGS, NULL},
    {"reset", (PyCFunction)CMachine_reset, METH_VARARGS | METH_KEYWORDS, NULL},
    {"step", (PyCFunction)CMachine_step, METH_NOARGS, NULL},
    {"get_limb_hex_str", (PyCFunction)CMachine_get_limb_hex_str, METH_VARARGS, NULL},
    {"get_xlen_hex_str", (PyCFunction)CMachine_get_xlen_hex_str, METH_VARARGS, NULL},
    {"get_full_dmem", (PyCFunction)CMachine_get_full_dmem, METH_NOARGS, NULL},
    {"stat_record_instr", (PyCFunction)CMachine_stat_record_instr, METH_VARARGS, NULL},
    {"get_breakpoints", (PyCFunction)CMachine_get_breakpoints, METH_NOARGS, NULL},
    {"toggle_breakpoint", (PyCFunction)CMachine_toggle_breakpoint, METH_VARARGS, NULL},
    {"set_breakpoint", (PyCFunction)CMachine_set_breakpoint, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

/* ------------------------------------------------------------------ */
/* Getset table                                                        */
/* ------------------------------------------------------------------ */
static PyGetSetDef CMachine_getset[] = {
    {"finishFlag", (getter)CMachine_get_finishFlag, (setter)CMachine_set_finishFlag, NULL, NULL},
    {"loop_stack", (getter)CMachine_get_loop_stack_py, NULL, NULL, NULL},
    {"call_stack", (getter)CMachine_get_call_stack_py, NULL, NULL, NULL},
    {"ctx", (getter)CMachine_get_ctx, (setter)CMachine_set_ctx, NULL, NULL},
    {"stats", (getter)CMachine_get_stats_py, (setter)CMachine_set_stats_py, NULL, NULL},
    {"pc", (getter)CMachine_get_pc_prop, (setter)CMachine_set_pc_prop, NULL, NULL},
    {"stop_addr", (getter)CMachine_get_stop_addr_prop, (setter)CMachine_set_stop_addr_prop, NULL, NULL},
    {"mod", (getter)CMachine_get_mod, (setter)CMachine_set_mod, NULL, NULL},
    {"dmp", (getter)CMachine_get_dmp_prop, (setter)CMachine_set_dmp_prop, NULL, NULL},
    {"rfp", (getter)CMachine_get_rfp_prop, (setter)CMachine_set_rfp_prop, NULL, NULL},
    {"lc", (getter)CMachine_get_lc_prop, (setter)CMachine_set_lc_prop, NULL, NULL},
    {"rnd", (getter)CMachine_get_rnd_prop, (setter)CMachine_set_rnd_prop, NULL, NULL},
    {"acc", (getter)CMachine_get_acc_prop, (setter)CMachine_set_acc_prop, NULL, NULL},
    {"r", (getter)CMachine_get_r, NULL, NULL, NULL},
    {"gpr", (getter)CMachine_get_gpr_arr, NULL, NULL, NULL},
    {"dmem", (getter)CMachine_get_dmem_prop, (setter)CMachine_set_dmem_prop, NULL, NULL},
    {"imem", (getter)CMachine_get_imem_prop, NULL, NULL, NULL},
    {"init_dmem", (getter)CMachine_get_init_dmem_prop, NULL, NULL, NULL},
    {"breakpoints", (getter)CMachine_get_breakpoints, NULL, NULL, NULL},
    /* Constants */
    {"XLEN", (getter)CMachine_get_XLEN, NULL, NULL, NULL},
    {"LIMBS", (getter)CMachine_get_LIMBS, NULL, NULL, NULL},
    {"NUM_REGS", (getter)CMachine_get_NUM_REGS, NULL, NULL, NULL},
    {"NUM_GPRS", (getter)CMachine_get_NUM_GPRS, NULL, NULL, NULL},
    {"GPR_WIDTH", (getter)CMachine_get_GPR_WIDTH, NULL, NULL, NULL},
    {"DMEM_DEPTH", (getter)CMachine_get_DMEM_DEPTH, NULL, NULL, NULL},
    {"IMEM_DEPTH", (getter)CMachine_get_IMEM_DEPTH, NULL, NULL, NULL},
    {"I_TYPE_IMM_WIDTH", (getter)CMachine_get_I_TYPE_IMM_WIDTH, NULL, NULL, NULL},
    {"LOOP_STACK_SIZE", (getter)CMachine_get_LOOP_STACK_SIZE, NULL, NULL, NULL},
    {"CALL_STACK_SIZE", (getter)CMachine_get_CALL_STACK_SIZE, NULL, NULL, NULL},
    {"CSR_FLAG", (getter)CMachine_get_CSR_FLAG, NULL, NULL, NULL},
    {"CSR_MOD_BASE", (getter)CMachine_get_CSR_MOD_BASE, NULL, NULL, NULL},
    {"CSR_RNG", (getter)CMachine_get_CSR_RNG, NULL, NULL, NULL},
    {"WSR_MOD", (getter)CMachine_get_WSR_MOD, NULL, NULL, NULL},
    {"WSR_RND", (getter)CMachine_get_WSR_RND, NULL, NULL, NULL},
    {"DEFAULT_DUMP_FILENAME", (getter)CMachine_get_DEFAULT_DUMP_FILENAME, NULL, NULL, NULL},
    {"xlen_mask", (getter)CMachine_get_xlen_mask_prop, NULL, NULL, NULL},
    {"limb_mask", (getter)CMachine_get_limb_mask_prop, NULL, NULL, NULL},
    {"half_limb_mask", (getter)CMachine_get_half_limb_mask_prop, NULL, NULL, NULL},
    {"hw_mask", (getter)CMachine_get_hw_mask_prop, NULL, NULL, NULL},
    {"qw_mask", (getter)CMachine_get_qw_mask_prop, NULL, NULL, NULL},
    {"gpr_mask", (getter)CMachine_get_gpr_mask_prop, NULL, NULL, NULL},
    {"limb_width", (getter)CMachine_get_limb_width_prop, NULL, NULL, NULL},
    {"half_limb_width", (getter)CMachine_get_half_limb_width_prop, NULL, NULL, NULL},
    {"qw_width", (getter)CMachine_get_qw_width_prop, NULL, NULL, NULL},
    {"hw_width", (getter)CMachine_get_hw_width_prop, NULL, NULL, NULL},
    {"half_xlen_mask", (getter)CMachine_get_half_xlen_mask, NULL, NULL, NULL},
    {"reg_idx_width", (getter)CMachine_get_reg_idx_width, NULL, NULL, NULL},
    {"reg_idx_mask", (getter)CMachine_get_reg_idx_mask, NULL, NULL, NULL},
    {"dmem_idx_width", (getter)CMachine_get_dmem_idx_width, NULL, NULL, NULL},
    {"dmem_idx_mask", (getter)CMachine_get_dmem_idx_mask, NULL, NULL, NULL},
    /* Flags as direct properties */
    {"M", (getter)CMachine_get_M, (setter)CMachine_set_M, NULL, NULL},
    {"L", (getter)CMachine_get_L, (setter)CMachine_set_L, NULL, NULL},
    {"Z", (getter)CMachine_get_Z, (setter)CMachine_set_Z, NULL, NULL},
    {"C", (getter)CMachine_get_C_flag, (setter)CMachine_set_C_flag, NULL, NULL},
    {"XM", (getter)CMachine_get_XM, (setter)CMachine_set_XM, NULL, NULL},
    {"XL", (getter)CMachine_get_XL, (setter)CMachine_set_XL, NULL, NULL},
    {"XZ", (getter)CMachine_get_XZ, (setter)CMachine_set_XZ, NULL, NULL},
    {"XC", (getter)CMachine_get_XC, (setter)CMachine_set_XC, NULL, NULL},
    {"force_break", (getter)CMachine_get_force_break, (setter)CMachine_set_force_break, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/* ------------------------------------------------------------------ */
/* Type definition                                                     */
/* ------------------------------------------------------------------ */
static PyTypeObject CMachineType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_machine.CMachine",
    .tp_doc = "C implementation of the Machine class for ot_dsim.",
    .tp_basicsize = sizeof(CMachine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)CMachine_init,
    .tp_dealloc = (destructor)CMachine_dealloc,
    .tp_methods = CMachine_methods,
    .tp_getset = CMachine_getset,
};

/* ------------------------------------------------------------------ */
/* Module definition                                                   */
/* ------------------------------------------------------------------ */

/* CallStackUnderrun is forward-declared near the top of the file */

static struct PyModuleDef machinemodule = {
    PyModuleDef_HEAD_INIT,
    "_machine",
    "C Machine core for ot_dsim.",
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit__machine(void) {
    PyObject *m = PyModule_Create(&machinemodule);
    if (!m) return NULL;

    if (PyModule_AddIntConstant(m, "ABI_VERSION", OT_DSIM_MACHINE_ABI_VERSION) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyType_Ready(&CMachineType) < 0)
        return NULL;

    Py_INCREF(&CMachineType);
    if (PyModule_AddObject(m, "CMachine", (PyObject *)&CMachineType) < 0) {
        Py_DECREF(&CMachineType);
        Py_DECREF(m);
        return NULL;
    }

    /* Create CallStackUnderrun as subclass of OverflowError */
    CallStackUnderrun = PyErr_NewException("_machine.CallStackUnderrun", PyExc_OverflowError, NULL);
    Py_XINCREF(CallStackUnderrun);
    if (PyModule_AddObject(m, "CallStackUnderrun", CallStackUnderrun) < 0) {
        Py_XDECREF(CallStackUnderrun);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
