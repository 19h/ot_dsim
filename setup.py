import os
import sys
from setuptools import setup, find_packages, Extension

# ---------------------------------------------------------------------------
# C extension modules
# ---------------------------------------------------------------------------

# Common compiler flags – portable, no intrinsics.
extra_compile_args = []
if sys.platform != "win32":
    extra_compile_args = [
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
    ]

# 1. _cops: 256-bit arithmetic helpers (existing)
_cops_ext = Extension(
    "ot_dsim._cops",
    sources=["csrc/ot_dsim_cops.c"],
    extra_compile_args=extra_compile_args,
)

# 2. _machine: C-accelerated Machine core (new)
_machine_ext = Extension(
    "ot_dsim._machine",
    sources=["csrc/ot_dsim_machine.c"],
    extra_compile_args=extra_compile_args,
)

# Allow building without the C extensions (pure-Python fallback) by setting
# the environment variable OT_DSIM_PURE_PYTHON=1
_extensions = []
if not os.environ.get("OT_DSIM_PURE_PYTHON"):
    _extensions = [_cops_ext, _machine_ext]

setup(
    name="ot_dsim",
    version="0.3.0",
    description="Simulator for bignum cryptographic accelerator coprocessor (dcrypto / OTBN)",
    packages=["ot_dsim", "ot_dsim.bignum_lib"],
    package_dir={"ot_dsim": "."},
    ext_modules=_extensions,
    install_requires=[
        "pycryptodome",
        "tabulate",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
