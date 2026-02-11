#!/usr/bin/env python3
"""Benchmark C Machine vs pure-Python Machine on an RSA workload."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORKLOAD = """
from ot_dsim.bignum_lib.machine import _USE_C_MACHINE
import sim_rsa_tests as t

t.ENABLE_TRACE_DUMP = False
t.init_dmem()
t.load_program_otbn_asm()
t.breakpoints = {}

msg = t.get_msg_val("ot_dsim benchmark message")
enc = t.rsa_encrypt(t.RSA_N[768], 3, msg)
dec = t.rsa_decrypt(t.RSA_N[768], 3, t.RSA_D[768], enc)

if dec != msg:
    raise RuntimeError("RSA round-trip failed")

print(f"backend={'c' if _USE_C_MACHINE else 'py'}")
print(f"inst={t.inst_cnt} cycles={t.cycle_cnt}")
""".strip()


def _run_once(python_exe: str, repo_root: Path, force_pure_python: bool):
    env = os.environ.copy()
    if force_pure_python:
        env["OT_DSIM_PURE_PYTHON"] = "1"
    else:
        env.pop("OT_DSIM_PURE_PYTHON", None)

    start = time.perf_counter()
    proc = subprocess.run(
        [python_exe, "-c", WORKLOAD],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        raise RuntimeError(
            f"workload run failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    backend = None
    inst = None
    cycles = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("backend="):
            backend = line.split("=", 1)[1]
        if line.startswith("inst=") and "cycles=" in line:
            parts = line.split()
            inst = int(parts[0].split("=", 1)[1])
            cycles = int(parts[1].split("=", 1)[1])

    if backend is None or inst is None or cycles is None:
        raise RuntimeError(f"unexpected workload output:\n{proc.stdout}\n{proc.stderr}")

    return elapsed, backend, inst, cycles


def _summary(values):
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _run_backend(label, python_exe, repo_root, runs, force_pure_python):
    durations = []
    backends = set()
    inst_cycles = set()

    for _ in range(runs):
        elapsed, backend, inst, cycles = _run_once(
            python_exe, repo_root, force_pure_python
        )
        durations.append(elapsed)
        backends.add(backend)
        inst_cycles.add((inst, cycles))

    return {
        "label": label,
        "durations": durations,
        "summary": _summary(durations),
        "backends": backends,
        "inst_cycles": inst_cycles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="timed runs per backend")
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs per backend")
    parser.add_argument(
        "--python", default=sys.executable, help="python executable to use"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    for _ in range(args.warmup):
        _run_once(args.python, repo_root, force_pure_python=False)
        _run_once(args.python, repo_root, force_pure_python=True)

    c_result = _run_backend(
        "c-machine",
        args.python,
        repo_root,
        args.runs,
        force_pure_python=False,
    )
    py_result = _run_backend(
        "pure-python",
        args.python,
        repo_root,
        args.runs,
        force_pure_python=True,
    )

    print(f"Runs per backend: {args.runs}")
    print(f"Warmup runs: {args.warmup}")
    print()

    for result in (c_result, py_result):
        s = result["summary"]
        print(
            f"{result['label']:<12} mean={s['mean']:.3f}s "
            f"min={s['min']:.3f}s max={s['max']:.3f}s stdev={s['stdev']:.3f}s"
        )
        print(f"  backend markers: {sorted(result['backends'])}")
        print(f"  inst/cycles: {sorted(result['inst_cycles'])}")

    speedup = py_result["summary"]["mean"] / c_result["summary"]["mean"]
    print()
    print(f"Speedup (pure-python / c-machine): {speedup:.2f}x")


if __name__ == "__main__":
    main()
