#!/usr/bin/env python3
"""
NumPy mirror of ScalarStepper::advance, for debugging.

Same tableau, same shifted coefficients, same truncated separable IF kernel
(exp(-2a) I_n(2a), n <= n_IF), same central-advection stencil, same register
order. Run it on an IF_HERK_Validate run directory and it replays every level
from phi0.npy and compares with the C++ phi(T):

  * C++ vs mirror at round-off  -> the C++ implements the intended algorithm
  * mirror vs reference.py      -> the algorithm itself converges as expected

so a failed convergence test can be pinned on the implementation or on the
formulation. Pure Python loops over stages; use small or quasi-1D grids.

Usage:  ifrk_prototype.py RUN_DIR [--levels N]
"""
import argparse
import json
import os

import numpy as np
from scipy.special import ive

R3 = np.sqrt(3.0)
A = np.array([[0.0, 0.0, 0.0],
              [0.5, 0.0, 0.0],
              [R3 / 3.0, (3.0 - R3) / 3.0, 0.0]])
B = np.array([(3.0 + R3) / 6.0, -R3 / 3.0, (3.0 + R3) / 6.0])
C = np.array([0.0, 0.5, 1.0])


def shifted_tableau():
    """1-based a~, c~, gaps as in IFHERK::setupRKCoeffs (index 0 unused)."""
    s = len(B)
    at = np.zeros((s + 1, s + 1))
    ct = np.zeros(s + 1)
    for i in range(1, s):
        ct[i] = C[i]
        at[i, 1:] = A[i, :]
    ct[s] = 1.0
    at[s, 1:] = B
    gap = np.zeros(s + 1)
    gap[1:] = np.diff(np.concatenate(([0.0], ct[1:])))
    return s, at, ct, gap


def apply_if(f, alpha, n_if):
    if alpha <= 0.0:
        return f
    h = ive(np.arange(n_if + 1), 2.0 * alpha)
    for ax in range(f.ndim):
        g = h[0] * f
        for m in range(1, n_if + 1):
            g = g + h[m] * (np.roll(f, -m, axis=ax) + np.roll(f, m, axis=ax))
        f = g
    return f


def N_linear(phi, c, dx):
    out = np.zeros_like(phi)
    for ax in range(phi.ndim):
        out += c[ax] * (np.roll(phi, -1, axis=ax) - np.roll(phi, 1, axis=ax)) / (2.0 * dx[ax])
    return out


def advance(phi, dt, nu, c, dx, n_if):
    s, at, _, gap = shifted_tableau()
    alpha = gap * dt * nu / dx[0] ** 2
    w = [None] * (s + 1)
    q = phi.copy()
    stage = phi.copy()
    for i in range(1, s + 1):
        g = -at[i, i] * dt * N_linear(stage, c, dx)
        if i > 1:
            q = apply_if(q, alpha[i - 1], n_if)
            for j in range(1, i):
                w[j] = apply_if(w[j], alpha[i - 1], n_if)
        r = q + g
        for j in range(1, i):
            r = r + dt * at[i, j] * w[j]
        stage = apply_if(r, alpha[i], n_if)
        w[i] = g / (at[i, i] * dt)
    return stage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--levels", type=int, default=None, help="replay only the first N levels")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "run_info.json")) as f:
        info = json.load(f)
    phi0 = np.load(os.path.join(args.run_dir, info["phi0"]))
    dx, c, nu, n_if = info["dx"], info["c"], info["nu"], info["n_IF"]

    runs = info["runs"][: args.levels] if args.levels else info["runs"]
    print(f"{'dt':>10} {'max|C++ - mirror|':>18}")
    for run in runs:
        phi = phi0.copy()
        for _ in range(run["nsteps"]):
            phi = advance(phi, run["dt"], nu, c, dx, n_if)
        cpp = np.load(os.path.join(args.run_dir, run["file"]))
        print(f"{run['dt']:10.3e} {np.max(np.abs(cpp - phi)):18.3e}")


if __name__ == "__main__":
    main()
