#!/usr/bin/env python3
"""
Errors and observed orders for an IF_HERK_Validate run directory.

Case 1, scalar linear convection-diffusion (c = 0: pure diffusion).
The reference is the exact solution of the SEMI-DISCRETE system,

    dphi/dt = L phi,   L = -c . D0 + nu * Lap_h,

obtained mode by mode in Fourier space from the sampled phi0.npy the run
wrote, with the symbols of the stencils actually used:

    central advection   D0_d  ->  i sin(k_d dx_d) / dx_d
    7-point Laplacian   Lap_h ->  -sum_d 4 sin^2(k_d dx_d / 2) / dx_d^2

Because the spatial operator is identical, every bit of the error is
temporal (plus IF truncation, bounded by IF_eps per application).

Usage:  reference.py RUN_DIR [--plot]
"""
import argparse
import json
import os
import sys

import numpy as np


def semi_discrete_symbol(shape, dx, c, nu):
    """lambda(k) on the FFT grid of an array with the given shape."""
    lam = np.zeros(shape, dtype=complex)
    for d, (n, h) in enumerate(zip(shape, dx)):
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
        kd = k * h
        term = -1j * c[d] * np.sin(kd) / h - nu * 4.0 * np.sin(0.5 * kd) ** 2 / h**2
        bshape = [1] * len(shape)
        bshape[d] = n
        lam = lam + term.reshape(bshape)
    return lam


def reference_solution(phi0, dx, c, nu, T):
    lam = semi_discrete_symbol(phi0.shape, dx, c, nu)
    return np.real(np.fft.ifftn(np.fft.fftn(phi0) * np.exp(lam * T)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--plot", action="store_true", help="log-log error vs dt (needs matplotlib)")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "run_info.json")) as f:
        info = json.load(f)

    if info["problem"] != "linear_advection":
        sys.exit(f"reference.py: no reference for problem '{info['problem']}' yet")

    phi0 = np.load(os.path.join(args.run_dir, info["phi0"]))
    dx, c, nu, T = info["dx"], info["c"], info["nu"], info["T"]
    ref = reference_solution(phi0, dx, c, nu, T)
    scale = np.max(np.abs(ref))

    pure_diffusion = all(abs(cd) == 0.0 for cd in c)
    print(f"{info['spacedim']}D, n_cell = {info['n_cell']}, c = {c}, nu = {nu}, T = {T}, "
          f"n_IF = {info['n_IF']}, IF_eps = {info['IF_eps']:.1e}")
    print(f"max|ref| = {scale:.4e}\n")
    print(f"{'dt':>10} {'CFL':>7} {'steps':>6} {'Linf err':>11} {'order':>6} {'L2 err':>11} {'order':>6}")

    dts, einf, el2 = [], [], []
    for run in info["runs"]:
        phi = np.load(os.path.join(args.run_dir, run["file"]))
        diff = phi - ref
        dts.append(run["dt"])
        einf.append(np.max(np.abs(diff)))
        el2.append(np.sqrt(np.mean(diff**2)))

        def order(e):
            if pure_diffusion or len(e) < 2 or e[-1] == 0.0 or e[-2] == 0.0:
                return "     -"
            return f"{np.log(e[-2] / e[-1]) / np.log(dts[-2] / dts[-1]):6.2f}"

        print(f"{run['dt']:10.3e} {run['cfl']:7.3f} {run['nsteps']:6d} "
              f"{einf[-1]:11.3e} {order(einf)} {el2[-1]:11.3e} {order(el2)}")

    print()
    if pure_diffusion:
        print("Pure diffusion: the error should be independent of dt, at the "
              "round-off / IF-truncation level.")
        print(f"  max Linf error = {max(einf):.3e} (relative {max(einf) / scale:.3e})")
    else:
        if len(dts) >= 2:
            p = np.polyfit(np.log(dts), np.log(einf), 1)[0]
            print(f"Least-squares order (Linf, all levels): {p:.2f}   (expected: 3 for linear advection)")
        if any(r["cfl"] > np.sqrt(3.0) for r in info["runs"]):
            print("Note: levels with CFL > sqrt(3) are beyond the linear stability limit.")

    seed = info.get("ic", {}).get("seed_amp", 0.0)
    if seed != 0.0:
        # amplitude of the (k,k,k), k dx = pi/2 mode the stability seed excites;
        # the semi-discrete system (nu = 0) keeps it at exactly 1
        if all(n % 4 == 0 for n in phi0.shape):
            m = tuple(n // 4 for n in phi0.shape)
            a0 = abs(np.fft.fftn(phi0)[m])
            print("\nStability seed, (k,k,k) mode with k dx = pi/2:")
            for run in info["runs"]:
                aT = abs(np.fft.fftn(np.load(os.path.join(args.run_dir, run["file"])))[m])
                verdict = "GROWS" if aT > 1.01 * a0 else "bounded"
                print(f"  CFL {run['cfl']:.3f}: |a(T)| / |a(0)| = {aT / a0:.3e}   ({verdict})")
        else:
            print("\nStability seed: n_cell must be divisible by 4 to measure the seeded mode")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.loglog(dts, einf, "o-", label="Linf")
        plt.loglog(dts, el2, "s-", label="L2")
        if not pure_diffusion:
            d = np.array(dts)
            plt.loglog(d, einf[-1] * (d / d[-1]) ** 3, "k--", label="slope 3")
        plt.xlabel(r"$\Delta t$"); plt.ylabel("Error"); plt.legend(); plt.grid(True, which="both")
        out = os.path.join(args.run_dir, "convergence.png")
        plt.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
