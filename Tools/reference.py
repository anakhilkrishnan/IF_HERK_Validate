#!/usr/bin/env python3
"""
Errors and observed orders for an IF_HERK_Validate run directory.

The reference depends on the problem (read from run_info.json):

  linear_advection   Exact solution of the semi-discrete system, mode by mode
                     in Fourier space, from the sampled initial field and the
                     symbols of the stencils actually used:
                       central advection  -> i sum_d c_d sin(k_d dx_d) / dx_d
                       Laplacian          -> -sum_d 4 sin^2(k_d dx_d / 2) / dx_d^2
                     Expected: order 3 (c = 0: error independent of dt).

  scalar_burgers     The same semi-discrete system (operators.py) integrated
                     by scipy's DOP853 at tight tolerance -- an integrator
                     completely independent of the C++. Expected: order 2.

  vector_burgers     No independent reference exists in 3D, so the run is
                     compared with its own reference run at dt_min / 2^m
                     (time.ref_refine = m). Expected: order 2. The exact
                     checks for this case are in check_reduction.py.

In all cases the spatial operator is the same as the code's, so every bit of
the measured error is temporal.

Usage:  reference.py RUN_DIR [--plot] [--save-ref]
"""
import argparse
import os
import sys

import numpy as np

import operators as ops

EXPECTED = {"linear_advection": 3, "scalar_burgers": 2, "vector_burgers": 2}


def fourier_reference(phi0, dx, c, nu, T):
    lam = np.zeros(phi0.shape, dtype=complex)
    for d, (n, h) in enumerate(zip(phi0.shape, dx)):
        kd = 2.0 * np.pi * np.fft.fftfreq(n, d=h) * h
        term = -1j * c[d] * np.sin(kd) / h - nu * 4.0 * np.sin(0.5 * kd) ** 2 / h**2
        bshape = [1] * phi0.ndim
        bshape[d] = n
        lam = lam + term.reshape(bshape)
    return np.real(np.fft.ifftn(np.fft.fftn(phi0) * np.exp(lam * T)))


def ode_reference(phi0, dx, c, nu, T, N):
    from scipy.integrate import solve_ivp
    shape = phi0.shape

    def rhs(_t, y):
        p = y.reshape(shape)
        return (-N(p, c, dx) + nu * ops.laplacian(p, dx)).ravel()

    sol = solve_ivp(rhs, (0.0, T), phi0.ravel(), method="DOP853", rtol=1e-13, atol=1e-13)
    if not sol.success:
        sys.exit(f"reference.py: solve_ivp failed: {sol.message}")
    print(f"(reference: DOP853, {sol.nfev} right-hand-side evaluations)")
    return sol.y[:, -1].reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--plot", action="store_true", help="log-log error vs dt (needs matplotlib)")
    ap.add_argument("--save-ref", action="store_true", help="save the reference as ref_<field>.npy")
    args = ap.parse_args()

    info, init = ops.load_run(args.run_dir)
    problem = info["problem"]
    dx, c, nu, T, n_cell = info["dx"], info["c"], info["nu"], info["T"], info["n_cell"]

    # ---- reference ------------------------------------------------------------
    if problem == "linear_advection":
        ref = {"phi": fourier_reference(init["phi"], dx, c, nu, T)}
    elif problem == "scalar_burgers":
        ref = {"phi": ode_reference(init["phi"], dx, c, nu, T, ops.N_scalar_burgers)}
    elif problem == "vector_burgers":
        if info.get("reference_run") is None:
            sys.exit("reference.py: vector_burgers needs a reference run; set time.ref_refine "
                     "(e.g. 4 for dt_min/16)")
        ref = ops.load_fields(args.run_dir, info["reference_run"])
        print(f"(reference: own run at dt = {info['reference_run']['dt']:.4e}; "
              f"self-convergence, not independent)")
    else:
        sys.exit(f"reference.py: unknown problem '{problem}'")

    # face-centred fields: count every periodic point once
    ref = {k: ops.drop_duplicate_faces(v, n_cell) for k, v in ref.items()}
    if args.save_ref:
        for k, v in ref.items():
            np.save(os.path.join(args.run_dir, f"ref_{k}.npy"), v)

    scale = max(np.max(np.abs(v)) for v in ref.values())
    pure_diffusion = problem == "linear_advection" and all(cd == 0.0 for cd in c)

    print(f"{problem}, {info['spacedim']}D, n_cell = {n_cell}, nu = {nu}, T = {T}, "
          f"n_IF = {info['n_IF']}, cell Re = {info.get('cell_re', float('nan')):.2f}")
    print(f"fields = {info['fields']}, max|ref| = {scale:.4e}\n")
    print(f"{'dt':>10} {'CFL':>7} {'steps':>6} {'Linf err':>11} {'order':>6} {'L2 err':>11} {'order':>6}")

    dts, einf, el2 = [], [], []
    for run in info["runs"]:
        fld = {k: ops.drop_duplicate_faces(v, n_cell) for k, v in ops.load_fields(args.run_dir, run).items()}
        # max over fields; L2 = rms over all points of all fields
        einf.append(max(np.max(np.abs(fld[k] - ref[k])) for k in ref))
        el2.append(np.sqrt(np.mean(np.concatenate([((fld[k] - ref[k]) ** 2).ravel() for k in ref]))))
        dts.append(run["dt"])

        def order(e):
            if pure_diffusion or len(e) < 2 or e[-1] == 0.0 or e[-2] == 0.0:
                return "     -"
            return f"{np.log(e[-2] / e[-1]) / np.log(dts[-2] / dts[-1]):6.2f}"

        print(f"{run['dt']:10.3e} {run['cfl']:7.3f} {run['nsteps']:6d} "
              f"{einf[-1]:11.3e} {order(einf)} {el2[-1]:11.3e} {order(el2)}")

    print()
    if pure_diffusion:
        print("Pure diffusion: the error should sit at the round-off / IF-truncation level for")
        print("every dt (IF truncation at the coarsest dt, accumulated round-off at the finest).")
        print(f"  max Linf error = {max(einf):.3e} (relative {max(einf) / scale:.3e})")
    elif len(dts) >= 2:
        p = np.polyfit(np.log(dts), np.log(einf), 1)[0]
        print(f"Least-squares order (Linf, all levels): {p:.2f}   (expected: {EXPECTED[problem]})")
        if any(r["cfl"] > np.sqrt(3.0) for r in info["runs"]):
            print("Note: levels with CFL > sqrt(3) are beyond the inviscid linear stability limit "
                  "(the IF damping can still keep them stable).")

    seed = info.get("ic", {}).get("seed_amp", 0.0)
    if seed != 0.0 and "phi" in init:
        # amplitude of the (k,k,k), k dx = pi/2 mode the stability seed excites;
        # the semi-discrete system (nu = 0) keeps it at exactly 1
        phi0 = init["phi"]
        if all(n % 4 == 0 for n in phi0.shape):
            m = tuple(n // 4 for n in phi0.shape)
            a0 = abs(np.fft.fftn(phi0)[m])
            print("\nStability seed, (k,k,k) mode with k dx = pi/2:")
            for run in info["runs"]:
                aT = abs(np.fft.fftn(ops.load_fields(args.run_dir, run)["phi"])[m])
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
            q = EXPECTED[problem]
            plt.loglog(d, einf[-1] * (d / d[-1]) ** q, "k--", label=f"slope {q}")
        plt.xlabel("dt"); plt.ylabel("error"); plt.legend(); plt.grid(True, which="both")
        out = os.path.join(args.run_dir, "convergence.png")
        plt.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
