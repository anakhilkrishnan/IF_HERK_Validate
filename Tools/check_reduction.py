#!/usr/bin/env python3
"""
Exact reductions of vector Burgers to scalar Burgers, checked to round-off.

Run the paired inputs first (same grid, nu, T and dt sweep), then

    check_reduction.py VECTOR_RUN_DIR SCALAR_RUN_DIR

The mode follows the vector run's ic.type:

  quasi1d   U = [u(x), 0, 0]. The vorticity vanishes and the rotational-form
            kernel reduces exactly to the scalar stencil sum (u+^2 - u-^2)/4dx.
            The scalar run must sample at the same x-face points
            (ic.stagger = 1 0 0, c = 1 0 0). Checks: u == phi, v == w == 0.

  diagonal  U = g(i+j+k)[1 1 1]. x-, y- and z-faces with the same (i,j,k) lie
            on the same diagonal, the discrete vorticity vanishes identically,
            and every component must equal a scalar diagonal run (c = 1 1 1).
            Exercises all three face types and every stencil direction.

Both checks are exact identities of the discretisation, so the differences
must be at round-off, many orders below the temporal error. They are checked
at t = 0 (sampling) and after every level of the dt sweep.
"""
import sys

import numpy as np

import operators as ops

TOL_REL = 1e-12   # relative to max|phi|; generous for accumulated round-off


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    vinfo, vinit = ops.load_run(sys.argv[1])
    sinfo, sinit = ops.load_run(sys.argv[2])

    if vinfo["problem"] != "vector_burgers" or sinfo["problem"] != "scalar_burgers":
        sys.exit("check_reduction.py: expected a vector_burgers run and a scalar_burgers run")
    mode = vinfo["ic"]["type"]
    if mode not in ("quasi1d", "diagonal"):
        sys.exit(f"check_reduction.py: vector ic.type is '{mode}'; need quasi1d or diagonal")

    for key in ("n_cell", "nu", "T"):
        if vinfo[key] != sinfo[key]:
            sys.exit(f"check_reduction.py: runs differ in {key}: {vinfo[key]} vs {sinfo[key]}")
    vdts = [r["dt"] for r in vinfo["runs"]]
    sdts = [r["dt"] for r in sinfo["runs"]]
    if vdts != sdts:
        sys.exit("check_reduction.py: the two runs use different dt sweeps")

    if mode == "quasi1d":
        expect_c, expect_stagger, expect_type = [1.0] + [0.0] * (vinfo["spacedim"] - 1), \
                                                [1] + [0] * (vinfo["spacedim"] - 1), "modes"
    else:
        expect_c, expect_stagger, expect_type = [1.0] * vinfo["spacedim"], [0] * vinfo["spacedim"], "diagonal"
    if sinfo["c"] != expect_c or sinfo["ic"]["stagger"] != expect_stagger or sinfo["ic"]["type"] != expect_type:
        sys.exit(f"check_reduction.py: the scalar run must use ic.type = {expect_type}, "
                 f"prob.c = {expect_c}, ic.stagger = {expect_stagger}")

    n_cell = vinfo["n_cell"]
    fields = vinfo["fields"]

    def compare(vfld, sfld, label):
        phi = sfld["phi"]
        scale = max(1.0, np.max(np.abs(phi)))
        worst = 0.0
        parts = []
        for d, name in enumerate(fields):
            comp = ops.drop_duplicate_faces(vfld[name], n_cell)
            if mode == "quasi1d" and d > 0:
                diff = np.max(np.abs(comp))          # must stay exactly zero
                parts.append(f"max|{name}| = {diff:.1e}")
            else:
                diff = np.max(np.abs(comp - phi))
                parts.append(f"|{name} - phi| = {diff:.1e}")
            worst = max(worst, diff / scale)
        ok = worst <= TOL_REL
        print(f"  {label:<12} {'PASS' if ok else 'FAIL'}   " + ",  ".join(parts))
        return ok

    print(f"{mode} reduction: {sys.argv[1]}  vs  {sys.argv[2]}   (tolerance {TOL_REL:.0e} x max|phi|)")
    ok = compare(vinit, sinit, "t = 0")
    for vr, sr in zip(vinfo["runs"], sinfo["runs"]):
        ok &= compare(ops.load_fields(sys.argv[1], vr), ops.load_fields(sys.argv[2], sr),
                      f"dt = {vr['dt']:.2e}")
    print("\nAll levels agree to round-off." if ok else
          "\nMISMATCH: the vector kernel does not reduce to the scalar stencil (or the runs differ).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
