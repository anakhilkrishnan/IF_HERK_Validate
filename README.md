# IF_HERK_Validate

Advection-diffusion test bench for the IF-HERK time integrator of LGF_NSE_GPU
(Liska & Colonius 2016). The integrator core is problem-agnostic and is meant
to be linked back into LGF_NSE_GPU unchanged.

Implemented so far: **Case 1**, scalar linear convection-diffusion (and its
pure-diffusion sub-case), cell-centred, periodic.

## Layout

```
IF_HERK_Validate/
├── CMakeLists.txt
├── Src/
│   ├── IFHERK_core/            # portable: knows nothing about the problem
│   │   ├── RKCoefficients.H    # from NSE_core (one comment changed)
│   │   ├── IFHERK.H            # shifted tableau, IF tables, applyIF
│   │   └── ifherk.cpp
│   ├── ADV_core/               # problem-specific
│   │   ├── ProblemConfig.H     # ParmParse inputs (cf. SolverConfig.H)
│   │   ├── ScalarOperators.H   # device kernels for N(phi) (cf. SpatialDiscretization.H)
│   │   ├── InitialCondition.H  # device-callable phi0 (cf. InitialVorticity.H)
│   │   ├── ScalarStepper.H     # unconstrained stage loop (cf. ProjectionWorkspace)
│   │   ├── scalarStepper.cpp
│   │   └── FieldIO.H           # gather-and-write .npy for the Python tools
│   └── ADV_run/
│       └── main.cpp            # dt sweep
├── Inputs/
│   ├── inputs_advection          # order test, 3D
│   ├── inputs_diffusion          # c = 0, dt-independence test
│   ├── inputs_advection_quasi1d  # thin domain in the 3D build
│   └── inputs_stability          # sqrt(3) limit
└── Tools/
    ├── reference.py            # semi-discrete Fourier reference, errors, orders
    └── ifrk_prototype.py       # NumPy mirror of ScalarStepper::advance
```

## Build

```bash
# against an installed AMReX
cmake -S . -B build -DIFV_DIM=3 -DAMReX_ROOT=/path/to/amrex/install
# or fetch AMReX in-tree
cmake -S . -B build -DIFV_DIM=3 -DIFV_FETCH_AMREX=ON [-DAMReX_GPU_BACKEND=CUDA]
cmake --build build -j
```

`IFV_DIM` selects a 1D, 2D or 3D build (separate build directories). The code
uses the `AMREX_D_DECL`/`AMREX_D_TERM` style throughout, so all three compile.

## Run

```bash
./build/Src/ADV_run/adv3d.ex Inputs/inputs_advection
python3 Tools/reference.py adv_run
```

The run writes `phi0.npy`, one `phi_l<l>.npy` per dt level and `run_info.json`.
Any input can be overridden on the command line, e.g. `time.dt_max=0.02`.

**Why the reference is exact.** `reference.py` builds the reference from the
sampled `phi0.npy` and the Fourier symbols of the stencils actually used
(central advection, 7-point Laplacian). Since it solves the same semi-discrete
system, all of the measured error is temporal. The initial condition therefore
needs no Python twin and can be anything.

**Expected results.**

| test | expectation |
|---|---|
| `inputs_diffusion` | error independent of dt, round-off / IF-truncation level |
| `inputs_advection` | order 3 |
| `inputs_advection_quasi1d` | order 3; same errors as a 1D build |
| `inputs_stability` | seed grows at CFL 1.83, stays bounded at CFL 1.53 |

CFL is reported as `sum_d |c_d| dt / dx`; the limit for the central stencil
is sqrt(3). With seeded runs, `reference.py` also reports the growth factor of
the seeded mode.

Checked on a CPU build (AMReX 26.09): quasi-1D and 32^3 (8 boxes) advection
give order 3.00 at every level; pure diffusion stays at ~1e-13 for every dt;
the seed grows by ~7e5 at CFL 1.83 and decays at CFL 1.53; C++ and
`ifrk_prototype.py` agree to ~1e-14. 1D and 2D builds compile warning-free.

**When a test fails.** `Tools/ifrk_prototype.py RUN_DIR` replays every level in
NumPy with the same kernel, stencil and register order. If C++ and the mirror
agree to round-off, the problem is in the formulation; if not, it is in the
implementation. An error that plateaus as dt shrinks usually means a sampling
or stencil mismatch, not an integrator bug.

## Design notes

- **`IFHERK` owns only what NSE also uses:** the shifted tableau, the IF tables
  and `applyIF`. The stage loop stays with the problem (`ScalarStepper` here,
  `ProjectionWorkspace` in NSE), since the projection sits inside each stage.
- **Caller-owned scratch.** `applyIF(fld, buff, idx)` ping-pongs through a buffer
  the caller owns, so one `IFHERK` serves cell-, face- and edge-centred data.
- **`setTimeStep(dt)`** rebuilds the tables (dt enters through alpha); this is
  what allows a CFL-adaptive dt later. It never grows `n_IF`, so construct with
  the largest dt you will use.
- **`setGeometry(geom)`** updates the periodicity on a snug-domain regrid and
  rebuilds the tables only if dx changed.
- **Automatic `n_IF`** (`ifherk.n_IF = 0`): the smallest half-width whose kernel
  tail is below `IF_eps` for every sub-step. The tables themselves are computed
  with the same series as the NSE code, so for a given `n_IF` they match
  bit-for-bit.
- **Thin domains** must be at least `n_IF` cells thick in every direction, so
  that periodic ghost filling never needs more than one period. This is
  checked at start-up.

## Porting back to LGF_NSE_GPU

`ProjectionWorkspace` replaces its `rk_*` members, `rk_unique_gaps`,
`if_table`, `setupRKCoeffs`, `precomputeIFs` and `applyIF` with one `IFHERK`
member, built in the initializer list as
`ifherk(getRKButcher(), geom_in, config.invRe, config.set_dt, config.n_IF, config.int_fact_eps)`:

| before | after |
|---|---|
| `aT(i,j)`, `rk_if_idx(i)` | `ifherk.aT(i,j)`, `ifherk.ifIdx(i)` |
| `applyIF(q, idx)` | `ifherk.applyIF(q, IF_buff, idx)` |
| `n_IF` (config) | `ifherk.nGhostRequired()` |
| `regridOnto(...)` | also call `ifherk.setGeometry(new_geom)` |

`fcompare` on plotfiles from before and after confirms nothing changed.

Note: in the current `ProjectionWorkspace::advanceTimeStep`, `dt_in` is
unused. The step always runs with the constructor's `config.set_dt`, which
also sized the IF tables. With `IFHERK` this becomes `ifherk.setTimeStep(dt_in)`
at the top of the step (cheap: a few tables of `n_IF + 1` entries).

## Next

- Scalar Burgers (conservative central flux) + `solve_ivp` reference: order 2.
- Face-centred vector Burgers reusing `nonLinearTerm` + grad(1/2 P(u,u)),
  quasi-1D check against scalar Burgers to round-off, genuinely 3D
  self-convergence.
