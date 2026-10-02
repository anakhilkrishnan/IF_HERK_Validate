# IF_HERK_Validate

Advection-diffusion test bench for the IF-HERK time integrator of LGF_NSE_GPU
(Liska & Colonius 2016). The integrator core is problem-agnostic and is meant
to be linked back into LGF_NSE_GPU unchanged.

Implemented, all periodic:

- **Case 1:** scalar linear convection-diffusion (and pure diffusion), cell-centred.
- **Case 2.1:** scalar Burgers, conservative central flux, cell-centred.
- **Case 2.2:** vector Burgers, face-centred, with the LGF_NSE_GPU kernel
  (`nonLinearTerm` for omega x U, plus grad 1/2|U|^2). 2D and 3D builds only.

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
│   │   ├── ScalarOperators.H   # device kernels for scalar N(phi)
│   │   ├── VectorOperators.H   # vector Burgers kernel, built on NSE_shared
│   │   ├── InitialCondition.H  # device-callable initial fields (cf. InitialVorticity.H)
│   │   ├── ScalarStepper.H     # unconstrained stage loop, cell-centred scalar
│   │   ├── scalarStepper.cpp
│   │   ├── VectorStepper.H     # unconstrained stage loop, face-centred vector
│   │   ├── vectorStepper.cpp   #   (ProjectionWorkspace minus the projection)
│   │   └── FieldIO.H           # gather-and-write .npy for the Python tools
│   ├── NSE_shared/             # verbatim copies from LGF_NSE_GPU -- do not edit here
│   │   └── SpatialDiscretization.H
│   └── ADV_run/
│       └── main.cpp            # dt sweep (scalar and vector drivers)
├── Inputs/
│   ├── inputs_advection          # order test, 3D
│   ├── inputs_diffusion          # c = 0, dt-independence test
│   ├── inputs_advection_quasi1d  # thin domain in the 3D build
│   ├── inputs_stability          # sqrt(3) limit
│   ├── inputs_scalar_burgers     # 3D scalar Burgers order test
│   ├── inputs_vector_burgers     # generic 3D vector Burgers, self-convergence
│   ├── inputs_{vector,scalar}_burgers_q1d    # quasi-1D reduction pair
│   └── inputs_{vector,scalar}_burgers_diag   # diagonal 3D reduction pair
└── Tools/
    ├── operators.py            # NumPy versions of the scalar stencils (shared)
    ├── reference.py            # references, errors, orders (all problems)
    ├── check_reduction.py      # vector vs scalar Burgers, to round-off
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

The run writes the initial field (`phi0.npy`, or `u0.npy`, `v0.npy`, `w0.npy`),
one file per field per dt level (`phi_l<l>.npy`), an optional reference run
(`*_ref.npy`, with `time.ref_refine`), and `run_info.json`, which tells the
Python tools everything else. Any input can be overridden on the command line,
e.g. `time.dt_max=0.02`.

The Burgers reduction checks run in pairs:

```bash
./adv3d.ex Inputs/inputs_vector_burgers_q1d
./adv3d.ex Inputs/inputs_scalar_burgers_q1d
python3 Tools/check_reduction.py vb_q1d sb_q1d      # same for _diag: vb_diag sb_diag
```

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
| `inputs_scalar_burgers` (and the `_q1d`, `_diag` partners) | order 2 vs an independent DOP853 reference |
| `inputs_vector_burgers` | order 2 vs its own dt_min/16 run |
| `check_reduction.py vb_q1d sb_q1d` | u = phi and v = w = 0, to round-off |
| `check_reduction.py vb_diag sb_diag` | u = v = w = phi, to round-off |

CFL is reported as `sum_d |c_d| dt / dx`; the limit for the central stencil
is sqrt(3). With seeded runs, `reference.py` also reports the growth factor of
the seeded mode.

Checked on a CPU build (AMReX 26.09): quasi-1D and 32^3 (8 boxes) advection
give order 3.00 at every level; pure diffusion stays at ~1e-13 for every dt;
the seed grows by ~7e5 at CFL 1.83 and decays at CFL 1.53; C++ and
`ifrk_prototype.py` agree to ~1e-14. Scalar Burgers converges at order 2 (quasi-1D,
diagonal and generic 3D); the quasi-1D and diagonal vector runs match their
scalar partners to ~1e-15 at every level (bitwise at t = 0); generic 3D vector
Burgers at 32^3 gives order 2.00. 1D and 2D builds compile warning-free.

**What the Burgers checks do and don't cover.** Each check has a blind spot,
and they are chosen so the blind spots don't overlap, except one:

- *Quasi-1D* catches direction mix-ups (a planted bug using the x-gradient of
  1/2|U|^2 in the y-component makes v jump from 0 to 1.2), but only for data
  varying in x.
- *Diagonal* exercises all three face types and every stencil direction, but on
  diagonal data all directions look alike, so it cannot see direction mix-ups
  (the same planted bug passes it).
- *Generic 3D* is the only test with omega != 0, so it is the only one that
  exercises omega x U, and only by self-convergence: it would converge happily
  to the solution of a consistently wrong omega x U.

That last gap is the remaining one. Closing it exactly would take the linearised
vector operator, [N(U0 + eps V) - N(U0 - eps V)] / (2 eps), whose 3x3 Fourier
symbol gives an exact reference; it was judged not worth the effort here, since
`nonLinearTerm` is used unchanged from LGF_NSE_GPU, where it is already exercised.

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

## Inputs reference (beyond Case 1)

| key | meaning |
|---|---|
| `prob.problem` | `linear_advection`, `scalar_burgers`, `vector_burgers` |
| `prob.c` | linear: velocity; scalar Burgers: direction weights c_d in sum_d c_d d(phi^2/2)/dx_d |
| `ic.type` | `modes` (default), `diagonal` (scalar or vector), `quasi1d` (vector: [u(x), 0, 0]) |
| `ic.mean` | mean added to the scalar `modes`, `quasi1d` and `diagonal` fields |
| `ic.mean_vec` | per-component means of the vector `modes` field |
| `ic.stagger` | scalar only: sample a direction at face positions (pairs with a vector run) |
| `time.ref_refine` | m > 0: extra reference run at dt_min / 2^m |
