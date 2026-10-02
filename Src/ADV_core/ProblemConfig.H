#ifndef PROBLEMCONFIG_H_
#define PROBLEMCONFIG_H_

#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_Print.H>
#include <AMReX_Math.H>

#include <string>
#include <cmath>

// ---------------------------------------------------------------------------
// ProblemConfig: all run inputs, read once through ParmParse
// (the analogue of SolverConfig.H in LGF_NSE_GPU)
//
// Case 1, scalar linear convection-diffusion:
//     dphi/dt + c . grad(phi) = nu lap(phi)
// with c = 0 giving the pure-diffusion sub-case.
// Case 2.1 scalar Burgers equation:
//     dphi/dt + div(1/2 phi^2 c) = nu lap(phi)       (c: direction weights)
// Case 2.2 vector Burgers equation (face-centred, 2D/3D builds only):
//     dU/dt + (U . grad) U = nu lap(U),  in rotational form
//     (U . grad) U = omega x U + grad(1/2 |U|^2), as in LGF_NSE_GPU
// ---------------------------------------------------------------------------

enum class ProblemType { LinearAdvection, ScalarBurgers, VectorBurgers };
enum class ICType      { Modes, Diagonal, Quasi1D };

struct ProblemConfig
{
    // ---- geometry: periodic in every direction ----------------------------
    // Cells are cubic (the IF assumes dx == dy == dz): dx = Lx / n_cell[0], and
    // direction d spans [0, n_cell[d] * dx). Unequal n_cell gives thin domains,
    // e.g. n_cell = 128 16 16 for quasi-1D runs in the 3D build.
    amrex::IntVect n_cell {AMREX_D_DECL(64, 64, 64)};
    amrex::Real    Lx = 2.0 * amrex::Math::pi<amrex::Real>();
    int            max_grid_size = 32;

    // ---- physics ------------------------------------------------------------
    std::string problem = "linear_advection";
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> adv_vel {AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Real nu = 0.05;

    // ---- initial condition --------------------------------------------------
    std::string ic_type   = "modes";   // "modes" | "diagonal" | "quasi1d" (see InitialCondition.H)
    amrex::IntVect ic_active {AMREX_D_DECL(1, 1, 1)};   // 0: phi0 independent of that direction
    amrex::Real ic_mean = 0.0; // adding mean flow to create both translation and attenuation in burgers
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ic_mean_vec {AMREX_D_DECL(0.4, 0.3, 0.2)}; // vector "modes" means
    amrex::IntVect ic_stagger {AMREX_D_DECL(0, 0, 0)}; // scalar only: 1 samples that direction at faces
    amrex::Real noise_amp = 0.0;       // broadband, deterministic noise (pure-diffusion test)
    amrex::Real seed_amp  = 0.0;       // (k,k,k) mode at k dx = pi/2 (stability test)

    // ---- time integration / convergence sweep -----------------------------
    amrex::Real T        = 1.0;        // final time
    amrex::Real dt_max   = 0.04;       // coarsest dt
    int         n_levels = 5;          // dt_l = dt_max / 2^l, l = 0 .. n_levels-1
    int         ref_refine = 0;        // > 0: extra reference run at dt_min / 2^ref_refine
    int         n_IF     = 0;          // IF half-width; <= 0 -> automatic from IF_eps
    amrex::Real IF_eps   = 1.0e-13;    // kernel truncation tolerance

    // ---- output -------------------------------------------------------------
    std::string out_dir        = "run";
    bool        write_plotfile = false;
    int         verbose        = 1;

    static ProblemConfig fromParmParse ()
    {
        ProblemConfig cfg;

        {
            amrex::ParmParse pp("geom");
            amrex::Vector<int> nc;
            if (pp.queryarr("n_cell", nc)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nc.size() >= AMREX_SPACEDIM,
                    "geom.n_cell needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {cfg.n_cell[d] = nc[d]; }
            }
            pp.query("Lx",cfg.Lx);
            pp.query("max_grid_size",cfg.max_grid_size);
        }
        {
            amrex::ParmParse pp("prob");
            pp.query("problem",cfg.problem);
            amrex::Vector<amrex::Real> cv;
            if (pp.queryarr("c", cv)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cv.size() >= AMREX_SPACEDIM,
                    "prob.c needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {cfg.adv_vel[d] = cv[d]; }
            }
            pp.query("nu",cfg.nu);
        }
        {
            amrex::ParmParse pp("ic");
            pp.query("type",cfg.ic_type);
            amrex::Vector<int> act;
            if (pp.queryarr("active", act)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(act.size() >= AMREX_SPACEDIM,
                    "ic.active needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {cfg.ic_active[d] = act[d]; }
            }
            pp.query("mean", cfg.ic_mean);
            amrex::Vector<amrex::Real> mv;
            if (pp.queryarr("mean_vec", mv)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mv.size() >= AMREX_SPACEDIM,
                    "ic.mean_vec needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {cfg.ic_mean_vec[d] = mv[d]; }
            }
            amrex::Vector<int> stg;
            if (pp.queryarr("stagger", stg)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(stg.size() >= AMREX_SPACEDIM,
                    "ic.stagger needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {cfg.ic_stagger[d] = stg[d]; }
            }
            pp.query("noise_amp",cfg.noise_amp);
            pp.query("seed_amp",cfg.seed_amp);
        }
        {
            amrex::ParmParse pp("time");
            pp.query("T",cfg.T);
            pp.query("dt_max",cfg.dt_max);
            pp.query("n_levels",cfg.n_levels);
            pp.query("ref_refine",cfg.ref_refine);
        }
        {
            amrex::ParmParse pp("ifherk");
            pp.query("n_IF",cfg.n_IF);
            pp.query("IF_eps",cfg.IF_eps);
        }
        {
            amrex::ParmParse pp("out");
            pp.query("dir",cfg.out_dir);
            pp.query("write_plotfile",cfg.write_plotfile);
            pp.query("verbose",cfg.verbose);
        }

        cfg.validate();
        return cfg;
    }

    void validate () const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(problem == "linear_advection" || problem == "scalar_burgers"
                                         || problem == "vector_burgers",
            "prob.problem: 'linear_advection', 'scalar_burgers' or 'vector_burgers'");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ic_type == "modes" || ic_type == "diagonal" || ic_type == "quasi1d",
            "ic.type: 'modes', 'diagonal' or 'quasi1d'");
#if AMREX_SPACEDIM == 1
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!isVector(),
            "vector_burgers needs a 2D or 3D build (the NSE stencils assume >= 2D)");
#endif
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(icType() != ICType::Quasi1D || isVector(),
            "ic.type = quasi1d is the vector field [u(x), 0, 0]; for the scalar use "
            "ic.type = modes with ic.active = 1 0 0 (and ic.stagger = 1 0 0 to match the vector run)");
        if (icType() == ICType::Diagonal) {
            for (int d = 1; d < AMREX_SPACEDIM; ++d) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell[d] == n_cell[0],
                    "ic.type = diagonal needs equal n_cell (periodic along x + y + z)");
            }
        }
        if (isVector()) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ic_stagger[d] == 0,
                    "ic.stagger is for scalar runs only (vector components are already staggered)");
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ref_refine >= 0, "time.ref_refine must be >= 0");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Lx > 0.0, "geom.Lx must be positive");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nu >= 0.0, "prob.nu must be non-negative");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(T > 0.0 && dt_max > 0.0 && n_levels >= 1,
            "need T > 0, dt_max > 0, n_levels >= 1");
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell[d] > 0, "geom.n_cell must be positive");
        }
    }

    ProblemType problemType () const
    {
        if (problem == "scalar_burgers") { return ProblemType::ScalarBurgers; }
        if (problem == "vector_burgers") { return ProblemType::VectorBurgers; }
        return ProblemType::LinearAdvection;
    }
    bool isVector () const { return problemType() == ProblemType::VectorBurgers; }

    ICType icType () const
    {
        if (ic_type == "diagonal") { return ICType::Diagonal; }
        if (ic_type == "quasi1d")  { return ICType::Quasi1D; }
        return ICType::Modes;
    }

    // level index of the optional reference run (dt_min / 2^ref_refine)
    int refLevel () const { return n_levels - 1 + ref_refine; }

    amrex::Real dx () const { return Lx / n_cell[0]; }

    amrex::Real dt (int level) const { return dt_max / std::pow(2.0, level); }

    int nSteps (int level) const
    {
        const amrex::Real ratio = T / dt(level);
        const int n = static_cast<int>(std::lround(ratio));
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(ratio - n) < 1.0e-9 * ratio,
            "T must be an integer multiple of every dt in the sweep");
        return n;
    }

    // sum_d |c_d| dt / dx: the linear stability limit of the scheme is sqrt(3)
    amrex::Real cfl (amrex::Real dt_in) const
    {
        amrex::Real s = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { s += std::abs(adv_vel[d]); }
        return s * dt_in / dx();
    }

    // cfl overload for scalar burgers
    amrex::Real cfl (amrex::Real dt_in, amrex::Real phi_max) const
    {
        amrex::Real s = 0.0;
        // selects the correct directions
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { s += std::abs(adv_vel[d]); }
        return s * phi_max * dt_in / dx();
    }   

    // compute cell-Re
    amrex::Real cellRe(amrex::Real phi_max) const
    {
        return phi_max * dx() / nu;
    }
};

#endif
