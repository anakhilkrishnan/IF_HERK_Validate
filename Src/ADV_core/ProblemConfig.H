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
// ---------------------------------------------------------------------------

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
    std::string problem = "linear_advection";            // only case implemented so far
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> adv_vel {AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Real nu = 0.05;

    // ---- initial condition --------------------------------------------------
    std::string ic_type   = "modes";   // "modes": smooth multi-mode field (see InitialCondition.H)
    amrex::IntVect ic_active {AMREX_D_DECL(1, 1, 1)};   // 0: phi0 independent of that direction
    amrex::Real noise_amp = 0.0;       // broadband, deterministic noise (pure-diffusion test)
    amrex::Real seed_amp  = 0.0;       // (k,k,k) mode at k dx = pi/2 (stability test)

    // ---- time integration / convergence sweep -----------------------------
    amrex::Real T        = 1.0;        // final time
    amrex::Real dt_max   = 0.04;       // coarsest dt
    int         n_levels = 5;          // dt_l = dt_max / 2^l, l = 0 .. n_levels-1
    int         n_IF     = 0;          // IF half-width; <= 0 -> automatic from IF_eps
    amrex::Real IF_eps   = 1.0e-13;    // kernel truncation tolerance

    // ---- output -------------------------------------------------------------
    std::string out_dir        = "run";
    bool        write_plotfile = false;
    int         verbose        = 1;

    static ProblemConfig fromParmParse ()
    {
        ProblemConfig c;

        {
            amrex::ParmParse pp("geom");
            amrex::Vector<int> nc;
            if (pp.queryarr("n_cell", nc)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nc.size() >= AMREX_SPACEDIM,
                    "geom.n_cell needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { c.n_cell[d] = nc[d]; }
            }
            pp.query("Lx", c.Lx);
            pp.query("max_grid_size", c.max_grid_size);
        }
        {
            amrex::ParmParse pp("prob");
            pp.query("problem", c.problem);
            amrex::Vector<amrex::Real> cv;
            if (pp.queryarr("c", cv)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cv.size() >= AMREX_SPACEDIM,
                    "prob.c needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { c.adv_vel[d] = cv[d]; }
            }
            pp.query("nu", c.nu);
        }
        {
            amrex::ParmParse pp("ic");
            pp.query("type", c.ic_type);
            amrex::Vector<int> act;
            if (pp.queryarr("active", act)) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(act.size() >= AMREX_SPACEDIM,
                    "ic.active needs AMREX_SPACEDIM entries");
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { c.ic_active[d] = act[d]; }
            }
            pp.query("noise_amp", c.noise_amp);
            pp.query("seed_amp", c.seed_amp);
        }
        {
            amrex::ParmParse pp("time");
            pp.query("T", c.T);
            pp.query("dt_max", c.dt_max);
            pp.query("n_levels", c.n_levels);
        }
        {
            amrex::ParmParse pp("ifherk");
            pp.query("n_IF", c.n_IF);
            pp.query("IF_eps", c.IF_eps);
        }
        {
            amrex::ParmParse pp("out");
            pp.query("dir", c.out_dir);
            pp.query("write_plotfile", c.write_plotfile);
            pp.query("verbose", c.verbose);
        }

        c.validate();
        return c;
    }

    void validate () const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(problem == "linear_advection",
            "prob.problem: only 'linear_advection' is implemented");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ic_type == "modes", "ic.type: only 'modes' is implemented");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Lx > 0.0, "geom.Lx must be positive");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nu >= 0.0, "prob.nu must be non-negative");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(T > 0.0 && dt_max > 0.0 && n_levels >= 1,
            "need T > 0, dt_max > 0, n_levels >= 1");
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_cell[d] > 0, "geom.n_cell must be positive");
        }
    }

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
};

#endif
