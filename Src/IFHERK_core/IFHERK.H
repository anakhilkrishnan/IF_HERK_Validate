#ifndef IFHERK_H_
#define IFHERK_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>

#include <RKCoefficients.H>

// ---------------------------------------------------------------------------
// IFHERK: the problem-agnostic half of the integrating-factor half-explicit
// Runge-Kutta scheme of Liska & Colonius (2016).
//
// Owns
//   -> the shifted, 1-based tableau a~_{i,j}, c~_i (paper Eqs. 23, 27-30)
//   -> the stage -> unique-sub-step map, and one 1D heat-kernel table per
//      unique sub-step: h_alpha(n) = exp(-2 alpha) I_n(2 alpha), n = 0..n_IF
//   -> applyIF: H = E(alpha) applied in place as d separable 1D sweeps
//
// Does NOT own the stage loop. The loop lives with the problem
// (ScalarStepper here, ProjectionWorkspace in LGF_NSE_GPU), because what
// happens between the IF applications (nothing, or a projection) is
// problem-specific.
//
// Assumes dx == dy == dz, as the NSE code does.
// ---------------------------------------------------------------------------

class IFHERK
{
public:
    static constexpr int MAX_STAGES = RKButcher::MAX_STAGES;

    // n_IF <= 0 selects the stencil half-width automatically: the smallest n
    // whose kernel tail 2*sum_{m>n} h(m) is below IF_eps for every sub-step
    // at this dt. Size the integrator with the LARGEST dt you will use:
    // setTimeStep() never grows n_IF, since that would outgrow the caller's
    // ghost cells.
    IFHERK (const RKButcher& tableau, const amrex::Geometry& geom,
            amrex::Real nu, amrex::Real dt, int n_IF, amrex::Real IF_eps);

    // dt enters every table through alpha = gap * dt * nu / dx^2
    void setTimeStep (amrex::Real dt_in);

    // new geometry (e.g. snug-domain regrid): periodicity always updated,
    // tables rebuilt only if dx changed
    void setGeometry (const amrex::Geometry& geom);

    // shifted tableau, 1-based
    int         nStages ()          const { return rk_stages; }
    amrex::Real aT   (int i, int j) const { return rk_at(i,j); }
    amrex::Real cT   (int i)        const { return rk_ct(i);   }
    amrex::Real gapT (int i)        const { return rk_gap(i);  }
    int         ifIdx(int i)        const { return rk_if_idx(i); }   // -1: H = identity

    int         nGhostRequired ()   const { return n_IF; }
    amrex::Real timeStep ()         const { return dt; }
    amrex::Real viscosity ()        const { return nu; }
    amrex::Real maxKernelTail ()    const;   // largest tail over the current tables

    void printSummary () const;

    // H applied in place: separable 1D sweeps, ping-ponging through buff.
    // buff is caller-owned scratch with the same BoxArray/DM/IndexType as fld.
    // Public because it launches a device lambda (CUDA extended-lambda rule).
    void applyIF (amrex::MultiFab& fld, amrex::MultiFab& buff, int if_idx) const;

    // vector overload: launches nothing itself, so it can live in the header
    template <std::size_t N>
    void applyIF (amrex::Array<amrex::MultiFab, N>& fld,
                  amrex::Array<amrex::MultiFab, N>& buff, int if_idx) const
    {
        for (std::size_t c = 0; c < N; ++c) { applyIF(fld[c], buff[c], if_idx); }
    }

private:
    void setupRKCoeffs (const RKButcher& tableau);   // from projectionWorkspace.cpp
    void precomputeIFs ();                           // from projectionWorkspace.cpp; dt, nu now members

    // exp(-z) * I_n(z), z = 2 alpha, n = 0..nmax; same series as the NSE code
    static amrex::Vector<amrex::Real> besselKernel (amrex::Real alpha, int nmax);
    static amrex::Real kernelTail (const amrex::Vector<amrex::Real>& h, int n);

    amrex::Periodicity period;
    amrex::Real dx  = 0.0;
    amrex::Real dx2 = 0.0;
    amrex::Real nu  = 0.0;
    amrex::Real dt  = 0.0;
    amrex::Real IF_eps = 0.0;
    int n_IF      = 0;
    int rk_stages = 0;

    amrex::Array2D<amrex::Real, 1, MAX_STAGES, 1, MAX_STAGES> rk_at;   // a~_{i,j}
    amrex::Array1D<amrex::Real, 1, MAX_STAGES>                rk_ct;   // c~_i
    amrex::Array1D<amrex::Real, 1, MAX_STAGES>                rk_gap;  // c~_i - c~_{i-1}
    amrex::Array1D<int, 1, MAX_STAGES>                        rk_if_idx;

    amrex::Vector<amrex::Real>                           rk_unique_gaps;
    amrex::Vector<amrex::Real>                           if_tail;    // per unique gap
    amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> if_table;   // [gap][0..n_IF]
};

#endif
