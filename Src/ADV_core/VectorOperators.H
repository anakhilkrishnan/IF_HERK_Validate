#ifndef VECTOROPERATORS_H_
#define VECTOROPERATORS_H_

#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

// verbatim copy of the LGF_NSE_GPU stencils (Src/NSE_shared)
#include <SpatialDiscretization.H>

// ---------------------------------------------------------------------------
// Device kernels for vector Burgers on the staggered (face-centred) grid.
//
//   dU/dt = -N(U) + nu lap(U),     N(U) = omega x U + grad(1/2 |U|^2)
//
// omega x U is nonLinearTerm<dim> from the NSE code, unchanged. In NSE the
// gradient part is absorbed into the pressure; Burgers has no pressure, so it
// is added back here with the NSE gradient stencil (cell -> face).
//
// |U|^2 at a cell centre averages the squares on the cell's two faces in
// each direction:
//     KE = 1/2 sum_d 1/2 (u_d(lo)^2 + u_d(hi)^2)
// For U = [u(x), 0, 0] this makes the x-component at face i exactly
//     (KE_i - KE_{i-1}) / dx = (u_{i+1}^2 - u_{i-1}^2) / (4 dx),
// the scalar Burgers stencil (ScalarOperators.H).
// ---------------------------------------------------------------------------

// cell-centred kinetic energy; reads only the cell's own (valid) faces
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real cellKineticEnergy (int i, int j, int k,
                               amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& vel)
{
    amrex::Real s = 0.0;
    AMREX_D_TERM(s += vel[0](i,j,k) * vel[0](i,j,k) + vel[0](i+1,j,k) * vel[0](i+1,j,k);,
                 s += vel[1](i,j,k) * vel[1](i,j,k) + vel[1](i,j+1,k) * vel[1](i,j+1,k);,
                 s += vel[2](i,j,k) * vel[2](i,j,k) + vel[2](i,j,k+1) * vel[2](i,j,k+1);)
    return 0.25 * s;
}

// N_d(U) at face (i,j,k) of component d. d is uniform across a launch, so
// the branch costs nothing; it avoids templating the launch site.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real vectorBurgersN (int d, int i, int j, int k,
                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& vel,
                            amrex::Array4<amrex::Real const> const& ke)
{
    if (d == 0) {
        return nonLinearTerm<0>(i, j, k, invdx, vel) + discreteGradientC2F<0>(i, j, k, invdx, ke);
    }
#if AMREX_SPACEDIM >= 2
    if (d == 1) {
        return nonLinearTerm<1>(i, j, k, invdx, vel) + discreteGradientC2F<1>(i, j, k, invdx, ke);
    }
#endif
#if AMREX_SPACEDIM == 3
    if (d == 2) {
        return nonLinearTerm<2>(i, j, k, invdx, vel) + discreteGradientC2F<2>(i, j, k, invdx, ke);
    }
#endif
    return 0.0;
}

#endif
