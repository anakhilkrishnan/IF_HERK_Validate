#ifndef SCALAROPERATORS_H_
#define SCALAROPERATORS_H_

#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

// ---------------------------------------------------------------------------
// Device kernels for the explicit operator N(phi) of the scalar tests
// (the analogue of SpatialDiscretization.H).
//
// Convention, as in LGF_NSE_GPU:  dphi/dt = -N(phi) + nu lap(phi),
// with the viscous term handled exactly by the integrating factor, so only
// N is ever evaluated explicitly.
//
// Every kernel reads one ghost cell.
// ---------------------------------------------------------------------------

// Linear advection, cell-centred, 2nd-order central:
//   N(phi) = sum_d c_d (phi_{+1} - phi_{-1}) / (2 dx_d)
// Fourier symbol: i sum_d c_d sin(k_d dx_d) / dx_d   (Tools/reference.py)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real linearAdvectionCentral (int i, int j, int k,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& c,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                                    amrex::Array4<amrex::Real const> const& phi)
{
    amrex::ignore_unused(j, k);
    amrex::Real adv = 0.0;
    AMREX_D_TERM(adv += c[0] * 0.5 * invdx[0] * (phi(i+1,j,k) - phi(i-1,j,k));,
                 adv += c[1] * 0.5 * invdx[1] * (phi(i,j+1,k) - phi(i,j-1,k));,
                 adv += c[2] * 0.5 * invdx[2] * (phi(i,j,k+1) - phi(i,j,k-1));)
    return adv;
}

// scalar Burgers flux
//   N(u) = sum_d dim_sel_d (u_{+1}^2 - u_{-1}^2) / (4 dx_d)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real scalarBurgersCentral (int i, int j, int k,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dim_sel,
                                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx,
                                    amrex::Array4<amrex::Real const> const& phi)
{
    amrex::ignore_unused(j, k);
    amrex::Real adv = 0.0;
    AMREX_D_TERM(adv += dim_sel[0] * 0.25 * invdx[0] * (phi(i+1,j,k) * phi(i+1,j,k) - phi(i-1,j,k) * phi(i-1,j,k));,
                 adv += dim_sel[1] * 0.25 * invdx[1] * (phi(i,j+1,k) * phi(i,j+1,k) - phi(i,j-1,k) * phi(i,j-1,k));,
                 adv += dim_sel[2] * 0.25 * invdx[2] * (phi(i,j,k+1) * phi(i,j,k+1) - phi(i,j,k-1) * phi(i,j,k-1));)
    return adv;
}

#endif
