#ifndef INITIALCONDITION_H_
#define INITIALCONDITION_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>
#include <AMReX_GpuQualifiers.H>

#include <ProblemConfig.H>

#include <cstdint>

// ---------------------------------------------------------------------------
// phi_0 for the scalar tests (the analogue of InitialVorticity.H).
//
// The reference in Tools/reference.py is built from the SAMPLED phi_0 that
// the run writes out, so nothing here needs a Python twin: the reference is
// exact for whatever this produces.
//
// "modes": smooth, couples the directions, distinct wavenumbers in each
// direction (so an indexing bug that swaps x and y cannot hide behind
// symmetry). Angles are normalised per direction, X_d = 2 pi x_d / L_d, so the
// field is periodic on any domain; ic.active = 0 drops a direction entirely
// (quasi-1D/2D runs in the 3D build, and the same formula as a 1D/2D build).
//
//   phi_0 = mean + sin(X + 0.3) cos(2Y) + 0.5 sin(3Z) + 0.25 sin(2X + Z)
//         + noise_amp * (deterministic noise in [-1, 1])
//         + seed_amp  * cos(pi/2 (i + j + k))
//
// The noise is a hash of the global index, so it does not depend on the box
// decomposition or the rank count. The seed is the (k,k,k) mode with
// k dx = pi/2, where the central-advection eigenvalue peaks: it sits exactly
// on the sqrt(3) CFL limit.
//
// ic.stagger (scalar only) samples a direction at face positions instead of
// cell centres, so a scalar run can be compared index-by-index with a
// face-centred vector run (the quasi-1D check).
//
// Vector Burgers (one call per component, see fillInitialCondition):
//
// "modes": generic field with mean flow, different structure in each
//   component, and non-zero vorticity (Taylor-Green keeps w = 0 and Beltrami
//   flows have omega x U = 0, so neither exercises the full kernel):
//     u = m0 + sin X cos Y + 0.5 sin 2Z
//     v = m1 + sin Y cos Z + 0.5 sin 2X
//     w = m2 + sin Z cos X + 0.5 sin 2Y        (m = ic.mean_vec)
//
// "quasi1d": U = [phi_0, 0, 0], phi_0 the scalar "modes" field (use
//   ic.active = 1 0 0). omega = 0 and the kernel reduces exactly to the
//   scalar Burgers stencil.
//
// "diagonal" (scalar and vector): a profile of the diagonal index
//   s = i + j + k only,
//     g(s) = mean + sin(theta + 0.3) + 0.5 sin(2 theta),
//     theta = 2 pi (s + 1) dx / L,
//   used for every vector component. x-, y- and z-faces with the same (i,j,k)
//   all lie at x + y + z = (s + 1) dx, so this samples U = f(x+y+z)[1 1 1]
//   exactly, and computing it from the integer s keeps the components
//   bitwise equal. The discrete vorticity then vanishes identically, and the
//   vector run must reproduce a scalar diagonal run (c = 1 1 1) to round-off.
//
// Indices are wrapped into the periodic domain before sampling, so the
// duplicated periodic faces of a face-centred field start bitwise equal.
// ---------------------------------------------------------------------------

struct ICParams
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> L;
    amrex::GpuArray<int,         AMREX_SPACEDIM> active;
    amrex::GpuArray<int,         AMREX_SPACEDIM> ncell;      // for periodic wrapping
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> mean_vec;
    int type;                                                // ICType as int
    amrex::Real mean;
    amrex::Real noise_amp;
    amrex::Real seed_amp;
};

// uniform in [-1, 1], from the global cell index
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hashNoise (int i, int j, int k)
{
    std::uint32_t h = static_cast<std::uint32_t>(i) * 73856093u
                    ^ static_cast<std::uint32_t>(j) * 19349663u
                    ^ static_cast<std::uint32_t>(k) * 83492791u;
    h ^= h >> 16;  h *= 0x7feb352du;
    h ^= h >> 15;  h *= 0x846ca68bu;
    h ^= h >> 16;
    return 2.0 * (static_cast<amrex::Real>(h) / 4294967295.0) - 1.0;
}

// nodal[d] = 1 for a face/node-centred direction, 0 for cell-centred,
// so the same function serves the face-centred vector fields later
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initialPhi (int i, int j, int k,
                        amrex::GpuArray<int, AMREX_SPACEDIM> const& nodal,
                        ICParams const& p)
{
    constexpr amrex::Real twopi = 2.0 * amrex::Math::pi<amrex::Real>();
    const int idx[3] = {i, j, k};

    amrex::Real X[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::Real x = (idx[d] + 0.5 * (1 - nodal[d])) * p.dx[d];
        X[d] = p.active[d] ? twopi * x / p.L[d] : 0.0;
    }

    amrex::Real phi = p.mean 
                    + std::sin(X[0] + 0.3) * std::cos(2.0 * X[1])
                    + 0.5  * std::sin(3.0 * X[2])
                    + 0.25 * std::sin(2.0 * X[0] + X[2]);

    if (p.noise_amp != 0.0) { phi += p.noise_amp * hashNoise(i, j, k); }

    if (p.seed_amp != 0.0)
    {
        int s = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { s += idx[d]; }
        phi += p.seed_amp * std::cos(0.5 * amrex::Math::pi<amrex::Real>() * s);
    }
    return phi;
}

// "diagonal": a function of the diagonal index s only
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real diagonalProfile (int s, ICParams const& p)
{
    constexpr amrex::Real twopi = 2.0 * amrex::Math::pi<amrex::Real>();
    const amrex::Real theta = twopi * (s + 1) * p.dx[0] / p.L[0];
    return p.mean + std::sin(theta + 0.3) + 0.5 * std::sin(2.0 * theta);
}

// vector "modes", component comp
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initialVelModes (int comp, int i, int j, int k,
                             amrex::GpuArray<int, AMREX_SPACEDIM> const& nodal,
                             ICParams const& p)
{
    constexpr amrex::Real twopi = 2.0 * amrex::Math::pi<amrex::Real>();
    const int idx[3] = {i, j, k};

    amrex::Real X[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::Real x = (idx[d] + 0.5 * (1 - nodal[d])) * p.dx[d];
        X[d] = p.active[d] ? twopi * x / p.L[d] : 0.0;
    }

    const int a = comp, b = (comp + 1) % 3, c = (comp + 2) % 3;
    amrex::Real m = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { if (d == comp) { m = p.mean_vec[d]; } }
    return m + std::sin(X[a]) * std::cos(X[b]) + 0.5 * std::sin(2.0 * X[c]);
}

// comp < 0: the scalar field; comp = d: vector component d
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initialValue (int comp, int i, int j, int k,
                          amrex::GpuArray<int, AMREX_SPACEDIM> const& nodal,
                          ICParams const& p)
{
    if (p.type == static_cast<int>(ICType::Diagonal))
    {
        const int idx[3] = {i, j, k};
        int s = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { s += idx[d]; }
        return diagonalProfile(s, p);
    }
    if (comp < 0) { return initialPhi(i, j, k, nodal, p); }
    if (p.type == static_cast<int>(ICType::Quasi1D))
    {
        return (comp == 0) ? initialPhi(i, j, k, nodal, p) : 0.0;
    }
    return initialVelModes(comp, i, j, k, nodal, p);
}

inline ICParams makeICParams (const amrex::Geometry& geom, const ProblemConfig& cfg)
{
    ICParams p;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        p.dx[d]     = geom.CellSize(d);
        p.L[d]      = geom.ProbLength(d);
        p.active[d] = cfg.ic_active[d];
        p.ncell[d]  = geom.Domain().length(d);
        p.mean_vec[d] = cfg.ic_mean_vec[d];
    }
    p.type = static_cast<int>(cfg.icType());
    p.mean = cfg.ic_mean;
    p.noise_amp = cfg.noise_amp;
    p.seed_amp  = cfg.seed_amp;
    return p;
}

// fill the valid region of mf (any IndexType) and its periodic ghosts.
// comp < 0: the scalar field (ic.stagger applies); comp = d: vector component d
inline void fillInitialCondition (amrex::MultiFab& mf, const amrex::Geometry& geom,
                                  const ProblemConfig& cfg, int comp = -1)
{
    const ICParams p = makeICParams(geom, cfg);

    amrex::GpuArray<int, AMREX_SPACEDIM> nodal;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        nodal[d] = (mf.ixType().nodeCentered(d) || (comp < 0 && cfg.ic_stagger[d])) ? 1 : 0;
    }

    for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& a = mf.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // wrap into [0, n): duplicated periodic faces get identical values
            int w[3] = {i, j, k};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { w[d] = ((w[d] % p.ncell[d]) + p.ncell[d]) % p.ncell[d]; }
            a(i,j,k) = initialValue(comp, w[0], w[1], w[2], nodal, p);
        });
    }
    mf.FillBoundary(geom.periodicity());
}

#endif
