#include <VectorStepper.H>
#include <VectorOperators.H>

#include <AMReX_BLProfiler.H>

VectorStepper::VectorStepper (const amrex::Geometry& geom_in, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm, const ProblemConfig& cfg,
                              amrex::Real dt)
    : geom(geom_in),
      ifherk(getRKButcher(), geom_in, cfg.nu, dt, cfg.n_IF, cfg.IF_eps)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cfg.isVector(), "VectorStepper: problem is not a vector problem");

    n_ghost = std::max(ifherk.nGhostRequired(), 1);

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().length(d) >= n_ghost,
            "domain thinner than the IF stencil half-width in some direction: "
            "increase n_cell there (or raise IF_eps / lower dt)");
    }

    const int s = ifherk.nStages();
    defineFaceVector(stage, ba, dm, n_ghost);
    defineFaceVector(q,     ba, dm, n_ghost);
    defineFaceVector(r,     ba, dm, n_ghost);
    defineFaceVector(buff,  ba, dm, n_ghost);
    w.resize(s + 1);                                       // w[0] left undefined
    for (int j = 1; j <= s; ++j) { defineFaceVector(w[j], ba, dm, n_ghost); }

    ke.define(ba, dm, 1, 1);
    ke.setVal(0.0);
}

void VectorStepper::defineFaceVector (VecMF& v, const amrex::BoxArray& ba,
                                      const amrex::DistributionMapping& dm, int ngrow)
{
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        v[d].define(amrex::convert(ba, amrex::IntVect::TheDimensionVector(d)), dm, 1, ngrow);
        v[d].setVal(0.0);
    }
}

void VectorStepper::setTimeStep (amrex::Real dt_in)
{
    ifherk.setTimeStep(dt_in);
}

void VectorStepper::fillBoundary (VecMF& v) const
{
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { v[d].FillBoundary(geom.periodicity()); }
}

void VectorStepper::computeKineticEnergy (const VecMF& st)
{
    BL_PROFILE("<Compute> VectorStepper::computeKineticEnergy()");

    for (amrex::MFIter mfi(ke, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& ke_arr = ke.array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel;
        for (int c = 0; c < AMREX_SPACEDIM; ++c) { vel[c] = st[c].const_array(mfi); }

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            ke_arr(i,j,k) = cellKineticEnergy(i, j, k, vel);
        });
    }
    ke.FillBoundary(geom.periodicity());
}

void VectorStepper::computeGStage (const VecMF& st, int i)
{
    BL_PROFILE("<Compute> VectorStepper::computeGStage()");

    computeKineticEnergy(st);

    const auto invdx = geom.InvCellSizeArray();
    const amrex::Real coef = -ifherk.aT(i,i) * ifherk.timeStep();

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        for (amrex::MFIter mfi(w[i][d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            auto const& g_arr  = w[i][d].array(mfi);
            auto const& ke_arr = ke.const_array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel;
            for (int c = 0; c < AMREX_SPACEDIM; ++c) { vel[c] = st[c].const_array(mfi); }

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2, int j2, int k2)
            {
                g_arr(i2,j2,k2) = coef * vectorBurgersN(d, i2, j2, k2, invdx, vel, ke_arr);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Eqs. (25)-(30) with the constraint removed:  U^i = H^i r^i
// Line for line the same as ScalarStepper::advance, per component.
// ---------------------------------------------------------------------------

void VectorStepper::advance (VecMF& U)
{
    BL_PROFILE("<Compute> VectorStepper::advance()");

    for (int d = 0; d < AMREX_SPACEDIM; ++d) { AMREX_ALWAYS_ASSERT(U[d].nGrow() >= n_ghost); }

    const int s  = ifherk.nStages();
    const amrex::Real dt = ifherk.timeStep();

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        amrex::MultiFab::Copy(stage[d], U[d], 0, 0, 1, 0);      // U^0 = U_k
        amrex::MultiFab::Copy(q[d],     U[d], 0, 0, 1, 0);      // q^1 = U_k, Eq. (29)
    }

    for (int i = 1; i <= s; ++i)
    {
        // --- g^i into slot i. Fresh; never aged. Eq. (28) ------------------
        fillBoundary(stage);
        computeGStage(stage, i);

        // --- age q and w[1..i-1] by H^{i-1}. Eqs. (29), (30) ---------------
        if (i > 1)
        {
            const int idx = ifherk.ifIdx(i-1);
            ifherk.applyIF(q, buff, idx);
            for (int j = 1; j <= i-1; ++j) { ifherk.applyIF(w[j], buff, idx); }
        }

        // --- r^i = q^i + dt sum_{j<i} a~_{i,j} w^{i,j} + g^i. Eq. (27) -----
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            amrex::MultiFab::Copy(r[d], q[d], 0, 0, 1, 0);
            for (int j = 1; j <= i-1; ++j)
            { amrex::MultiFab::Saxpy(r[d], dt * ifherk.aT(i,j), w[j][d], 0, 0, 1, 0); }
            amrex::MultiFab::Saxpy(r[d], 1.0, w[i][d], 0, 0, 1, 0);   // g^i, coeff 1
        }

        // --- U^i = H^i r^i -------------------------------------------------
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { amrex::MultiFab::Copy(stage[d], r[d], 0, 0, 1, 0); }
        ifherk.applyIF(stage, buff, ifherk.ifIdx(i));

        // --- w^{i,i} = g^i / (a~_{i,i} dt). Eq. (30) ------------------------
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { w[i][d].mult(1.0 / (ifherk.aT(i,i) * dt), 0); }
    }

    for (int d = 0; d < AMREX_SPACEDIM; ++d) { amrex::MultiFab::Copy(U[d], stage[d], 0, 0, 1, 0); }
    fillBoundary(U);
}
