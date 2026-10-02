#include <ScalarStepper.H>
#include <ScalarOperators.H>

#include <AMReX_BLProfiler.H>
#include <AMReX_Print.H>

ScalarStepper::ScalarStepper (const amrex::Geometry& geom_in, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm, const ProblemConfig& cfg,
                              amrex::Real dt)
    : problem(cfg.problem),
      geom(geom_in),
      adv_vel(cfg.adv_vel),
      ifherk(getRKButcher(), geom_in, cfg.nu, dt, cfg.n_IF, cfg.IF_eps)
{
    // the kernel branches in computeGStage cover the scalar problems only
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!cfg.isVector(), "ScalarStepper: problem is a vector problem");

    n_ghost = std::max(ifherk.nGhostRequired(), 1);

    // Periodic FillBoundary is only trusted here for ghost widths up to one
    // period; thin domains must therefore be at least n_ghost cells thick.
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().length(d) >= n_ghost,
            "domain thinner than the IF stencil half-width in some direction: "
            "increase n_cell there (or raise IF_eps / lower dt)");
    }

    const int s = ifherk.nStages();
    stage.define(ba, dm, 1, n_ghost);
    q    .define(ba, dm, 1, n_ghost);
    r    .define(ba, dm, 1, n_ghost);
    buff .define(ba, dm, 1, n_ghost);
    stage.setVal(0.0);  q.setVal(0.0);  r.setVal(0.0);  buff.setVal(0.0);

    w.resize(s + 1);                                 // w[0] left undefined
    for (int j = 1; j <= s; ++j)
    {
        w[j].define(ba, dm, 1, n_ghost);
        w[j].setVal(0.0);
    }
}

void ScalarStepper::setTimeStep (amrex::Real dt_in)
{
    ifherk.setTimeStep(dt_in);
}

void ScalarStepper::computeGStage (const amrex::MultiFab& st, int i)
{
    BL_PROFILE("<Compute> ScalarStepper::computeGStage()");

    const auto invdx = geom.InvCellSizeArray();
    const auto c     = adv_vel;
    const amrex::Real coef = -ifherk.aT(i,i) * ifherk.timeStep();

    for (amrex::MFIter mfi(w[i], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& g_arr = w[i].array(mfi);
        auto const& phi   = st.const_array(mfi);

        // branch for scalar burgers and linear convection
        if (problem == "linear_advection")
        {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2, int j2, int k2)
            {
                g_arr(i2,j2,k2) = coef * linearAdvectionCentral(i2, j2, k2, c, invdx, phi);
            });
        }
        else if (problem == "scalar_burgers")
        {
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i2, int j2, int k2)
            {
                g_arr(i2,j2,k2) = coef * scalarBurgersCentral(i2, j2, k2, c, invdx, phi);
            });
        }
        
    }
}

// ---------------------------------------------------------------------------
// Eqs. (25)-(30) with the constraint removed:  phi^i = H^i r^i
// ---------------------------------------------------------------------------

void ScalarStepper::advance (amrex::MultiFab& phi)
{
    BL_PROFILE("<Compute> ScalarStepper::advance()");

    AMREX_ALWAYS_ASSERT(phi.nGrow() >= n_ghost);

    const int s  = ifherk.nStages();
    const amrex::Real dt = ifherk.timeStep();
    const auto per = geom.periodicity();

    amrex::MultiFab::Copy(stage, phi, 0, 0, 1, 0);          // phi^0 = phi_k
    amrex::MultiFab::Copy(q,     phi, 0, 0, 1, 0);          // q^1   = phi_k, Eq. (29)

    for (int i = 1; i <= s; ++i)
    {
        // --- g^i into slot i. Fresh; never aged. Eq. (28) ------------------
        stage.FillBoundary(per);
        computeGStage(stage, i);

        // --- age q and w[1..i-1] by H^{i-1}. Eqs. (29), (30) ---------------
        if (i > 1)
        {
            const int idx = ifherk.ifIdx(i-1);
            ifherk.applyIF(q, buff, idx);
            for (int j = 1; j <= i-1; ++j) { ifherk.applyIF(w[j], buff, idx); }
        }

        // --- r^i = q^i + dt sum_{j<i} a~_{i,j} w^{i,j} + g^i. Eq. (27) -----
        amrex::MultiFab::Copy(r, q, 0, 0, 1, 0);
        for (int j = 1; j <= i-1; ++j)
        { amrex::MultiFab::Saxpy(r, dt * ifherk.aT(i,j), w[j], 0, 0, 1, 0); }
        amrex::MultiFab::Saxpy(r, 1.0, w[i], 0, 0, 1, 0);   // g^i, coeff 1

        // --- phi^i = H^i r^i ---------------------------------------------
        amrex::MultiFab::Copy(stage, r, 0, 0, 1, 0);
        ifherk.applyIF(stage, buff, ifherk.ifIdx(i));

        // --- w^{i,i} = g^i / (a~_{i,i} dt). Eq. (30) ------------------------
        w[i].mult(1.0 / (ifherk.aT(i,i) * dt), 0);
    }

    amrex::MultiFab::Copy(phi, stage, 0, 0, 1, 0);
    phi.FillBoundary(per);
}
