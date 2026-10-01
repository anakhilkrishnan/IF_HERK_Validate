#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_Print.H>

#include <ProblemConfig.H>
#include <InitialCondition.H>
#include <ScalarStepper.H>
#include <FieldIO.H>

#include <fstream>
#include <iomanip>
#include <string>

// ---------------------------------------------------------------------------
// Convergence sweep for the scalar tests.
//
// For each dt_l = dt_max / 2^l: reset phi to phi_0, take T/dt_l steps, write
// phi(T). phi_0 itself is written once, and Tools/reference.py builds the
// semi-discrete reference from that exact sampled field.
//
//   <out.dir>/phi0.npy
//   <out.dir>/phi_l<l>.npy       l = 0 .. n_levels-1
//   <out.dir>/run_info.json
// ---------------------------------------------------------------------------

namespace {

void writeRunInfo (const ProblemConfig& cfg, const amrex::Geometry& geom,
                   const IFHERK& ifh, const amrex::Vector<amrex::Real>& dts,
                   const amrex::Vector<int>& nsteps, const amrex::Vector<amrex::Real>& wall)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream f(cfg.out_dir + "/run_info.json");
    f << std::setprecision(17);
    auto arr = [&f](auto get, int n) {
        f << "[";
        for (int d = 0; d < n; ++d) { f << (d ? ", " : "") << get(d); }
        f << "]";
    };

    f << "{\n";
    f << "  \"spacedim\": " << AMREX_SPACEDIM << ",\n";
    f << "  \"problem\": \"" << cfg.problem << "\",\n";
    f << "  \"n_cell\": ";   arr([&](int d){ return geom.Domain().length(d); }, AMREX_SPACEDIM); f << ",\n";
    f << "  \"dx\": ";       arr([&](int d){ return geom.CellSize(d); },        AMREX_SPACEDIM); f << ",\n";
    f << "  \"c\": ";        arr([&](int d){ return cfg.adv_vel[d]; },          AMREX_SPACEDIM); f << ",\n";
    f << "  \"nu\": " << cfg.nu << ",\n";
    f << "  \"T\": " << cfg.T << ",\n";
    f << "  \"n_IF\": " << ifh.nGhostRequired() << ",\n";
    f << "  \"IF_eps\": " << cfg.IF_eps << ",\n";
    f << "  \"ic\": {\"type\": \"" << cfg.ic_type << "\", \"noise_amp\": " << cfg.noise_amp
      << ", \"seed_amp\": " << cfg.seed_amp << "},\n";
    f << "  \"phi0\": \"phi0.npy\",\n";
    f << "  \"runs\": [\n";
    for (int l = 0; l < dts.size(); ++l)
    {
        f << "    {\"level\": " << l << ", \"dt\": " << dts[l] << ", \"nsteps\": " << nsteps[l]
          << ", \"cfl\": " << cfg.cfl(dts[l]) << ", \"wall_s\": " << wall[l]
          << ", \"file\": \"phi_l" << l << ".npy\"}" << (l + 1 < dts.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
}

} // namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        const ProblemConfig cfg = ProblemConfig::fromParmParse();

        // ---- geometry: periodic everywhere, cubic cells ------------------------
        const amrex::Real dx = cfg.dx();
        amrex::Box domain(amrex::IntVect(0), cfg.n_cell - 1);
        amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                          {AMREX_D_DECL(cfg.n_cell[0] * dx,
                                        cfg.n_cell[1] * dx,
                                        cfg.n_cell[2] * dx)});
        amrex::Array<int, AMREX_SPACEDIM> is_periodic {AMREX_D_DECL(1, 1, 1)};
        amrex::Geometry geom(domain, rb, amrex::CoordSys::cartesian, is_periodic);

        amrex::BoxArray ba(domain);
        ba.maxSize(cfg.max_grid_size);
        amrex::DistributionMapping dm(ba);

        // ---- dt sweep ---------------------------------------------------------------
        amrex::Vector<amrex::Real> dts(cfg.n_levels);
        amrex::Vector<int>         nsteps(cfg.n_levels);
        amrex::Vector<amrex::Real> wall(cfg.n_levels, 0.0);
        for (int l = 0; l < cfg.n_levels; ++l) { dts[l] = cfg.dt(l); nsteps[l] = cfg.nSteps(l); }

        // sized with the largest dt, so n_IF (and the ghost width) covers every level
        ScalarStepper stepper(geom, ba, dm, cfg, dts[0]);
        const int ng = stepper.nGhostRequired();

        if (cfg.verbose > 0)
        {
            amrex::Print() << "IF_HERK_Validate: " << cfg.problem << ", "
                           << AMREX_SPACEDIM << "D, n_cell = " << cfg.n_cell
                           << ", boxes = " << ba.size() << ", ghosts = " << ng << "\n";
            stepper.integrator().printSummary();
        }

        amrex::MultiFab phi0(ba, dm, 1, ng);
        amrex::MultiFab phi (ba, dm, 1, ng);
        fillInitialCondition(phi0, geom, cfg);

        amrex::UtilCreateDirectory(cfg.out_dir, 0755);
        writeNpy(phi0, geom, cfg.out_dir + "/phi0.npy");
        if (cfg.write_plotfile)
        {
            amrex::WriteSingleLevelPlotfile(cfg.out_dir + "/plt_ic", phi0, {"phi"}, geom, 0.0, 0);
        }

        for (int l = 0; l < cfg.n_levels; ++l)
        {
            stepper.setTimeStep(dts[l]);
            amrex::MultiFab::Copy(phi, phi0, 0, 0, 1, ng);

            amrex::ParallelDescriptor::Barrier();
            const amrex::Real t0 = amrex::second();
            for (int n = 0; n < nsteps[l]; ++n) { stepper.advance(phi); }
            amrex::Gpu::streamSynchronize();
            wall[l] = amrex::second() - t0;
            amrex::ParallelDescriptor::ReduceRealMax(wall[l]);

            writeNpy(phi, geom, cfg.out_dir + "/phi_l" + std::to_string(l) + ".npy");
            if (cfg.write_plotfile)
            {
                amrex::WriteSingleLevelPlotfile(cfg.out_dir + "/plt_l" + std::to_string(l),
                                                phi, {"phi"}, geom, cfg.T, nsteps[l]);
            }

            if (cfg.verbose > 0)
            {
                amrex::Print() << std::scientific << std::setprecision(4)
                               << "  level " << l << ": dt = " << dts[l]
                               << ", CFL = " << cfg.cfl(dts[l])
                               << ", steps = " << nsteps[l]
                               << ", max|phi(T)| = " << phi.norm0(0, 0, false)
                               << ", wall = " << wall[l] << " s\n";
            }
        }

        writeRunInfo(cfg, geom, stepper.integrator(), dts, nsteps, wall);
        amrex::Print() << "Wrote " << cfg.out_dir << "/ ; run Tools/reference.py "
                       << cfg.out_dir << " for errors and orders\n";
    }
    amrex::Finalize();
}
