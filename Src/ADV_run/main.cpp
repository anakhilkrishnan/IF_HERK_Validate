#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_Print.H>

#include <ProblemConfig.H>
#include <InitialCondition.H>
#include <ScalarStepper.H>
#include <FieldIO.H>
#if AMREX_SPACEDIM >= 2
#include <VectorStepper.H>
#endif

#include <fstream>
#include <iomanip>
#include <string>

// ---------------------------------------------------------------------------
// Convergence sweep for the scalar and vector tests.
//
// For each dt_l = dt_max / 2^l: reset the field to its initial condition,
// take T/dt_l steps, write the result. With time.ref_refine = m > 0, one more
// run at dt_min / 2^m is written as the self-convergence reference.
//
//   <out.dir>/<f>0.npy                initial field, f = phi | u, v, w
//   <out.dir>/<f>_l<l>.npy            l = 0 .. n_levels-1
//   <out.dir>/<f>_ref.npy             optional reference run
//   <out.dir>/run_info.json
//
// The Python tools read run_info.json for everything else.
// ---------------------------------------------------------------------------

namespace {

struct LevelRun
{
    int         level;
    amrex::Real dt;
    int         nsteps;
    std::string tag;          // "l<l>" or "ref"
    amrex::Real cfl  = 0.0;
    amrex::Real wall = 0.0;
};

amrex::Vector<LevelRun> makeLevels (const ProblemConfig& cfg)
{
    amrex::Vector<LevelRun> lv;
    for (int l = 0; l < cfg.n_levels; ++l) { lv.push_back({l, cfg.dt(l), cfg.nSteps(l), "l" + std::to_string(l)}); }
    if (cfg.ref_refine > 0)
    {
        const int L = cfg.refLevel();
        lv.push_back({L, cfg.dt(L), cfg.nSteps(L), "ref"});
    }
    return lv;
}

void writeRunInfo (const ProblemConfig& cfg, const amrex::Geometry& geom, const IFHERK& ifh,
                   const amrex::Vector<std::string>& fields, const amrex::Vector<LevelRun>& lv,
                   amrex::Real cell_re)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream f(cfg.out_dir + "/run_info.json");
    f << std::setprecision(17);
    auto arr = [&f](auto get, int n) {
        f << "[";
        for (int d = 0; d < n; ++d) { f << (d ? ", " : "") << get(d); }
        f << "]";
    };
    auto files = [&f, &fields](const std::string& suffix) {
        f << "{";
        for (int c = 0; c < fields.size(); ++c)
        { f << (c ? ", " : "") << "\"" << fields[c] << "\": \"" << fields[c] << suffix << ".npy\""; }
        f << "}";
    };
    auto run = [&](const LevelRun& r) {
        f << "{\"level\": " << r.level << ", \"dt\": " << r.dt << ", \"nsteps\": " << r.nsteps
          << ", \"cfl\": " << r.cfl << ", \"wall_s\": " << r.wall << ", \"files\": ";
        files("_" + r.tag);
        f << "}";
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
    f << "  \"cell_re\": " << cell_re << ",\n";
    f << "  \"ic\": {\"type\": \"" << cfg.ic_type << "\", \"mean\": " << cfg.ic_mean
      << ", \"mean_vec\": ";  arr([&](int d){ return cfg.ic_mean_vec[d]; }, AMREX_SPACEDIM);
    f << ", \"active\": ";    arr([&](int d){ return cfg.ic_active[d]; },   AMREX_SPACEDIM);
    f << ", \"stagger\": ";   arr([&](int d){ return cfg.ic_stagger[d]; },  AMREX_SPACEDIM);
    f << ", \"noise_amp\": " << cfg.noise_amp << ", \"seed_amp\": " << cfg.seed_amp << "},\n";
    f << "  \"fields\": [";
    for (int c = 0; c < fields.size(); ++c) { f << (c ? ", " : "") << "\"" << fields[c] << "\""; }
    f << "],\n";
    f << "  \"init\": ";  files("0");  f << ",\n";
    f << "  \"runs\": [\n";
    int n_main = 0;
    for (const auto& r : lv) { if (r.tag != "ref") { ++n_main; } }
    for (int l = 0; l < n_main; ++l)
    {
        f << "    ";  run(lv[l]);  f << (l + 1 < n_main ? "," : "") << "\n";
    }
    f << "  ],\n";
    f << "  \"reference_run\": ";
    if (lv.size() > n_main) { run(lv.back()); } else { f << "null"; }
    f << "\n}\n";
}

amrex::Geometry makeGeometry (const ProblemConfig& cfg)
{
    // periodic everywhere, cubic cells
    const amrex::Real dx = cfg.dx();
    amrex::Box domain(amrex::IntVect(0), cfg.n_cell - 1);
    amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                      {AMREX_D_DECL(cfg.n_cell[0] * dx,
                                    cfg.n_cell[1] * dx,
                                    cfg.n_cell[2] * dx)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic {AMREX_D_DECL(1, 1, 1)};
    return amrex::Geometry(domain, rb, amrex::CoordSys::cartesian, is_periodic);
}

void printHeader (const ProblemConfig& cfg, const amrex::BoxArray& ba, int ng,
                  const IFHERK& ifh, amrex::Real cell_re)
{
    amrex::Print() << "IF_HERK_Validate: " << cfg.problem << ", " << AMREX_SPACEDIM
                   << "D, n_cell = " << cfg.n_cell << ", boxes = " << ba.size()
                   << ", ghosts = " << ng << ", ic = " << cfg.ic_type
                   << ", cell Re = " << cell_re << "\n";
    if (cell_re > 2.0)
    {
        amrex::Print() << "  WARNING: cell Re > 2 -- central differencing will produce grid-scale\n"
                       << "  wiggles. The temporal order test is still valid (the reference has them\n"
                       << "  too), but the solution is not physically meaningful.\n";
    }
    ifh.printSummary();
}

void printLevel (const LevelRun& r, amrex::Real max_end)
{
    amrex::Print() << std::scientific << std::setprecision(4)
                   << "  " << (r.tag == "ref" ? "ref    " : "level " + std::to_string(r.level))
                   << ": dt = " << r.dt << ", CFL = " << r.cfl << ", steps = " << r.nsteps
                   << ", max|field(T)| = " << max_end << ", wall = " << r.wall << " s\n";
}

// ---- scalar problems: linear advection, scalar Burgers ----------------------

void runScalar (const ProblemConfig& cfg, const amrex::Geometry& geom,
                const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    auto lv = makeLevels(cfg);

    // sized with the largest dt, so n_IF (and the ghost width) covers every level
    ScalarStepper stepper(geom, ba, dm, cfg, lv[0].dt);
    const int ng = stepper.nGhostRequired();

    amrex::MultiFab phi0(ba, dm, 1, ng);
    amrex::MultiFab phi (ba, dm, 1, ng);
    fillInitialCondition(phi0, geom, cfg);

    // advection speed for CFL and cell Re: c itself, or c scaled by the field
    // (viscous Burgers never exceeds its initial maximum)
    const amrex::Real phi0_max = phi0.norm0(0, 0, false);
    const bool burgers = (cfg.problemType() == ProblemType::ScalarBurgers);
    amrex::Real c_max = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { c_max = std::max(c_max, std::abs(cfg.adv_vel[d])); }
    const amrex::Real cell_re = (cfg.nu > 0.0) ? cfg.cellRe(burgers ? c_max * phi0_max : c_max) : 0.0;

    if (cfg.verbose > 0) { printHeader(cfg, ba, ng, stepper.integrator(), cell_re); }

    amrex::UtilCreateDirectory(cfg.out_dir, 0755);
    writeNpy(phi0, geom, cfg.out_dir + "/phi0.npy");
    if (cfg.write_plotfile)
    { amrex::WriteSingleLevelPlotfile(cfg.out_dir + "/plt_ic", phi0, {"phi"}, geom, 0.0, 0); }

    for (auto& r : lv)
    {
        r.cfl = burgers ? cfg.cfl(r.dt, phi0_max) : cfg.cfl(r.dt);

        stepper.setTimeStep(r.dt);
        amrex::MultiFab::Copy(phi, phi0, 0, 0, 1, ng);

        amrex::ParallelDescriptor::Barrier();
        const amrex::Real t0 = amrex::second();
        for (int n = 0; n < r.nsteps; ++n) { stepper.advance(phi); }
        amrex::Gpu::streamSynchronize();
        r.wall = amrex::second() - t0;
        amrex::ParallelDescriptor::ReduceRealMax(r.wall);

        writeNpy(phi, geom, cfg.out_dir + "/phi_" + r.tag + ".npy");
        if (cfg.write_plotfile)
        { amrex::WriteSingleLevelPlotfile(cfg.out_dir + "/plt_" + r.tag, phi, {"phi"}, geom, cfg.T, r.nsteps); }

        if (cfg.verbose > 0) { printLevel(r, phi.norm0(0, 0, false)); }
    }

    writeRunInfo(cfg, geom, stepper.integrator(), {"phi"}, lv, cell_re);
}

// ---- vector Burgers ------------------------------------------------------------

#if AMREX_SPACEDIM >= 2
void writeVectorPlotfile (const std::string& name, const VectorStepper::VecMF& U,
                          const amrex::Geometry& geom, const amrex::BoxArray& ba,
                          const amrex::DistributionMapping& dm, amrex::Real time, int step)
{
    // plotfiles need cell-centred data: average each component's two faces
    amrex::MultiFab cc(ba, dm, AMREX_SPACEDIM, 0);
    amrex::average_face_to_cellcenter(cc, 0, amrex::GetArrOfConstPtrs(U));
    amrex::WriteSingleLevelPlotfile(name, cc, {AMREX_D_DECL("u", "v", "w")}, geom, time, step);
}

void runVector (const ProblemConfig& cfg, const amrex::Geometry& geom,
                const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
{
    auto lv = makeLevels(cfg);
    const amrex::Vector<std::string> fields {AMREX_D_DECL("u", "v", "w")};

    VectorStepper stepper(geom, ba, dm, cfg, lv[0].dt);
    const int ng = stepper.nGhostRequired();

    VectorStepper::VecMF U0, U;
    VectorStepper::defineFaceVector(U0, ba, dm, ng);
    VectorStepper::defineFaceVector(U,  ba, dm, ng);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { fillInitialCondition(U0[d], geom, cfg, d); }

    // CFL: sum_d max|u_d| dt / dx (the per-direction speeds add, as for c);
    // cell Re from the largest component
    amrex::Real speed = 0.0, u_max = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const amrex::Real m = U0[d].norm0(0, 0, false);
        speed += m;
        u_max  = std::max(u_max, m);
    }
    const amrex::Real cell_re = (cfg.nu > 0.0) ? cfg.cellRe(u_max) : 0.0;

    if (cfg.verbose > 0) { printHeader(cfg, ba, ng, stepper.integrator(), cell_re); }

    amrex::UtilCreateDirectory(cfg.out_dir, 0755);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { writeNpy(U0[d], geom, cfg.out_dir + "/" + fields[d] + "0.npy"); }
    if (cfg.write_plotfile) { writeVectorPlotfile(cfg.out_dir + "/plt_ic", U0, geom, ba, dm, 0.0, 0); }

    for (auto& r : lv)
    {
        r.cfl = speed * r.dt / cfg.dx();

        stepper.setTimeStep(r.dt);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { amrex::MultiFab::Copy(U[d], U0[d], 0, 0, 1, ng); }

        amrex::ParallelDescriptor::Barrier();
        const amrex::Real t0 = amrex::second();
        for (int n = 0; n < r.nsteps; ++n) { stepper.advance(U); }
        amrex::Gpu::streamSynchronize();
        r.wall = amrex::second() - t0;
        amrex::ParallelDescriptor::ReduceRealMax(r.wall);

        amrex::Real max_end = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            writeNpy(U[d], geom, cfg.out_dir + "/" + fields[d] + "_" + r.tag + ".npy");
            max_end = std::max(max_end, U[d].norm0(0, 0, false));
        }
        if (cfg.write_plotfile)
        { writeVectorPlotfile(cfg.out_dir + "/plt_" + r.tag, U, geom, ba, dm, cfg.T, r.nsteps); }

        if (cfg.verbose > 0) { printLevel(r, max_end); }
    }

    writeRunInfo(cfg, geom, stepper.integrator(), fields, lv, cell_re);
}
#endif

} // namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        const ProblemConfig cfg = ProblemConfig::fromParmParse();

        const amrex::Geometry geom = makeGeometry(cfg);
        amrex::BoxArray ba(geom.Domain());
        ba.maxSize(cfg.max_grid_size);
        amrex::DistributionMapping dm(ba);

        if (cfg.isVector())
        {
#if AMREX_SPACEDIM >= 2
            runVector(cfg, geom, ba, dm);
#endif
        }
        else
        {
            runScalar(cfg, geom, ba, dm);
        }

        amrex::Print() << "Wrote " << cfg.out_dir << "/ ; run Tools/reference.py "
                       << cfg.out_dir << " for errors and orders\n";
    }
    amrex::Finalize();
}
