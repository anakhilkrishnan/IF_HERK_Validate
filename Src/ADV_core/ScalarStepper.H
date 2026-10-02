#ifndef SCALARSTEPPER_H_
#define SCALARSTEPPER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <string>

#include <IFHERK.H>
#include <ProblemConfig.H>

// ---------------------------------------------------------------------------
// ScalarStepper: the IF-HERK stage loop for a cell-centred scalar, with no
// constraint (the analogue of ProjectionWorkspace::advanceTimeStep with the
// Poisson solve and G dhat removed). Registers mirror the NSE code:
//
//   q      : the aged initial state  (q^1 = phi_k,  q^i = H^{i-1} q^{i-1})
//   w[j]   : w^{.,j}, 1-based; slot i holds g^i until the end of stage i
//   r      : r^i = q^i + dt sum_{j<i} a~_{i,j} w^{i,j} + g^i
//   stage  : phi^i = H^i r^i
//   buff   : sweep scratch handed to IFHERK::applyIF
// ---------------------------------------------------------------------------

class ScalarStepper
{
public:
    // dt: the LARGEST dt that will be used (it sizes n_IF when automatic)
    ScalarStepper (const amrex::Geometry& geom, const amrex::BoxArray& ba,
                   const amrex::DistributionMapping& dm, const ProblemConfig& cfg,
                   amrex::Real dt);

    // ghost cells phi must carry: max(IF half-width, advection stencil)
    int  nGhostRequired () const { return n_ghost; }

    void setTimeStep (amrex::Real dt_in);

    // one step, in place; phi needs nGhostRequired() ghosts
    void advance (amrex::MultiFab& phi);

    const IFHERK& integrator () const { return ifherk; }

    // g^i = -a~_{i,i} dt N(phi^{i-1})  ->  w[i]
    // Public because it launches a device lambda (CUDA extended-lambda rule).
    void computeGStage (const amrex::MultiFab& st, int i);

private:

    std::string problem;
    amrex::Geometry geom;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> adv_vel; // serves as direction selector for scalar burgers

    IFHERK ifherk;
    int n_ghost = 1;

    amrex::MultiFab stage, q, r, buff;
    amrex::Vector<amrex::MultiFab> w;   // 1-based; w[0] unused
};

#endif
