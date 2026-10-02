#ifndef VECTORSTEPPER_H_
#define VECTORSTEPPER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>

#include <IFHERK.H>
#include <ProblemConfig.H>

// ---------------------------------------------------------------------------
// VectorStepper: the IF-HERK stage loop for a face-centred vector field
// (component d on d-faces), with no constraint. This is
// ProjectionWorkspace::advanceTimeStep with the Poisson solve and G dhat
// removed, and the same registers as ScalarStepper, one MultiFab per
// component:
//
//   q      : the aged initial state  (q^1 = U_k,  q^i = H^{i-1} q^{i-1})
//   w[j]   : w^{.,j}, 1-based; slot i holds g^i until the end of stage i
//   r      : r^i = q^i + dt sum_{j<i} a~_{i,j} w^{i,j} + g^i
//   stage  : U^i = H^i r^i
//   buff   : sweep scratch handed to IFHERK::applyIF
//   ke     : cell-centred 1/2 |U|^2, recomputed for every g^i
//
// Built in 2D and 3D only (the NSE stencils assume at least 2D).
// ---------------------------------------------------------------------------

class VectorStepper
{
public:
    using VecMF = amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>;

    // dt: the LARGEST dt that will be used (it sizes n_IF when automatic)
    VectorStepper (const amrex::Geometry& geom, const amrex::BoxArray& ba,
                   const amrex::DistributionMapping& dm, const ProblemConfig& cfg,
                   amrex::Real dt);

    // component d on d-faces of the cell BoxArray ba
    static void defineFaceVector (VecMF& v, const amrex::BoxArray& ba,
                                  const amrex::DistributionMapping& dm, int ngrow);

    int  nGhostRequired () const { return n_ghost; }

    void setTimeStep (amrex::Real dt_in);

    // one step, in place; every component needs nGhostRequired() ghosts
    void advance (VecMF& U);

    const IFHERK& integrator () const { return ifherk; }

    // Public because they launch device lambdas (CUDA extended-lambda rule).
    void computeKineticEnergy (const VecMF& st);      // st -> ke (+ ghosts)
    void computeGStage (const VecMF& st, int i);       // g^i = -a~_{i,i} dt N(U^{i-1}) -> w[i]

private:
    void fillBoundary (VecMF& v) const;

    amrex::Geometry geom;
    IFHERK ifherk;
    int n_ghost = 1;

    VecMF stage, q, r, buff;
    amrex::Vector<VecMF> w;    // 1-based; w[0] unused
    amrex::MultiFab ke;
};

#endif
