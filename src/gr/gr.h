/* gr/gr.h -- general-relativistic hydrodynamics: compile-time constants, data structures and host prototypes.
 *
 * GR hydro: Lupi 2023, MNRAS, 519, 1115.
 *
 * This header is included from declarations/allvars.h right after the math types (Vec3, Mat3, SymmetricTensor2), which
 * the GR structures are built on; it holds no function bodies. The device-callable physics (metric, tensor algebra, EOS,
 * conserved variables, con2prim) lives in gr/gr_functions.h and gr/gr_con2prim_functions.h, included from
 * declarations/cell_data.h; the integration of the conserved variables in gr/gr_kick_functions.h (with the source terms in
 * gr/gr_source_functions.h); the Riemann solvers in gr/gr_riemann_functions.h. The host-only routines are in gr/gr_utils.cc.
 *
 * Port to the Kokkos code of the GR module of the C/C++ GIZMO branches (A. Lupi, G. Fedrigo). */
#ifndef GR_GR_H
#define GR_GR_H

#include "../math_types/vec3.h"
#include "../math_types/mat3.h"
#include "../math_types/symmetric_tensor2.h"

/* kick/drift of the conserved variables: implicit solver tolerance and iteration cap (otherwise a predictor-corrector) */
#define IMPLICIT_SOLVER (1e-8)
#define IMPLICIT_MAX_NITER 100
#define GR_EPS_TOLERANCE (1e-8) /* finite-difference step of the metric derivatives */

#define PARTICLE (25487837)           /* ID of one particle for which the GR solver prints diagnostics (debug) */
#define GR_GAMMA_LIMITER (1000)       /* maximum Lorentz factor enforced inside GAMMA_LIMITER_RADIUS */
#define GAMMA_LIMITER_RADIUS (0.0)    /* radius (from the origin) inside which the Lorentz factor is limited */
/* time step of the gas (core/timestep.cc), on top of the Courant and acceleration criteria (not in the original code, whose 10% limit on the change of tau is removed) */
#define GR_TIMESTEP_ENERGY_FRAC (0.02) /* the kinetic-energy error of a first-order sub-step, (1/2) rho0 h* W^3 (|a| dt)^2, is at most this fraction of rho0 u */


/* --- 3+1 metric coefficients at one point --- */
struct gr_metric
{
    SymmetricTensor2<double> gamma;   /*!< covariant spatial metric gamma_ij */
    SymmetricTensor2<double> ugamma;  /*!< contravariant spatial metric gamma^ij */
    Vec3<double> beta;                /*!< contravariant shift beta^i */
    double alpha;                     /*!< lapse */
    double gdet;                      /*!< sqrt of the spatial metric determinant */
    int limiter;                      /*!< 1 where the Lorentz factor is limited to GR_GAMMA_LIMITER (inside GAMMA_LIMITER_RADIUS): set with the metric, so that no caller needs the position */
};

/* --- a symmetric 4-tensor in the 3+1 split (the covariant 4-metric g_{mu nu}, its derivatives, the stress-energy tensor T^{mu nu}) --- */
struct gr_sym4
{
    double tt;                        /*!< tt component */
    Vec3<double> ti;                  /*!< ti components */
    SymmetricTensor2<double> ij;      /*!< ij components (for the metric: the covariant spatial metric) */
};

/* --- Conserved and primitive variables of one cell (the conserved ones are densitized, i.e. include sqrt(gamma)) --- */
struct gr_conservatives
{
    double D;                         /*!< sqrt(gamma) rho0 W */
    Vec3<double> S;                   /*!< sqrt(gamma) S_i (covariant momentum density) */
    double tau;                       /*!< sqrt(gamma) (energy density - D) */
    double p;                         /*!< total (gas + magnetic) pressure, carried along for the Riemann solvers */
};

struct gr_primitives
{
    double rho0;                      /*!< rest-mass density */
    Vec3<double> v;                   /*!< velocity measured by the Eulerian observer (contravariant) */
    double u;                         /*!< specific internal energy */
    double p;                         /*!< gas pressure */
    int u_kept;                       /*!< set by gr_get_primitive_variables: 1 if no admissible u exists (tau < tau_min) and the previous u was kept */
};

/* scalars shared by the con2prim Newton-Raphson iteration */
struct kernel_NR
{
    double Bsq, QdotBsq, Qtsq, Qdotn, D, half_Bsq;
};

#if (GR_METRIC>0)
#define EHradius(M,a) ((M)*(1+sqrt(1-(a)[0]*(a)[0]-(a)[1]*(a)[1]-(a)[2]*(a)[2]))) /* event-horizon radius of a Kerr hole of mass M and spin vector a (dimensionless) */
#endif

/* --- host routines (gr/gr_utils.cc) --- */
#if (GR_METRIC<0)
void gr_load_metric_table(void); /* reads All.GR_MetricTableFile into All.GR_MetricTable (begrun) */
#endif
#ifdef PARTICLE_EXCISION
/* apply_excision() (gravity/analytic_gravity.h, called once per step) uses check_boundary() to remove the cells inside GR_EXCISION_RADIUS horizon radii */
void check_boundary(int target);
void smooth_conservatives_inside_horizon(int i, double r, double r_EH, double M);
#endif
void gr_con2prim_report_failure(const char *where, int code, const struct gr_conservatives *cons, const struct gr_primitives *prims, const struct gr_metric *gr); /* prints the state of a failed inversion (host) */

/* --- the integration of the conserved variables is in gr/gr_kick_functions.h (device-callable: gr_kick_impl, gr_drift_impl, gr_predict_impl) --- */
void compute_initial_accelerations(void); /* host (gr/gr_utils.cc): the first kick of every cell, to seed HydroAccel/DtInternalEnergy from the initial fluxes */

#endif /* GR_GR_H */
