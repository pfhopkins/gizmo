/* ags_functions.h — GPU-callable AGS helpers (particle size/volume) and
 * AGS_ATOMIC_{ADD,STORE} dispatch macros for j-side writes in the
 * AGSForce GPU port.
 *
 * Get_Particle_Size_AGS / get_particle_volume_ags used to be host-only
 * with implicit reliance on the global `P`. They are now KOKKOS_INLINE_FUNCTION
 * and take P explicitly so both the CPU tree-walk and the GPU parallel_for
 * lambda can call them uniformly. Inside a GPU kernel the caller passes the
 * SharedSpace P mirror; inside CPU code the caller passes the global P.
 *
 * AGS_ATOMIC_{ADD,STORE} selects backend atomics at compile time. On the GPU
 * path (Kokkos+GPU) we use Kokkos::atomic_{add,store}. On the CPU
 * tree-walk path there is no macro that captures `#pragma omp atomic` (pragmas
 * can't live inside macros), so we provide a plain assignment and the CPU
 * caller is expected to wrap the statement with `#pragma omp atomic` directly.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef AGS_FUNCTIONS_H
#define AGS_FUNCTIONS_H

#include "../declarations/allvars.h"

/* KOKKOS_INLINE_FUNCTION falls back to `inline` outside GPU TUs. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif


#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE

/* Effective particle size based on AGS_KernelRadius and NumNgb. */
KOKKOS_INLINE_FUNCTION
double Get_Particle_Size_AGS_P(int i, const struct particle_data *P_arr)
{
#if (NUMDIMS == 1)
    return 2.00000 * P_arr[i].AGS_KernelRadius / P_arr[i].NumNgb;
#endif
#if (NUMDIMS == 2)
    return 1.77245 * P_arr[i].AGS_KernelRadius / P_arr[i].NumNgb;
#endif
#if (NUMDIMS == 3)
    return 1.61199 * P_arr[i].AGS_KernelRadius / P_arr[i].NumNgb;
#endif
}


/* Particle volume = L^NUMDIMS where L = Get_Particle_Size_AGS. */
KOKKOS_INLINE_FUNCTION
double get_particle_volume_ags_P(int j, const struct particle_data *P_arr)
{
    double L_j = Get_Particle_Size_AGS_P(j, P_arr);
#if (NUMDIMS==1)
    return L_j;
#elif (NUMDIMS==2)
    return L_j*L_j;
#else
    return L_j*L_j*L_j;
#endif
}

/* Whether ags_density must solve for this particle's KernelRadius.  Reads only. */
KOKKOS_INLINE_FUNCTION
int ags_density_solves_for_P(int i, const struct particle_data *P_arr)
{
    int default_to_return = 0; // default to not being active - needs to be pro-actively 'activated' by some physics
#ifdef ADAPTIVE_GRAVSOFT_FORALL
    default_to_return = 1;
    if(!((1 << P_arr[i].Type) & (ADAPTIVE_GRAVSOFT_FORALL))) /* particle is NOT one of the designated 'adaptive' types */
    {
        default_to_return = 0;
    } else {default_to_return = 1;} /* particle is AGS-active */
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    if(P_arr[i].Type==0)
    {
        default_to_return = 0; // don't actually need to do the loop //
    }
#endif
#ifdef DM_SIDM
    if((1 << P_arr[i].Type) & (DM_SIDM)) {default_to_return = 1;}
#endif
#if defined(CBE_INTEGRATOR)
    if(CBE_INTEGRATOR_DOES_TYPE(P_arr[i].Type)) {default_to_return = 1;}
#elif defined(DM_FUZZY)
    if(P_arr[i].Type == 1) {default_to_return = 1;}
#endif
    if(P_arr[i].TimeBin < 0) {default_to_return = 0;} /* check our 'marker' for particles which have finished iterating to an KernelRadius solution (if they have, dont do them again) */
    return default_to_return;
}

/* ags_density_solves_for_P, which also sets AGS_KernelRadius and AGS_zeta for particles that do not
 * iterate -- a safety default, idempotent, and touching only this particle's own fields. */
KOKKOS_INLINE_FUNCTION
int ags_density_isactive_P(int i, struct particle_data *P_arr)
{
#ifdef ADAPTIVE_GRAVSOFT_FORALL
    if(!((1 << P_arr[i].Type) & (ADAPTIVE_GRAVSOFT_FORALL))) /* particle is NOT one of the designated 'adaptive' types */
    {
        P_arr[i].AGS_KernelRadius = All.ForceSoftening[P_arr[i].Type];
        P_arr[i].AGS_zeta = 0;
    }
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    if(P_arr[i].Type==0) {P_arr[i].AGS_KernelRadius = P_arr[i].KernelRadius;} // gas sees gas, these are identical
#endif
    return ags_density_solves_for_P(i, P_arr);
}


/* maximum allowed softening */
KOKKOS_INLINE_FUNCTION
double ags_return_maxsoft_P(int i, const struct particle_data *P_arr)
{
    double maxsoft = All.MaxKernelRadius; // user-specified maximum: nothing is allowed to exceed this
#ifdef PMGRID /* Maximum allowed gravitational softening when using the TreePM method. The quantity is given in units of the scale used for the force split (PM_ASMTH) */
    maxsoft = DMIN(maxsoft, 1e3 * 0.5 * All.Asmth[0]); /* no more than 1/2 the size of the largest PM cell, times a 'safety factor' which can be pretty big */
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32) && defined(SINK_PARTICLES) && !defined(SINGLE_STAR_SINK_DYNAMICS)
    /* MaxAccretionRadius is defined in params.txt in PHYSICAL units. Enforce the
     * recommended-policy invariant "nothing exceeds MaxKernelRadius" via DMIN
     * (never > All.MaxKernelRadius, already the line-1 seed): keeps the Mode-B
     * per-type node band (capped at MaxKernelRadius) a valid upper bound on this
     * sink's AGS leaf reach. No effect where SinkMaxAccretionRadius < MaxKernelRadius
     * (the recommended + every-tested config). */
    if(P_arr[i].Type == 5) {maxsoft = DMIN(maxsoft, All.SinkMaxAccretionRadius / All.cf_atime);}
#endif
    return maxsoft;
}


/* minimum allowed softening */
KOKKOS_INLINE_FUNCTION
double ags_return_minsoft_P(int i, const struct particle_data *P_arr)
{
    double minsoft = All.ForceSoftening[P_arr[i].Type]; // this is the user-specified minimum
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
    minsoft = DMIN(All.MinKernelRadius, minsoft);
#endif
    return minsoft;
}

/* Fuzzy-DM drift/kick. Lives here rather than in sidm/dm_fuzzy_loop.h because that
 * header includes <Kokkos_Core.hpp> unconditionally and sidm/dm_fuzzy.cc is a host TU;
 * pulling Kokkos into a host TU breaks the CUDA build while compiling fine on the
 * OpenMP backend. Everything below touches only this particle's AGS_* fields.
 * DM_FUZZY implies AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE, so the enclosing gate holds. */
#ifdef DM_FUZZY
KOKKOS_INLINE_FUNCTION
void do_dm_fuzzy_drift_kick_P(int i, double dt, int mode, struct particle_data *pp)
{
    if(mode==0)
    {
        // calculate various energies: quantum potential QP0, 'stored' numerical pressure NQ0, kinetic energy KE0
        double dNQ=pp[i].AGS_Dt_Numerical_QuantumPotential*dt, NQ0=pp[i].AGS_Numerical_QuantumPotential, NQ1=NQ0+dNQ, KE0=0.5*pp[i].Mass*pp[i].Vel.norm_sq()*All.cf_a2inv;
        double f00 = 0.5 * All.ScalarField_hbar_over_mass; // this encodes the coefficient with the mass of the particle: units vel*L = hbar / particle_mass
        double d2rho = pp[i].AGS_Gradients2_Density[0][0] + pp[i].AGS_Gradients2_Density[1][1] + pp[i].AGS_Gradients2_Density[2][2]; // laplacian
        double drho2 = pp[i].AGS_Gradients_Density.norm_sq();
        double QP0 = (f00*f00 / pp[i].AGS_Density) * (d2rho - 0.5*drho2/pp[i].AGS_Density); // quantum 'potential'
        NQ1 = DMAX(0,DMAX(NQ1,0.1*NQ0)); NQ1 = DMIN(NQ1,1.1*DMAX(DMAX(KE0+NQ0,fabs(QP0)),KE0+NQ0+QP0)); // limit kick to not produce unphysical energy over-or-under-shoot
        pp[i].AGS_Numerical_QuantumPotential = NQ1;
    }

#if (DM_FUZZY > 0) /* if using direct-wavefunction integration methods */
    double vol_inv = pp[i].AGS_Density / pp[i].Mass;
    if(mode == 0)
    {
        //double psimag_mass_old = (pp[i].AGS_Psi_Re*pp[i].AGS_Psi_Re + pp[i].AGS_Psi_Im*pp[i].AGS_Psi_Im) * vol_inv;
        pp[i].AGS_Psi_Re += pp[i].AGS_Dt_Psi_Re * dt;
        pp[i].AGS_Psi_Im += pp[i].AGS_Dt_Psi_Im * dt;
        double mass_old = pp[i].Mass, dmass = pp[i].AGS_Dt_Psi_Mass * dt, mass_new = mass_old + dmass;
        dmass = DMIN(DMAX(dmass,-0.5*mass_old),0.5*mass_old);
        mass_new = mass_old + dmass;
        double psimag_mass_new = (pp[i].AGS_Psi_Re*pp[i].AGS_Psi_Re + pp[i].AGS_Psi_Im*pp[i].AGS_Psi_Im) * vol_inv;
#if (DM_FUZZY == 2)
        mass_new = psimag_mass_new; /* uses direct [NON-MASS-CONSERVING] integration of psi field */
#endif
        double psi_corr_fac = sqrt(mass_new / (MIN_REAL_NUMBER + psimag_mass_new));
        pp[i].Mass = mass_new; pp[i].AGS_Psi_Re *= psi_corr_fac; pp[i].AGS_Psi_Im *= psi_corr_fac;

        pp[i].AGS_Density = pp[i].Mass * vol_inv;
        pp[i].AGS_Psi_Re_Pred = pp[i].AGS_Psi_Re;
        pp[i].AGS_Psi_Im_Pred = pp[i].AGS_Psi_Im;
    } else {
        /* in drift mode, AGS_Density should automatically be drifted already by the predictor step, but not the other quantities here */
        pp[i].AGS_Psi_Re_Pred += pp[i].AGS_Dt_Psi_Re * dt;
        pp[i].AGS_Psi_Im_Pred += pp[i].AGS_Dt_Psi_Im * dt;
        pp[i].AGS_Density *= 1. + DMIN(DMAX(pp[i].AGS_Dt_Psi_Mass*dt/pp[i].Mass,-0.5),0.5);
    }
#endif
}
#endif /* DM_FUZZY */

#endif /* AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE */

/* Whether a drift advances this particle's search length by its own Particle_DivVel (drift_particle_impl): a gas
 * cell's KernelRadius, a grain's, and the AGS_KernelRadius of a particle ags_density solves for.  A tree node's
 * divVmax, which grows every radius band the node holds, is the largest such member's growth rate (its divergence
 * times its dilation factor, particle_radius_growth_rate), so it must come from exactly these.  Reads only. */
KOKKOS_INLINE_FUNCTION
int particle_radius_drifts_with_divergence_P(int i, const struct particle_data *P_arr)
{
    if(P_arr[i].Type == 0) {return 1;}
#ifdef GRAIN_FLUID
    if((1 << P_arr[i].Type) & (GRAIN_PTYPES)) {return 1;}
#endif
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    if(ags_density_solves_for_P(i, P_arr)) {return 1;}
#endif
    return 0;
}


#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
/* Result of inverting an AGS face moment matrix. cn_expansion is the signal
 * that drives neighbor-count expansion (a large sentinel when the raw matrix
 * is singular, because matrix_invert_ndims() reports CN=1 for a zero inverse);
 * cn_final + status are diagnostics (status: 0=clean, 1=regularized via
 * diagonal-loading, 2=could-not-condition / near-zero raw matrix, warned). */
struct AgsNvtInversion { double cn_expansion; double cn_final; int status; };
/* Invert the accumulated raw moment matrix into the symmetric, fully-populated
 * face operator used by the CBE/dm_fuzzy face construction and the DMGrad
 * gradient estimator. Single source of truth for AGS face NV_T inversion;
 * mirrors the hydro density path (DensitySpec::after_iter_body, hydro/density_loop.h) including the
 * diagonal-loading preconditioning loop.
 *
 * Inputs : sym6 = upper-triangle raw moment components {00,01,02,11,12,22}
 *          h    = AGS kernel size (AGS_KernelRadius), used to normalize the
 *                 matrix into dimensionless units so the regularization scale
 *                 matches the matrix scale (it divides out of the result).
 * Output : Tinv_out = full 3x3 inverse of the raw moment matrix.
 * Returns: cn_expansion (neighbor-expansion signal), cn_final + status (diag).
 *
 * Robustness: matrix_invert_ndims() returns Tinv=0 and
 * ConditionNumber=1 for an exactly-singular matrix (Mat3::invert sets inv=0 on
 * det==0). So acceptance CANNOT key off the condition number alone — we also
 * require a finite, non-zero inverse, and we report a large cn_expansion for
 * the singular case so the neighbor iteration expands (a singular matrix needs
 * MORE neighbors, the opposite of what CN=1 would imply). */
KOKKOS_INLINE_FUNCTION
AgsNvtInversion ags_invert_nvt_for_faces(const double sym6[6], double h, double Tinv_out[3][3])
{
    AgsNvtInversion result; result.cn_expansion = 1.0; result.cn_final = 1.0; result.status = 0;
    const double ConditionNumber_threshold = 10. * CONDITION_NUMBER_DANGER;

    /* Normalize to dimensionless units (divides out of the final inverse). */
    double dimensional_NV_T_normalizer = pow(h, 2 - NUMDIMS);
    double inv_norm = ((dimensional_NV_T_normalizer != 0) && !isnan(dimensional_NV_T_normalizer))
                      ? 1.0 / dimensional_NV_T_normalizer : 1.0;
    MyDouble NV_T[3][3], Tinv[3][3];
    NV_T[0][0] = sym6[0]*inv_norm; NV_T[0][1] = sym6[1]*inv_norm; NV_T[0][2] = sym6[2]*inv_norm;
    NV_T[1][0] = sym6[1]*inv_norm; NV_T[1][1] = sym6[3]*inv_norm; NV_T[1][2] = sym6[4]*inv_norm;
    NV_T[2][0] = sym6[2]*inv_norm; NV_T[2][1] = sym6[4]*inv_norm; NV_T[2][2] = sym6[5]*inv_norm;

    double trace_initial = NV_T[0][0] + NV_T[1][1] + NV_T[2][2];
    /* Absolute floor on the trace so the conditioning step makes progress even
     * for a (near-)zero raw matrix (no usable neighbors). */
    double trace_for_cond = DMAX(trace_initial, MIN_REAL_NUMBER);
    double conditioning_term_to_add = 1.05 * (trace_for_cond / NUMDIMS) / ConditionNumber_threshold;
    if(!(conditioning_term_to_add > 0)) {conditioning_term_to_add = MIN_REAL_NUMBER;}

    const int MAXIT = 50; int ok = 0, regularized = 0; int j,k;
    for(int it = 0; it < MAXIT; it++)
    {
        double ConditionNumber = matrix_invert_ndims(NV_T, Tinv);
        double frob_inv = 0; int finite_inv = 1;
        for(j=0;j<3;j++) {for(k=0;k<3;k++) {double e = Tinv[j][k]; if(!isfinite(e)) {finite_inv = 0;} frob_inv += e*e;}}
        int inversion_ok = finite_inv && (frob_inv > 0) && isfinite(ConditionNumber);
        if(it == 0) {result.cn_expansion = inversion_ok ? ConditionNumber : ConditionNumber_threshold;}
        if(inversion_ok && (ConditionNumber < ConditionNumber_threshold)) {
            result.cn_final = ConditionNumber;
            result.status = regularized ? 1 : 0;
            ok = 1;
            break;
        }
        regularized = 1;
        for(j=0;j<NUMDIMS;j++) {NV_T[j][j] += conditioning_term_to_add;}
        conditioning_term_to_add *= 1.2;
        if(!(conditioning_term_to_add > 0) || !isfinite(conditioning_term_to_add)) {conditioning_term_to_add = MIN_REAL_NUMBER;}
    }
    /* Un-normalize: stored = Tinv / normalizer = inverse of the raw moment
     * matrix. matrix_invert_ndims returns the FULL inverse, so all 9 entries
     * are populated (no upper-triangular-only state). */
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {Tinv_out[j][k] = Tinv[j][k] * inv_norm;}}
    if(!ok)
    {
        result.status = 2; result.cn_final = result.cn_expansion;
        PRINT_WARNING("AGS NV_T inversion could not be conditioned after %d diagonal-load steps "
                      "(raw trace=%g); storing best-effort regularized inverse. The CBE/DMGrad "
                      "faces for this particle are degraded -- check its neighbor count.",
                      MAXIT, trace_initial);
    }
    return result;
}
#endif /* AGS_FACE_CALCULATION_IS_ACTIVE */

/* Kokkos atomics inside a device lambda. */
#define AGS_ATOMIC_ADD(ptr, val)   Kokkos::atomic_add((ptr), (val))
#define AGS_ATOMIC_STORE(ptr, val) Kokkos::atomic_store((ptr), (val))

#endif /* AGS_FUNCTIONS_H */
