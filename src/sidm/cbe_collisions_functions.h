/* cbe_collisions_functions.h -- intra-cell CBE collision operators.
 *
 * Local (single-cell, no neighbors) collision operators applied to the per-cell
 * velocity basis functions, operator-split from the CBE transport/gravity solve
 * (see sec. "Collision Operators & Local Mixing" of the CBE method paper). The
 * full phase-space DF f_a(v) is carried in each cell, so the operator is applied
 * exactly within the cell -- no inter-cell communication, no super-particle
 * coarse-graining.
 *
 * Model dispatch is CBE_COLLISION_MODEL (declarations/precompiler_logic.h). The
 * one worked example is isotropic elastic hard-sphere scattering, relaxing the
 * local DF toward the cell Maxwell-Boltzmann fixed point via a BGK closure
 * (Bhatnagar-Gross-Krook 1954); users add other operators (velocity-dependent,
 * anisotropic, inelastic/annihilation) as additional dispatch branches.
 *
 * Frame note: the persistent per-basis moments are stored RELATIVE to the cell
 * bulk frame V=P[i].Vel (feedback_validation frame rule). The BGK update below is
 * frame-COVARIANT -- it uses only the stored relative momenta/second-moments
 * consistently (cell bulk from Sum_b p_b, per-basis from p_b, T_b), so it is
 * applied directly to the stored moments with NO frame conversion. Only the
 * collision RATE nu is converted to physical units.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef CBE_COLLISIONS_FUNCTIONS_H
#define CBE_COLLISIONS_FUNCTIONS_H

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#include "cbe_integrator_functions.h"   /* moment-slot accessors: cbe_basis_p_r/_w, cbe_basis_T_r/_w */

#if defined(CBE_INTEGRATOR) && defined(CBE_INTEGRATOR_COLLISIONS)

/* --------------------------------------------------------------------------
 * Isotropic elastic hard-sphere BGK relaxation toward the local Maxwellian.
 *
 * Given the cell moments (m_a, <v_a>, S^com_a) built from the bases, the BGK
 * closure relaxes f_a toward the isotropic Gaussian f_a^eq with variance
 * sigma_eq^2 = Tr(S^com)/D on the collision timescale 1/nu_a, integrated in
 * closed form over dt as a convex mixing with fraction eta = 1 - exp(-nu*dt).
 * The per-basis convex update (masses fixed) is
 *   <v_b>   -> (1-eta) <v_b> + eta <v_a>
 *   S_b     -> (1-eta) S_b + eta sigma_eq^2 I + eta(1-eta) dv_b (x) dv_b
 * with dv_b = <v_b> - <v_a>. The last term is the variance-of-the-mean of the
 * convex mixture; it makes the cell-summed (m_a, p_a, T_a) EXACTLY invariant
 * (mass/momentum/kinetic-energy conservation), while the anisotropic part of T
 * decays toward isotropy as physical elastic scattering requires.
 *
 * Inputs (all physical unless noted):
 *   moments[b][]        per-basis moment rows (MODIFIED in place; stored frame)
 *   rho_phys            physical mass density of the cell (m_a / V_a * cf_a3inv)
 *   cross_sec_per_mass  sigma/mu_p in code units (area per mass); 0 => no-op
 *   cf_atime            scale factor a (physical peculiar v = v_code / cf_atime)
 *   dt_phys             physical timestep
 * Returns 1 if the update was applied (eta>0), else 0.
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
int cbe_collision_bgk_isotropic(double moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
                                double rho_phys, double cross_sec_per_mass,
                                double cf_atime, double dt_phys)
{
    if(!(rho_phys > 0) || !(cross_sec_per_mass > 0) || !(dt_phys > 0) || !(cf_atime > 0)) return 0;

    /* cell moments (stored/relative frame): m_a, p_a, raw 2nd-moment T_a */
    double m_a = 0, p_a[3] = {0,0,0}, T_a[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for(int b = 0; b < CBE_INTEGRATOR_NBASIS; b++) {
        const double m_b = moments[b][0];
        if(!(m_b > 0)) continue;
        m_a += m_b;
        for(int k = 0; k < NUMDIMS; k++) p_a[k] += cbe_basis_p_r(moments[b], k);
        for(int a = 0; a < NUMDIMS; a++)
            for(int c = a; c < NUMDIMS; c++) { double t = cbe_basis_T_r(moments[b], a, c); T_a[a][c] += t; T_a[c][a] += (a==c)?0:t; }
    }
    if(!(m_a > 0)) return 0;
    const double inv_ma = 1.0 / m_a;
    double u_a[3] = {0,0,0};
    for(int k = 0; k < NUMDIMS; k++) u_a[k] = p_a[k] * inv_ma;

    /* cell central second-moment trace (code velocity^2): Tr(S^com) = Tr(T_a)/m_a - |u_a|^2 */
    double Scom_tr = 0;
    for(int k = 0; k < NUMDIMS; k++) Scom_tr += T_a[k][k] * inv_ma - u_a[k] * u_a[k];
    if(!(Scom_tr > 0)) return 0;                              /* cold cell: no thermal target, nu->0 */
    const double sigeq2 = Scom_tr / NUMDIMS;                  /* per-dim equilibrium dispersion (code) */

    /* collision rate nu = rho_phys * (sigma/mu) * <|v_rel|>, with the hard-sphere
     * Maxwellian mean relative speed <|v_rel|> = sqrt(16 kT/(pi mu)) = 4 sigma_eq/sqrt(pi).
     * sigma_eq is physical: sigma_eq_phys = sqrt(sigeq2)/cf_atime (v_phys = v_code/a). */
    const double sigeq_phys = sqrt(sigeq2) / cf_atime;
    const double vrel_phys  = 4.0 * sigeq_phys / 1.7724538509055159;   /* /sqrt(pi) */
    const double nu = rho_phys * cross_sec_per_mass * vrel_phys;
    const double eta = 1.0 - exp(-nu * dt_phys);
    if(!(eta > 0)) return 0;

    /* per-basis convex update (masses fixed); all in the stored frame */
    for(int b = 0; b < CBE_INTEGRATOR_NBASIS; b++) {
        const double m_b = moments[b][0];
        if(!(m_b > 0)) continue;
        const double inv_mb = 1.0 / m_b;
        double u_b[3] = {0,0,0}, dv[3] = {0,0,0}, u_new[3] = {0,0,0};
        for(int k = 0; k < NUMDIMS; k++) {
            u_b[k]  = cbe_basis_p_r(moments[b], k) * inv_mb;
            dv[k]   = u_b[k] - u_a[k];                        /* frame-invariant */
            u_new[k] = (1.0 - eta) * u_b[k] + eta * u_a[k];
            cbe_basis_p_w(moments[b], k, m_b * u_new[k]);
        }
        for(int a = 0; a < NUMDIMS; a++)
            for(int c = a; c < NUMDIMS; c++) {
                const double S_old = cbe_basis_T_r(moments[b], a, c) * inv_mb - u_b[a] * u_b[c];
                const double S_new = (1.0 - eta) * S_old
                                   + ((a == c) ? eta * sigeq2 : 0.0)
                                   + eta * (1.0 - eta) * dv[a] * dv[c];
                cbe_basis_T_w(moments[b], a, c, m_b * (S_new + u_new[a] * u_new[c]));
            }
    }
    return 1;
}

/* --------------------------------------------------------------------------
 * Model dispatch. Returns 1 if an operator ran, else 0 (no-op). Add new models
 * as branches here; keep each fully local (single-cell) and (m,p,Tr T)-conserving.
 * -------------------------------------------------------------------------- */
KOKKOS_INLINE_FUNCTION
int cbe_collision_apply(double moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS],
                        double rho_phys, double cross_sec_per_mass,
                        double cf_atime, double dt_phys, int model)
{
    switch(model) {
        case CBE_COLLISION_MODEL_BGK_ISOTROPIC:
            return cbe_collision_bgk_isotropic(moments, rho_phys, cross_sec_per_mass, cf_atime, dt_phys);
        default:
            return 0;
    }
}

#endif  /* CBE_INTEGRATOR && CBE_INTEGRATOR_COLLISIONS */
#endif  /* CBE_COLLISIONS_FUNCTIONS_H */
