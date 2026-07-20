/* sink_functions.h — GPU-callable wrappers for sink helper routines.
 *
 * Provides KOKKOS_INLINE_FUNCTION versions of sink_vesc and
 * sink_check_boundedness that take explicit struct refs instead of particle
 * indices, so they can be called from device lambdas in sink_environment_gpu.cc.
 *
 * These mirror the CPU implementations in sink.cc exactly.  The #define All
 * redirect in gpu_all_mirror.h ensures All.* accesses go to All_dev when
 * compiled as a GPU TU.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef SINK_FUNCTIONS_H
#define SINK_FUNCTIONS_H

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef SINK_PARTICLES

#include "../galaxy_sf/stellar_evolution_functions.h"

/* Note: caller must have already included kernel.h, particle_data.h, and allvars.h
 * before including this header (no include guards on those files). */

KOKKOS_INLINE_FUNCTION
static double sink_vesc_gpu(const struct particle_data& kp_j,
                            const struct gas_cell_data& kc_j,
                            double mass_i, double r_code, double sink_softening)
{
    double cs_to_add = 10. / UNIT_VEL_IN_KMS;
#if defined(SINK_SEED_GROWTH_TESTS) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    cs_to_add = 0;
#endif
    double m_eff = mass_i + kp_j.Mass;
#if defined(SINK_SEED_GROWTH_TESTS) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    double gas_density = -1;
    if(kp_j.Type == 0) { gas_density = kc_j.Density; }
#ifdef GRAIN_FLUID
    if((1 << kp_j.Type) & GRAIN_PTYPES) { gas_density = kp_j.Gas_Density; }
#endif
    if(gas_density > 0) { m_eff += 4. * M_PI * r_code * r_code * r_code * gas_density; }
#endif
    double hinv = 1. / sink_softening, fac = 2. * All.G * m_eff / All.cf_atime;
#if defined(SINK_REPOSITION_ON_POTMIN)
    return sqrt(fac / r_code + cs_to_add * cs_to_add);
#endif
    return sqrt(fac * fabs(kernel_gravity(r_code * hinv, hinv, hinv * hinv * hinv, -1)) + cs_to_add * cs_to_add);
}


KOKKOS_INLINE_FUNCTION
static int sink_check_boundedness_gpu(const struct particle_data& kp_j,
                                      const struct gas_cell_data& kc_j,
                                      double vrel, double vesc,
                                      double dr_code, double sink_radius)
{
#if defined(SINK_REPOSITION_ON_POTMIN)
    if(kp_j.Type == 5) { return 1; }
#endif

    double cs = 0;
    if(kp_j.Type == 0) {
        double vA = kc_j.Alfven_speed();
        if(fabs(kc_j.gamma_eos_value() - 1) < 0.1) { cs = sqrt(vA*vA + 3.*kc_j.Pressure/kc_j.Density); }
        else                                         { cs = sqrt(vA*vA + 2.*kc_j.InternalEnergy); }
    }

#ifdef SINGLE_STAR_SINK_DYNAMICS
    double gas_density = -1;
    if(kp_j.Type == 0) { gas_density = kc_j.Density; }
#ifdef GRAIN_FLUID
    if((1 << kp_j.Type) & GRAIN_PTYPES) { gas_density = kp_j.Gas_Density; }
#endif
    if(gas_density > 0) {
        if((kp_j.Get_Particle_Size() > sink_radius * 1.396263) && (kp_j.Type == 0)) { return 0; }
#if defined(COOLING)
        double nHcgs = HYDROGEN_MASSFRAC * (gas_density * All.cf_a3inv * UNIT_DENSITY_IN_NHCGS);
        if(nHcgs > 1e13 && (cs > 0.1 * vrel || kp_j.Type != 0)) {
            double m_eff = 4. * M_PI * dr_code * dr_code * dr_code * gas_density;
            vesc = DMAX(sqrt(2 * All.G * m_eff / dr_code), vesc);
        }
#endif
    }
#endif  /* SINGLE_STAR_SINK_DYNAMICS */

    double v2 = (vrel*vrel + cs*cs) / (vesc*vesc);
    int bound = 0;
    if(v2 < 1) {
        double apocenter = dr_code / (1. - v2);
        double apocenter_max = 2. * SinkParticle_GravityKernelRadius;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
#ifdef GRAIN_FLUID
        if((1 << kp_j.Type) & (GRAIN_PTYPES)) { return (dr_code > 1.4 * sink_radius) ? 0 : 1; }
#endif
        return (dr_code > sink_radius) ? 0 : 1;
#endif
        /* Note: SINK_SEED_GROWTH_TESTS path omitted — incompatible with SINGLE_STAR_SINK_DYNAMICS
           (guarded by #if !defined(SINGLE_STAR_SINK_DYNAMICS) in the CPU version). */
        if(apocenter < apocenter_max) { bound = 1; }
    }
    return bound;
}

#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
/* cf_atime is threaded as a parameter (NOT read from All.cf_atime inline)
 * so the helper is safely callable from a device kernel — TRAP 1 (no
 * bare All.* in KOKKOS_INLINE_FUNCTION bodies). The caller in the runner
 * pair body passes scalars.common.cf_atime which was captured in
 * populate_call_scalars on the host.
 *
 * allow_sink_merger is `volatile int *` to match the caller's local
 * `volatile int allow_sink_merger` — required per [[feedback_gpu]] §D.3
 * to defeat the nvc++ device-lambda const-propagation bug on int gate
 * vars assigned conditionally. The named instance "allow_sink_merger"
 * is one of the known-vulnerable cases in that section's table. */
KOKKOS_INLINE_FUNCTION
static void sink_apply_binary_merge_away_limits(int local_eligible,
                                                int neighbor_eligible,
                                                double soft_j,
                                                double local_kernel_radius,
                                                double neighbor_kernel_radius,
                                                double local_mass,
                                                double neighbor_mass,
                                                double sink_radius,
                                                volatile int *allow_sink_merger,
                                                double *max_rmerge,
                                                double *max_mmerge,
                                                double cf_atime)
{
    if(local_eligible == 0) *allow_sink_merger = 0;
    if(neighbor_eligible == 0) *allow_sink_merger = 0;
    *max_mmerge = 10. * neighbor_mass;
    *max_rmerge = DMAX(*max_rmerge, soft_j);
    *max_rmerge = DMAX(*max_rmerge, DMIN(local_kernel_radius, neighbor_kernel_radius));
    *max_rmerge = DMIN(*max_rmerge, 10. * sink_radius);
    {
        double dt_min_orbit_yr = 100.;
        double rmax_dt = 0.000485 / (cf_atime * UNIT_LENGTH_IN_PC) *
            pow(((neighbor_mass + local_mass) * UNIT_MASS_IN_SOLAR / 100.) *
                (dt_min_orbit_yr * dt_min_orbit_yr / (100. * 100.)), 1./3.);
        *max_rmerge = DMAX(*max_rmerge, rmax_dt);
    }
}
#endif

/* GPU-callable version of sink_fb_angleweight_localcoupling.
 * Takes explicit particle refs instead of indices for device use. */
KOKKOS_INLINE_FUNCTION
static double sink_fb_angleweight_localcoupling_gpu(
    const struct particle_data& kp_j,
    const struct gas_cell_data& kc_j,
    double cos_theta, double r, double H_kernel)
{
    if(r <= 0 || H_kernel <= 0) return 0;
    double H_j = kp_j.KernelRadius;
    if((r >= H_j && r >= H_kernel) || (H_j <= 0) || (kp_j.Mass <= 0) || (kc_j.Density <= 0)) return 0;
    double hinv=1./H_kernel, hinv_j=1./H_j, hinv3_j=hinv_j*hinv_j*hinv_j;
    double wk_j=0, dwk_j=0, u_j=r*hinv_j, hinv4_j=hinv_j*hinv3_j;
    double V_j = kp_j.Mass / kc_j.Density;
    double hinv3=hinv*hinv*hinv, hinv4=hinv*hinv3, wk=0, dwk=0;
    double u = r * hinv;
    if(u < 1) { kernel_main(u, hinv3, hinv4, &wk, &dwk, 0); }
    if(u_j < 1) { kernel_main(u_j, hinv3_j, hinv4_j, &wk_j, &dwk_j, 0); }
    double V_i = 4.*M_PI/3. * H_kernel*H_kernel*H_kernel / (All.DesNumNgb * All.SinkNgbFactor);
    if(V_i < 0 || V_i != V_i) { V_i = 0; }
    if(V_j < 0 || V_j != V_j) { V_j = 0; }
    double face_area = fabs(V_i*V_i*dwk + V_j*V_j*dwk_j);
    wk = 0.5 * (1. - 1./sqrt(1. + face_area / (M_PI*r*r)));
#if defined(SINK_FB_VOLUMEWEIGHTED)
    wk = 0.5 * (V_j/V_i) * (V_i*wk + V_j*wk_j);
#endif
#if defined(SINK_FB_COLLIMATED)
    double costheta2=cos_theta*cos_theta, eps_width_jet=0.35;
    eps_width_jet *= eps_width_jet;
    wk *= eps_width_jet * (eps_width_jet + costheta2) / ((eps_width_jet + 1) * (eps_width_jet + (1-costheta2)));
#endif
    if(wk <= 0 || wk != wk) return 0;
    return wk;
}


#if defined(GRAVTREE_SOURCE_HOST_OWNER_TU) || (defined(GRAVTREE_SOURCE_DEVICE_TU) && defined(GRAVTREE_SOURCE_LAZY_SUPPORTED))
/* Sink accretion-luminosity helper cores shared by host and device; the host
 * externals in sink_util.cc are one-line wrappers around these. */

/* return the eddington accretion-rate = L_edd/(epsilon_r*c*c) - for a variable radiative efficiency, this scales off of the 'canonical' value given as the constant in the parameterfile, per usual convention in the literature */
KOKKOS_INLINE_FUNCTION double sink_eddington_mdot_core(double sink_mass)
{
    return (4*M_PI * GRAVITY_G_CGS * PROTONMASS_CGS / (All.SinkRadiativeEfficiency * C_LIGHT_CGS * THOMPSON_CX_CGS)) * sink_mass * UNIT_TIME_IN_CGS;
}

/* return the bh radiative efficiency - allows for models with dynamical or accretion-dependent radiative efficiencies */
KOKKOS_INLINE_FUNCTION double evaluate_sink_radiative_efficiency_core(double mdot, double mass, long pindex)
{
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL /* simple classic model where radiative efficiency declines linearly below critical eddington ratio of order 1% eddington, and super-Eddington accretion is also radiatively inefficient  */
    double lambda_0 = 0.01, lambda_1 = 2., lambda_eff = mdot/sink_eddington_mdot_core(mass), qfac = lambda_eff/lambda_0, qfac_he = lambda_eff/lambda_1;
    return All.SinkRadiativeEfficiency * (qfac/(1.+qfac)) * (1./(1.+qfac_he));
#endif
    return All.SinkRadiativeEfficiency; // default to constant
}

/* return the bh cosmic ray acceleration efficiency - allows for models with dynamical or accretion-dependent cosmic ray efficiencies */
KOKKOS_INLINE_FUNCTION double evaluate_sink_cosmicray_efficiency_core(double mdot, double mass, long pindex)
{
#ifdef SINK_COSMIC_RAYS
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL /* experiment with functions here to explore more/less efficient CR injection at lower/higher eddington ratios, reflecting hard/soft-type transition */
    /* current version with this flag scales this relative to the jet kinetic power */
    if(mdot/sink_eddington_mdot_core(mass) < 0.01) {return 0.5;} else {return 0.1;} /* high-Mdot uses standard SNe kinetic power, otherwise give unity power here */
#endif
    return All.Sink_CosmicRay_Injection_Efficiency; // default to constant
#endif
    return 0;
}

/* return the bh luminosity given some accretion rate and mass (allows for non-standard models: radiatively inefficient flows, stellar sinks, etc) */
KOKKOS_INLINE_FUNCTION double sink_lum_bol_core(double mdot, double mass, long pindex, struct particle_data *pp)
{
    double lum = evaluate_sink_radiative_efficiency_core(mdot,mass,pindex) * mdot * C_LIGHT_CODE*C_LIGHT_CODE; // this is automatically in -physical code units-
#ifdef SINGLE_STAR_SINK_DYNAMICS
    lum = calculate_individual_stellar_luminosity_core(mdot,mass,pindex,pp);
#endif
    return All.SinkFeedbackFactor * lum;
}

#endif /* GRAVTREE_SOURCE_HOST_OWNER_TU || (GRAVTREE_SOURCE_DEVICE_TU && GRAVTREE_SOURCE_LAZY_SUPPORTED) */

#endif /* SINK_PARTICLES */
#endif /* SINK_FUNCTIONS_H */
