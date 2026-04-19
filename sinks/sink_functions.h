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

#ifdef SINK_PARTICLES

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

#endif /* SINK_PARTICLES */
#endif /* SINK_FUNCTIONS_H */
