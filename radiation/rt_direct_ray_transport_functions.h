/* rt_direct_ray_transport_functions.h -- per-pair explicit radiation transport
 * along rays (Jiang et al. 2014 method). Replaces the fragment
 * radiation/rt_direct_ray_transport.h.
 *
 * Body guarded by RT_EVOLVE_INTENSITIES so callers invoke unconditionally.
 * Pure i-accumulation into out.Dt_Rad_Intensity (j-side removed with the
 * j_is_active_for_fluxes sweep).
 *
 * Templated on LocalT and OutT so both the hydro pass (hydro_data_in /
 * hydro_data_out) and the RT transport subcycle (transport INPUT_STRUCT_NAME /
 * OUTPUT_STRUCT_NAME) can call the function. LocalT must expose
 * Vel, ParticleVel (under HYDRO_MESHLESS_FINITE_VOLUME), and Rad_Intensity_Pred;
 * OutT must expose Dt_Rad_Intensity.
 *
 * Requires allvars.h included.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef RT_DIRECT_RAY_TRANSPORT_FUNCTIONS_H
#define RT_DIRECT_RAY_TRANSPORT_FUNCTIONS_H

template <typename LocalT, typename OutT>
KOKKOS_INLINE_FUNCTION
void rt_direct_ray_transport_compute_pair(
    const LocalT &local,
    const struct particle_data &Pj,
    const struct gas_cell_data &CPj,
    const Vec3<MyDouble> &VelPred_j,
    const Vec3<MyDouble> &ParticleVel_j,
    const Vec3<double> &Face_Area_Vec,
    double Face_Area_Norm,
    double V_i, double V_j,
    double Particle_Size_j,
    const double *tau_c_i,
    double dt_hydrostep,
    OutT &out)
{
#if defined(RT_EVOLVE_INTENSITIES)
    if(!(local.Mass>0 && Pj.Mass>0 && dt_hydrostep>0 && Face_Area_Norm>0)) { return; }

    double c_light_eff = C_LIGHT_CODE_REDUCED;
    double rsol_fac    = c_light_eff / C_LIGHT_CODE;
    double V_i_invphys = All.cf_a3inv / V_i;
    double V_j_invphys = All.cf_a3inv / V_j;
    double sigma_j     = Particle_Size_j * (CPj.Density * All.cf_a3inv);

    double vfluid_minus_vface_dotA = 0;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION<5)
    vfluid_minus_vface_dotA = dot(0.5*((ParticleVel_j + local.ParticleVel) - (local.Vel + VelPred_j)) / All.cf_atime, Face_Area_Vec);
#endif

    double cminusv_n_dotA[N_RT_INTENSITY_BINS] = {0};
    for(int k_angle=0; k_angle<N_RT_INTENSITY_BINS; k_angle++) {
        for(int k=0; k<3; k++) {
            cminusv_n_dotA[k_angle] += (c_light_eff * All.Rad_Intensity_Direction[k_angle][k]
                                        - rsol_fac * 0.5 * (local.Vel[k] + VelPred_j[k]) / All.cf_atime) * Face_Area_Vec[k];
        }
    }

    for(int k_freq=0; k_freq<N_RT_FREQ_BINS; k_freq++)
    {
        /* following Jiang et al., reduce advection speed when cell optical depth is large */
        double tau_c_j = CPj.Rad_Kappa[k_freq] * sigma_j;
        double q_tau   = 10. * 0.5 * (tau_c_i[k_freq] + tau_c_j);
        double a_tau   = 1;
        if(q_tau > 3.5)      { a_tau = 1. / q_tau; }
        else if(q_tau < 0.1) { a_tau = 1. - 0.25 * q_tau * q_tau; }
        else                 { a_tau = sqrt(1. - exp(-q_tau * q_tau)) / q_tau; }

        for(int k_angle=0; k_angle<N_RT_INTENSITY_BINS; k_angle++) {
            double scalar_ij = 0.5 * (local.Rad_Intensity_Pred[k_freq][k_angle] * V_i_invphys
                                    + CPj.Rad_Intensity_Pred[k_freq][k_angle] * V_j_invphys);
            double cmag = scalar_ij * (vfluid_minus_vface_dotA + a_tau * cminusv_n_dotA[k_angle]);
            out.Dt_Rad_Intensity[k_freq][k_angle] += cmag;
        }
    }
#else
    (void)local; (void)Pj; (void)CPj; (void)VelPred_j; (void)ParticleVel_j;
    (void)Face_Area_Vec; (void)Face_Area_Norm; (void)V_i; (void)V_j;
    (void)Particle_Size_j; (void)tau_c_i; (void)dt_hydrostep; (void)out;
#endif /* RT_EVOLVE_INTENSITIES */
}

#endif /* RT_DIRECT_RAY_TRANSPORT_FUNCTIONS_H */
