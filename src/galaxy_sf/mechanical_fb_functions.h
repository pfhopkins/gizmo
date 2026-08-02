/* mechanical_fb_functions.h — GPU-callable structs + per-pair kernel body
 * for the mechanical_fb default-scheme GPU port.
 *
 * Covers all 6 modes (-2, -1, 0, 1, 2, 3) of addFB_evaluate default scheme
 * INCLUDING cosmic-ray injection (CR_DYNAMICAL_INJECTION_IN_SNE +
 * COSMIC_RAY_FLUID + GALSF_FB_FIRE_STELLAREVOLUTION inject_cosmic_rays_into_delta
 * helper below; deltas atomically accumulated into MechFBGasDelta, scattered
 * to CellP[].CosmicRayEnergy/etc. host-side in verify_and_assign).
 *
 * The shared type MechFBGasDelta is used by BOTH the CPU addFB_evaluate path
 * (as the LocalGasMechFBInfoTemp element type) and the GPU kernel, so the
 * host-side verify_and_assign_local_mechfb_integrals scatter is unchanged.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#include "mechanical_fb_types.h"  /* provides struct MechFBGasDelta */
#if defined(COSMIC_RAY_FLUID)
#include "../eos/cosmic_ray_fluid/cosmic_ray_functions.h"  /* KOKKOS_INLINE CR helpers */
#endif
#if defined(GALSF_ISMDUSTCHEM_MODEL)
#include "../solids/ism_dust_chemistry_functions.h"  /* KOKKOS_INLINE ISMDustChem_Return_Mass_Where_Dust_Shocked (Phase D 2026-05-21 #20011-D fix) */
#endif

#ifdef GALSF_FB_MECHANICAL

#include "../mesh/neighbor_loop_runner.h"  /* NlrCommonScalars */

/* Cosmology + unit-factor + CR-rigidity scalars for mechfb kernel helpers.
 * Built host-side once per call via mechfb_fill_call_scalars (mechfb_loop.cc);
 * threaded into mechanical_fb_per_source_setup, mechanical_fb_pair_kernel, and
 * inject_cosmic_rays_into_delta so no helper reads bare All.* directly
 * (a per-TU All_dev mirror is unreliable). */
struct MechFBCallScalars {
    NlrCommonScalars common;            /* cf_atime, cf_a2inv, cf_a3inv, … */
    double unit_length_in_kpc;
    double unit_density_in_NHcgs;       /* unit density in n_H (protons/cm^3) */
    double unit_energy_in_cgs;
    double unit_mass_in_solar;
    double unit_vel_in_kms;
#ifdef METALS
    double SolarAbundances0;
#endif
#if defined(CR_DYNAMICAL_INJECTION_IN_SNE)
    /* Mirrors All.CosmicRay_SNeFraction (global_data_all_struct.h:408-410).
     * CR_DYNAMICAL_INJECTION_IN_SNE is defined whenever COSMIC_RAY_FLUID OR
     * COSMIC_RAY_SUBGRID_LEBRON is, and the upstream field is in the same
     * gate — the mirror must match (not be narrowed to COSMIC_RAY_FLUID only)
     * or the LEBRON-subgrid + FIRE_BHS path fails to compile (Phase D
     * 2026-05-21). */
    double CosmicRay_SNeFraction;
#endif
#if defined(COSMIC_RAY_FLUID) && defined(GALSF_FB_FIRE_STELLAREVOLUTION) && defined(CRFLUID_EVOLVE_SPECTRUM)
    double CR_global_min_rigidity_in_bin   [N_CR_PARTICLE_BINS];
    double CR_global_max_rigidity_in_bin   [N_CR_PARTICLE_BINS];
    double CR_global_rigidity_at_bin_center[N_CR_PARTICLE_BINS];
#endif
#ifdef SINK_WIND_SPAWN
    MyIDType SpawnedWindCellID;         /* mirrors All.SpawnedWindCellID */
#endif
};

/* Ownership-split j-side write target.
 *
 * Picks the per-gas delta buffer for j-side atomic writes from inside the
 * pair kernel. P[] layout convention used here:
 *   [0 ........... num_local_gas)       local gas cells   → home_gas_delta[j]
 *   [num_local_gas, num_local_particles) local non-gas    → mask-violation abort
 *   [num_local_particles, ...)          imported ghost    → ghost_gas_delta[j - num_local_particles]
 *
 *   - num_local_gas       = N_gas      (local gas count this rank)
 *   - num_local_particles = NumPart    (= ghost_get_num_local() pre-import)
 *
 * Note: previously this helper used `j < num_local_gas` to
 * route home vs ghost, which silently corrupted memory whenever
 * NumPart > N_gas (any sim with non-gas particles) because imported ghosts
 * land at indices >= NumPart, NOT >= N_gas. The middle range [N_gas, NumPart)
 * is local non-gas — gas-only neighbor mask in load_neighbor should filter
 * these out before we ever reach the kernel; defensive abort if we don't.
 *
 * Returns the per-pair MechFBGasDelta target for atomic accumulation.
 *
 * Historical note: the retired legacy evaluator passed
 * num_local_gas = num_local_particles = num_all and ghost_gas_delta=nullptr,
 * so every j fell in [0, num_all) → first branch → home_gas_delta[j], with
 * the ghost and gap branches unreachable. The MechFBSpec runner-port supplies
 * the real three-way split. */
KOKKOS_INLINE_FUNCTION
static struct MechFBGasDelta *mechfb_target_gas_delta(
    int j,
    int num_local_gas,
    int num_local_particles,
    struct MechFBGasDelta *home_gas_delta,
    struct MechFBGasDelta *ghost_gas_delta)
{
    if (j < num_local_gas) return &home_gas_delta[j];
    if (j >= num_local_particles) {
        if (ghost_gas_delta == nullptr) {
            Kokkos::abort("mechfb_target_gas_delta: ghost-side write reached "
                          "(j >= num_local_particles) but ghost_gas_delta is "
                          "null. reset_per_iter_device_context must allocate "
                          "Aux::d_gas_iter whenever ghost_get_num_ghosts() > 0.");
        }
        return &ghost_gas_delta[j - num_local_particles];
    }
    Kokkos::abort("mechfb_target_gas_delta: gas-only kernel received local "
                  "non-gas neighbor j in [num_local_gas, num_local_particles) — "
                  "MechFBSpec::load_neighbor or legacy neighbor-list filter "
                  "should have rejected this before pair_kernel.");
    return home_gas_delta;  /* unreachable */
}

/* Per-source (star) input to kernel. Host packs once per mode via
 * particle2in_addFB_fromstars + P[i].Area_weighted_sum. */
struct MechFBLocalIn
{
    Vec3<MyDouble> Pos, Vel;
    MyDouble Msne;
    MyFloat KernelRadius, V_i, SNe_v_ejecta;
    MyFloat Area_weighted_sum[AREA_WEIGHTED_SUM_ELEMENTS];
    short int TimeBin;  /* source's TimeBin -- accumulated into gas-side
                         * max_source_wakeup so direct-dU receivers can be
                         * marked for positive wakeup at apply time. */
#ifdef METALS
    MyDouble yields[NUM_METAL_SPECIES];
#endif
};

/* Per-source (star) output from kernel. Area_weighted_sum[0..6] are written in
 * mode -2; Area_weighted_sum[7..11] in mode -1; M_coupled in modes >= 0. */
struct MechFBOut
{
    MyFloat M_coupled;
    MyFloat Area_weighted_sum[AREA_WEIGHTED_SUM_ELEMENTS];
};

/* Per-source per-mode precomputed state. Built in the lambda once per star-mode. */
struct MechFBSourceMode
{
    double wk_norm;
    double pnorm_sum;
    double momentum_to_couple_term_units;
    double U_thermal_residual_tocouple;
    double Esne51;
    double density_to_n;
    double r2max_phys;
    double h2;
    double hinv, hinv3, hinv4;
    double kernel_zero;
    int feedback_type_is_SNe;
    int retain_thermal_flag;
#if defined(CR_DYNAMICAL_INJECTION_IN_SNE)
    double CR_energy_to_inject;  /* total CR energy budget for this source in this mode */
#endif
};


#if defined(COSMIC_RAY_FLUID) && defined(GALSF_FB_FIRE_STELLAREVOLUTION)
/* Kernel-callable CR injection: writes per-bin CR energy / number / direction
 * deltas into MechFBGasDelta via Kokkos atomics instead of CellP directly.
 * Host scatter (in verify_and_assign_local_mechfb_integrals) normalizes direction and
 * applies to CellP[].CosmicRayEnergy/Pred/Number/Flux/FluxPred. */
KOKKOS_INLINE_FUNCTION
static void inject_cosmic_rays_into_delta(
    double CR_energy_to_inject, double injection_velocity, int source_type,
    int target, const double *dir,
    struct particle_data *P_arr, struct gas_cell_data *cell,
    /* Threaded per-call scalars; replaces bare-All reads of
     * CR_global_min/max/center rigidity arrays inside this body. */
    const struct MechFBCallScalars& scalars,
    int num_local_gas,
    int num_local_particles,
    struct MechFBGasDelta *home_gas_delta,
    struct MechFBGasDelta *ghost_gas_delta)
{
    if(CR_energy_to_inject <= 0) return;
    struct MechFBGasDelta *gd =
        mechfb_target_gas_delta(target, num_local_gas, num_local_particles,
                                home_gas_delta, ghost_gas_delta);
    double f_injected[N_CR_PARTICLE_BINS]; f_injected[0] = 1;
#if (N_CR_PARTICLE_BINS > 1)
    double sum_in = 0.0;
    for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {
        f_injected[k] = CR_energy_spectrum_injection_fraction(k, source_type, injection_velocity, 0, target, P_arr, cell);
        sum_in += f_injected[k];
    }
    if(sum_in > 0.0) { for(int k = 0; k < N_CR_PARTICLE_BINS; k++) f_injected[k] /= sum_in; }
    else             { for(int k = 0; k < N_CR_PARTICLE_BINS; k++) f_injected[k] = 1.0 / N_CR_PARTICLE_BINS; }
#endif
    double sum_dEcr = 0.0;
    for(int k = 0; k < N_CR_PARTICLE_BINS; k++)
    {
        double dEcr = evaluate_cr_transport_reductionfactor(target, k, 0, cell) * CR_energy_to_inject * f_injected[k];
        if(dEcr <= 0) continue;
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        double E_GeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k);
        double egy_slopemode = 1;
        double xm = scalars.CR_global_min_rigidity_in_bin[k] / scalars.CR_global_rigidity_at_bin_center[k];
        double xp = scalars.CR_global_max_rigidity_in_bin[k] / scalars.CR_global_rigidity_at_bin_center[k];
        double xm_e = xm, xp_e = xp;
        if(CR_check_if_bin_is_nonrelativistic(k)) { egy_slopemode = 2; xm_e = xm*xm; xp_e = xp*xp; }
        double slope_inj = CR_energy_spectrum_injection_fraction(k, source_type, injection_velocity, 1, target, P_arr, cell);
        double gamma_one = slope_inj + 1.0;
        double xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one);
        double ntot_inj = (dEcr / E_GeV) * ((gamma_one + egy_slopemode) / gamma_one) *
                          (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp_e - xm_gamma_one*xm_e);
        Kokkos::atomic_add(&gd->CR_number_injected[k], ntot_inj);
#endif
        Kokkos::atomic_add(&gd->CR_energy_injected[k], dEcr);
        sum_dEcr += dEcr;
    }
    if(sum_dEcr > 0) {
        for(int c = 0; c < 3; c++) Kokkos::atomic_add(&gd->CR_dir_weighted[c], sum_dEcr * dir[c]);
    }
}
#endif /* COSMIC_RAY_FLUID && GALSF_FB_FIRE_STELLAREVOLUTION */


/* Host-side helpers for the MechFBSpec runner-port.
 * Defined in galaxy_sf/mechfb_loop.cc (single source of truth;
 * originally shared with the now-retired legacy mechanical_fb_gpu.cc
 * evaluator). All read host P only — no All.* access from these helpers. */
void mech_fb_local_fill(int i, int loop_iteration,
                         struct MechFBLocalIn *loc);
void mech_fb_apply_aws_out(const struct MechFBOut *out,
                            int i, int loop_iteration);
void mech_fb_apply_source_mass_out(struct particle_data *P_arr,
                                    struct gas_cell_data *CellP_arr,
                                    int i,
                                    MyFloat M_coupled);


/* Device-callable mirror of mechanical_fb.cc:addFB_evaluate_active_check.
 * Must stay in sync with that CPU routine — it's the star-side filter used
 * both by the CPU code_block_xchange dispatch and the GPU per-mode mask. */
KOKKOS_INLINE_FUNCTION
static int mechanical_fb_star_active_check(int i, int fb_loop_iteration,
                                            struct particle_data *P_arr)
{
    if(P_arr[i].Type <= 1) return 0;
    if(P_arr[i].KernelRadius <= 0) return 0;
    if(P_arr[i].NumNgb <= 0) return 0;
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    if(P_arr[i].Type == 5 && !P_arr[i].do_gas_search_this_timestep) return 0;
#endif
    if(P_arr[i].SNe_ThisTimeStep > 0) { if(fb_loop_iteration < 0 || fb_loop_iteration == 0) return 1; }
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION
    if(P_arr[i].MassReturn_ThisTimeStep > 0) { if(fb_loop_iteration < 0 || fb_loop_iteration == 1) return 1; }
#ifdef GALSF_FB_FIRE_RPROCESS
    if(P_arr[i].RProcessEvent_ThisTimeStep > 0) { if(fb_loop_iteration < 0 || fb_loop_iteration == 2) return 1; }
#endif
#ifdef GALSF_FB_FIRE_AGE_TRACERS
    if(P_arr[i].AgeDeposition_ThisTimeStep > 0) { if(fb_loop_iteration < 0 || fb_loop_iteration == 3) return 1; }
#endif
#endif
#if defined(SINGLE_STAR_FB_WINDS) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    if(P_arr[i].wind_mode != 2 || P_arr[i].ProtoStellarStage != 5) return 0;
#endif
    return 0;
}

/* MechFBCallScalars is defined above (after the #ifdef GALSF_FB_MECHANICAL gate).
 * Helpers below are self-contained: include this header directly. */

KOKKOS_INLINE_FUNCTION
static void mechanical_fb_per_source_setup(
    const struct MechFBLocalIn& local, int loop_iteration,
    const struct MechFBCallScalars& scalars,
    struct MechFBSourceMode& m)
{
    m.h2 = (double)local.KernelRadius * (double)local.KernelRadius;
    double wk_dummy = 0;
    kernel_main(0.0, 1.0, 1.0, &m.kernel_zero, &wk_dummy, -1);
    kernel_hinv((double)local.KernelRadius, &m.hinv, &m.hinv3, &m.hinv4);
    /* Replace bare-All reads (All.cf_atime / All.cf_a3inv) and
     * macro reads (UNIT_*_IN_*) with threaded scalars. */
    double unitlength_in_kpc = scalars.unit_length_in_kpc * scalars.common.cf_atime;
    m.density_to_n = scalars.common.cf_a3inv * scalars.unit_density_in_NHcgs;
    double unit_egy_SNe = 1.0e51 / scalars.unit_energy_in_cgs;

    m.wk_norm   = 1.0 / (MIN_REAL_NUMBER + fabs((double)local.Area_weighted_sum[0]));
    m.pnorm_sum = 1.0 / (MIN_REAL_NUMBER + fabs((double)local.Area_weighted_sum[10]));
    double thermal_to_kinetic_ratio_universal = 2.54;
    m.retain_thermal_flag = 1;
    m.feedback_type_is_SNe = (loop_iteration == 0) ? 1 : 0;
    if(m.feedback_type_is_SNe == 0) thermal_to_kinetic_ratio_universal = 1.0e-2;
    double f_sedov_kin = 1.0 / (1.0 + thermal_to_kinetic_ratio_universal);
    m.momentum_to_couple_term_units = 0;
    m.U_thermal_residual_tocouple = 0;
    double Energy_injected_codeunits = 0.5 * (double)local.Msne * (double)local.SNe_v_ejecta * (double)local.SNe_v_ejecta;
    double v_ejecta_eff_init = (double)local.SNe_v_ejecta;

    if(((double)local.Area_weighted_sum[0] > MIN_REAL_NUMBER) && (loop_iteration >= 0))
    {
        double vba_2_eff = m.pnorm_sum * (double)local.Area_weighted_sum[7];
        v_ejecta_eff_init = sqrt((double)local.SNe_v_ejecta * (double)local.SNe_v_ejecta + vba_2_eff);
        Energy_injected_codeunits = 0.5 * (double)local.Msne * v_ejecta_eff_init * v_ejecta_eff_init;

        double p_terminal_multiplier_rhoZE =
            (m.pnorm_sum * (double)local.Area_weighted_sum[9]) * (Energy_injected_codeunits / unit_egy_SNe);
        double p_terminal = sqrt(f_sedov_kin) * (4.8e5 / (scalars.unit_mass_in_solar * scalars.unit_vel_in_kms)) * p_terminal_multiplier_rhoZE;
        if(m.feedback_type_is_SNe == 0) p_terminal = (double)local.Msne * v_ejecta_eff_init;

        double egy_0_norm_for_soln = f_sedov_kin * Energy_injected_codeunits;
        double S1 = m.pnorm_sum * (double)local.Area_weighted_sum[8];
        double S2 = m.pnorm_sum * m.pnorm_sum * (double)local.Area_weighted_sum[11];
        double S1_2 = S1 * S1, S2_E = 4.0 * S2 * egy_0_norm_for_soln;
        double p0_egycon = 1.0 / (2.0 * S2) * (-S1 + sqrt(S1_2 + S2_E));
        if((S2_E < 0.01 * S1_2) && (S1 > 0.0)) p0_egycon = egy_0_norm_for_soln / S1 * (1.0 - S2_E / (4.0 * S1_2));

        double p_coupled = DMIN(p0_egycon, p_terminal);
        double egy_RHS = p_coupled * p_coupled * S2 + p_coupled * S1;
        m.momentum_to_couple_term_units = p_coupled;
        m.U_thermal_residual_tocouple = Energy_injected_codeunits - egy_RHS;
    }

#if defined(CR_DYNAMICAL_INJECTION_IN_SNE)
    /* Mirror CPU lines 548-555 in mechanical_fb.cc: compute CR energy budget
     * for this source at current velocity, subtracting it from the ejecta KE.
     * All.CosmicRay_SNeFraction and UNIT_VEL_IN_KMS routed through
     * scalars (precomputed in populate_call_scalars). */
    m.CR_energy_to_inject = 0;
    if(v_ejecta_eff_init > 1000.0 / scalars.unit_vel_in_kms)
    {
        double post_cr_corr = sqrt(1.0 - scalars.CosmicRay_SNeFraction);
        v_ejecta_eff_init *= post_cr_corr;
        m.CR_energy_to_inject = (scalars.CosmicRay_SNeFraction / (1.0 - scalars.CosmicRay_SNeFraction)) *
                                 0.5 * (double)local.Msne * v_ejecta_eff_init * v_ejecta_eff_init;
        /* also reduce Energy_injected_codeunits correspondingly (CPU version does
         * this implicitly via v_ejecta_eff *= post_cr_corr; we recompute). */
        Energy_injected_codeunits = 0.5 * (double)local.Msne * v_ejecta_eff_init * v_ejecta_eff_init;
    }
#endif

    m.Esne51 = Energy_injected_codeunits / unit_egy_SNe;
    double r2max = 2.0 / unitlength_in_kpc;
    m.r2max_phys = r2max * r2max;
}


KOKKOS_INLINE_FUNCTION
static void mechanical_fb_pair_kernel(
    const struct MechFBLocalIn& local,
    const struct MechFBSourceMode& m,
    int loop_iteration,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    /* Threaded per-call scalars: cosmology factors,
     * SolarAbundances[0], unit conversions, CR rigidity arrays. Replaces
     * bare All.* reads inside this body. Caller (runner-port pair_kernel)
     * builds it host-side via populate_call_scalars. */
    const struct MechFBCallScalars& scalars,
    /* j-side ownership-split write target.
     * Three-way index split via mechfb_target_gas_delta (see helper above):
     *   home_gas_delta       : local gas range [0, num_local_gas)
     *   ghost_gas_delta      : imported-ghost range [num_local_particles, ...);
     *                          may be nullptr only when num_local_particles is
     *                          set such that the ghost branch is unreachable
     *                          (legacy passes num_local_gas = num_local_particles
     *                          = num_all, so j < num_local_gas always holds).
     *   middle gap            : local non-gas range, gas-only mask should filter
     *                          out, defensive abort if not.
     * oracle_dry_run : if true, i-side accum (myout.*) still completes but
     *                  every j-side atomic_add is suppressed (oracle brute-pass). */
    struct MechFBGasDelta *home_gas_delta,
    struct MechFBGasDelta *ghost_gas_delta,
    int  num_local_gas,
    int  num_local_particles,
    const Vec3<double>& dp,  /* = local.Pos - P[j].Pos, nearest_xyz-corrected */
    double r2,
    struct MechFBOut& myout)
{
    const bool is_ghost = (j >= num_local_particles);
    if(P[j].Type != 0) return;
#ifdef SINK_WIND_SPAWN
    if(P[j].ID == scalars.SpawnedWindCellID) return; /* don't couple feedback onto sink-spawned jet cells */
#endif
    double Mass_j = P[j].Mass;
    if(Mass_j <= 0) return;
    if(r2 <= 0) return;
    double h2j = (double)P[j].KernelRadius * (double)P[j].KernelRadius;
    if((r2 > m.h2) && (r2 > h2j)) return;
    if(r2 > m.r2max_phys) return;
    double r = sqrt(r2);
    if(r <= 0) return;

    double rho_j = CellP[j].Density;
    double u = r * m.hinv;
    double hinv_j = 1.0 / (double)P[j].KernelRadius;
    double hinv3_j = hinv_j * hinv_j * hinv_j;
    double hinv4_j = hinv_j * hinv3_j;
    double V_j = (rho_j > 0) ? Mass_j / rho_j : 0;
    double u_j = r * hinv_j;
    double wk_self = 0, dwk_self = 0, wk_j = 0, dwk_j = 0;
    if(u < 1) kernel_main(u, m.hinv3, m.hinv4, &wk_self, &dwk_self, 1);
    if(u_j < 1) kernel_main(u_j, hinv3_j, hinv4_j, &wk_j, &dwk_j, 1);
    double V_i = (double)local.V_i;
    if(V_i < 0 || V_i != V_i) V_i = 0;
    if(V_j < 0 || V_j != V_j) V_j = 0;
    double face_area = fabs(V_i * V_i * dwk_self + V_j * V_j * dwk_j);
    double wk = 0.5 * (1 - 1 / sqrt(1 + face_area / (M_PI * r * r)));
    if(wk <= 0 || wk != wk) return;

    double wk_vec[AREA_WEIGHTED_SUM_ELEMENTS] = {0};
    wk_vec[0] = wk;
    if(dp[0] > 0) { wk_vec[1] = wk * dp[0] / r; wk_vec[2] = 0; } else { wk_vec[1] = 0; wk_vec[2] = wk * dp[0] / r; }
    if(dp[1] > 0) { wk_vec[3] = wk * dp[1] / r; wk_vec[4] = 0; } else { wk_vec[3] = 0; wk_vec[4] = wk * dp[1] / r; }
    if(dp[2] > 0) { wk_vec[5] = wk * dp[2] / r; wk_vec[6] = 0; } else { wk_vec[5] = 0; wk_vec[6] = wk * dp[2] / r; }

    if(loop_iteration == -2) {
        for(int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; k++) myout.Area_weighted_sum[k] += (MyFloat)wk_vec[k];
        return;
    }

    double InternalEnergy_j = CellP[j].InternalEnergy;
    double Vel_j[3]; for(int k = 0; k < 3; k++) Vel_j[k] = P[j].Vel[k];
    double InternalEnergy_j_0 = InternalEnergy_j, Mass_j_0 = Mass_j;
    double Vel_j_0[3]; for(int k = 0; k < 3; k++) Vel_j_0[k] = Vel_j[k];
#ifdef METALS
    double Metallicity_j[NUM_METAL_SPECIES], Metallicity_j_0[NUM_METAL_SPECIES];
    for(int k = 0; k < NUM_METAL_SPECIES; k++) Metallicity_j[k] = P[j].Metallicity[k];
    for(int k = 0; k < NUM_METAL_SPECIES; k++) Metallicity_j_0[k] = Metallicity_j[k];
#endif

    if(loop_iteration < 0) {  /* loop_iteration == -1 */
        double pnorm = 0, pvec[3] = {0}, vel_ba_2 = 0, cos_vel_ba_pcoupled = 0;
        for(int k = 0; k < 3; k++)
        {
            double q = 0; int i1 = 2 * k + 1, i2 = i1 + 1;
            double q_i1 = fabs((double)local.Area_weighted_sum[i1]);
            double q_i2 = fabs((double)local.Area_weighted_sum[i2]);
            if((q_i1 > MIN_REAL_NUMBER) && (q_i2 > MIN_REAL_NUMBER)) {
                double rr = q_i2 / q_i1;
                double rr2 = rr * rr;
                if(wk_vec[i1] != 0) q += m.wk_norm * wk_vec[i1] * sqrt(0.5 * (1.0 + rr2));
                else                q += m.wk_norm * wk_vec[i2] * sqrt(0.5 * (1.0 + 1.0 / rr2));
            } else {
                q += m.wk_norm * (wk_vec[i1] + wk_vec[i2]);
            }
            pvec[k] = -q;
            pnorm += pvec[k] * pvec[k];
        }
        pnorm = sqrt(pnorm);
        for(int k = 0; k < 3; k++)
        {
            double v_ba = (Vel_j[k] - (double)local.Vel[k]) / scalars.common.cf_atime;
            vel_ba_2 += v_ba * v_ba;
            if(pnorm > 0) cos_vel_ba_pcoupled += v_ba * pvec[k] / pnorm;
        }
        double mu_inv = 1.0 / (1.0 + pnorm * (double)local.Msne / Mass_j);
        wk_vec[7]  = pnorm * vel_ba_2 * mu_inv;
        wk_vec[8]  = pnorm * cos_vel_ba_pcoupled * mu_inv;
        wk_vec[10] = pnorm;
        wk_vec[11] = 0.5 * pnorm * pnorm * mu_inv / Mass_j;
        double n0 = DMAX(0.001, rho_j * m.density_to_n);
#ifdef METALS
        double z0 = DMAX(0.01, Metallicity_j[0] / scalars.SolarAbundances0);
#else
        double z0 = 1.0;
#endif
        double z0_term = pow(z0, -0.18); if(z0 > 1.0) z0_term = pow(z0, -0.12);
        double p_term_prefac = pow(n0, -0.143) * z0_term;
        wk_vec[9] = pnorm * p_term_prefac;

        for(int k = 7; k < AREA_WEIGHTED_SUM_ELEMENTS; k++) myout.Area_weighted_sum[k] += (MyFloat)wk_vec[k];
        return;
    }

    /* loop_iteration >= 0: actual coupling */
    double wk_local_norm = wk * m.wk_norm;
    if(wk_local_norm <= 0 || wk_local_norm != wk_local_norm) return;

    double pnorm = 0, pvec[3] = {0};
    for(int k = 0; k < 3; k++)
    {
        double q = 0; int i1 = 2 * k + 1, i2 = i1 + 1;
        double q_i1 = fabs((double)local.Area_weighted_sum[i1]);
        double q_i2 = fabs((double)local.Area_weighted_sum[i2]);
        if((q_i1 > MIN_REAL_NUMBER) && (q_i2 > MIN_REAL_NUMBER)) {
            double rr = q_i2 / q_i1;
            double rr2 = rr * rr;
            if(wk_vec[i1] != 0) q += m.wk_norm * wk_vec[i1] * sqrt(0.5 * (1.0 + rr2));
            else                q += m.wk_norm * wk_vec[i2] * sqrt(0.5 * (1.0 + 1.0 / rr2));
        } else {
            q += m.wk_norm * (wk_vec[i1] + wk_vec[i2]);
        }
        pvec[k] = -q;
        pnorm += pvec[k] * pvec[k];
    }
    pnorm = sqrt(pnorm);
    pnorm *= m.pnorm_sum;
    for(int k = 0; k < 3; k++) pvec[k] *= m.pnorm_sum;
    double dM_ejecta_in = pnorm * (double)local.Msne;
    double mj_preshock = Mass_j;
    double massratio_ejecta = dM_ejecta_in / (dM_ejecta_in + Mass_j);

    volatile int couple_anything_but_scalar_mass_and_metals = 1; /* volatile: nvc++ folds to initial value otherwise */
    /* rho update (local copy, used for dust shock below) */
    if((double)P[j].KernelRadius <= 0) {
        if(rho_j > 0) rho_j *= (1 + dM_ejecta_in / Mass_j);
        else          rho_j  = dM_ejecta_in * m.hinv3;
    } else {
        rho_j += m.kernel_zero * dM_ejecta_in * hinv3_j;
    }
    rho_j *= 1 + dM_ejecta_in / Mass_j;
    Mass_j += dM_ejecta_in;
    myout.M_coupled += (MyFloat)dM_ejecta_in;

#ifdef METALS
    {
        int kmax_std = NUM_METAL_SPECIES;
#if defined(NUM_AGE_TRACERS)
        kmax_std = NUM_METAL_SPECIES - NUM_AGE_TRACERS;
#endif
        for(int k = 0; k < kmax_std; k++) {
            Metallicity_j[k] = (1 - massratio_ejecta) * Metallicity_j[k] + massratio_ejecta * (double)local.yields[k];
        }
#ifdef GALSF_FB_FIRE_AGE_TRACERS
        if(loop_iteration == 3) {
            for(int k = NUM_METAL_SPECIES - NUM_AGE_TRACERS; k < NUM_METAL_SPECIES; k++) {
                Metallicity_j[k] += pnorm * (double)local.yields[k] / Mass_j;
            }
        } else {
            for(int k = NUM_METAL_SPECIES - NUM_AGE_TRACERS; k < NUM_METAL_SPECIES; k++) {
                Metallicity_j[k] = (1 - massratio_ejecta) * Metallicity_j[k] + massratio_ejecta * (double)local.yields[k];
            }
        }
#endif
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION
        if(loop_iteration >= 2) couple_anything_but_scalar_mass_and_metals = 0;
#endif
    }
#endif /* METALS */

#if defined(GALSF_ISMDUSTCHEM_MODEL)
    double Mass_Where_Dust_Shocked_pair = 0;
    if(m.feedback_type_is_SNe == 1) {Mass_Where_Dust_Shocked_pair = ISMDustChem_Return_Mass_Where_Dust_Shocked(rho_j, pnorm * m.Esne51, mj_preshock, Metallicity_j[0]);}
#endif

    double KE_initial = 0, KE_final = 0;
    if(couple_anything_but_scalar_mass_and_metals)
    {
#if defined(COSMIC_RAY_FLUID) && defined(GALSF_FB_FIRE_STELLAREVOLUTION)
        /* Phase 2: CR injection via delta struct (host scatter applies in
         * verify_and_assign_local_mechfb_integrals). j-side oracle gate
         * applied externally below; suppress CR atomics when oracle is on. */
        double crdir[3] = { -dp[0] / r, -dp[1] / r, -dp[2] / r };
#if defined(CR_DYNAMICAL_INJECTION_IN_SNE)
        double cr_to_inject = pnorm * m.CR_energy_to_inject;
#else
        double cr_to_inject = 0;
#endif
        inject_cosmic_rays_into_delta(cr_to_inject, (double)local.SNe_v_ejecta, loop_iteration,
                                       j, crdir, P, CellP, scalars,
                                       num_local_gas, num_local_particles,
                                       home_gas_delta, ghost_gas_delta);
#endif
        double mom_prefactor = scalars.common.cf_atime * m.momentum_to_couple_term_units / Mass_j;
        if(mom_prefactor > 0) {
            for(int k = 0; k < 3; k++) {
                double d_vel = mom_prefactor * pvec[k] + massratio_ejecta * ((double)local.Vel[k] - Vel_j[k]);
                KE_initial += Vel_j_0[k] * Vel_j_0[k];
                Vel_j[k] += d_vel;
                KE_final += Vel_j[k] * Vel_j[k];
            }
            KE_initial *= 0.5 * mj_preshock * scalars.common.cf_a2inv;
            KE_final   *= 0.5 * Mass_j     * scalars.common.cf_a2inv;
        }
        double d_Egy_internal = pnorm * m.U_thermal_residual_tocouple;
        if(m.retain_thermal_flag == 0) d_Egy_internal = 0;
        d_Egy_internal /= Mass_j;
        if(d_Egy_internal > 0) InternalEnergy_j += d_Egy_internal;
    }

    /* J-side oracle gate (Phase 4 / Wave 3 / 3e.1): i-side accum (myout.*)
     * above always completes for both production + oracle passes; j-side
     * atomic writes below are suppressed during the oracle brute pass so
     * the oracle's separate per-active AccumData buffer isn't contaminated
     * by writes to the production gas_delta. The CR injection branch above
     * has its own `if (!oracle_dry_run)` gate (CR helper takes home/ghost). */
    /* atomic accumulation into the per-gas delta struct */
    struct MechFBGasDelta *gd =
        mechfb_target_gas_delta(j, num_local_gas, num_local_particles,
                                home_gas_delta, ghost_gas_delta);
    Kokkos::atomic_add(&gd->N_injected, 1);
    /* Track max wakeup_val across all source events into this receiver,
     * for the host-side direct-dU branch in mechanical_fb.cc to apply as
     * P[j].wakeup = max(P[j].wakeup, max_source_wakeup). Standard hydro-
     * convention encoding: wakeup_val = source.TimeBin + 1. */
    Kokkos::atomic_max(&gd->max_source_wakeup, (int)local.TimeBin + 1);
    Kokkos::atomic_add(&gd->m_injected, Mass_j - Mass_j_0);
    Kokkos::atomic_add(&gd->TE_injected, Mass_j * InternalEnergy_j - Mass_j_0 * InternalEnergy_j_0);
    Kokkos::atomic_add(&gd->KE_injected, KE_final - KE_initial);
#ifdef METALS
    for(int k = 0; k < NUM_METAL_SPECIES; k++) {
        Kokkos::atomic_add(&gd->Z_injected[k], Mass_j * Metallicity_j[k] - Mass_j_0 * Metallicity_j_0[k]);
    }
#endif
    for(int k = 0; k < 3; k++) {
        Kokkos::atomic_add(&gd->p_injected[k], (Mass_j * Vel_j[k] - Mass_j_0 * Vel_j_0[k]) / scalars.common.cf_atime);
    }
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    Kokkos::atomic_add(&gd->Mass_Where_Dust_Shocked, Mass_Where_Dust_Shocked_pair);
#endif
}

#endif /* GALSF_FB_MECHANICAL */
