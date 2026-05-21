/* density_gpu.cc — GPU-accelerated hydro evaluator via Kokkos.
 *
 * GPU translation unit (Kokkos/nvcc_wrapper). Provides hydro_evaluate_gpu()
 * — Riemann/flux loop (B4).
 *
 * The legacy density_evaluate_gpu / density_gpu_session_* path was retired
 * in Phase 4.B.2 when density() migrated to the runner template
 * (hydro/density_loop.cc). The legacy gradient_evaluate_gpu walker was
 * retired in hydro corridor commit 7 (GradientsSpec) + commit 7b cleanup
 * — gradients now live in hydro/gradients_loop.{h,cc}.
 *
 * Architecture follows the cooling.cc pattern:
 *   - declarations/gpu_all_mirror.h provides static __managed__ AllDeviceMirror
 *     per TU with device-pass `#define All AllDeviceMirror` redirect (93897f62)
 *   - Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE> for device-accessible data
 *   - CPU gather → GPU kernel → CPU scatter
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

/* Standard and Kokkos headers BEFORE global_data_all_struct.h
 * (macros.h #define terminate(x) conflicts with std::terminate in <exception>) */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
#include "../core/timestep_functions.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/sfc_tiles_functions.h"
#include "../system/gpu_particles_arena.h"
#include "gradient_functions.h"
#include "hydro_structs.h"
#include "../system/eigen_symmetric.h"
#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#ifndef HYDRO_SPH
#include "reimann.h"
#endif
/* Define xchange macro names so hydro_functions.h can reference the structs */
#define INPUT_STRUCT_NAME hydro_data_in
#define OUTPUT_STRUCT_NAME hydro_data_out
#include "hydro_functions.h"
#undef INPUT_STRUCT_NAME
#undef OUTPUT_STRUCT_NAME


/* TILE_PERIODIC_X/Y/Z defined in sfc_tiles.h (included via gpu_neighbor_list.h) */

/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */


/* ================================================================
   GPU neighbor list construction (B1)
   ================================================================
   Strategy:
   - Build tiles + BVH on CPU (recursive, serial — fine for ~100s of tiles)
   - Copy tiles, BVH, pool, active_indices to SharedSpace
   - Pass 1: Kokkos::parallel_for counting neighbors per particle
   - Exclusive prefix scan → CSR offsets
   - Pass 2: Kokkos::parallel_for filling neighbor indices
   ================================================================ */

/* Structs and build/free functions are in mesh/gpu_neighbor_list.h/.cc,
   shared with the symmetric neighbor list build in accel.cc. */
#include "../mesh/gpu_neighbor_list.h"

/* Density's spatial index is now the module-shared g_step_sidx
 * (mesh/gpu_neighbor_list.cc). */

/* gradient_evaluate_gpu retired in hydro corridor commit 7 (GradientsSpec) +
 * cleanup commit 7b. Gradients now run via the runner Spec contract — see
 * hydro/gradients_loop.{h,cc}. */

/* ================================================================
   GPU hydro force kernel (B3b)
   ----------------------------------------------------------------
   Kernel writes (host-visible, by index):
     i-side (active sources, indexed by aa->ii):
       kout[aa]                            — per-active output (flux deltas, MaxSignalVel, etc.)
     j-side (neighbors, indexed by j = neighbors[idx]):
       P_gpu[j].wakeup                     — HYDRO_ATOMIC_MAX
     #ifdef HYDRO_MESHLESS_FINITE_VOLUME
       CellP_gpu[j].dMass                  — HYDRO_ATOMIC_ADD
     #endif
     scalar:
       *NeedToWakeup                       — HYDRO_ATOMIC_STORE on wakeup trigger

   Sparse-scatter target: walk neighbors[0..csr_total_pairs] and scatter
   only the (deduplicated by idempotent assignment) j fields above. NEVER
   memcpy the full P/CellP arrays — those are populated host->arena once
   per arena cycle and not mutated by the hydro kernel.
   ================================================================ */

void hydro_evaluate_gpu(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                        int num_total, int *active_indices_host, int num_active,
                        int64_t *csr_offsets_host, int *csr_neighbors_host, int64_t csr_total_pairs,
                        void *out_host_void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    struct hydro_data_out *out_host = (struct hydro_data_out *)out_host_void;
    double t_hyd_start = my_second(); /* Phase 7 sub-bucket timing */

    /* Persistent decomp-scoped arena (Step 13 Phase 1). Replaces per-call
     * SharedSpace alloc+memcpy. Fast path skips memcpy when arena is valid. */
    gpu_particles_arena_set_site("density_gpu_hydro");
    gpu_particles_arena_acquire(num_total, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();
    double t_hyd_arena = my_second();

    /* Copy CSR neighbor list to SharedSpace (offsets 64-bit; neighbor values int) */
    int64_t *d_offsets = (int64_t *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((size_t)(num_active + 1) * sizeof(int64_t));
    int *d_neighbors = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((size_t)((csr_total_pairs > 0) ? csr_total_pairs : 1) * sizeof(int));
    int *d_active = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(int));
    memcpy(d_offsets, csr_offsets_host, (size_t)(num_active + 1) * sizeof(int64_t));
    memcpy(d_neighbors, csr_neighbors_host, (size_t)csr_total_pairs * sizeof(int));
    memcpy(d_active, active_indices_host, num_active * sizeof(int));
    double t_hyd_csr_copy = my_second();

    /* Copy TimeBinActive to SharedSpace */
    int *d_TimeBinActive = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(TIMEBINS * sizeof(int));
    memcpy(d_TimeBinActive, TimeBinActive, TIMEBINS * sizeof(int));

    /* Wakeup flag in SharedSpace */
    int *d_NeedToWakeup = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sizeof(int));
    *d_NeedToWakeup = 0;

    /* Output array in SharedSpace */
    struct hydro_data_out *d_out = (struct hydro_data_out *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(((num_active > 0) ? num_active : 1) * sizeof(struct hydro_data_out));

    PRINT_STATUS("  GPU hydro: %d active, %lld pairs", num_active, (long long)csr_total_pairs);

    /* Verification harness (β): GIZMO_VERIFY_KERNEL_WRITES=1 snapshots the
     * arena fields the kernel is permitted to write (P[j].wakeup, CellP[j].dMass
     * MFV-only) so the post-kernel sparse scatter can prove the kernel touched
     * ONLY j's that appear in csr_neighbors_host[]. O(N) cost, off by default.
     * Used once after each new wrapper sparse-scatter conversion to confirm
     * the kernel-writes header comment is complete. */
    static int gizmo_verify_init = 0, gizmo_verify_on = 0;
    if(!gizmo_verify_init) {
        gizmo_verify_init = 1;
        gizmo_verify_on = (getenv("GIZMO_VERIFY_KERNEL_WRITES") != nullptr) ? 1 : 0;
    }
    short int *verify_snap_wakeup = nullptr;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    MyDouble *verify_snap_dMass = nullptr;
#endif
    if(gizmo_verify_on && num_total > 0) {
        verify_snap_wakeup = (short int *) malloc((size_t)num_total * sizeof(short int));
        for(int j = 0; j < num_total; j++) verify_snap_wakeup[j] = P_gpu[j].wakeup;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        verify_snap_dMass = (MyDouble *) malloc((size_t)num_total * sizeof(MyDouble));
        for(int j = 0; j < num_total; j++) verify_snap_dMass[j] = CellP_gpu[j].dMass;
#endif
    }

#if defined(GALSF_ISMDUSTCHEM_MODEL)
    /* Pre-compute ISMDustChem passive scalar diffusion values on host.
     * return_ismdustchem_species_of_interest_for_diffusion_and_yields() reads
     * All.* grain tables not mirrored to All_dev, so it is not device-callable. */
    double *d_ismdc = nullptr;
    const int n_ismdc = NUM_ISMDUSTCHEM_PASSIVE_SCALARS;
    if(n_ismdc > 0 && num_active > 0) {
        double *ismdc_host = new double[(size_t)num_active * n_ismdc];
        for(int aa = 0; aa < num_active; aa++) {
            int ii = active_indices_host[aa];
            for(int ki = 0; ki < n_ismdc; ki++) {
                ismdc_host[aa * n_ismdc + ki] =
                    return_ismdustchem_species_of_interest_for_diffusion_and_yields(
                        ii, ISMDUSTCHEM_SPECIES_OFFSET_IN_METALLICITY + ki, 0, CellP_host);
            }
        }
        d_ismdc = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>((size_t)num_active * n_ismdc * sizeof(double));
        memcpy(d_ismdc, ismdc_host, (size_t)num_active * n_ismdc * sizeof(double));
        delete[] ismdc_host;
    }
#endif

    /* GPU hydro force kernel */
    {
        int64_t *offsets = d_offsets;
        int *neighbors = d_neighbors;
        int *active = d_active;
        struct particle_data *kp = P_gpu;
        struct gas_cell_data *kc = CellP_gpu;
        struct hydro_data_out *kout = d_out;
        int *kTimeBinActive = d_TimeBinActive;
        int *kNeedWakeup = d_NeedToWakeup;
#if defined(GALSF_ISMDUSTCHEM_MODEL)
        double *ismdc = d_ismdc;
        const int ismdc_n = n_ismdc;
#endif

        Kokkos::parallel_for("hydro_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];

            /* particle2in equivalent: load searching particle data */
            struct hydro_data_in local;
            memset(&local, 0, sizeof(local));
            local.Pos = kp[ii].Pos;
            local.Vel = kc[ii].VelPred;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            local.ParticleVel = kc[ii].VelPred;
#endif
            local.KernelRadius = kp[ii].KernelRadius;
            local.Mass = kp[ii].Mass;
            local.Density = kc[ii].Density;
            local.Pressure = kc[ii].Pressure;
            local.ConditionNumber = kc[ii].ConditionNumber;
#ifdef MHD_CONSTRAINED_GRADIENT
            if(kc[ii].FlagForConstrainedGradients == 0) {local.ConditionNumber *= -1;} /* sign encodes gradient flag status, matching particle2in_hydra */
#endif
            local.FaceClosureError = kc[ii].FaceClosureError;
            local.InternalEnergyPred = kc[ii].InternalEnergyPred;
            local.SoundSpeed = kc[ii].effective_soundspeed();
            local.dt_hydrostep_i = get_particle_timestep_in_physical(ii, kp);
            local.DrkernNgbFactor = kp[ii].DrkernNgbFactor;
            local.Gradients.Density = kc[ii].Gradients.Density;
            local.Gradients.Pressure = kc[ii].Gradients.Pressure;
            local.Gradients.Velocity = kc[ii].Gradients.Velocity;
            local.NV_T = kc[ii].NV_T;
            local.TimeBin = kp[ii].TimeBin;
#ifdef MAGNETIC
            local.BPred = kc[ii].Bfield(); /* Bfield() = BPred * Density/Mass: convert from mass-weighted storage to physical B */
            local.Gradients.B = kc[ii].Gradients.B;
#ifdef MHD_BATTERY_MECHANISMS
#if (MHD_BATTERY_MECHANISMS & 1)
            local.Gradients.ElectronNumberDensity = kc[ii].Gradients.ElectronNumberDensity;
            local.Gradients.ElectronTemperature = kc[ii].Gradients.ElectronTemperature;
#endif
#endif
#ifdef DIVBCLEANING_DEDNER
            local.PhiPred = kc[ii].PhiPred / kp[ii].Mass;
            local.Gradients.Phi = kc[ii].Gradients.Phi;
#endif
#ifdef MHD_MODIFIED_GRADIENT
            local.MG_cgcoeff = kc[ii].MG_cgcoeff;
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
            local.Gradients.InternalEnergy = kc[ii].Gradients.InternalEnergy;
#endif
#ifdef DOGRAD_SOUNDSPEED
            local.Gradients.SoundSpeed = kc[ii].Gradients.SoundSpeed;
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
            for(int k=0;k<NUM_METAL_SPECIES;k++) {local.Metallicity[k] = kp[ii].Metallicity[k];}
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k=0;k<N_CR_PARTICLE_BINS;k++) {local.CosmicRayPressure[k] = Get_Gas_CosmicRayPressure(ii, k, kc);}
#endif
#ifdef CONDUCTION
            local.Kappa_Conduction = kc[ii].Kappa_Conduction;
#endif
#ifdef VISCOSITY
            local.Eta_ShearViscosity = kc[ii].Eta_ShearViscosity;
            local.Zeta_BulkViscosity = kc[ii].Zeta_BulkViscosity;
#endif
#ifdef TURB_DIFFUSION
            local.TD_DiffCoeff = kc[ii].TD_DiffCoeff;
#endif
#if defined(KERNEL_CRK_FACES)
            for(int k=0;k<16;k++) {local.Tensor_CRK_Face_Corrections[k] = kc[ii].Tensor_CRK_Face_Corrections[k];}
#endif
#ifdef HYDRO_SPH
            local.DrkernHydroSumFactor = kc[ii].DrkernHydroSumFactor;
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
            local.alpha = kc[ii].alpha_limiter * kc[ii].alpha;
#else
            local.alpha = kc[ii].alpha_limiter;
#endif
#endif
#ifdef HYDRO_PRESSURE_SPH
            local.EgyWtRho = kc[ii].EgyWtDensity;
#endif
#ifdef MHD_NON_IDEAL
            local.Eta_MHD_OhmicResistivity_Coeff = kc[ii].Eta_MHD_OhmicResistivity_Coeff;
            local.Eta_MHD_HallEffect_Coeff = kc[ii].Eta_MHD_HallEffect_Coeff;
            local.Eta_MHD_AmbiPolarDiffusion_Coeff = kc[ii].Eta_MHD_AmbiPolarDiffusion_Coeff;
#endif
#if defined(SPH_TP12_ARTIFICIAL_RESISTIVITY)
            local.Balpha = kc[ii].Balpha;
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
            for(int k=0;k<NUM_METAL_SPECIES;k++) {local.Gradients.Metallicity[k] = kc[ii].Gradients.Metallicity[k];}
#endif
#if defined(GALSF_ISMDUSTCHEM_MODEL) && (defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME)))
            /* local.Metallicity field gated by TURB_DIFF_METALS||(METALS&&MFV)
             * in hydro_structs.h:129 — original bare ISMDUSTCHEM gate here
             * was over-broad and never compiled for METALS+MFM+ISMDC. */
            for(int ki=0;ki<ismdc_n;ki++) {local.Metallicity[ISMDUSTCHEM_SPECIES_OFFSET_IN_METALLICITY+ki] = ismdc[aa*ismdc_n+ki];}
#endif
#ifdef CHIMES_TURB_DIFF_IONS
            for(int k=0;k<ChimesGlobalVars.totalNumberOfSpecies;k++) {local.ChimesNIons[k] = kc[ii].ChimesNIons[k];}
#endif
#if defined(RT_SOLVER_EXPLICIT) && defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
            for(int k=0;k<N_RT_FREQ_BINS;k++) {local.Gradients.Rad_E_gamma_ET[k] = kc[ii].Gradients.Rad_E_gamma_ET[k];}
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            for(int k=0;k<N_RT_FREQ_BINS;k++) {
                for(int k2=0;k2<3;k2++) {local.Gradients.Rad_E_gamma_Grad[k][k2] = kc[ii].Gradients.Rad_E_gamma_Grad[k][k2];}
                for(int k2=0;k2<3;k2++) {for(int k3=0;k3<3;k3++) {local.Gradients.Rad_Flux_Grad[k][k2][k3] = kc[ii].Gradients.Rad_Flux_Grad[k][k2][k3];}}
            }
#endif
#ifdef RT_SOLVER_EXPLICIT
            for(int k=0;k<N_RT_FREQ_BINS;k++) {
                local.Rad_E_gamma[k] = kc[ii].Rad_E_gamma_Pred[k];
                local.Rad_Kappa[k] = kc[ii].Rad_Kappa[k];
                local.RT_DiffusionCoeff[k] = rt_diffusion_coefficient(ii, k, kc);
#if defined(RT_EVOLVE_FLUX) || defined(HYDRO_SPH)
                local.ET[k] = kc[ii].ET[k];
#endif
#ifdef RT_EVOLVE_FLUX
                local.Rad_Flux[k] = kc[ii].Rad_Flux_Pred[k];
#endif
#if defined(RT_EVOLVE_INTENSITIES)
                for(int k2=0;k2<N_RT_INTENSITY_BINS;k2++) {local.Rad_Intensity_Pred[k][k2] = kc[ii].Rad_Intensity_Pred[k][k2];}
#endif
            }
#ifdef RT_INFRARED
            local.Radiation_Temperature = kc[ii].Radiation_Temperature;
#endif
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k=0;k<N_CR_PARTICLE_BINS;k++) {
                local.CosmicRayDiffusionCoeff[k] = kc[ii].CosmicRayDiffusionCoeff[k];
                local.CosmicRayFlux[k] = kc[ii].CosmicRayFluxPred[k];
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
                for(int k2=0;k2<2;k2++) {local.CosmicRayAlfvenEnergy[k][k2] = kc[ii].CosmicRayAlfvenEnergyPred[k][k2];}
#endif
#ifdef CRFLUID_EVOLVE_SPECTRUM
                local.CR_number_to_energy_ratio[k] = kc[ii].CosmicRay_Number_in_Bin[k] / (kc[ii].CosmicRayEnergy[k] + MIN_REAL_NUMBER);
                local.CR_number_to_energy_ratio[k] *= kc[ii].Flux_Number_to_Energy_Correction_Factor[k];
#endif
            }
#endif
#if defined(EOS_ELASTIC) || defined(EOS_ANEOS)
            local.CompositionType = kc[ii].CompositionType;
#endif
#ifdef EOS_ELASTIC
            for(int k=0;k<3;k++) {for(int k2=0;k2<3;k2++) {local.Elastic_Stress_Tensor[k][k2] = kc[ii].Elastic_Stress_Tensor_Pred[k][k2];}}
#endif
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1)
            local.Damage = kc[ii].Damage;
#endif
#ifdef GALSF_SUBGRID_WINDS
            local.DelayTime = kc[ii].DelayTime;
#endif

            /* Initialize output and workspace */
            struct hydro_data_out out;
            memset(&out, 0, sizeof(out));
            struct kernel_hydra kernel;
            memset(&kernel, 0, sizeof(kernel));
            struct Conserved_var_Riemann Fluxes;

            kernel.h_i = local.KernelRadius;
            kernel.sound_i = local.SoundSpeed;
            kernel.spec_egy_u_i = local.InternalEnergyPred;
            out.MaxSignalVel = kernel.sound_i; /* must match hydro_evaluate.h line 82 */
#ifdef MAGNETIC
            {
                double fac_magnetic_pressure_loc = 1.0 / All.cf_atime; /* B*B*fac = pressure units */
                kernel.b2_i = local.BPred.norm_sq(); /* raw B^2, without fac_magnetic_pressure */
                kernel.alfven2_i = kernel.b2_i * fac_magnetic_pressure_loc / local.Density;
                kernel.alfven2_i = DMIN(kernel.alfven2_i, 1000. * kernel.sound_i * kernel.sound_i);
            }
#endif

            /* Accumulate over symmetric neighbors */
            for(int64_t idx = offsets[aa]; idx < offsets[aa + 1]; idx++)
            {
                int j = neighbors[idx];
                memset(&Fluxes, 0, sizeof(Fluxes));
                hydro_accumulate_neighbor(local, out, kernel, Fluxes, j,
                                          local.dt_hydrostep_i, kp, kc,
                                          kTimeBinActive, kNeedWakeup,
                                          /*allow_j_writes=*/true);
            }

            /* Store output for this particle */
            kout[aa] = out;
        }); /* end KOKKOS_LAMBDA */
        Kokkos::fence();

        gizmo_gpu_check_last_error("hydro kernel", num_active);
    }
    double t_hyd_kernel = my_second();

    /* Copy output back to host */
    memcpy(out_host, d_out, num_active * sizeof(struct hydro_data_out));

    /* Copy wakeup flag back */
    if(*d_NeedToWakeup) NeedToWakeupParticles_local = 1;
    double t_hyd_out_copy = my_second();

    /* SPARSE scatter (replaces former full-num_total loop). The kernel writes
     * only P_gpu[j].wakeup (atomic_max) and, MFV-only, CellP_gpu[j].dMass
     * (atomic_add) — and only for j's that appear in the CSR neighbors[].
     * Walk the host-side CSR directly; idempotent assignment makes duplicate
     * j's across active-i ranges correct without a dedupe pass.
     * Ghost writeback below handles MPI communication of these deltas. */
    for(int64_t idx = 0; idx < csr_total_pairs; idx++) {
        int j = csr_neighbors_host[idx];
        P_host[j].wakeup = P_gpu[j].wakeup;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP_host[j].dMass = CellP_gpu[j].dMass;
#endif
    }
    double t_hyd_scatter = my_second();

    /* Verification harness (β): for any untouched j, P_gpu[j] must match the
     * pre-kernel snapshot. Any violation means the kernel-writes header is
     * incomplete (a write the audit missed). Off by default. */
    if(verify_snap_wakeup) {
        char *touched = (char *) calloc((size_t)num_total, sizeof(char));
        for(int64_t idx = 0; idx < csr_total_pairs; idx++) touched[csr_neighbors_host[idx]] = 1;
        long wakeup_violations = 0;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        long dMass_violations = 0;
#endif
        for(int j = 0; j < num_total; j++) {
            if(touched[j]) continue;
            if(P_gpu[j].wakeup != verify_snap_wakeup[j]) wakeup_violations++;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            if(CellP_gpu[j].dMass != verify_snap_dMass[j]) dMass_violations++;
#endif
        }
        if(wakeup_violations) {
            printf("[VERIFY_KERNEL_WRITES] hydro: %ld untouched j's had P_gpu.wakeup mutated (rank=%d, num_total=%d, csr_pairs=%lld)\n",
                   wakeup_violations, ThisTask, num_total, (long long)csr_total_pairs);
            fflush(stdout);
        }
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        if(dMass_violations) {
            printf("[VERIFY_KERNEL_WRITES] hydro: %ld untouched j's had CellP_gpu.dMass mutated (rank=%d)\n",
                   dMass_violations, ThisTask);
            fflush(stdout);
        }
        free(verify_snap_dMass);
#endif
        free(touched);
        free(verify_snap_wakeup);
    }

    /* Cleanup SharedSpace (P/CellP owned by arena — do NOT free here).
     * The scatter above syncs wakeup+dMass; any other arena-side writes by
     * the kernel may be unscattered, so invalidate to force the next kernel
     * to re-acquire from host. */
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    if(d_ismdc) { Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_ismdc); }
#endif
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_NeedToWakeup);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_TimeBinActive);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_neighbors);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_offsets);
    gpu_particles_arena_invalidate();

    /* Phase 7 sub-buckets — env-gated; no-op when GIZMO_VERBOSE_DIAG off */
    {
        double t_postloop_end = my_second();
        gizmo_step_phase_record("hydro_arena",       timediff(t_hyd_start,    t_hyd_arena));
        gizmo_step_phase_record("hydro_csr_copy",    timediff(t_hyd_arena,    t_hyd_csr_copy));
        gizmo_step_phase_record("hydro_kernel",      timediff(t_hyd_csr_copy, t_hyd_kernel));
        gizmo_step_phase_record("hydro_out_copy",    timediff(t_hyd_kernel,   t_hyd_out_copy));
        gizmo_step_phase_record("hydro_scatter",     timediff(t_hyd_out_copy, t_hyd_scatter));
        gizmo_step_phase_record("hydro_postloop",    timediff(t_hyd_scatter,  t_postloop_end));
    }
}

