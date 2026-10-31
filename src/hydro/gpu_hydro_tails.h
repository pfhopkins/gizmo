/* hydro/gpu_hydro_tails.h -- the host work between the gradients pass and the hydro pass, on the device.
 *
 * With GPU_PCIE_OPTIMIZATIONS the rank's own particles stay on the device after a big-step gradients pass
 * (gizmo_particles_keep_owned_on_device) instead of making a bulk round trip over PCIe for three host loops:
 * the gradients writeback, the gradients finalization loop and the hydro preloop. Those loops then run as
 * kernels (hydro/gpu_hydro_tails.cc) on the SAME bodies the host loops run (this header and
 * hydro/gradients_tail.h), so the two paths cannot drift apart. Bitwise contract: the bodies admitted here use
 * only + - * / sqrt, fabs and compares, and the device TU is compiled without FMA contraction (Makefile);
 * a configuration whose bodies call a transcendental or a host-only routine is not admitted and always takes
 * the host loops. GIZMO_GPU_GRADTAIL=0 in the environment selects the host loops in an admitted build.
 */
#ifndef GIZMO_GPU_HYDRO_TAILS_H
#define GIZMO_GPU_HYDRO_TAILS_H

/* Admitted configurations: meshless finite mass, and none of the options that add a transcendental, a
 * host-only call or an unaudited field to the three bodies. */
#if defined(GPU_PCIE_OPTIMIZATIONS) && defined(HYDRO_MESHLESS_FINITE_MASS) && !defined(HYDRO_MESHLESS_FINITE_VOLUME) && !defined(HYDRO_SPH) \
    && !defined(MAGNETIC) && !defined(MHD_CONSTRAINED_GRADIENT) && !defined(MHD_NON_IDEAL) && !defined(MHD_BATTERY_MECHANISMS) \
    && !defined(SPH_TP12_ARTIFICIAL_RESISTIVITY) && !defined(SPHAV_CD10_VISCOSITY_SWITCH) && !defined(KERNEL_CRK_FACES) \
    && !defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORALL) && !defined(DO_DENSITY_AROUND_NONGAS_PARTICLES) \
    && !defined(DOGRAD_INTERNAL_ENERGY) && !defined(DOGRAD_SOUNDSPEED) && !defined(ENERGY_ENTROPY_SWITCH_IS_ACTIVE) \
    && !defined(CONDUCTION) && !defined(VISCOSITY) && !defined(TURB_DIFFUSION) && !defined(TURB_DIFF_DYNAMIC) && !defined(TURB_DIFF_METALS) \
    && !defined(COSMIC_RAY_FLUID) && !defined(RADTRANSFER) && !defined(RT_SOLVER_EXPLICIT) && !defined(RT_COMPGRAD_EDDINGTON_TENSOR) \
    && !defined(TWO_TEMPERATURE_PLASMA) && !defined(TRANSPORT_SUBCYCLE) && !defined(OUTPUT_SHOCK_MACH_NUMBER) \
    && !defined(EOS_ELASTIC) && !defined(EOS_TILLOTSON) && !defined(EOS_ANEOS) && !defined(GALSF) && !defined(SINGLE_STAR_SINK_FORMATION)
#define GIZMO_HYDRO_TAILS_DEVICE 1
#else
#define GIZMO_HYDRO_TAILS_DEVICE 0
#endif

/* The shared bodies are host+device functions only where the device TU instantiates them; everywhere else
 * they are the plain host code they were. */
#if GIZMO_HYDRO_TAILS_DEVICE && defined(__CUDACC__)
#define HYDRO_TAIL_FUNCTION static inline __host__ __device__
#else
#define HYDRO_TAIL_FUNCTION static inline
#endif

/* Device entry points (hydro/gpu_hydro_tails.cc). Each returns 1 when it did the work on the device and 0 when
 * the caller must run its host loop: always 0 in a configuration that is not admitted, with GIZMO_GPU_GRADTAIL=0,
 * or when the particles are not being kept on the device. */
#ifdef __cplusplus
extern "C" {
#endif
/* After the gradients pass, before its writeback: keep the rank's own particles on the device for the writeback,
 * the finalization loop and the hydro preloop (n_active rows; corridor_list: the pass consumed the hydro
 * corridor's list). Moves the imported ghosts to the host, which the ghost refresh before the hydro pass writes. */
int gizmo_gradients_tail_begin(int corridor_list, int n_active);
/* The gradients finalization loop over the active list, when the writeback ran on the device. */
int gizmo_gradients_finalize_on_device(void);
/* The hydro preloop over the active list, while the rank's own particles are kept on the device. */
int gizmo_hydro_preloop_on_device(void);
#ifdef __cplusplus
}
#endif

/* Body of hydro_force_initial_operations_preloop for one active gas cell i. */
HYDRO_TAIL_FUNCTION void hydro_preloop_cell(struct particle_data *Pp, struct gas_cell_data *Cp, int i)
{
    int k = 0; (void) k;
    Cp[i].MaxSignalVel = MIN_REAL_NUMBER;
#ifdef MHD_NON_IDEAL_EVOLVED_FLUX
    {   /* the face fluxes must see the evolved non-ideal field relaxed toward the current from THIS step's gradients, as the
           standard solver uses them: the drift predicted it with the previous step's gradients, which lags the non-ideal
           EMF one step behind the ideal one. Re-predict it from the kicked value over the second half-step with the fresh
           gradients -- the same update the second half-step kick then applies to the kicked value. */
        integertime ti_step = Pp[i].integertime_step();
        double dt_second_half = (ti_step - ti_step/2) * unit_integertime_in_physical(i, Pp);
        Cp[i].NonIdealMHDFluxPred = Cp[i].NonIdealMHDFlux;
        nonideal_mhd_flux_update_driftkick(i, dt_second_half, 1, Pp, Cp);
    }
#endif
#ifdef OUTPUT_SHOCK_MACH_NUMBER
    Cp[i].ShockMachNumber = 0;
#endif
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    Cp[i].MaxKineticEnergyNgb = MIN_REAL_NUMBER;
#endif
    Cp[i].DtInternalEnergy = 0; //Cp[i].dInternalEnergy = 0;//manifest-indiv-timestep-debug//
#if defined(TWO_TEMPERATURE_PLASMA) && (TWO_TEMPERATURE_PLASMA & 4) && defined(CONDUCTION)
    Cp[i].DtInternalEnergy_FromConduction = 0;
#endif
    Cp[i].HydroAccel = {};
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    Cp[i].DtMass = 0; Cp[i].dMass = 0; Cp[i].GravWorkTerm = {};
#endif
#if defined(TURB_DIFF_METALS) || (defined(METALS) && defined(HYDRO_MESHLESS_FINITE_VOLUME))
    for(k=0;k<NUM_METAL_SPECIES;k++) {Cp[i].Dyield[k] = 0;}
#endif
#if defined(RT_SOLVER_EXPLICIT)
#if defined(RT_EVOLVE_ENERGY)
    for(k=0;k<N_RT_FREQ_BINS;k++) {Cp[i].Dt_Rad_E_gamma[k] = 0;}
#ifdef TRANSPORT_SUBCYCLE
    for(k=0;k<N_RT_FREQ_BINS;k++) {Cp[i].Dt_Rad_E_gamma_Work[k] = 0;}
#endif
#endif
#if defined(RT_EVOLVE_FLUX)
    for(k=0;k<N_RT_FREQ_BINS;k++) {Cp[i].Dt_Rad_Flux[k] = {};}
#endif
#if defined(RT_INFRARED)
    Cp[i].Dt_Rad_E_gamma_T_weighted_IR = 0;
#endif
#if defined(RT_EVOLVE_FLUX)
    for(k=0;k<N_RT_FREQ_BINS;k++) {Cp[i].Dt_Rad_Flux[k] = {};}
#endif
#if defined(RT_EVOLVE_INTENSITIES)
    for(k=0;k<N_RT_FREQ_BINS;k++) {int k_dir; for(k_dir=0;k_dir<N_RT_INTENSITY_BINS;k_dir++) {Cp[i].Dt_Rad_Intensity[k][k_dir] = 0;}}
#endif
#endif
#ifdef MAGNETIC
    Cp[i].divB = 0; Cp[i].Face_Area = {};
#ifdef DIVBCLEANING_DEDNER
    Cp[i].DtB_PhiCorr = {};
#endif
#ifndef HYDRO_SPH
    Cp[i].DtB = {};
#ifdef DIVBCLEANING_DEDNER
    Cp[i].DtPhi = 0;
#endif
#endif
#endif // magnetic //
#ifdef COSMIC_RAY_FLUID
    Cp[i].Face_DivVel_ForAdOps = 0;
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    Cp[i].DtCREgyNewInjectionFromShocks = 0;
#endif
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        Cp[i].DtCosmicRayEnergy[k] = 0;
#if defined(TRANSPORT_SUBCYCLE) && !defined(CRFLUID_EVOLVE_SCATTERINGWAVES) && !defined(CRFLUID_EVOLVE_SPECTRUM)
        Cp[i].DtCosmicRayEnergy_StreamingLoss[k] = 0;
#endif
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        Cp[i].DtCosmicRay_Number_in_Bin[k] = 0;
#endif
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
        int kAlf; for(kAlf=0;kAlf<2;kAlf++) {Cp[i].DtCosmicRayAlfvenEnergy[k][kAlf] = 0;}
#endif
    }
#endif
    Pp[i].wakeup = 0;
}

#endif /* GIZMO_GPU_HYDRO_TAILS_H */
