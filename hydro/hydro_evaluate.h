/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
/*! This function is the 'core' of the hydro force computation. A target
*  particle is specified which may either be local, or reside in the
*  communication buffer.
*   In this routine, we find the gas particle neighbors, and do the loop over
*  neighbors to calculate the hydro fluxes. The actual flux calculation,
*  and the returned values, should be in PHYSICAL (not comoving) units */
/*!
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
/*!   -- this subroutine writes to shared memory [updating -some- essential neighbor values, setting wakeups, etc.]:
  this should ideally be avoided whenever possible; need to protect these write operations for openmp below.
  the legacy 'j_is_active_for_fluxes' symmetric-dispatch path has been removed; each particle now
  accumulates all fluxes as the i-side and conservation is preserved by the symmetric neighbor
  list visiting each pair from both directions. the remaining shared writes
  (MFV dMass conservation, wakeups) are either guarded by a timestep criterion or use OMP atomics. */
/* --------------------------------------------------------------------------------- */

#include "hydro_pair_types.h"
#include "compute_finitevol_faces_functions.h"
#include "hydro_core_meshless_functions.h"
#include "hydro_core_sph_functions.h"
#include "conduction_functions.h"
#include "viscosity_functions.h"
#include "nonideal_mhd_functions.h"
#include "../solids/elastic_stress_tensor_force_functions.h"
#include "../turb/turbulent_diffusion_functions.h"
#include "../turb/chimes_turbulent_ion_diffusion_functions.h"
#include "../eos/cosmic_ray_fluid/cosmic_ray_diffusion_functions.h"
#include "../radiation/rt_direct_ray_transport_functions.h"
#include "../radiation/rt_diffusion_explicit_functions.h"

int hydro_force_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int j, k, n, startnode, numngb, kernel_mode, listindex;
    double hinv_i,hinv3_i,hinv4_i,hinv_j,hinv3_j,hinv4_j,V_i,V_j,dt_hydrostep_i,dt_hydrostep_j,dt_hydrostep,r2,rinv,rinv_soft,u,Particle_Size_i;
    double v_hll = 0;
    struct kernel_hydra kernel;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    struct Conserved_var_Riemann Fluxes;
    listindex = 0;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));
    memset(&kernel, 0, sizeof(struct kernel_hydra));
    memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
#ifndef HYDRO_SPH
    struct Input_vec_Riemann Riemann_vec;
    struct Riemann_outputs Riemann_out;
    memset(&Riemann_vec, 0, sizeof(struct Input_vec_Riemann));
    memset(&Riemann_out, 0, sizeof(struct Riemann_outputs));
    double face_area_dot_vel;
    face_area_dot_vel = 0;
#endif
    double face_vel_i=0, face_vel_j=0, Face_Area_Norm=0; Vec3<double> Face_Area_Vec;

#ifdef HYDRO_MESHLESS_FINITE_MASS
    double epsilon_entropic_eos_big, epsilon_entropic_eos_small;
    epsilon_entropic_eos_big = 0.5; // can be anything from (small number=more diffusive, less accurate entropy conservation) to ~1.1-1.3 (least diffusive, most noisy)
    epsilon_entropic_eos_small = 1.e-3; // should be << epsilon_entropic_eos_big
#if defined(FORCE_ENTROPIC_EOS_BELOW)
    epsilon_entropic_eos_small = FORCE_ENTROPIC_EOS_BELOW; // if set manually
#elif !defined(SELFGRAVITY_OFF)
    epsilon_entropic_eos_small = 1.e-2; epsilon_entropic_eos_big = 0.6; // with gravity larger tolerance behaves better on hydrostatic equilibrium problems //
#endif
#endif

    if(mode == 0)
    {
        particle2in_hydra(&local, target, loop_iteration); // this setup allows for all the fields we need to define (don't hard-code here)
    }
    else
    {
        local = DATAGET_NAME[target]; // this setup allows for all the fields we need to define (don't hard-code here)
    }

    /* certain particles should never enter the loop: check for these */
    if(local.Mass <= 0) return 0;
    if(local.Density <= 0) return 0;
#ifdef GALSF_SUBGRID_WINDS
    if(local.DelayTime > 0) {return 0;}
#endif

    /* --------------------------------------------------------------------------------- */
    /* pre-define Particle-i based variables (so we save time in the loop below) */
    /* --------------------------------------------------------------------------------- */
    kernel.sound_i = local.SoundSpeed;
    kernel.spec_egy_u_i = local.InternalEnergyPred;
    kernel.h_i = local.KernelRadius;
    kernel_hinv(kernel.h_i, &hinv_i, &hinv3_i, &hinv4_i);
    hinv_j=hinv3_j=hinv4_j=0;
    V_i = local.Mass / local.Density;
    Particle_Size_i = pow(V_i,1./NUMDIMS) * All.cf_atime; // in physical, used below in some routines //
    out.MaxSignalVel = kernel.sound_i;
    kernel_mode = 0; /* need dwk and wk */
    double cnumcrit2; cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - local.ConditionNumber*local.ConditionNumber;
#if defined(HYDRO_SPH)
#ifdef HYDRO_PRESSURE_SPH
    kernel.p_over_rho2_i = local.Pressure / (local.EgyWtRho*local.EgyWtRho);
#else
    kernel.p_over_rho2_i = local.Pressure / (local.Density*local.Density);
#endif
#endif

    /* magfluxv / resistivity_heatflux lifted out of MAGNETIC+HYDRO_SPH so the
       hydro_core_sph function can take them as output args. Zero otherwise. */
    Vec3<double> magfluxv = {};
    double resistivity_heatflux = 0;
#ifdef MAGNETIC
    kernel.b2_i = local.BPred.norm_sq();
#if defined(HYDRO_SPH)
    kernel.mf_i = local.Mass * fac_magnetic_pressure / (local.Density * local.Density);
    kernel.mf_j = local.Mass * fac_magnetic_pressure;
#endif
    kernel.alfven2_i = kernel.b2_i * fac_magnetic_pressure / local.Density;
    kernel.alfven2_i = DMIN(kernel.alfven2_i, 1000. * kernel.sound_i*kernel.sound_i);
    double vcsa2_i = kernel.sound_i*kernel.sound_i + kernel.alfven2_i;
#endif // MAGNETIC //

#ifdef RT_SOLVER_EXPLICIT
    double tau_c_i[N_RT_FREQ_BINS]; for(k=0;k<N_RT_FREQ_BINS;k++) {tau_c_i[k] = Particle_Size_i * local.Rad_Kappa[k]*local.Density*All.cf_a3inv;}
#endif

    /* --------------------------------------------------------------------------------- */
    /* Now start the actual hydrodynamic force computation for this particle */
    /* --------------------------------------------------------------------------------- */
    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = DATAGET_NAME[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            /* --------------------------------------------------------------------------------- */
            /* get the neighbor list */
            /* --------------------------------------------------------------------------------- */
            numngb = ngb_treefind_pairs_threads(local.Pos, kernel.h_i, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb < 0) {return -2;}

            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if(P[j].Mass <= 0) {continue;}
                if(CellP[j].Density <= 0) {continue;}
#ifdef GALSF_SUBGRID_WINDS
                if(CellP[j].DelayTime > 0) {continue;} /* no hydro forces for decoupled wind particles */
#endif

                /* check if I need to compute this pair-wise interaction from "i" to "j", or skip it and let it be computed from "j" to "i" */
                dt_hydrostep_j = get_particle_timestep_in_physical(j);
                dt_hydrostep = DMAX(dt_hydrostep_i , dt_hydrostep_j); // this is used for flux-limiting, so we always want to be more conservative and use the larger timestep //
                kernel.dp = local.Pos - P[j].Pos;
                nearest_xyz(kernel.dp); /* find the closest image in the given box size  */
                r2 = kernel.dp.norm_sq();
                kernel.h_j = P[j].KernelRadius;

                /* force applied for all particles inside each-others kernels! */
                if((r2 >= kernel.h_i * kernel.h_i) && (r2 >= kernel.h_j * kernel.h_j)) continue;
                if(r2 <= 0) continue;

                /* --------------------------------------------------------------------------------- */
                /* ok, now we definitely have two interacting particles */
                /* --------------------------------------------------------------------------------- */

                /* --------------------------------------------------------------------------------- */
                /* calculate a couple basic properties needed: separation, velocity difference (needed for timestepping) */
                kernel.r = sqrt(r2);
#ifdef HYDRO_REGULAR_GRID
                if(kernel.r > 1.1 * Particle_Size_i * sqrt(NUMDIMS)) continue; // only do interactions for the immediate neighbors //
#endif
                rinv = 1 / kernel.r;
                /* we require a 'softener' to prevent numerical madness in interpolating functions */
                rinv_soft = 1.0 / sqrt(r2 + 0.0001*kernel.h_i*kernel.h_i);
                Vec3<MyDouble> VelPred_j = CellP[j].VelPred; // set the velocity of neighbor
                NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos,P[j].Pos,VelPred_j,-1); /* in a shearing box, wrap velocities for shearing boxes if needed [literally does nothing if not shearing box here] */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                Vec3<MyDouble> ParticleVel_j = CellP[j].VelPred; // set the com-element velocity of neighbor
                NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos,P[j].Pos,ParticleVel_j,-1); /* wrap velocities for shearing boxes if needed */
#endif
                kernel.dv = local.Vel - VelPred_j;
                kernel.rho_ij_inv = 2.0 / (local.Density + CellP[j].Density);
                double Particle_Size_j; Particle_Size_j = P[j].Get_Particle_Size() * All.cf_atime; /* physical units */
                V_j = P[j].Mass / CellP[j].Density; /* neighbor volume, needed by sub-module functions below */

                /* --------------------------------------------------------------------------------- */
                /* sound speed, relative velocity, and signal velocity computation */
                kernel.sound_j = CellP[j].effective_soundspeed();
                kernel.vsig = kernel.sound_i + kernel.sound_j;
                /* magneticspeed_i/j declared unconditionally so hydro_core_sph can take them
                   as plain args; zero under !MAGNETIC (unused on that path). */
                double magneticspeed_i = 0, magneticspeed_j = 0;
#ifdef COSMIC_RAY_FLUID
                double CosmicRayPressure_j[N_CR_PARTICLE_BINS]; for(k=0;k<N_CR_PARTICLE_BINS;k++) {CosmicRayPressure_j[k] = Get_Gas_CosmicRayPressure(j, k, CellP);} /* compute this for use below */
                //double Streaming_Loss_Term = 0; // alternative evaluation of streaming+diffusion losses: still experimental //
#endif
                /* BPred_j / PhiPred_j declared unconditionally so the hydro_core and
                   nonideal_mhd functions can take them as plain args. */
                Vec3<double> BPred_j = {};
                double PhiPred_j = 0;
#ifdef MAGNETIC
                BPred_j = CellP[j].Bfield(); /* defined j b-field in appropriate units for everything */
                NGB_SHEARBOX_BOUNDARY_BCORR_(local.Pos,P[j].Pos,BPred_j,-1); /* in a shearing box, wrap magnetic fields for shearing boxes if needed [literally does nothing if not shearing box here] */
#ifdef DIVBCLEANING_DEDNER
                PhiPred_j = Get_Gas_PhiField(j); /* define j phi-field in appropriate units */
#endif
                kernel.b2_j = BPred_j.norm_sq();
                kernel.alfven2_j = kernel.b2_j * fac_magnetic_pressure / CellP[j].Density;
                kernel.alfven2_j = DMIN(kernel.alfven2_j, 1000. * kernel.sound_j*kernel.sound_j);
                double vcsa2_j = kernel.sound_j*kernel.sound_j + kernel.alfven2_j;
                double Bpro2_j = dot(BPred_j, kernel.dp) / kernel.r;
                Bpro2_j *= Bpro2_j;
                magneticspeed_j = sqrt(0.5 * (vcsa2_j + sqrt(DMAX((vcsa2_j*vcsa2_j -
                        4 * kernel.sound_j*kernel.sound_j * Bpro2_j*fac_magnetic_pressure/CellP[j].Density), 0))));
                double Bpro2_i = dot(local.BPred, kernel.dp) / kernel.r;
                Bpro2_i *= Bpro2_i;
                magneticspeed_i = sqrt(0.5 * (vcsa2_i + sqrt(DMAX((vcsa2_i*vcsa2_i -
                        4 * kernel.sound_i*kernel.sound_i * Bpro2_i*fac_magnetic_pressure/local.Density), 0))));
                kernel.vsig = magneticspeed_i + magneticspeed_j;
                Bpro2_i /= kernel.b2_i; Bpro2_j /= kernel.b2_j;
#endif
                kernel.vdotr2 = dot(kernel.dp, kernel.dv);
                // hubble-flow correction: need in -code- units, hence extra a2 appearing here //
                if(All.ComovingIntegrationOn) {kernel.vdotr2 += All.cf_hubble_a2 * r2;}
                if(kernel.vdotr2 < 0)
                {
#if defined(HYDRO_SPH) || defined(HYDRO_MESHLESS_FINITE_VOLUME)
                    kernel.vsig -= 3 * fac_mu * kernel.vdotr2 * rinv;
#else
                    kernel.vsig -= fac_mu * kernel.vdotr2 * rinv;
#endif
                }
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
                double KE = kernel.dv.norm_sq();
                if(KE > out.MaxKineticEnergyNgb) {out.MaxKineticEnergyNgb = KE;}
#endif
                /* mdot_estimated: set inside MFM/MFV Riemann core when TURB_DIFF_METALS
                   is enabled; declared unconditionally so turb-diffusion functions can take
                   it as a plain argument (value is 0 when not set). */
                double mdot_estimated = 0;
                /* tensile_correction_factor: declared unconditionally so the elastic
                   function can take it as a plain arg (value 0 when flags off). */
                double tensile_correction_factor = 0;
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
                tensile_correction_factor = get_negative_pressure_tensilecorrfac(kernel.r, kernel.h_i, kernel.h_j);
#endif
                
                /* --------------------------------------------------------------------------------- */
                /* calculate the kernel functions (centered on both 'i' and 'j') */
                if(kernel.r < kernel.h_i)
                {
                    u = kernel.r * hinv_i;
                    kernel_main(u, hinv3_i, hinv4_i, &kernel.wk_i, &kernel.dwk_i, kernel_mode);
                }
                else
                {
                    kernel.dwk_i = 0;
                    kernel.wk_i = 0;
                }
                if(kernel.r < kernel.h_j)
                {
                    kernel_hinv(kernel.h_j, &hinv_j, &hinv3_j, &hinv4_j);
                    u = kernel.r * hinv_j;
                    kernel_main(u, hinv3_j, hinv4_j, &kernel.wk_j, &kernel.dwk_j, kernel_mode);
                }
                else
                {
                    kernel.dwk_j = 0;
                    kernel.wk_j = 0;
                }

                /* --------------------------------------------------------------------------------- */
                /* with the overhead numbers above calculated, we now 'feed into' the "core"
                    hydro computation (SPH, meshless godunov, etc -- doesn't matter, should all take the same inputs)
                    the core code is -inserted- here from the appropriate .h file, depending on the mode
                    the code has been compiled in */
                /* --------------------------------------------------------------------------------- */
#ifdef HYDRO_SPH
                hydro_core_sph_compute_pair(local, j, P, CellP, VelPred_j, BPred_j, PhiPred_j,
                                            kernel, r2, V_j,
                                            magneticspeed_i, magneticspeed_j,
                                            fac_mu, fac_vsic_fix, tensile_correction_factor,
                                            dt_hydrostep, magfluxv, resistivity_heatflux, Fluxes);
#else
                {
                    const double *cr_pressure_j_ptr = nullptr;
#ifdef COSMIC_RAY_FLUID
                    cr_pressure_j_ptr = CosmicRayPressure_j;
#endif
                    Vec3<MyDouble> particle_vel_j_for_core = {};
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                    particle_vel_j_for_core = ParticleVel_j;
#endif
                    hydro_core_meshless_compute_pair(local, j, P, CellP, VelPred_j, particle_vel_j_for_core,
                                                     BPred_j, PhiPred_j, kernel, rinv, r2, V_i, V_j,
                                                     Particle_Size_i, Particle_Size_j, cnumcrit2,
                                                     fac_magnetic_pressure, tensile_correction_factor,
                                                     epsilon_entropic_eos_big, epsilon_entropic_eos_small,
                                                     cr_pressure_j_ptr,
                                                     Face_Area_Vec, Face_Area_Norm,
                                                     face_vel_i, face_vel_j, face_area_dot_vel,
                                                     mdot_estimated, Riemann_vec, Riemann_out, Fluxes, out);
                }
#endif

#ifdef FREEZE_HYDRO
                memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
#endif




                /* the following macros are useful for all the diffusion operations below: this is the diffusion term associated
                    with the HLL reimann problem solution. This adds numerical diffusion (albeit limited to the magnitude of the
                    physical diffusion coefficients), but stabilizes the relevant equations */
#ifdef HYDRO_SPH
                face_vel_i = dot(local.Vel, kernel.dp) / (kernel.r * All.cf_atime);
                face_vel_j = dot(VelPred_j, kernel.dp) / (kernel.r * All.cf_atime);
                // SPH: use the sph 'effective areas' oriented along the lines between particles and direct-difference gradients
                Face_Area_Norm = local.Mass * P[j].Mass * fabs(kernel.dwk_i+kernel.dwk_j) / (local.Density * CellP[j].Density) * All.cf_atime*All.cf_atime;
                Face_Area_Vec = kernel.dp * (Face_Area_Norm / kernel.r);
#endif

                /* bhat/bhat_mag declared unconditionally so per-pair physics functions
                   can take them as plain args without needing #ifdef MAGNETIC in signatures. */
                Vec3<double> bhat = {};
                double bhat_mag = 0;
#ifdef MAGNETIC
                bhat = 0.5 * (local.BPred + BPred_j) * All.cf_a2inv;
                bhat_mag = bhat.norm_sq();
                if(bhat_mag>0) {bhat_mag=sqrt(bhat_mag); bhat /= bhat_mag;}
                v_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(magneticspeed_i,magneticspeed_j);
#else
                v_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(kernel.sound_i,kernel.sound_j);
#endif

                /* Per-pair physics sub-modules. Functions guard their bodies with the
                   relevant #ifdef so callers invoke unconditionally (no-op when disabled). */
                double face_density_for_diffusion = 0;
#if defined(SAVE_FACE_DENSITY) && !defined(HYDRO_SPH)
                face_density_for_diffusion = Riemann_out.Face_Density;
#endif
                elastic_stress_tensor_force_compute_pair(local, P[j], CellP[j], VelPred_j, kernel, rinv,
                                                         Face_Area_Vec, Face_Area_Norm,
                                                         tensile_correction_factor, dt_hydrostep, Fluxes);
                nonideal_mhd_compute_pair(local, P[j], CellP[j], BPred_j, kernel, rinv,
                                          Face_Area_Vec, Face_Area_Norm, v_hll, bhat, bhat_mag,
                                          dt_hydrostep, Fluxes);
                conduction_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                        v_hll, bhat, bhat_mag, dt_hydrostep, Fluxes);
                viscosity_compute_pair(local, P[j], CellP[j], VelPred_j, kernel, rinv,
                                       Face_Area_Vec, Face_Area_Norm, v_hll, bhat, bhat_mag,
                                       dt_hydrostep, Fluxes);

                turb_diff_metals_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                              face_density_for_diffusion, v_hll, dt_hydrostep, mdot_estimated, out);
                chimes_turb_diff_ions_compute_pair(local, P[j], CellP[j], kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                                   face_density_for_diffusion, v_hll, dt_hydrostep, mdot_estimated, out);

                {
                    /* Riemann_out / face_area_dot_vel exist only under !HYDRO_SPH */
                    double riemann_S_M_val = 0, face_area_dot_vel_val = 0;
#ifndef HYDRO_SPH
                    riemann_S_M_val = Riemann_out.S_M;
                    face_area_dot_vel_val = face_area_dot_vel;
#endif
                    const double *cr_pressure_j_ptr = nullptr;
#ifdef COSMIC_RAY_FLUID
                    cr_pressure_j_ptr = CosmicRayPressure_j;
#endif
                    cosmic_ray_diffusion_compute_pair(local, j, P, CellP, VelPred_j, kernel,
                                                      Face_Area_Vec, Face_Area_Norm, V_i, V_j,
                                                      Particle_Size_i, Particle_Size_j,
                                                      face_vel_i, face_vel_j,
                                                      face_area_dot_vel_val, riemann_S_M_val,
                                                      cr_pressure_j_ptr, bhat, dt_hydrostep, Fluxes, out);
                }
                {
                    const double *tau_c_i_ptr = nullptr;
#if defined(RT_SOLVER_EXPLICIT) && (N_RT_FREQ_BINS > 0)
                    tau_c_i_ptr = tau_c_i;
#endif
                    Vec3<MyDouble> particle_vel_j_for_rt = {};
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                    particle_vel_j_for_rt = ParticleVel_j;
#endif
                    rt_direct_ray_transport_compute_pair(local, P[j], CellP[j], VelPred_j, particle_vel_j_for_rt,
                                                         Face_Area_Vec, Face_Area_Norm, V_i, V_j, Particle_Size_j,
                                                         tau_c_i_ptr, dt_hydrostep, out);
                    rt_diffusion_explicit_compute_pair(local, j, P, CellP, VelPred_j, particle_vel_j_for_rt,
                                                       kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                                       V_i, V_j, Particle_Size_i, Particle_Size_j,
                                                       face_vel_i, face_vel_j, tau_c_i_ptr, dt_hydrostep, out);
                }


                /* --------------------------------------------------------------------------------- */
                /* now we will actually assign the hydro variables for the evolution step */
                /* --------------------------------------------------------------------------------- */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                double dmass_holder = Fluxes.rho * dt_hydrostep_i, dmass_limiter;
                if(dmass_holder > 0) {dmass_limiter=P[j].Mass;} else {dmass_limiter=local.Mass;}
                dmass_limiter *= 0.1;
                if(fabs(dmass_holder) > dmass_limiter) {dmass_holder *= dmass_limiter / fabs(dmass_holder);}
                if(local.dt_hydrostep_i < dt_hydrostep_j) {
                    out.dMass += dmass_holder;
                    #pragma omp atomic
                    CellP[j].dMass -= dmass_holder; // machine-accurate mass conservation across different timesteps: thread-safe
                }
                if(local.dt_hydrostep_i == dt_hydrostep_j) {
                    out.dMass += 0.5*dmass_holder;
                    #pragma omp atomic
                    CellP[j].dMass -= 0.5*dmass_holder;
                }
                 /* this gets subtracted here to ensure the exchange is exact */
                out.DtMass += Fluxes.rho;
                Vec3<double> gravwork = kernel.dp * Fluxes.rho;
                out.GravWorkTerm += gravwork;
#ifdef METALS   /* if we have mass fluxes, we need to have metal fluxes if we're using them (or any other passive scalars) */
                if(Fluxes.rho > 0) {out.Dyield[k] += (P[j].Metallicity[k] - local.Metallicity[k]) * dmass_holder;}
#endif
#endif
                #ifdef GIZMO_DEBUG_RT_COOLING
                /* HYDRO_PAIR_DIAG: per-pair flux diagnostic for tracked particle IDs.
                   Print Fluxes.p and Fluxes.v AFTER Riemann+conduction+viscosity+RT but BEFORE accumulation.
                   Also print key neighbor quantities to check for corrupted EOS values. */
                {static int hpp_step=0; if(P[target].ID == 1000 && hpp_step < 1) {
                    printf("[HYDRO_PAIR] ID_i=1000 j_ID=%llu n=%d Fp=%.10e Fv=%.8e/%.8e/%.8e cs_j=%.6e u_j=%.6e P_j=%.6e rho_j=%.6e T_j=%.6e gamma_j=%.6e\n",
                        (unsigned long long)P[j].ID, n, Fluxes.p, Fluxes.v[0], Fluxes.v[1], Fluxes.v[2],
                        kernel.sound_j, CellP[j].InternalEnergyPred, CellP[j].Pressure, CellP[j].Density, CellP[j].Temperature, CellP[j].Gamma);
                    if(n == numngb-1) {hpp_step++; printf("[HYDRO_PAIR_TOTAL] ID=1000 total_DtU=%.10e total_Acc=%.8e/%.8e/%.8e numngb=%d\n", out.DtInternalEnergy, out.Acc[0], out.Acc[1], out.Acc[2], numngb); fflush(stdout);}
                }}
                #endif /* GIZMO_DEBUG_RT_COOLING */
                out.Acc += Fluxes.v;
                out.DtInternalEnergy += Fluxes.p;
#ifdef MAGNETIC
#ifndef HYDRO_SPH
                out.Face_Area += Face_Area_Vec;
#endif
#ifndef FREEZE_HYDRO
                out.DtB += Fluxes.B;
                out.divB += Fluxes.B_normal_corrected;
#if defined(DIVBCLEANING_DEDNER) && defined(HYDRO_MESHLESS_FINITE_VOLUME) // mass-based phi-flux
                out.DtPhi += Fluxes.phi;
#endif
#ifdef HYDRO_SPH
                out.DtInternalEnergy += dot(magfluxv, local.Vel) / All.cf_atime;
                out.DtInternalEnergy += resistivity_heatflux;
#else
                double wt_face_sum = Face_Area_Norm * (-face_area_dot_vel+face_vel_i);
                double du_mag_pres = 0.5 * kernel.b2_i*All.cf_a2inv*All.cf_a2inv * wt_face_sum;
                out.DtInternalEnergy += du_mag_pres;
#ifdef DIVBCLEANING_DEDNER
                double du_dedner = 0;
                for(k=0; k<3; k++)
                {
                    out.DtB_PhiCorr[k] += Riemann_out.phi_normal_db * Face_Area_Vec[k];
                    out.DtB[k] += Riemann_out.phi_normal_mean * Face_Area_Vec[k];
                    double du_ded_k = Riemann_out.phi_normal_mean * Face_Area_Vec[k] * local.BPred[k]*All.cf_a2inv;
                    out.DtInternalEnergy += du_ded_k;
                    du_dedner += du_ded_k;
                }
                /* HYDRO_MAGDTU_DIAG: per-pair magnetic+Dedner DtU corrections */
#ifdef GIZMO_DEBUG_RT_COOLING
                {static int hmag_step=0; if(P[target].ID == 1000 && hmag_step < 1) {
                    printf("[HYDRO_MAGDTU] ID_i=1000 j_ID=%llu n=%d du_mag=%.10e du_ded=%.10e b2_i=%.6e wt=%.6e phi=%.6e DtU=%.10e\n",
                        (unsigned long long)P[j].ID, n, du_mag_pres, du_dedner, kernel.b2_i, wt_face_sum, Riemann_out.phi_normal_mean, out.DtInternalEnergy);
                    if(n == numngb-1) {hmag_step++;}
                }}
#endif
#endif
#ifdef MHD_NON_IDEAL
                out.DtInternalEnergy += dot(local.BPred, bflux_from_nonideal_effects) * All.cf_a2inv;
#endif
#endif
#endif
#endif // magnetic //

                /* --------------------------------------------------------------------------------- */
                /* don't forget to save the signal velocity for time-stepping! */
                /* --------------------------------------------------------------------------------- */
                if(kernel.vsig > out.MaxSignalVel) {out.MaxSignalVel = kernel.vsig;}
                if(!(TimeBinActive[P[j].TimeBin]))
                {
                    if(kernel.vsig > WAKEUP*CellP[j].MaxSignalVel) {
                        #pragma omp atomic write
                        P[j].wakeup = (short int)(local.TimeBin + 1);
                        #pragma omp atomic write
                        NeedToWakeupParticles_local = 1;
                    }
                }


            } // for(n = 0; n < numngb; n++) //
        } // while(startnode >= 0) //
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;}	/* open it */
            }
        } // if(mode == 1) //
    } // while(startnode >= 0) //

    /* Now collect the result at the right place */
    if(mode == 0) {out2particle_hydra(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}
