/* grain_drag.cc — grain drag force evaluation with CPU/GPU adaptive dispatch.
 *
 * GPU translation unit (Kokkos/nvcc_wrapper).
 * Provides grain_drag_evaluate() which runs the per-particle grain drag/Lorentz
 * force computation with an OpenMP tiny-N path and a Kokkos large-N path.
 *
 * Called from apply_grain_dragforce() in grain_physics.cc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

/* Standard and Kokkos headers BEFORE global_data_all_struct.h */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"

#include "../declarations/allvars.h"
#include "../declarations/multifluid_helpers.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"


#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"


#if defined(DO_FLUID_ALTSPECIES_DRAG_CALCULATION)

#ifdef GRAIN_EVOLUTION
#include "grain_evolution_functions.h"
#endif

/* Per-particle grain drag kernel. Operates on compact arrays pp[idx].
 * Modifies pp[idx].GravAccel, pp[idx].Grain_AccelTimeMin, and
 * (if GRAIN_BACKREACTION) pp[idx].Grain_DeltaMomentum in-place.
 * Under GRAIN_EVOLUTION, also runs the per-superparticle local-step
 * operators (sputter; later: COND/SUBL) at the end of the kernel so they
 * share the same Gas_* and dt computation, and the same back-reaction
 * writeback machinery (the generic grainbackrx ghost-writeback bundle in
 * solids/grain_physics_loop.cc) as the drag itself. */
KOKKOS_FUNCTION
static void grain_drag_kernel(int idx, struct particle_data *pp, struct gas_cell_data *cellp)
{
    int k;
    /* Check drag eligibility and set defaults for non-eligible particles */
    const bool drag_eligible_source = IS_PARTICLE_DRAGVALID(pp[idx].Type, pp[idx].FluidType);
    if(!drag_eligible_source) {pp[idx].Grain_AccelTimeMin = MAX_REAL_NUMBER; return;}
#ifdef BOX_BND_PARTICLES
    if(pp[idx].ID <= 0) return;
#endif
    if(!( drag_eligible_source && (pp[idx].Mass > 0) )) return;

#if defined(SINK_PARTICLES) && defined(GRAIN_FLUID)
    /* SwallowID/SwallowTime on Type==0 belong to sink physics; do not reset
     * here for Type==0 dust-fluid sources. */
    if(drag_eligible_source) {
        pp[idx].SwallowID = 0;
#ifdef SINGLE_STAR_SINK_DYNAMICS
        pp[idx].SwallowTime = MAX_REAL_NUMBER;
#endif
    }
#endif
#if defined(GRAIN_BACKREACTION)
    pp[idx].Grain_DeltaMomentum = {};
#endif
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
    for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { pp[idx].Grain_DeltaVolatileMass[kv] = 0; }
    pp[idx].Grain_DeltaInternalEnergyHeating = 0;
#endif

    pp[idx].Grain_AccelTimeMin = MAX_REAL_NUMBER; /* safe default: drag not computable yet (e.g. dt==0 at IC) */
    double dt = get_particle_timestep_in_physical(idx, pp);
    double vgas_mag = sqrt((pp[idx].Gas_Velocity - pp[idx].Vel).norm_sq()) / All.cf_atime;
    int grain_subtype = 1;
#if defined(PIC_MHD)
    /* MHD_PIC_SubType is only meaningful for Type=3-style grain particles;
     * Type==0 dust-fluid sources keep the default subtype = 1 (Epstein/Stokes). */
    if(drag_eligible_source) {grain_subtype = pp[idx].MHD_PIC_SubType;}
#endif



    if((grain_subtype <= 2) && (dt > 0) && (pp[idx].Gas_Density > 0) && (vgas_mag > 0))
    {
        double gamma_eff = GAMMA_DEFAULT;
        double mu_gas = molecular_weight_estimator_for_gas_around_grain(pp[idx].Gas_Temperature, pp[idx].Gas_fion);
        double cs = sqrt(gamma_eff * pp[idx].Gas_Temperature / (mu_gas * U_TO_TEMP_UNITS));
        double rho_gas = pp[idx].Gas_Density * All.cf_a3inv;
        double x0 = 0.469993*sqrt(gamma_eff) * vgas_mag/cs;
        double tstop_inv = MAX_REAL_NUMBER;

#if defined(GRAIN_FLUID) || defined(HYDRO_MULTIFLUID_DUST_DRAG) /* below block calculates grain-gas drag, only makes sense for dust modules specifically */
        double grain_size_cgs = 1.e-5;
        double grain_rho_cgs = 1.;
#if defined(GRAIN_FLUID)
        grain_size_cgs = pp[idx].Grain_Size;
        grain_rho_cgs = All.Grain_Internal_Density;
#endif
        double R_grain_cgs = grain_size_cgs, R_grain_code = R_grain_cgs / UNIT_LENGTH_IN_CGS;
        double rho_grain_physical = grain_rho_cgs, rho_grain_code = rho_grain_physical / UNIT_DENSITY_IN_CGS;
        tstop_inv = 1.59577/sqrt(gamma_eff) * rho_gas * cs / (R_grain_code * rho_grain_code);
#ifdef DO_FLUID_DRAG_CALCULATION_WITHBFIELDS
        double cs_cgs = cs * UNIT_VEL_IN_CGS;
        double tau_draine_sutin = R_grain_cgs * (2.3*PROTONMASS_CGS) * (cs_cgs*cs_cgs) / (gamma_eff * ELECTRONCHARGE_CGS*ELECTRONCHARGE_CGS);
        double Z_grain = -DMAX(1./(1. + sqrt(1.0e-3/tau_draine_sutin)), 2.5*tau_draine_sutin);
        if(isnan(Z_grain)||(Z_grain>=0)) {Z_grain=0;}
#endif
/* The temperature and ionized fraction of the gas the grain is coupled to are interpolated onto it
   by the density loop, so the blocks below read them rather than guessing from a sound speed. */
#ifdef GRAIN_EPSTEIN_STOKES
        if(grain_subtype == 0 || grain_subtype == 1)
        {
            double mu = mu_gas*PROTONMASS_CGS, temperature = pp[idx].Gas_Temperature;
            double cross_section = GRAIN_EPSTEIN_STOKES * 2.0e-15 * (1. + 70./temperature);
            cross_section /= UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS;
            double n_mol = rho_gas * UNIT_MASS_IN_CGS / mu, mean_free_path = 1 / (n_mol * cross_section);
            double corr_mfp = R_grain_code / ((9./4.) * mean_free_path);
            if(corr_mfp > 1) {tstop_inv /= corr_mfp;}
        }
#endif
#if defined(DO_FLUID_DRAG_CALCULATION_WITHBFIELDS) && defined(GRAIN_EPSTEIN_STOKES)
        if(grain_subtype == 1)
        {
            double a_Coulomb = sqrt(2.*gamma_eff*gamma_eff*gamma_eff/(9.*M_PI));
            double tstop_Coulomb_inv = 0.797885/sqrt(gamma_eff) * rho_gas * cs / (R_grain_code * rho_grain_code);
            tstop_Coulomb_inv /= (1. + a_Coulomb*(vgas_mag/cs)*(vgas_mag/cs)*(vgas_mag/cs)) * sqrt(1.+x0*x0);
            tstop_Coulomb_inv *= (Z_grain/tau_draine_sutin) * (Z_grain/tau_draine_sutin) / 17.;
            double T_Kelvin = pp[idx].Gas_Temperature, f_ion_to_use;
            if(pp[idx].Gas_fion >= 0) {f_ion_to_use = DMAX(0., DMIN(1., pp[idx].Gas_fion));}
            else {f_ion_to_use = (T_Kelvin > 1000.) ? exp(-15000./T_Kelvin) : 0;} /* no chemistry in this configuration */
            tstop_Coulomb_inv *= f_ion_to_use;
            tstop_inv += tstop_Coulomb_inv;
        }
#endif
#elif defined(HYDRO_MULTIFLUID_IONNEUTRAL)
        /* Draine ambipolar drag: tstop_inv = (<sigma v>_in / (m_i + m_n)) * rho_neutral.
         * Hardcoded HI/H+ values; edit two constants for other regimes (HCO+/H2 etc.). */
        double sigmav_in_cgs          = 1.9e-9;                /* HI/H+ momentum-transfer rate coefficient (Draine 1980; ~T^0.5 dep neglected v1) */
        double m_ion_plus_neutral_cgs = 2.0 * PROTONMASS_CGS;  /* H+/HI; replace for other regimes */
        double gamma_AD_cgs  = sigmav_in_cgs / m_ion_plus_neutral_cgs;
        double gamma_AD_code = gamma_AD_cgs * UNIT_DENSITY_IN_CGS * UNIT_TIME_IN_CGS;
        tstop_inv = gamma_AD_code * rho_gas;
#endif /* closes grain-fluid-specific drag calculation block */

        double external_forcing[3]={0}, eps=MIN_REAL_NUMBER;
        pp[idx].Grain_AccelTimeMin = DMAX(1./(eps+tstop_inv), sqrt(pp[idx].Get_Particle_Size()*All.cf_atime/(eps+vgas_mag*tstop_inv)));

#ifdef DO_FLUID_DRAG_CALCULATION_WITHBFIELDS
        if(grain_subtype == 1)
        {
            double grain_charge_cinv = 0;
            double lorentz_units = UNIT_B_IN_GAUSS;
            lorentz_units *= (ELECTRONCHARGE_CGS/C_LIGHT_CGS) * UNIT_VEL_IN_CGS / UNIT_MASS_IN_CGS;
            lorentz_units /= UNIT_VEL_IN_CGS / UNIT_TIME_IN_CGS;
#if defined(GRAIN_FLUID) || defined(HYDRO_MULTIFLUID_DUST_DRAG) /* below block calculates grain-gas drag, only makes sense for dust modules specifically */
            double grain_mass = (4.*M_PI/3.) * R_grain_code*R_grain_code*R_grain_code * rho_grain_code;
            grain_charge_cinv = Z_grain / grain_mass * lorentz_units;
#ifdef GRAIN_RDI_TESTPROBLEM
            if(All.Grain_Charge_Parameter != 0) {grain_charge_cinv = -All.Grain_Charge_Parameter*sqrt(1.)/((All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS)*(All.Grain_Size_Max/UNIT_LENGTH_IN_CGS)) * pow(All.Grain_Size_Max/pp[idx].Grain_Size,2);}
#endif
#endif
            /* this part defines the non-relativistic Boris integrator */
            Vec3<double> bhat, efield={}; double bmag=0, efield_coeff=0;
            bhat = pp[idx].Gas_B * All.cf_a2inv; bmag = bhat.norm_sq();
            Vec3<double> dv = (pp[idx].Vel - pp[idx].Gas_Velocity) / All.cf_atime;
            if(bmag>0) {bmag=sqrt(bmag); bhat /= bmag;} else {bmag=0;}

            double lorentz_coeff = (0.5*dt) * bmag * grain_charge_cinv;
            Vec3<double> v_m = dv + efield * (0.5*efield_coeff);
            Vec3<double> vcrosst = cross(v_m, bhat);
            double tL=1./(eps+0.5*bmag*fabs(grain_charge_cinv)), vgasXB_mag=vcrosst.norm_sq();
            pp[idx].Grain_AccelTimeMin = DMIN(pp[idx].Grain_AccelTimeMin, DMAX(tL, sqrt(pp[idx].Get_Particle_Size()*All.cf_atime/(eps+sqrt(vgasXB_mag)/tL))));
            Vec3<double> v_t = v_m + vcrosst * lorentz_coeff;
            vcrosst = cross(v_t, bhat);
            Vec3<double> v_p = v_m + vcrosst * (2.*lorentz_coeff/(1.+lorentz_coeff*lorentz_coeff));
            v_p += efield * (0.5*efield_coeff);
            Vec3<double> ef_update = (v_p - dv) / dt;
            for(k=0;k<3;k++) {external_forcing[k] += ef_update[k];}
        }
#endif

        /* update conservative variables */
        double delta_egy=0, delta_mom[3]={0}, xf=0, dt_tinv=dt*tstop_inv, C1=-(1+sqrt(1+x0*x0))/x0;
        if(dt_tinv < 100.) {double C2 = C1*exp(dt_tinv); xf=-2*C2/(C2*C2-1);}
        double slow_fac = 1 - xf/x0;
        for(k=0; k<3; k++)
        {
            double v_init = pp[idx].Vel[k] / All.cf_atime, v_gas_i = pp[idx].Gas_Velocity[k] / All.cf_atime;
            double vel_new = v_init + slow_fac * (v_gas_i - v_init);
            double vdrift = 0, dv_k;
            if(tstop_inv > 0) {vdrift = external_forcing[k] / (tstop_inv * sqrt(1+x0*x0));}
            dv_k = slow_fac * (v_gas_i - v_init + vdrift);
            if(isnan(vdrift)||isnan(slow_fac)) {dv_k = 0;}
            vel_new = v_init + dv_k;
            delta_mom[k] = pp[idx].Mass * (vel_new - v_init);
            delta_egy += 0.5*pp[idx].Mass * (vel_new*vel_new - v_init*v_init);
#ifdef GRAIN_BACKREACTION
            pp[idx].Grain_DeltaMomentum[k] = delta_mom[k] * All.cf_atime;
#endif
            pp[idx].GravAccel[k] += dv_k / (dt * All.cf_a2inv);
        }
    } /* closes check for gas density, dt, vmag > 0, subtype valid */

#ifdef PIC_MHD
    /* PIC_SPEEDOFLIGHT_REDUCTION default is set globally in precompiler_logic.h
     * so this and the PIC timestep limiter (core/timestep.cc) stay consistent. */
    if((grain_subtype >= 3) && (dt > 0) && (pp[idx].Gas_Density>0) && (vgas_mag > 0))
    {
        double reduced_C = PIC_SPEEDOFLIGHT_REDUCTION * C_LIGHT_CODE;
        double charge_to_mass_ratio_dimensionless = All.PIC_Charge_to_Mass_Ratio;
        double dt_cgs = dt * UNIT_TIME_IN_CGS;
        double lorentz_units = 1./C_LIGHT_CGS;
#ifdef PIC_MHD_NEW_RSOL_METHOD
        lorentz_units *= PIC_SPEEDOFLIGHT_REDUCTION;
#endif
        Vec3<double> bhat={}; double Bmag=0;
        bhat = pp[idx].Gas_B * All.cf_a2inv; Bmag = bhat.norm_sq();
        Vec3<double> beta_true_gas = pp[idx].Gas_Velocity / (All.cf_atime*C_LIGHT_CODE);
        if(Bmag>0) {Bmag=sqrt(Bmag); bhat /= Bmag;} else {Bmag=0;}
        double Bmag_cgs = Bmag * UNIT_B_IN_GAUSS;
        double efield_coeff = (0.5*dt_cgs) * (charge_to_mass_ratio_dimensionless * ELECTRONCHARGE_CGS/PROTONMASS_CGS) * Bmag_cgs * lorentz_units;
        Vec3<double> efield = cross(beta_true_gas, bhat) * (-1.0);
        Vec3<double> beta_true_cr_0 = pp[idx].Vel / (All.cf_atime*reduced_C);
        double beta2_true_cr = beta_true_cr_0.norm_sq();
        double gamma_0 = 1/sqrt(1-beta2_true_cr);
        Vec3<double> betagamma_0 = beta_true_cr_0 * gamma_0;

        Vec3<double> betagamma_m = betagamma_0 + efield * efield_coeff;
        double lorentz_coeff_tau = efield_coeff / sqrt(1+betagamma_m.norm_sq());
        Vec3<double> beta_b_crosst = cross(betagamma_m, bhat);
        Vec3<double> betagamma_t = betagamma_m + beta_b_crosst * lorentz_coeff_tau;
        beta_b_crosst = cross(betagamma_t, bhat);
        Vec3<double> betagamma_p = betagamma_m + beta_b_crosst * (2.*lorentz_coeff_tau/(1.+lorentz_coeff_tau*lorentz_coeff_tau));
        betagamma_p += efield * efield_coeff;
        double betagamma2_p = betagamma_p.norm_sq(), gamma_f = sqrt(1+betagamma2_p);
        Vec3<double> beta_true_cr_f = betagamma_p / gamma_f;

        for(k=0;k<3;k++)
        {
#ifdef GRAIN_BACKREACTION
            double delta_momentum = pp[idx].Mass * C_LIGHT_CODE * (beta_true_cr_f[k]*gamma_f - beta_true_cr_0[k]*gamma_0) * All.cf_atime;
#ifdef PIC_MHD_NEW_RSOL_METHOD
            delta_momentum /= PIC_SPEEDOFLIGHT_REDUCTION;
#endif
            pp[idx].Grain_DeltaMomentum[k] += delta_momentum;
#endif
            pp[idx].GravAccel[k] += (beta_true_cr_f[k]-beta_true_cr_0[k]) * reduced_C / (dt * All.cf_a2inv);
        }
    }
#endif

#ifdef GRAIN_EVOLUTION
    /* Per-superparticle local operators. Co-located with the
     * drag kernel because (a) every Gas_* field they need is already in
     * registers here, (b) any back-reaction they generate piggybacks on
     * the existing GRAIN_BACKREACTION writeback infrastructure rather than
     * requiring a parallel pass. Gated on grain_subtype <= 2 so PIC
     * particles (subtype >= 3) -- whose Composition[] would be meaningless
     * -- are skipped. Restricted to GRAIN_PTYPES sources: Type==0 dust-fluid
     * sources will not be compatible in COND/SUBL/sputter */
    if(drag_eligible_source && (grain_subtype <= 2) && (dt > 0)) {
        grain_evolution_local_step(idx, pp, dt);
    }
#endif
}


/* Dispatch function: tiny-N → OMP host path; large-N → GPU compact path.
 * Caller (apply_grain_dragforce) pre-filters active_indices to GRAIN_PTYPES only,
 * so num_active here is the true active grain count. */
void grain_drag_evaluate(struct particle_data *P_host, struct gas_cell_data *CellP_host,
                         int *active_indices, int num_active)
{
    if(num_active <= 0) return;

    if(num_active < GPU_MIN_PARTICLES_FOR_OFFLOAD)
    {
        /* Tiny-N CPU/OMP path: dispatch directly on host, no GPU launch, no Kokkos
           allocation, no compact gather/scatter.  Each thread writes only its own grain
           particle (unique indices from caller) so this is thread-safe.
           GIZMO_GPU_ENSURE_ALL_FRESH is NOT called here: host code reads the host All
           extern directly; the device-pass-only #define All AllDeviceMirror is inert. */
        PRINT_STATUS("  grain drag (OMP): %d active grains", num_active);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int a = 0; a < num_active; a++)
            grain_drag_kernel(active_indices[a], P_host, CellP_host);
        return;
    }

    /* Large-N GPU path: compact gather → kernel → scatter. */
    GIZMO_GPU_ENSURE_ALL_FRESH();
    const size_t grain_stage_bytes = (size_t) num_active * (sizeof(struct particle_data)
                                                            + sizeof(struct gas_cell_data));
    struct particle_data *compact_P = (struct particle_data *) gizmo_gpu_alloc_shared((size_t) num_active * sizeof(struct particle_data), NULL);
    struct gas_cell_data *compact_Cell = (struct gas_cell_data *) gizmo_gpu_alloc_shared((size_t) num_active * sizeof(struct gas_cell_data), NULL);
    if(!compact_P || !compact_Cell) {
        if(compact_Cell) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_Cell);}
        if(compact_P)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);}
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "grain drag: could not stage %d grains (%.1f MB); no drag is applied",
                 num_active, (double) grain_stage_bytes / (1024.0 * 1024.0));
        gizmo_request_controlled_stop(7718, msg, __FILE__, __LINE__, __FUNCTION__);
        return;
    }
    /* compact_Cell is zeroed and stays zeroed: caller guarantees active_indices are
       all GRAIN_PTYPES (not gas), so no valid CellP entry exists for any of them, and
       the kernel reads none -- every gas quantity it needs is interpolated onto the
       grain itself in the density loop. */
    memset(compact_Cell, 0, num_active * sizeof(struct gas_cell_data));
    for(int j = 0; j < num_active; j++)
        compact_P[j] = P_host[active_indices[j]];

    PRINT_STATUS("  grain drag (GPU): %d active grains", num_active);

    {
        struct particle_data *kp = compact_P;
        struct gas_cell_data *kc = compact_Cell;
        gizmo_gpu_kernel_launch("grain_drag", num_active, KOKKOS_LAMBDA(int j) {
            grain_drag_kernel(j, kp, kc);
        });
    }

    /* Scatter back only the fields grain_drag_kernel writes. */
    for(int j = 0; j < num_active; j++) {
        int ii = active_indices[j];
        P_host[ii].GravAccel = compact_P[j].GravAccel;
        P_host[ii].Grain_AccelTimeMin = compact_P[j].Grain_AccelTimeMin;
#if defined(SINK_PARTICLES) && defined(GRAIN_FLUID)
        P_host[ii].SwallowID = compact_P[j].SwallowID;
#ifdef SINGLE_STAR_SINK_DYNAMICS
        P_host[ii].SwallowTime = compact_P[j].SwallowTime;
#endif
#endif
#if defined(GRAIN_BACKREACTION)
        P_host[ii].Grain_DeltaMomentum = compact_P[j].Grain_DeltaMomentum;
#endif
#ifdef GRAIN_EVOLUTION
        P_host[ii].Mass       = compact_P[j].Mass;
        P_host[ii].Grain_Size = compact_P[j].Grain_Size;
#if (GRAIN_EVOLUTION & (32|64))
        for(int s = 0; s < GRAIN_NUM_SPECIES; s++) { P_host[ii].Composition[s] = compact_P[j].Composition[s]; }
        for(int kv = 0; kv < GRAIN_NUM_VOLATILE_SPECIES; kv++) { P_host[ii].Grain_DeltaVolatileMass[kv] = compact_P[j].Grain_DeltaVolatileMass[kv]; }
        P_host[ii].Grain_DeltaInternalEnergyHeating = compact_P[j].Grain_DeltaInternalEnergyHeating;
#endif
#endif
    }

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_Cell);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);
}


/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */

#else /* stubs */

void grain_drag_evaluate(struct particle_data *, struct gas_cell_data *, int *, int) {}

#endif /* DO_FLUID_ALTSPECIES_DRAG_CALCULATION */
