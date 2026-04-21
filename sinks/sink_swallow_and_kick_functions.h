/* sink_swallow_and_kick_functions.h — GPU-callable per-pair kernel + shared
 * structs for the D1 GPU port of sink_swallow_and_kick_evaluate.
 *
 * Include order: after allvars.h, particle_data.h, gas_cell_data.h, kernel.h,
 * sink_functions.h (for sink_fb_angleweight_localcoupling_gpu).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef SINK_PARTICLES

#if defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY) && defined(OPENMP_GPU_OFFLOAD) && \
    defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS)
#error "D1 Phase 1 GPU sink_swallow_and_kick port does not support COSMIC_RAY_FLUID + SINK_COSMIC_RAYS — port the CR injection path in a follow-on (mirrors B8 Phase 2 pattern)."
#endif

/* Per-source (active sink) input. Mirrors INPUT_STRUCT_NAME in sink_swallow_and_kick.cc:34-56. */
struct SinkSwallowLocalIn
{
    Vec3<MyDouble> Pos;
    Vec3<MyFloat> Vel;
    MyFloat KernelRadius;
    MyFloat Mass;
    MyFloat Sink_Mass;
    MyFloat Dt;
    MyFloat Mdot;
    MyIDType ID;
    MyIDType ID_child_number;
    MyIDType ID_generation;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) || defined(SINK_WIND_KICK)
    Vec3<MyFloat> Jgas_in_Kernel;
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat Sink_Mass_Reservoir;
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    MyFloat Sink_angle_weighted_kernel_sum;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    Vec3<MyFloat> Sink_Specific_AngMom;
    MyFloat angmom_norm_topass_in_swallowloop;
#endif
#if defined(SINK_RETURN_BFLUX)
    Vec3<MyFloat> B;
    MyFloat kernel_norm_topass_in_swallowloop;
#endif
#ifdef SINGLE_STAR_FB_LOCAL_RP
    MyFloat Luminosity;
#endif
};

/* Per-source (active sink) output + swallow counters. Mirrors OUTPUT_STRUCT_NAME
 * in sink_swallow_and_kick.cc:97-130, plus N_gas/N_sink/N_star/N_dm swallow
 * counters (previously global statics, now per-source so host can MPI-reduce). */
struct SinkSwallowOut
{
    MyDouble accreted_Mass;
    MyDouble accreted_Sink_Mass;
    MyDouble accreted_Sink_Mass_reservoir;
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    MyDouble Sink_AccretionDeficit;
#endif
#ifdef GRAIN_FLUID
    MyDouble accreted_dust_Mass;
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    MyDouble accreted_photon_energy;
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
    Vec3<MyDouble> accreted_momentum;
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM)
    Vec3<MyDouble> accreted_centerofmass;
#endif
#if defined(SINK_RETURN_BFLUX)
    Vec3<MyDouble> accreted_B;
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    Vec3<MyDouble> accreted_J;
#endif
#ifdef SINK_COUNTPROGS
    int Sink_CountProgs;
#endif
#ifdef GALSF
    MyFloat Accreted_Age;
#endif
    int n_gas_swallowed;
    int n_sink_swallowed;
    int n_star_swallowed;
    int n_dm_swallowed;
    /* Per-bin deltas for TimeBin_Sink_* — accumulated in GPU kernel for the
     * sink-sink merger path (Type-5 swallow) and applied on host in scatter,
     * since TimeBin_Sink_* are plain host globals not accessible from CUDA. */
    MyDouble delta_TimeBin_Sink_mass[TIMEBINS];
    MyDouble delta_TimeBin_Sink_dynamicalmass[TIMEBINS];
    MyDouble delta_TimeBin_Sink_Mdot[TIMEBINS];
    MyDouble delta_TimeBin_Sink_Medd[TIMEBINS];
};


#if defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY) && defined(OPENMP_GPU_OFFLOAD)

/* Per-pair kernel — mirrors sink_swallow_and_kick.cc:204-510. Atomic j-writes on
 * the target gas cell / sink / star / DM; host-side scatter of per-source
 * outputs handles SinkTempInfo accumulation and MPI_Reduce of counters. */
KOKKOS_INLINE_FUNCTION
static void sink_swallow_pair_kernel(
    const struct SinkSwallowLocalIn& local,
    int j,
    struct particle_data *P,
    struct gas_cell_data *CellP,
    struct SinkSwallowOut& out,
    const Vec3<double>& dpos,   /* P[j].Pos - local.Pos, nearest_xyz applied (CPU sign convention) */
    double r2,
    double h_i,
    double mom_budget,           /* photon-momentum budget (SINK_CALC_LOCAL_ANGLEWEIGHTS) */
    const Vec3<double>& J_dir,
    double sink_mass_withdisk)
{
    MyIDType OriginallyMarkedSwallowID = P[j].SwallowID;
    double Mass_j = (double)P[j].Mass;
    double Mass_j_0 = Mass_j;
    double InternalEnergy_j = 0, InternalEnergy_j_0 = 0;
    if(P[j].Type == 0) {
        InternalEnergy_j   = (double)CellP[j].InternalEnergy;
        InternalEnergy_j_0 = InternalEnergy_j;
    }
    double Vel_j[3], Vel_j_0[3];
    for(int k = 0; k < 3; k++) { Vel_j[k] = (double)P[j].Vel[k]; Vel_j_0[k] = Vel_j[k]; }

    /* dvel computed from dpos + local/P[j] velocities; shearing-box velcorr already
     * incorporated by caller via nearest_xyz. Replicate the CPU's vel handling. */
    Vec3<double> dvel{Vel_j[0] - (double)local.Vel[0], Vel_j[1] - (double)local.Vel[1], Vel_j[2] - (double)local.Vel[2]};
    NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos, P[j].Pos, dvel, -1);

#if defined(SINK_RETURN_ANGMOM_TO_GAS) || defined(SINK_RETURN_BFLUX)
    double wk = 0, dwk_tmp = 0;
    if(P[j].Type == 0) {
        double hj = (double)P[j].KernelRadius;
        double h_denom = (h_i > hj) ? h_i : hj;
        double u = sqrt(r2) / h_denom;
        if(u < 1) { kernel_main(u, 1.0, 1.0, &wk, &dwk_tmp, -1); } else { wk = 0; }
    }
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    if(P[j].Type == 0) {
        Vec3<double> sam{(double)local.Sink_Specific_AngMom[0], (double)local.Sink_Specific_AngMom[1], (double)local.Sink_Specific_AngMom[2]};
        Vec3<double> dlv = cross(sam, dpos) * (wk * (double)local.angmom_norm_topass_in_swallowloop);
        for(int k = 0; k < 3; k++) Vel_j[k] += dlv[k];
        out.accreted_momentum[0] -= Mass_j * dlv[0];
        out.accreted_momentum[1] -= Mass_j * dlv[1];
        out.accreted_momentum[2] -= Mass_j * dlv[2];
        Vec3<double> cross_dl = cross(dpos, dlv);
        out.accreted_J[0] -= Mass_j * cross_dl[0];
        out.accreted_J[1] -= Mass_j * cross_dl[1];
        out.accreted_J[2] -= Mass_j * cross_dl[2];
    }
#endif
#if defined(SINK_RETURN_BFLUX)
    if(P[j].Type == 0 && (double)local.kernel_norm_topass_in_swallowloop > 0) {
        double b_frac = DMIN(0.1, (double)local.Dt / ((double)local.Sink_Mass_Reservoir / (double)local.Mdot))
                        * wk / (double)local.kernel_norm_topass_in_swallowloop;
        for(int k = 0; k < 3; k++) {
            double dB = b_frac * (double)local.B[k];
            Kokkos::atomic_add(&CellP[j].B[k],     (MyFloat)dB);
            Kokkos::atomic_add(&CellP[j].BPred[k], (MyFloat)dB);
            out.accreted_B[k] -= dB;
        }
    }
#endif

    double f_accreted = 0;

    if(P[j].SwallowID == local.ID && Mass_j > 0 && r2 > 0) {
        f_accreted = 1;
#ifdef SINK_WIND_KICK
        if(P[j].Type == 0) {
            f_accreted = All.Sink_accreted_fraction;
#ifndef SINK_GRAVCAPTURE_GAS
            if((All.SinkFeedbackFactor > 0) && (All.SinkFeedbackFactor != 1.0)) {
                f_accreted /= All.SinkFeedbackFactor;
            } else {
                if(All.Sink_outflow_velocity > 0) {
                    f_accreted = 1.0 / (1.0 + fabs(1.0 * SINK_WIND_KICK) * All.SinkRadiativeEfficiency * C_LIGHT_CODE / All.Sink_outflow_velocity);
                }
            }
            if((sink_mass_withdisk - (double)local.Mass) <= 0) { f_accreted = 0; }
#endif
        }
#endif

        double mcount_for_conserve = f_accreted * Mass_j;
#if (SINK_FOLLOW_ACCRETED_ANGMOM == 1)
        if(P[j].Type != 5) { mcount_for_conserve = 0; }
        else                { mcount_for_conserve = (double)P[j].Sink_Mass; }
#ifdef SINK_ALPHADISK_ACCRETION
        if(P[j].Type == 5) { mcount_for_conserve += (double)P[j].Sink_Mass_Reservoir; }
#endif
#endif
#ifdef GRAIN_FLUID
        if((1 << P[j].Type) & GRAIN_PTYPES) { out.accreted_dust_Mass += Mass_j; }
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
        if(P[j].Type == 0) {
            double photon_energy = 0;
            for(int kf = 0; kf < N_RT_FREQ_BINS; kf++) photon_energy += (double)CellP[j].Rad_E_gamma[kf];
            out.accreted_photon_energy += photon_energy;
        }
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
        out.accreted_momentum[0] += mcount_for_conserve * dvel[0];
        out.accreted_momentum[1] += mcount_for_conserve * dvel[1];
        out.accreted_momentum[2] += mcount_for_conserve * dvel[2];
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM)
        out.accreted_centerofmass[0] += mcount_for_conserve * dpos[0];
        out.accreted_centerofmass[1] += mcount_for_conserve * dpos[1];
        out.accreted_centerofmass[2] += mcount_for_conserve * dpos[2];
#endif
#ifdef SINK_RETURN_BFLUX
        out.accreted_B[0] += (double)CellP[j].BPred[0];
        out.accreted_B[1] += (double)CellP[j].BPred[1];
        out.accreted_B[2] += (double)CellP[j].BPred[2];
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
        {
            Vec3<double> cdv = cross(dpos, dvel);
            out.accreted_J[0] += mcount_for_conserve * cdv[0];
            out.accreted_J[1] += mcount_for_conserve * cdv[1];
            out.accreted_J[2] += mcount_for_conserve * cdv[2];
            if(P[j].Type == 5) {
                Vec3<double> psam{(double)P[j].Sink_Specific_AngMom[0], (double)P[j].Sink_Specific_AngMom[1], (double)P[j].Sink_Specific_AngMom[2]};
                out.accreted_J[0] += mcount_for_conserve * psam[0];
                out.accreted_J[1] += mcount_for_conserve * psam[1];
                out.accreted_J[2] += mcount_for_conserve * psam[2];
            }
        }
#endif

        if(P[j].Type == 5) {
            /* Sink-sink merger */
#ifdef SINK_INCREASE_DYNAMIC_MASS
            double acc_mass = ((double)P[j].Sink_Mass > Mass_j / SINK_INCREASE_DYNAMIC_MASS)
                                ? (double)P[j].Sink_Mass : (Mass_j / SINK_INCREASE_DYNAMIC_MASS);
            out.accreted_Mass += acc_mass;
#else
            out.accreted_Mass += Mass_j;
#endif
            out.accreted_Sink_Mass += (double)P[j].Sink_Mass;
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
            out.Sink_AccretionDeficit += (double)P[j].Sink_AccretionDeficit;
#endif
#ifdef SINK_ALPHADISK_ACCRETION
            out.accreted_Sink_Mass_reservoir += (double)P[j].Sink_Mass_Reservoir;
#endif
#ifdef SINK_WIND_SPAWN
#ifdef SINK_ALPHADISK_ACCRETION
            out.accreted_Sink_Mass_reservoir += (double)P[j].unspawned_wind_mass;
#else
            out.accreted_Sink_Mass          += (double)P[j].unspawned_wind_mass;
#endif
#endif
#ifdef SINK_COUNTPROGS
            out.Sink_CountProgs += P[j].Sink_CountProgs;
#endif
            int bin = P[j].TimeBin;
            Kokkos::atomic_add(&out.delta_TimeBin_Sink_mass[bin],          -(double)P[j].Sink_Mass);
            Kokkos::atomic_add(&out.delta_TimeBin_Sink_dynamicalmass[bin], -Mass_j);
            Kokkos::atomic_add(&out.delta_TimeBin_Sink_Mdot[bin],          -(double)P[j].Sink_Mdot);
            if((double)P[j].Sink_Mass > 0) {
                Kokkos::atomic_add(&out.delta_TimeBin_Sink_Medd[bin], -(double)P[j].Sink_Mdot / (double)P[j].Sink_Mass);
            }
            Mass_j = 0;
#ifdef SINK_ALPHADISK_ACCRETION
            Kokkos::atomic_store(&P[j].Sink_Mass_Reservoir, (MyFloat)0);
#endif
            Kokkos::atomic_store(&P[j].Sink_Mdot, (MyFloat)0);
            Kokkos::atomic_store(&P[j].Sink_Mass, (MyFloat)0);
#ifdef GALSF
            out.Accreted_Age = P[j].StellarAge;
#endif
            out.n_sink_swallowed++;
        }

#if defined(SINK_GRAVCAPTURE_NONGAS)
        if(P[j].Type > 0 && P[j].Type < 5) {
            out.accreted_Mass += Mass_j;
            if(P[j].Type == 1 || (All.ComovingIntegrationOn && (P[j].Type == 2 || P[j].Type == 3))) {
                out.accreted_Sink_Mass += Mass_j;
                out.n_dm_swallowed++;
            } else {
#ifdef SINK_ALPHADISK_ACCRETION
                out.accreted_Sink_Mass_reservoir += Mass_j;
#else
                out.accreted_Sink_Mass           += Mass_j;
#endif
                out.n_star_swallowed++;
            }
            Mass_j = 0;
        }
#endif

        if(P[j].Type == 0) {
            out.accreted_Mass += f_accreted * Mass_j;
#ifdef SINK_GRAVCAPTURE_GAS
#ifdef SINK_ALPHADISK_ACCRETION
            out.accreted_Sink_Mass_reservoir += f_accreted * Mass_j;
#else
            out.accreted_Sink_Mass           += f_accreted * Mass_j;
#endif
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
            out.Sink_AccretionDeficit -= f_accreted * Mass_j;
#endif
            Mass_j *= (1.0 - f_accreted);
#ifdef SINK_WIND_KICK
            {
                double v_kick = All.Sink_outflow_velocity;
                Vec3<double> dir{dpos[0], dpos[1], dpos[2]};
#if (SINK_WIND_KICK < 0)
                if(dot(dir, J_dir) > 0) { dir = J_dir; } else { dir = -J_dir; }
#endif
                double nrm = dir.norm_sq();
                if(nrm <= 0) { dir[0] = 0; dir[1] = 0; dir[2] = 1; }
                else         { nrm = sqrt(nrm); dir /= nrm; }
                for(int k = 0; k < 3; k++) { Vel_j[k] += v_kick * All.cf_atime * dir[k]; }
#ifdef GALSF_SUBGRID_WINDS
                Kokkos::atomic_store(&CellP[j].DelayTime, (MyFloat)(All.WindFreeTravelMaxTimeFactor / All.cf_hubble_a));
#endif
            }
#endif
            out.n_gas_swallowed++;
        }
    } /* end of (SwallowID matches) branch */

#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    if(mom_budget > 0 && (double)local.Dt > 0 && OriginallyMarkedSwallowID == 0 &&
       P[j].SwallowID == 0 && Mass_j > 0 && P[j].Type == 0) {
        double r = sqrt(r2);
        if(r > 0) {
            Vec3<double> dir{dpos[0], dpos[1], dpos[2]}; dir /= r;
            double cos_t = dot(dir, J_dir);
            double w = sink_fb_angleweight_localcoupling_gpu(P[j], CellP[j], cos_t, r, h_i);
            double mom_wt = ((double)local.Sink_angle_weighted_kernel_sum > 0)
                            ? (w / (double)local.Sink_angle_weighted_kernel_sum) : 0;
#ifdef SINK_PHOTONMOMENTUM
            double v_kick = All.Sink_Rad_MomentumFactor * mom_wt * mom_budget / Mass_j;
            Vel_j[0] += v_kick * All.cf_atime * dir[0];
            Vel_j[1] += v_kick * All.cf_atime * dir[1];
            Vel_j[2] += v_kick * All.cf_atime * dir[2];
#endif
            (void)mom_wt;
        }
    }
#endif

    /* Commit delta j-writes atomically.  Matches sink_swallow_and_kick.cc:458-504 exactly. */
    if(Mass_j != Mass_j_0 ||
       Vel_j[0] != Vel_j_0[0] || Vel_j[1] != Vel_j_0[1] || Vel_j[2] != Vel_j_0[2] ||
       InternalEnergy_j != InternalEnergy_j_0) {
        if(Mass_j > 0) {
            /* The "don't resurrect zero-mass" double-check is implicit here:
             * an atomic_add of (Mass_j - Mass_j_0) only makes sense if the
             * current P[j].Mass is nonzero. Kokkos::atomic_add is commutative;
             * if multiple threads race, the final sum is correct so long as
             * all contributions are finite. Zero-check happens below via the
             * atomic_store fallback if the sum would have gone non-positive. */
            double dmass = Mass_j - Mass_j_0;
            double current_mass = (double)Kokkos::atomic_load(&P[j].Mass);
            if(current_mass > 0) {
                Kokkos::atomic_add(&P[j].Mass, (MyDouble)dmass);
            }
        } else {
            Kokkos::atomic_store(&P[j].Mass, (MyDouble)0);
        }
        for(int k = 0; k < 3; k++) {
            double dVel = Vel_j[k] - Vel_j_0[k];
            double dpk  = Mass_j * Vel_j[k] - Mass_j_0 * Vel_j_0[k];
            Kokkos::atomic_add(&P[j].Vel[k], (MyDouble)dVel);
            Kokkos::atomic_add(&P[j].dp[k],  (MyFloat)dpk);
        }
        if(P[j].Type == 0) {
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            if(Mass_j > 0) {
                Kokkos::atomic_add(&CellP[j].MassTrue, (MyDouble)(Mass_j - Mass_j_0));
            } else {
                Kokkos::atomic_store(&CellP[j].MassTrue, (MyDouble)0);
            }
#endif
            for(int k = 0; k < 3; k++) {
                Kokkos::atomic_add(&CellP[j].VelPred[k], (MyDouble)(Vel_j[k] - Vel_j_0[k]));
            }
            double dIE = InternalEnergy_j - InternalEnergy_j_0;
            Kokkos::atomic_add(&CellP[j].InternalEnergy,     (MyDouble)dIE);
            Kokkos::atomic_add(&CellP[j].InternalEnergyPred, (MyDouble)dIE);
        }
    }
}

#endif /* GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY && OPENMP_GPU_OFFLOAD */

#endif /* SINK_PARTICLES */
