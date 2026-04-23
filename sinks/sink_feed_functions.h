/* sink_feed_functions.h — GPU-callable per-pair kernel for sink_feed_evaluate (C3).
 *
 * Ports sink_feed_evaluate (sinks/sink_feed.cc) to a KOKKOS_INLINE_FUNCTION
 * so it can be called from a device lambda in sink_feed_gpu.cc.
 *
 * i-particles are active sinks (Type 5); j-particles are neighbors matched by
 * SINK_NEIGHBOR_BITFLAG (gas + sink + other types depending on config).
 *
 * Key j-writes (GPU-safe via Kokkos::atomic_exchange):
 *   P[j].SwallowID   — set-if-zero semantics; last-writer-wins acceptable
 *   CellP[j].Injected_Sink_Energy  — additive (SINK_THERMALFEEDBACK only)
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef SINK_FEED_FUNCTIONS_H
#define SINK_FEED_FUNCTIONS_H

#ifdef SINK_PARTICLES

#include "../declarations/gpu_rng.h"
#include "sink_functions.h"

struct SinkFeedLocalIn {
    Vec3<MyDouble> Pos;
    Vec3<MyFloat>  Vel;
    MyFloat KernelRadius;
    MyFloat Mass;
    MyFloat Sink_Mass;
    MyFloat Density;
    MyFloat Mdot;
    MyFloat Dt;
    MyIDType ID;
#ifdef SINK_GRAVCAPTURE_GAS
    MyFloat mass_to_swallow_edd;
#endif
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
    MyFloat edd_p;  /* precomputed: min(1, 1/eddington_factor) — probability of swallow per gas particle */
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    MyFloat Sink_AccretionDeficit;
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    MyFloat SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    MyFloat AGS_KernelRadius;
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat Sink_Mass_Reservoir;
#endif
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
    Vec3<MyFloat> Jgas_in_Kernel;
#endif
#ifdef SINK_THERMALFEEDBACK
    MyFloat thermal_energy;  /* precomputed: sink_lum_bol(Mdot,Sink_Mass,-1)*Dt */
#endif
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    int Sink_eligible_for_binary_merge_away;
#endif
};

struct SinkFeedOut {
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
    double Sink_angle_weighted_kernel_sum;
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
    double Sink_PotentialMinimumOfNeighbors;
    Vec3<double> Sink_PotentialMinimumOfNeighborsPos;
#endif
};


KOKKOS_INLINE_FUNCTION
static void sink_feed_pair_kernel(const struct SinkFeedLocalIn& local,
                                  int j,
                                  struct particle_data * KOKKOS_RESTRICT kp,
                                  struct gas_cell_data * KOKKOS_RESTRICT kc,
                                  double r2,
                                  const Vec3<double>& dpos,
                                  const Vec3<double>& dvel,
                                  struct SinkFeedOut& out,
                                  double& mass_markedswallow,
                                  uint64_t rng_step,
                                  const double * KOKKOS_RESTRICT softening_kernel_radius,
                                  const int * KOKKOS_RESTRICT sink_binary_merge_eligible)
{
    if(kp[j].Mass <= 0) return;

    double h_i = (double)local.KernelRadius;
    double h_i2 = h_i * h_i;
    double soft_j = softening_kernel_radius ? softening_kernel_radius[j] : SinkParticle_GravityKernelRadius;
    double heff_j = DMAX((double)kp[j].KernelRadius, soft_j);
    if(r2 >= h_i2 && r2 >= heff_j * heff_j) return;

    double r = sqrt(r2);
    if(r <= 0) return;

    double vrel_sq = dvel.norm_sq();
    double vrel = sqrt(vrel_sq) / All.cf_atime;

    double ags_h_i = SinkParticle_GravityKernelRadius;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    ags_h_i = (double)local.AGS_KernelRadius;
#endif

    double sink_radius = SinkParticle_GravityKernelRadius;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    sink_radius = (double)local.SinkRadius;
#endif

    double vesc = sink_vesc_gpu(kp[j], kc[j], (double)local.Mass, r, ags_h_i);

    MyIDType SwallowID_j = Kokkos::atomic_load(&kp[j].SwallowID);

#ifdef SINK_REPOSITION_ON_POTMIN
    {
        double boundedness_function = kp[j].Potential + 0.5 * vrel * vrel * All.cf_atime;
        if(boundedness_function < 0) {
            double wt_rsoft = r / (3. * SinkParticle_GravityKernelRadius);
            boundedness_function *= 1. / (1. + wt_rsoft * wt_rsoft);
        }
        double potential_function = boundedness_function;
        if(potential_function < out.Sink_PotentialMinimumOfNeighbors &&
           kp[j].Type != 0 && kp[j].Type != 5) {
            out.Sink_PotentialMinimumOfNeighbors    = potential_function;
            out.Sink_PotentialMinimumOfNeighborsPos = kp[j].Pos;
        }
    }
#endif

    /* ---- sink-sink merger check ---- */
    if(kp[j].Type == 5) {
        if(((local.ID != kp[j].ID) || (r2 > 0)) &&
           (SwallowID_j == 0) && (kp[j].Sink_Mass < local.Sink_Mass)) {
#ifdef SINGLE_STAR_SINK_DYNAMICS
            int allow_sink_merger = 1;
            if(r >= 1.0001 * kp[j].Min_Distance_to_Sink)  allow_sink_merger = 0;
            if(r >= heff_j)                                 allow_sink_merger = 0;
            if(kp[j].Mass > local.Mass)                    allow_sink_merger = 0;
            if((kp[j].Mass == local.Mass) && (kp[j].ID > local.ID)) allow_sink_merger = 0;
            double max_rmerge = 1.0 * sink_radius;
            double max_mmerge = 10. * kp[j].Sink_Formation_Mass;
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
            sink_apply_binary_merge_away_limits(local.Sink_eligible_for_binary_merge_away,
                                                sink_binary_merge_eligible ? sink_binary_merge_eligible[j] : 0,
                                                soft_j, (double)local.KernelRadius,
                                                (double)kp[j].KernelRadius,
                                                (double)local.Mass, (double)kp[j].Mass,
                                                sink_radius, &allow_sink_merger,
                                                &max_rmerge, &max_mmerge);
#else
            if(!sink_check_boundedness_gpu(kp[j], kc[j], vrel, vesc, r, sink_radius)) allow_sink_merger = 0;
#endif
            if(r >= max_rmerge)              allow_sink_merger = 0;
            if(kp[j].Mass > max_mmerge)     allow_sink_merger = 0;
            if(allow_sink_merger == 1)
#endif
            {
                if(vrel < vesc) { SwallowID_j = local.ID; }
            }
        }
    }

    /* ---- grav-capture check (non-Type5) ---- */
#if defined(SINK_GRAVCAPTURE_GAS) || defined(SINK_GRAVCAPTURE_NONGAS)
    if(kp[j].Type != 5 && SwallowID_j < local.ID) {
        int do_gravcap = 1;
#ifdef SINGLE_STAR_SINK_DYNAMICS
        {
            double eps = DMAX(r, DMAX(heff_j, ags_h_i) * KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER);
            if(eps * eps * eps / (kp[j].Mass + (double)local.Mass) > kp[j].SwallowTime) do_gravcap = 0;
        }
#endif
#ifdef SINK_ALPHADISK_ACCRETION
        if((double)local.Sink_Mass_Reservoir >= SINK_ALPHADISK_ACCRETION * (double)local.Sink_Mass) do_gravcap = 0;
#endif
        if(do_gravcap && vrel < vesc) {
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            {
                double spec_mom = dot(dvel, dpos);
                spec_mom = r2 * vrel * vrel - spec_mom * spec_mom * All.cf_a2inv;
                if(spec_mom >= All.G * ((double)local.Mass + kp[j].Mass) * sink_radius) do_gravcap = 0;
            }
#endif
            if(do_gravcap && sink_check_boundedness_gpu(kp[j], kc[j], vrel, vesc, r, sink_radius)) {
#ifdef SINK_GRAVCAPTURE_NONGAS
                if(kp[j].Type != 0) { SwallowID_j = local.ID; }
#endif
#ifdef SINK_GRAVCAPTURE_GAS
                if(kp[j].Type == 0) {
#if defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
                    double p = (double)local.edd_p;  /* precomputed on CPU */
#ifdef SINK_WIND_KICK
                    if(All.Sink_accreted_fraction > 0) p /= All.Sink_accreted_fraction;
#endif
                    double w = gizmo_gpu_rand_double((uint64_t)kp[j].ID, rng_step);
                    if(w < p) { SwallowID_j = local.ID; }
#else
                    SwallowID_j = local.ID;
#endif
                }
#endif
            }
        }
    }
#endif /* SINK_GRAVCAPTURE_GAS || SINK_GRAVCAPTURE_NONGAS */

    /* ---- kernel quantities for gas pairs ---- */
    if(kp[j].Type == 0) {
        double hinv  = 1. / h_i, hinv3 = hinv * hinv * hinv;
        double u = r * hinv, wk = 0, dwk = 0;
        if(u < 1) { kernel_main(u, hinv3, hinv * hinv3, &wk, &dwk, -1); }

#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
        if(SwallowID_j < local.ID) {
            double dm_toacc = (double)local.Sink_AccretionDeficit - mass_markedswallow;
            double f_accreted = 1.;
#ifdef SINK_WIND_KICK
            if(All.Sink_accreted_fraction > 0) f_accreted = All.Sink_accreted_fraction;
#endif
            double p = 0;
            if(dm_toacc > 0 && local.Density > 0) { p = dm_toacc * wk / (double)local.Density; }
#ifdef SINK_WIND_KICK
            if(f_accreted > 0) {
                p /= f_accreted;
                double sink_mass_withdisk = (double)local.Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
                sink_mass_withdisk += (double)local.Sink_Mass_Reservoir;
#endif
                if(sink_mass_withdisk < (double)local.Mass && (double)local.Density > 0) {
                    p = ((1. - f_accreted) / f_accreted) * (double)local.Mdot * (double)local.Dt * wk / (double)local.Density;
                }
            }
#endif
            double w = gizmo_gpu_rand_double((uint64_t)kp[j].ID ^ (uint64_t)local.ID, rng_step + 1);
            if(w < p) {
                SwallowID_j = local.ID;
                mass_markedswallow += kp[j].Mass * f_accreted;
            }
        }
#endif /* SINK_SWALLOWGAS && !SINK_GRAVCAPTURE_GAS */

#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
        if((double)local.Dt > 0 && r > 0 && SwallowID_j == 0 && kp[j].Mass > 0) {
            Vec3<double> J_dir = (Vec3<double>)local.Jgas_in_Kernel;
            double norm_j2 = J_dir.norm_sq();
            if(norm_j2 > 0) { J_dir *= 1. / sqrt(norm_j2); } else { J_dir = {0, 0, 1}; }
            double cos_theta = dot(dpos / r, J_dir);
            out.Sink_angle_weighted_kernel_sum +=
                sink_fb_angleweight_localcoupling_gpu(kp[j], kc[j], cos_theta, r, h_i);
        }
#endif

#ifdef SINK_THERMALFEEDBACK
        if((double)local.Density > 0 && wk > 0) {
            double dE = (wk / (double)local.Density) * (double)local.thermal_energy * kp[j].Mass;
            Kokkos::atomic_add(&kc[j].Injected_Sink_Energy, dE);
        }
#endif
    }

    /* ---- commit SwallowID if set ---- */
    if(SwallowID_j > 0) {
        Kokkos::atomic_exchange(&kp[j].SwallowID, SwallowID_j);
    }
}

#endif /* SINK_PARTICLES */
#endif /* SINK_FEED_FUNCTIONS_H */
