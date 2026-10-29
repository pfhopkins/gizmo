#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "stella_star_mergers.h"
#if defined(STELLA_MERGER_PRODUCT)
#include "resolvedism_stellar_tables.h"
#include "resolvedism_star_age.h"
#endif

/* Phases 3-4 of KETJU's handle_mergers (galaxy_sf/ketju_coupling.cc, b1fa9895/33a286a2), shared with RUMI++ */
#ifdef STAR_MERGERS_SHARED

#ifdef STELLA_MERGER_PRODUCT
/* formation time (StellarAge convention) of a star that is age_yr old now */
static double star_formation_time_for_age(double age_yr)
{
    double age_gyr = age_yr * 1e-9;
    if(!All.ComovingIntegrationOn) return All.Time - age_gyr / UNIT_TIME_IN_GYR;
    double lo = 1e-6, hi = All.Time; /* time since a decreases with a */
    if(evaluate_time_since_t_initial_in_Gyr(lo) <= age_gyr) return lo;
    for(int it = 0; it < 80; it++) {double mid = 0.5 * (lo + hi); if(evaluate_time_since_t_initial_in_Gyr(mid) > age_gyr) lo = mid; else hi = mid;}
    return 0.5 * (lo + hi);
}
/* main-sequence fraction used for the rejuvenation */
static double star_ms_fraction(double M0, double Zb, double age_yr)
{
    double logM = log10(M0), logZ = log10(DMAX(Zb, 1e-10));
    double tab = age_yr - stellar_t_PMS(logM, logZ), t0 = stellar_t_MS_start(logM, logZ), t1 = stellar_t_MS_end(logM, logZ);
    if(!(t1 > t0)) return 1;
    return DMIN(DMAX((tab - t0) / (t1 - t0), 0.0), 1.0);
}
#endif

void star_mergers_apply(std::vector<star_merger_action> &actions, const char *tag)
{
    const int n_mergers = (int)actions.size(); (void)tag;
    /* ---- Phase 3: victims send their data to the survivors' tasks ---- */
#ifdef METALS
    const int n_met = NUM_METAL_SPECIES;
#else
    const int n_met = 0;
#endif
    /* [metals][MstarSampleIMF 0][MstarSampleIMF 4][birth Z][age yr] */
    const int buf_size = n_met + 4;
    struct survivor_acc {double M_old; std::vector<double> buf_sum; std::vector<std::vector<double>> victims; std::vector<double> victim_mass;};
    std::unordered_map<int, survivor_acc> acc; /* by local survivor index */

    for(int m = 0; m < n_mergers; m++) {
        star_merger_action &act = actions[m];
        std::vector<double> buf(buf_size, 0.0);
        if(act.task_merged == ThisTask) {
            int v = act.idx_merged; (void)v;
#ifdef METALS
            for(int k = 0; k < NUM_METAL_SPECIES; k++) buf[k] = P[v].Metallicity[k];
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
            if(P[v].Type == 4) {
                buf[n_met] = P[v].MstarSampleIMF[0];
#if (N_STELLAR_MASS >= 5)
                buf[n_met + 1] = P[v].MstarSampleIMF[4];
#endif
            }
#endif
#ifdef STELLA_MERGER_PRODUCT
            if(P[v].Type == 4) {buf[n_met + 2] = P[v].BirthMetallicity; buf[n_met + 3] = resolvedism_star_age_yr(v);}
#endif
        }
        if(act.task_merged != act.task_survivor) {
            if(act.task_merged == ThisTask) MPI_Send(buf.data(), buf_size, MPI_DOUBLE, act.task_survivor, 997, MPI_COMM_WORLD);
            if(act.task_survivor == ThisTask) MPI_Recv(buf.data(), buf_size, MPI_DOUBLE, act.task_merged, 997, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if(act.task_survivor == ThisTask) {
            survivor_acc &a = acc[act.idx_survivor];
            if(a.buf_sum.empty()) {a.M_old = act.M_survivor_old; a.buf_sum.assign(n_met, 0.0);}
            for(int k = 0; k < n_met; k++) a.buf_sum[k] += act.M_merged * buf[k];
            a.victims.push_back(buf); a.victim_mass.push_back(act.M_merged);
        }

        /* mark merged particle for removal */
        if(act.task_merged == ThisTask) {
            P[act.idx_merged].Mass = 0;
            TreeMomentsStaleFlag = 1;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
            P[act.idx_merged].MstarSampleIMF[0] = 0; /* inert ghost: no FB from the merged star */
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            P[act.idx_merged].UV_luminosity = 0; P[act.idx_merged].PE_luminosity = 0; P[act.idx_merged].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[act.idx_merged].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[act.idx_merged].OPT_luminosity = 0;
#endif
#endif
        }
    }

    /* ---- Phase 4: each survivor takes its victims' metal mass and (Stella) the merger-product state ---- */
    for(auto &it : acc) {
        int s_idx = it.first; survivor_acc &a = it.second; (void)s_idx;
        double M_sum = a.M_old; for(double mv : a.victim_mass) M_sum += mv;
#ifdef METALS
        for(int k = 0; k < NUM_METAL_SPECIES; k++) P[s_idx].Metallicity[k] = (a.M_old * P[s_idx].Metallicity[k] + a.buf_sum[k]) / M_sum;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        {double X_H_new = 1.0 - P[s_idx].Metallicity[0] - P[s_idx].Metallicity[MET_OF(ELEM_He)]; /* close sum(Met[1..]) = 1 via H */
         P[s_idx].Metallicity[MET_OF(ELEM_H)] = (MyFloat)DMAX(X_H_new, 0);}
#endif
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(P[s_idx].Type != 4) continue;
#ifdef STELLA_MERGER_PRODUCT
        if(P[s_idx].sampled && P[s_idx].MstarSampleIMF[0] > 0) {
            /* Product: a main-sequence star of the merged mass at the CoM, birth Z mass-weighted, rejuvenated after Hurley,
             * Pols & Tout 2002 eq. 80: its MS fraction is 0.1 sum_i M_i f_i / M (f_i the components' MS fractions). The wind
             * closure reference (MstarSampleIMF[4]) is reset so the product's cumulative wind table starts at its new age. */
            double Ms = a.M_old * UNIT_MASS_IN_SOLAR, Mnew = P[s_idx].Mass * UNIT_MASS_IN_SOLAR;
            double Zsum = Ms * P[s_idx].BirthMetallicity, fsum = Ms * star_ms_fraction(P[s_idx].MstarSampleIMF[0], P[s_idx].BirthMetallicity, resolvedism_star_age_yr(s_idx));
            for(size_t v = 0; v < a.victims.size(); v++) {
                const std::vector<double> &b = a.victims[v]; double Mv = a.victim_mass[v] * UNIT_MASS_IN_SOLAR;
                Zsum += Mv * b[n_met + 2];
                if(b[n_met] > 0) fsum += Mv * star_ms_fraction(b[n_met], b[n_met + 2], b[n_met + 3]); else fsum += Mv;
            }
            double Mtot = Ms; for(double mv : a.victim_mass) Mtot += mv * UNIT_MASS_IN_SOLAR;
            double Zb = Zsum / Mtot, logM = log10(Mnew), logZ = log10(DMAX(Zb, 1e-10));
            double t0 = stellar_t_MS_start(logM, logZ), t1 = stellar_t_MS_end(logM, logZ);
            double tab = t0 + 0.1 * (fsum / Mtot) * DMAX(t1 - t0, 0.0);
            P[s_idx].BirthMetallicity = Zb;
            P[s_idx].MstarSampleIMF[0] = Mnew;
            P[s_idx].StellarAge = star_formation_time_for_age(stellar_t_PMS(logM, logZ) + tab);
#ifdef GALSF_RESOLVEDISM_WINDS
            {
                double lage = log10(DMAX(tab, 100.0)), cum = 0;
                for(int kk = 0; kk < STBL_NELEM; kk++) cum += stellar_elem_ej_wind_cumulative(logM, logZ, lage, kk);
                P[s_idx].MstarSampleIMF[4] = (Mnew + cum) / UNIT_MASS_IN_SOLAR; /* closure: pending wind = cum(age) - (birth - M) = 0 now */
                P[s_idx].last_wind_log_age = lage; P[s_idx].M_current_old = Mnew; P[s_idx].WindMassAccum = 0; P[s_idx].WindMomentumAccum = 0;
            }
#elif (N_STELLAR_MASS >= 5)
            for(const std::vector<double> &b : a.victims) P[s_idx].MstarSampleIMF[4] += b[n_met + 1];
#endif
            printf("%s STAR MERGER PRODUCT: ID %llu M=%g Msun Zb=%g MS fraction %g -> age %g yr\n", tag, (unsigned long long)P[s_idx].ID,
                   Mnew, Zb, 0.1 * fsum / Mtot, stellar_t_PMS(logM, logZ) + tab);
            continue;
        }
#endif
        for(const std::vector<double> &b : a.victims) {
            if(b[n_met] > 0) P[s_idx].MstarSampleIMF[0] += b[n_met]; /* survivor carries the merged stellar mass; keeps its age */
#if (N_STELLAR_MASS >= 5)
            if(b[n_met + 1] > 0) P[s_idx].MstarSampleIMF[4] += b[n_met + 1]; /* birth mass too: wind closure reference */
#endif
        }
#endif
    }
}
#endif /* STAR_MERGERS_SHARED */
