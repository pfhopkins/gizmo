#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "ketju_star_mergers.h"
#include "../../galaxy_sf/single_star_record.h"
#ifdef SINGLE_STAR_MERGER_PRODUCT
#include "../../galaxy_sf/stellar_evolution_tables.h"
#endif

/* the survivor of a merger takes the victims' metals and drawn mass */
#ifdef STAR_MERGERS_SHARED

#ifdef SINGLE_STAR_MERGER_PRODUCT
/* formation time (StellarFormationTime convention) of a star that is age_yr old now */
static double star_formation_time_for_age(double age_yr)
{
    double age_gyr = age_yr * 1e-9;
    if(!All.ComovingIntegrationOn) {return All.Time - age_gyr / UNIT_TIME_IN_GYR;}
    double lo = 1e-6, hi = All.Time; /* time since a decreases with a */
    if(evaluate_time_since_t_initial_in_Gyr(lo) <= age_gyr) {return lo;}
    for(int it = 0; it < 80; it++) {double mid = 0.5 * (lo + hi); if(evaluate_time_since_t_initial_in_Gyr(mid) > age_gyr) {lo = mid;} else {hi = mid;}}
    return 0.5 * (lo + hi);
}
/* main-sequence fraction used for the rejuvenation */
static double star_ms_fraction(double M0, double Zb, double age_yr)
{
    double logM = log10(M0), logZ = log10(DMAX(Zb, 1e-10));
    double tab = age_yr - stellar_t_PMS(logM, logZ), t0 = stellar_t_MS_start(logM, logZ), t1 = stellar_t_MS_end(logM, logZ);
    if(!(t1 > t0)) {return 1;}
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
    const int buf_size = n_met + 3; /* [metals][drawn mass][birth Z][age yr] */
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
#ifdef GALSF_SINGLE_STARS_FROM_IMF
            if(P[v].StarIdx > 0) {buf[n_met] = SingleStars.M_drawn[single_star_slot(P[v].StarIdx)]; buf[n_met + 1] = SingleStars.Z_birth[single_star_slot(P[v].StarIdx)];}
#ifdef SINGLE_STAR_MERGER_PRODUCT
            if(P[v].StarIdx > 0) {buf[n_met + 2] = evaluate_stellar_age_Gyr(v) * 1.e9;}
#endif
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
        }
    }

    /* ---- Phase 4: each survivor takes its victims' metal mass and (Stella) the merger-product state ---- */
    for(auto &it : acc) {
        int s_idx = it.first; survivor_acc &a = it.second; (void)s_idx;
        double M_sum = a.M_old; for(double mv : a.victim_mass) M_sum += mv;
#ifdef METALS
        for(int k = 0; k < NUM_METAL_SPECIES; k++) P[s_idx].Metallicity[k] = (a.M_old * P[s_idx].Metallicity[k] + a.buf_sum[k]) / M_sum;
#endif
#ifdef GALSF_SINGLE_STARS_FROM_IMF
#ifdef SINGLE_STAR_MERGER_PRODUCT
        if(P[s_idx].StarIdx > 0 && SingleStars.M_drawn[single_star_slot(P[s_idx].StarIdx)] > 0 && SingleStars.event_bits[single_star_slot(P[s_idx].StarIdx)] == 0) {
            /* a main-sequence star of the merged mass, birth Z mass-weighted, MS fraction 0.1 sum_i M_i f_i / M (Hurley, Pols & Tout 2002 eq. 80) */
            int s = single_star_slot(P[s_idx].StarIdx);
            double Ms = a.M_old * UNIT_MASS_IN_SOLAR, Mnew = P[s_idx].Mass * UNIT_MASS_IN_SOLAR, Mtot = Ms;
            double Zsum = Ms * SingleStars.Z_birth[s], fsum = Ms * star_ms_fraction(SingleStars.M_drawn[s], SingleStars.Z_birth[s], evaluate_stellar_age_Gyr(s_idx) * 1.e9);
            for(size_t v = 0; v < a.victims.size(); v++) {
                const std::vector<double> &b = a.victims[v]; double Mv = a.victim_mass[v] * UNIT_MASS_IN_SOLAR;
                Mtot += Mv; Zsum += Mv * b[n_met + 1];
                if(b[n_met] > 0) {fsum += Mv * star_ms_fraction(b[n_met], b[n_met + 1], b[n_met + 2]);} else {fsum += Mv;}
            }
            double Zb = Zsum / Mtot, logM = log10(Mnew), logZ = log10(DMAX(Zb, 1e-10));
            double t0 = stellar_t_MS_start(logM, logZ), t1 = stellar_t_MS_end(logM, logZ), tab = t0 + 0.1 * (fsum / Mtot) * DMAX(t1 - t0, 0.0);
            SingleStars.Z_birth[s] = Zb; SingleStars.M_drawn[s] = Mnew;
            P[s_idx].StellarAge = star_formation_time_for_age(stellar_t_PMS(logM, logZ) + tab);
#ifdef GALSF_FB_MECHANICAL
            SingleStars.last_wind_log_age[s] = log10(DMAX(tab, 100.0)); /* the wind table restarts at the product's age */
#endif
            printf("%s STAR MERGER PRODUCT: ID %llu M=%g Msun Zb=%g MS fraction %g -> age %g yr\n", tag, (unsigned long long)P[s_idx].ID, Mnew, Zb, 0.1 * fsum / Mtot, stellar_t_PMS(logM, logZ) + tab);
            continue;
        }
#endif
        if(P[s_idx].StarIdx > 0) {for(const std::vector<double> &b : a.victims) {SingleStars.M_drawn[single_star_slot(P[s_idx].StarIdx)] += b[n_met];}} /* the survivor keeps its age */
#endif
    }
}
#endif /* STAR_MERGERS_SHARED */
