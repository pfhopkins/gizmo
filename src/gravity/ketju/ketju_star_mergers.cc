#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "ketju_star_mergers.h"
#include "../../galaxy_sf/single_star_record.h"

/* the survivor of a merger takes the victims' metals and drawn mass */
#ifdef STAR_MERGERS_SHARED

void star_mergers_apply(std::vector<star_merger_action> &actions, const char *tag)
{
    const int n_mergers = (int)actions.size(); (void)tag;
    /* ---- Phase 3: victims send their data to the survivors' tasks ---- */
#ifdef METALS
    const int n_met = NUM_METAL_SPECIES;
#else
    const int n_met = 0;
#endif
    const int buf_size = n_met + 1; /* [metals][drawn mass] */
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
            if(P[v].StarIdx > 0) {buf[n_met] = SingleStars.M_drawn[single_star_slot(P[v].StarIdx)];}
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
        if(P[s_idx].StarIdx > 0) {for(const std::vector<double> &b : a.victims) {SingleStars.M_drawn[single_star_slot(P[s_idx].StarIdx)] += b[n_met];}} /* the survivor keeps its age */
#endif
    }
}
#endif /* STAR_MERGERS_SHARED */
