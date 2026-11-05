/* Test-only audit of the gravity tree under KETJU (TEST_KETJU_TREE_MOTION_AUDIT): speed against the ancestors' vmax, box and
   centre of mass and bulk velocity of the nodes against their members, imported single-star leaves against the real stars. */
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../gravity/forcetree.h"
#include "../../gravity/gpu_gravity_tree.h"
#include "../../gravity/let_data.h"
#include "../../core/timestep_functions.h"

#ifdef TEST_KETJU_TREE_MOTION_AUDIT
namespace {
struct {long long vmax, vmax_chain, top_chain, box_chain, leaves, calls; double com_chain, vel_chain, vel_other, leaf_sum, leaf_max;} T;
Vec3<double> predicted_pos(int i)
{
    const double dt = (P[i].Ti_current == All.Ti_Current) ? 0 : get_drift_factor(P[i].Ti_current, All.Ti_Current, i, 0);
    const Vec3<double> v = particle_drift_velocity(i, P, CellP); Vec3<double> x;
    for(int j = 0; j < 3; j++) {x[j] = (double)P[i].Pos[j] + v[j] * dt;}
    return x;
}
}

/* after the bounds' flush, before the drift: chain members under every ancestor on every rank's copy (a kicked non-member's bound follows at the next tree update) */
void ketju_tree_audit_bounds(void)
{
    if(!Father || !Nodes || !Extnodes) {return;}
    std::vector<double> topmax(NTopnodes, 0.), truth(NTopnodes);
    for(int i = 0; i < NumPart && i < All.TreeParticleSlots; i++)
    {
        if(P[i].Mass <= 0 || Father[i] < 0) {continue;}
        const double vb = particle_motion_speed_bound(i, P, CellP);
        for(int no = Father[i]; no >= 0; no = Nodes[no].u.d.father)
        {
            if((double)Extnodes[no].vmax * (1 + 1e-6) < vb) {T.vmax++; if(P[i].KetjuIntegrated) {T.vmax_chain++;}}
            const int t = no - All.TreeNodeIndexBase;
            if(P[i].KetjuIntegrated && t >= 0 && t < NTopnodes && vb > topmax[t]) {topmax[t] = vb;}
        }
    }
    MPI_Allreduce(topmax.data(), truth.data(), NTopnodes, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for(int t = 0; t < NTopnodes; t++) {if((double)Extnodes[All.TreeNodeIndexBase + t].vmax * (1 + 1e-6) < truth[t]) {T.top_chain++;}}
}

/* after the tree update, before the walk; momentum still in P.dp has not reached the nodes */
void ketju_tree_audit_nodes(void)
{
    if(!Father || !Nodes || !Extnodes) {return;}
    const int base = All.TreeNodeIndexBase, nn = Numnodestree;
    std::vector<double> m(nn, 0.), mx(3 * nn, 0.), mv(3 * nn, 0.); std::vector<char> chain(nn, 0);
    for(int i = 0; i < NumPart && i < All.TreeParticleSlots; i++)
    {
        if(P[i].Mass <= 0 || Father[i] < 0) {continue;}
        const Vec3<double> x = predicted_pos(i);
        for(int no = Father[i]; no >= 0; no = Nodes[no].u.d.father)
        {
            const double half = 0.5 * ((double)Nodes[no].len + TREE_NODE_WIDENING_DELTA(Extnodes[no].vmax, get_drift_factor_undilated(Nodes[no].Ti_current, All.Ti_Current)));
            for(int j = 0; j < 3; j++) {if(P[i].KetjuIntegrated && fabs(x[j] - (double)Nodes[no].center[j]) > half * (1 + 1e-5)) {T.box_chain++; break;}}
            const int k = no - base; if(k < 0 || k >= nn) {continue;}
            m[k] += P[i].Mass; if(P[i].KetjuIntegrated) {chain[k] = 1;}
            for(int j = 0; j < 3; j++) {mx[3*k+j] += P[i].Mass * x[j]; mv[3*k+j] += P[i].Mass * (double)P[i].Vel[j] - (double)P[i].dp[j];}
        }
    }
    for(int k = 0; k < nn; k++)
    {
        const int no = base + k; if(m[k] <= 0 || (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))) {continue;}
        if(fabs(m[k] / (double)Nodes[no].u.d.mass - 1) > 1e-5) {continue;} /* mass change not yet refreshed */
        const double dt = get_drift_factor(Nodes[no].Ti_current, All.Ti_Current, no, 1);
        const int kicked = (Nodes[no].u.d.bitflags & (1 << BITFLAG_NODEHASBEENKICKED)) ? 1 : 0;
        double ds2 = 0, dv2 = 0;
        for(int j = 0; j < 3; j++)
        {
            const double vs = (double)Extnodes[no].vs[j] + (kicked ? (double)Extnodes[no].dp[j] / m[k] : 0.);
            const double ds = (double)Nodes[no].u.d.s[j] + vs * dt - mx[3*k+j] / m[k], dv = vs - mv[3*k+j] / m[k];
            ds2 += ds * ds; dv2 += dv * dv;
        }
        if(chain[k]) {T.com_chain = fmax(T.com_chain, sqrt(ds2)); T.vel_chain = fmax(T.vel_chain, sqrt(dv2));} else {T.vel_other = fmax(T.vel_other, sqrt(dv2));}
    }
}

/* after the walk: each imported single-star leaf where the walk put it, against the nearest real star */
void ketju_tree_audit_imports(void)
{
    struct xyz {double x[3];}; std::vector<xyz> mine;
    for(int i = 0; i < NumPart; i++) {if(P[i].Mass > 0) {Vec3<double> x = predicted_pos(i); mine.push_back({{x[0], x[1], x[2]}});}}
    int n = (int)mine.size() * (int)sizeof(xyz); std::vector<int> cnt(NTask), dis(NTask); MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int tot = 0; for(int t = 0; t < NTask; t++) {dis[t] = tot; tot += cnt[t];}
    std::vector<xyz> all(tot / sizeof(xyz)); MPI_Allgatherv(mine.data(), n, MPI_BYTE, all.data(), cnt.data(), dis.data(), MPI_BYTE, MPI_COMM_WORLD);
    const int fb = All.TreeNodeIndexBase + MaxNodes;
    for(int fs = 0; fs < Numforeignnodes; fs++)
    {
        const int no = fb + fs; if(ForeignLeafTag[fs] != LET_LEAF_TAG_REAL_LEAF || Nodes[no].u.d.mass <= 0) {continue;}
        const double dt = (Nodes[no].Ti_current == All.Ti_Current) ? 0 : get_drift_factor(Nodes[no].Ti_current, All.Ti_Current, no, 1);
        double best = 1e300;
        for(const xyz &t : all) {double r2 = 0; for(int j = 0; j < 3; j++) {const double d = (double)Nodes[no].u.d.s[j] + (double)Extnodes[no].vs[j] * dt - t.x[j]; r2 += d * d;} if(r2 < best) {best = r2;}}
        best = sqrt(best); T.leaves++; T.leaf_sum += best; T.leaf_max = fmax(T.leaf_max, best);
    }
}

void ketju_tree_audit_report(void)
{
    long long c[5] = {T.vmax, T.vmax_chain, T.top_chain, T.box_chain, T.leaves}, gc[5]; double d[4] = {T.com_chain, T.vel_chain, T.vel_other, T.leaf_max}, gd[4], ls = 0;
    MPI_Reduce(c, gc, 5, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); MPI_Reduce(d, gd, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); MPI_Reduce(&T.leaf_sum, &ls, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {printf("KETJU_TREE_AUDIT vmax %lld vmax_chain %lld top_chain %lld box_chain %lld com_chain %.4e vel_chain %.4e vel_other %.4e leaves %lld leaf_mean %.4e leaf_max %.4e\n",
        gc[0], gc[1], gc[2], gc[3], gd[0], gd[1], gd[2], gc[4], gc[4] ? ls / gc[4] : 0., gd[3]); fflush(stdout);}
}
#endif
