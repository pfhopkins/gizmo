/* Driver for test_stella_merger_remnant.py: begrun() on a small Stella box, then two gas cells become an 8 Msun compact
   remnant (sampled, MstarSampleIMF[0] = 0) and a live 3 Msun star; star_mergers_apply merges the live star into the remnant
   as KETJU hands it over (survivor mass already the sum). Prints PASS|FAIL lines. */
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "declarations/allvars.h"
#include "core/proto.h"
#include "galaxy_sf/stella_star_mergers.h"

static int n_fail = 0;
static void check(const char *name, bool ok, double got, double want) {if(!ok) {n_fail++;} printf("%s %s got=%.17g want=%.17g\n", ok ? "PASS" : "FAIL", name, got, want);}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    strcpy(ParameterFile, argv[1]);
    RestartFlag = 0; RestartSnapNum = -1;
    for(int i = 0; i < CPU_PARTS; i++) {All.CPU_Sum[i] = CPU_Step[i] = 0;}
    CPUThisRun = 0; WallclockTime = my_second();
    begrun();
    int s = 0, v = 1; /* survivor (remnant), victim (live star) */
    double Ms = 8.0 / UNIT_MASS_IN_SOLAR, Mv = 3.0 / UNIT_MASS_IN_SOLAR, Zs = 0.02, Zv = 0.01;
    for(int i : {s, v}) {P[i].Type = 4; P[i].sampled = 1; P[i].StellarAge = All.Time; P[i].BirthMetallicity = (i == s) ? Zs : Zv;
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {P[i].Metallicity[k] = (k == 0) ? ((i == s) ? Zs : Zv) : P[i].Metallicity[k];}}
    P[s].MstarSampleIMF[0] = 0;   P[s].Mass = Ms + Mv; /* remnant, mass already summed as after MSTAR's merger */
    P[v].MstarSampleIMF[0] = 3.0; P[v].Mass = Mv;
    std::vector<star_merger_action> acts(1);
    star_merger_action &a = acts[0];
    a.id_merged = P[v].ID; a.id_survivor = P[s].ID; a.type_merged = a.type_survivor = 4; a.task_merged = a.task_survivor = ThisTask;
    a.idx_merged = v; a.idx_survivor = s; a.M_merged = Mv; a.M_survivor_old = Ms; a.M_survivor_new = Ms + Mv;
    star_mergers_apply(acts, "TEST");
    fflush(stdout);
    check("survivor:stays_remnant", P[s].MstarSampleIMF[0] == 0, P[s].MstarSampleIMF[0], 0);
    check("survivor:mass_is_sum", fabs(P[s].Mass / (Ms + Mv) - 1) < 1e-12, P[s].Mass, Ms + Mv);
    check("survivor:metals_mass_weighted", fabs(P[s].Metallicity[0] / ((Ms * Zs + Mv * Zv) / (Ms + Mv)) - 1) < 1e-6, P[s].Metallicity[0], (Ms * Zs + Mv * Zv) / (Ms + Mv));
    check("victim:removed", P[v].Mass == 0 && P[v].MstarSampleIMF[0] == 0, P[v].Mass, 0);
    printf("DONE failed=%d\n", n_fail);
    fflush(stdout);
    MPI_Finalize();
    return n_fail;
}
