/* Unit test (T-C1) for structure/hbt/hbt_catalogue_core.h and hbt_catalogue_mpi.cc: per-type offsets of groups into
   the per-type member lists equal exclusive prefix sums of GroupLenType over all ranks in group order, and the
   per-type totals equal the list lengths. Exit 0 only if every check passes. */
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <mpi.h>
#include "structure/hbt/hbt_catalogue_core.h"
#include "structure/hbt/hbt_catalogue_mpi.h"

static uint64_t rng_state = 987654321;
static uint64_t rng_next() {rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state;}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, ntask; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ntask);
    int nfail = 0;
    for(int trial = 0; trial < 20; trial++)
    {
        /* groups distributed in group order over ranks, some ranks empty */
        const long ng_tot = 1 + (long)(rng_next() % 5000);
        std::vector<int> all(ng_tot * HBT_NTYPES);
        for(long i = 0; i < ng_tot * HBT_NTYPES; i++) {all[i] = (rng_next() % 3 == 0) ? 0 : (int)(rng_next() % 1000);}
        std::vector<long> first(ntask + 1);
        for(int t = 0; t <= ntask; t++) {first[t] = (t == 0) ? 0 : (t == ntask ? ng_tot : (long)(rng_next() % (uint64_t)(ng_tot + 1)));}
        for(int t = 1; t < ntask; t++) {if(first[t] < first[t - 1]) {first[t] = first[t - 1];}}
        const long ng = first[rank + 1] - first[rank];
        std::vector<long long> off(ng * HBT_NTYPES + 1), tot(HBT_NTYPES);
        hbt_group_type_offsets_global(all.data() + first[rank] * HBT_NTYPES, ng, off.data(), tot.data(), MPI_COMM_WORLD);
        /* serial reference: every rank holds the full table (same seed) */
        long long run[HBT_NTYPES] = {0}, bad = 0;
        for(long g = 0; g < ng_tot; g++)
            for(int t = 0; t < HBT_NTYPES; t++)
            {
                if(g >= first[rank] && g < first[rank + 1] && off[(g - first[rank]) * HBT_NTYPES + t] != run[t]) {bad++;}
                run[t] += all[g * HBT_NTYPES + t];
            }
        for(int t = 0; t < HBT_NTYPES; t++) {if(tot[t] != run[t]) {bad++;}}
        long long nloc[HBT_NTYPES]; hbt_group_type_totals(all.data(), ng_tot, nloc);
        for(int t = 0; t < HBT_NTYPES; t++) {if(nloc[t] != run[t]) {bad++;}}
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(bad) {nfail++;}
        if(rank == 0 && (trial < 3 || bad)) {printf("%s trial %d: %ld groups, offsets and totals (%lld mismatches)\n", bad ? "FAIL" : "PASS", trial, ng_tot, bad);}
    }
    /* member-type counts of a flat type array */
    {
        std::vector<int> ty(100000); long long want[HBT_NTYPES] = {0}, got[HBT_NTYPES];
        for(size_t i = 0; i < ty.size(); i++) {ty[i] = (int)(rng_next() % HBT_NTYPES); want[ty[i]]++;}
        hbt_count_members_by_type(ty.data(), (long)ty.size(), got);
        int ok = 1; for(int t = 0; t < HBT_NTYPES; t++) {ok &= (got[t] == want[t]);}
        if(!ok) {nfail++;}
        if(rank == 0) {printf("%s member counts by type\n", ok ? "PASS" : "FAIL");}
    }
    if(rank == 0) {printf("ranks=%d failures=%d\n", ntask, nfail);}
    MPI_Finalize();
    return nfail ? 1 : 0;
}
