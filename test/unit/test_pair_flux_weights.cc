// Pair-flux time weights (HYDRO_EXACT_PAIR_FLUX_EXCHANGE): a pair's closing and opening weights must tile time
// with no gap or overlap whatever the two cells' power-of-two steps do, including swaps of which cell is smaller.
#include "test_harness.h"
#include "../../hydro/pair_flux_weights.h"
#include <random>

struct Cell {long long S, N;};

/* runs two cells to time T; new steps from pick(cell, t); returns the number of syncs whose interval did not start where
   the previous one ended (new rule), and the same count for the old per-cell coverage rule */
template<class Pick> static void run_pair(long long T, Pick pick, int &bad_new, int &bad_old, long long &covered)
{
    Cell c[2]; long long cov_old[2] = {0, 0}, oldstep[2];
    for(int k = 0; k < 2; k++) {long long ti = pick(k, 0LL); c[k] = {0, ti}; oldstep[k] = ti;}
    long long end_new = pairflux_pair_opening_ti(0, c[0].N, c[1].N);
    int ev0 = (oldstep[0] <= oldstep[1]) ? 0 : 1; long long end_old = c[ev0].N/2;
    for(int k = 0; k < 2; k++) {cov_old[k] = c[k].N/2;}
    bad_new = bad_old = 0;
    while(true)
    {
        long long t = (c[0].N < c[1].N) ? c[0].N : c[1].N;
        if(t > T) {break;}
        int act[2] = {c[0].N == t, c[1].N == t};
        /* new rule: closing from the pair's previous evaluation */
        long long w2 = pairflux_pair_closing_ti(t, c[0].S, c[1].S);
        if(t - w2 != end_new) {bad_new++;}
        /* old rule: the active cell with the smaller old step evaluates, from its own coverage */
        int ev = act[0] && act[1] ? ((c[0].N - c[0].S) <= (c[1].N - c[1].S) ? 0 : 1) : (act[0] ? 0 : 1);
        if(cov_old[ev] != end_old) {bad_old++;}
        for(int k = 0; k < 2; k++) {if(act[k]) {long long ti = pick(k, t); c[k] = {t, t + ti}; cov_old[k] = t + ti/2;}}
        end_new = t + pairflux_pair_opening_ti(t, c[0].N, c[1].N);
        end_old = cov_old[ev];
        covered = end_new;
        if(act[0] && act[1] && t == T) {break;}
    }
}

TEST_CASE("pair weights: explicit swap at t=4 (steps 2,4 -> 4,2) tiles time; the per-cell rule does not") {
    int bad_new, bad_old; long long cov = 0;
    auto pick = [](int k, long long t) -> long long {if(t < 4) {return k == 0 ? 2 : 4;} return (k == 0 && t % 4 == 0) ? 4 : 2;};
    run_pair(16, pick, bad_new, bad_old, cov);
    CHECK(bad_new == 0);
    CHECK(bad_old > 0);
}

TEST_CASE("pair weights: random power-of-two steps (synchronised increases) always tile time") {
    std::mt19937_64 rng(12345);
    int total_bad_new = 0, total_bad_old = 0;
    for(int rep = 0; rep < 2000; rep++)
    {
        const int bmax = 6; long long T = 1LL << bmax;
        auto pick = [&](int, long long t) -> long long {
            int b = (int)(rng() % (bmax + 1));
            while(b > 0 && (t % (1LL << b)) != 0) {b--;} /* a step may only start where its bin is synchronised */
            return 1LL << b;};
        int bn, bo; long long cov = 0;
        run_pair(T, pick, bn, bo, cov);
        total_bad_new += bn; total_bad_old += bo;
    }
    CHECK(total_bad_new == 0);
    CHECK(total_bad_old > 0);
}

TEST_CASE("pair weights: opening plus closing of one interval add up to the interval") {
    for(long long d = 1; d < 64; d++) {CHECK(pairflux_pair_opening_ti(0, d, 1000) + pairflux_pair_closing_ti(d, 0, -5) == d);}
}

TEST_CASE("pair weights: a woken cell truncated without a drift (step [0,3], evaluated at t=8)") {
    long long S = 0, ti = 3, t = 8;
    long long applied = ti - ti/2;                             /* do_second_halfstep_kick: Ti_begstep + ti/2 .. Ti_begstep + ti */
    CHECK(pairflux_kick_closing_ti(ti) == applied);
    CHECK((t - S) - (t - S)/2 != applied);                     /* the step-start rule the corrections used before */
    CHECK(pairflux_next_sync_ti(t, S + ti, 4) == 12);          /* step end in the past: next sync of its bin */
    CHECK(pairflux_next_sync_ti(t, 16, 8) == 16);              /* ordinary inactive cell: its step end */
    CHECK(pairflux_pair_opening_ti(t, 12, pairflux_next_sync_ti(t, S + ti, 4)) >= 0);
}

TEST_MAIN()
