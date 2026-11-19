#pragma once
/* Integer-time weights of one pair's flux under HYDRO_EXACT_PAIR_FLUX_EXCHANGE. A pair is evaluated at every sync where
   either cell is active; the interval between two consecutive evaluations is split as the kicks split a step (opening
   d/2, closing d - d/2). S = start and N = end of a cell's current step on the integer timeline. */
static inline long long pairflux_pair_closing_ti(long long t, long long S_e, long long S_r)
{
    long long d = t - (S_e > S_r ? S_e : S_r); /* since the pair's previous evaluation */
    return d - d/2;
}
static inline long long pairflux_pair_opening_ti(long long t, long long N_e, long long N_r)
{
    long long d = (N_e < N_r ? N_e : N_r) - t; /* to the pair's next evaluation */
    return d/2;
}
