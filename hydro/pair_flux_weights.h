#pragma once
/* Integer-time weights of one pair's flux under HYDRO_EXACT_PAIR_FLUX_EXCHANGE. A pair is evaluated at every sync where
   either cell is active; the interval between two consecutive evaluations is split as the kicks split a step (opening
   d/2, closing d - d/2). S = start and N = end of a cell's current step on the integer timeline. */
static inline long long pairflux_pair_closing_ti(long long t, long long S_e, long long S_r)
{
    long long d = t - (S_e > S_r ? S_e : S_r); /* since the pair's previous evaluation */
    return d - d/2;
}
/* what a cell's own closing kick applies: the second half of its step, whatever the step's start (a woken cell's step can be
   truncated without being drifted, so its step need not end at the current sync) */
static inline long long pairflux_kick_closing_ti(long long ti_step) {return ti_step - ti_step/2;}
/* next evaluation of an inactive cell: its step end, or the next sync of its bin if that end has passed (truncated step) */
static inline long long pairflux_next_sync_ti(long long t, long long step_end, long long ti_bin)
{
    if(step_end > t || ti_bin <= 0) {return step_end > t ? step_end : t;}
    return (t / ti_bin + 1) * ti_bin;
}
static inline long long pairflux_pair_opening_ti(long long t, long long N_e, long long N_r)
{
    long long d = (N_e < N_r ? N_e : N_r) - t; /* to the pair's next evaluation */
    return d/2;
}
