#ifndef SINGLE_STAR_MERGERS_H
#define SINGLE_STAR_MERGERS_H
/* Stellar mergers handed back from the few-body integrator (KETJU or RUMI++): the victims' metals and drawn masses go to the survivor. */
#include <vector>
#if (defined(KETJU_REGULARIZATION) && (defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH))) || defined(STARS_INTEGRATED_BY_RUMI)
#define STAR_MERGERS_SHARED
#endif
#if ((defined(KETJU_REGULARIZATION) && defined(KETJU_MERGE_STARS)) || defined(STARS_INTEGRATED_BY_RUMI)) && defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_FB_MECHANICAL)
#define SINGLE_STAR_MERGER_PRODUCT /* collision products get table state (Hurley+2002 rejuvenation) */
#endif
#ifdef STAR_MERGERS_SHARED
/* one merger: the victim (merged away) and the survivor that takes its mass and metals */
struct star_merger_action {
    MyIDType id_merged, id_survivor;
    int type_merged, type_survivor;
    int task_merged, task_survivor;
    int idx_merged, idx_survivor;
    double M_merged, M_survivor_old, M_survivor_new;
};
/* Every task that owns a victim or a survivor calls it with the same list (MPI point-to-point between them);
   victims get mass 0 (removed at the next rearrangement); tag prefixes the product print */
void star_mergers_apply(std::vector<star_merger_action> &actions, const char *tag);
#endif
#endif
