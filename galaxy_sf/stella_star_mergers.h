/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef STELLA_STAR_MERGERS_H
#define STELLA_STAR_MERGERS_H
/* Stellar mergers handed back from an N-body integrator (KETJU, RUMI++): victims' metals, IMF masses and (Stella)
   the merged product's state go to the survivor. Shared by ketju_coupling.cc and rumi_coupling.cc. */
#include <vector>
#if (defined(KETJU_REGULARIZATION) && (defined(KETJU_MERGE_STARS) || defined(KETJU_MERGE_BH))) || defined(STARS_INTEGRATED_BY_RUMI)
#define STAR_MERGERS_SHARED
#endif
#if ((defined(KETJU_REGULARIZATION) && defined(KETJU_MERGE_STARS) && defined(KETJU_TYPE4_STARS)) || defined(STARS_INTEGRATED_BY_RUMI)) && defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES)
#define STELLA_MERGER_PRODUCT /* collision products get Stella table state (Hurley+2002 rejuvenation) */
#endif
#ifdef STAR_MERGERS_SHARED
/* one merger: the victim (merged away) and the survivor that takes its mass, metals and (Stella) the product state */
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
