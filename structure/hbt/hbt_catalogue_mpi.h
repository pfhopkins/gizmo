#ifndef HBT_CATALOGUE_MPI_H
#define HBT_CATALOGUE_MPI_H
/* Host MPI helpers of the keyed catalogue; no GIZMO globals (reused on gku_stella_dev as is). */
#include <mpi.h>
#include "hbt_catalogue_core.h"

/* groups held in group order across the ranks of comm (rank r holds a contiguous block after rank r-1):
   offsettype[g*6+t] is the global offset of local group g in the type-t member list, tot_global[t] the list length */
void hbt_group_type_offsets_global(const int *lentype, long ngroups, long long *offsettype, long long *tot_global, MPI_Comm comm);

#endif
