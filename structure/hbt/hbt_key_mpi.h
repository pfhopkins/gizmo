#ifndef HBT_KEY_MPI_H
#define HBT_KEY_MPI_H
/* Host MPI helpers for the SUBFIND-HBT particle key; no GIZMO globals (reused on gku_stella_dev as is). */
#include <mpi.h>
#include "hbt_key.h"

/* number of surplus copies (multiplicity - 1, summed) of keys over all tasks of comm; -1 on a count overflow */
long long hbt_count_duplicate_keys(const hbt_key_t *keys, long n, MPI_Comm comm);

#endif
