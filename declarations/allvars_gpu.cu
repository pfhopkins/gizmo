/*! \file allvars_gpu.cu
 *  \brief CUDA unified-memory definition of the global All struct.
 *
 *  This is the ONE translation unit (compiled by nvcc via nvcc_wrapper)
 *  that owns the __managed__ definition of All.  Every other CUDA TU
 *  (cooling.cc, future hydro/gravity GPU files, ...) includes allvars.h
 *  and gets an "extern __managed__" declaration — no special handling
 *  needed there.  Non-CUDA TUs get a plain "extern" and link against this
 *  managed symbol transparently via unified memory.
 *
 *  GIZMO_ALLVARS_GPU_OWNER must be defined before any include so that
 *  allvars.h (and global_data_all_struct.h via allvars.h) does NOT emit
 *  an extern declaration for All in this TU — having both a declaration
 *  and a __managed__ definition in the same TU causes nvcc to error.
 */

#define GIZMO_ALLVARS_GPU_OWNER
#include "global_data_all_struct.h"

__managed__ struct global_data_all_processes All;
