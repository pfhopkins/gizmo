/*
Special routines for computing thermodynamic properties of the hydrogen
molecule, assuming a standard 3:1 ortho:para mixture and accounting for
rotational, vibration, and translational degrees of freedom.

Equations follow Boley 2007, ApJ, 656, L89
*/

#include "../declarations/allvars.h"
#include <math.h>

/* All function bodies are now in hydrogen_molecule_functions.h (single source of truth).
   Redefine KOKKOS_INLINE_FUNCTION to produce non-inline, externally-visible symbols.
   On GPU builds: __host__ __device__ so the device linker can resolve them.
   On CPU builds: plain (no annotation). */
#undef KOKKOS_INLINE_FUNCTION
#ifdef GIZMO_GPU_COMPILER
#define KOKKOS_INLINE_FUNCTION __host__ __device__
#else
#define KOKKOS_INLINE_FUNCTION
#endif
#include "hydrogen_molecule_functions.h"
