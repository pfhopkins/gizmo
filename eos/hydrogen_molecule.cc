/*
Special routines for computing thermodynamic properties of the hydrogen
molecule, assuming a standard 3:1 ortho:para mixture and accounting for
rotational, vibration, and translational degrees of freedom.

Equations follow Boley 2007, ApJ, 656, L89
*/

#include "../declarations/allvars.h"
#include <math.h>

/* All function bodies are now in hydrogen_molecule_functions.h (single source of truth).
   Define KOKKOS_INLINE_FUNCTION as empty here so the functions are non-inline,
   providing externally-visible symbols for non-GPU TUs that link via proto.h.
   GPU TUs get the inline versions from cell_data.h -> hydrogen_molecule_functions.h. */
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "hydrogen_molecule_functions.h"
