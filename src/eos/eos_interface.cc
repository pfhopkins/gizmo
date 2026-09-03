#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eos.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"                  /* gizmo_gpu_alloc_shared */
#include "../system/gpu_particles_arena.h"  /* gpu_particles_uvm_free */

/* The tabulated-EOS table's lifetime, and nothing else.
   The routines that evaluate the equation of state -- the unit conversions,
   the range validation, the Newton inversion and the ideal-gas fallback --
   moved to eos/eos_functions.h so that a device kernel can call them: without
   relocatable device code a call cannot cross a translation unit, and the
   evaluation is a per-particle root-find, i.e. exactly the work that repays
   staging a batch to the device. Everything they need is passed to them, so
   the table below is named in this file alone. */

#ifdef EOS_HELMHOLTZ
#include "helmholtz/helmholtz.h"

/* Read-only after eos_init, so concurrent readers are safe. It lives in memory
   the host and the device can both read, and it is never copied afterwards:
   the interpolation below is re-entered on every Newton iteration, tens of
   times per particle, so a table that had to be staged per call would cost far
   more to move than the solve it serves. About 4.6 MB, once per rank. */
static HelmTable *helm_table = NULL;

/* The table, for whoever is about to evaluate the EOS. Host-only: call it
   before a dispatch and let the kernel capture the result. Null until
   eos_init has run, which begrun does before any drift or cooling. */
const HelmTable *helm_table_view(void) {return helm_table;}
#endif

#ifdef EOS_TABULATED
int eos_init(char const * eos_table_fname)
{
#ifdef EOS_HELMHOLTZ
  if(!helm_table) {
    helm_table = (HelmTable *) gizmo_gpu_alloc_shared(sizeof(HelmTable), "helm_table");
    if(!helm_table) {
      fprintf(stderr, "eos_init: could not allocate the Helmholtz table (%zu bytes)\n", sizeof(HelmTable));
      return 1;
    }
    memset(helm_table, 0, sizeof(HelmTable));
  }
  int ierr = helm_read_table(eos_table_fname, helm_table);
  if (ierr) {
    fprintf(stderr, "eos_init: failed to read Helmholtz table '%s'\n", eos_table_fname);
    return 1;
  }
#endif
  return 0;
}

int eos_cleanup()
{
#ifdef EOS_HELMHOLTZ
  /* Freed through the allocator that served it; free() on shared memory is
     undefined. */
  gpu_particles_uvm_free(helm_table);
  helm_table = NULL;
#endif
  return 0;
}
#endif
