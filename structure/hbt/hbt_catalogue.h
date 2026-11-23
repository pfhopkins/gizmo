#ifndef HBT_CATALOGUE_H
#define HBT_CATALOGUE_H
/* Keyed GADGET-4-format group catalogue (FOF_WRITE_KEYED_CATALOGUE): groups_NNN/fof_subhalo_tab_NNN.hdf5 with
   GADGET-4 dataset names and per-type member key lists in place of a group-ordered snapshot. */
#if defined(FOF) && defined(FOF_WRITE_KEYED_CATALOGUE)
#include "../fof.h"

struct hbt_member_t {int GrNr; int Type; MyIDType ID; MyIDType Child;};

/* Grp: this task's groups, sorted by GrNr and contiguous in GrNr across tasks; mem: this task's group members */
void hbt_write_keyed_catalogue(int num, group_properties *Grp, int ngroups, hbt_member_t *mem, long nmem);
#endif

#endif
