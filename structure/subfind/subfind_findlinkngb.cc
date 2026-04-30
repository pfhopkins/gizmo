
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../mesh/kernel.h"
/*!
* This file was originally part of the GADGET3 code developed by Volker Springel.
* It has been updated significantly by PFH for basic compatibility with GIZMO,
* as well as code cleanups, and accommodating new GIZMO functionality for various
* other operations. See notes in subfind.c and GIZMO User Guide for details.
*/


#ifdef SUBFIND 

#include "subfind.h"


void subfind_find_linkngb(void)
{
#ifdef OPENMP_GPU_OFFLOAD
  subfind_find_linkngb_modern();
  return;
#endif
}

int subfind_ngb_compare_dist(const void *a, const void *b)
{
  if(((struct r2data *) a)->r2 < (((struct r2data *) b)->r2)) {return -1;}
  if(((struct r2data *) a)->r2 > (((struct r2data *) b)->r2)) {return +1;}
  return 0;
}




#endif
