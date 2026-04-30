#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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


void subfind_find_nearesttwo(void)
{
  subfind_find_nearesttwo_modern();
  return;
}

#endif
