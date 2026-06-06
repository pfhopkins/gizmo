#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
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

#include "../fof.h"
#include "subfind.h"


/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct densdata_in
{
  Vec3<MyDouble> Pos;
  MyFloat KernelRadius;
  int NodeList[NODELISTLENGTH];
}
 *DensDataIn, *DensDataGet;


static struct densdata_out
{
  MyFloat Rho;
  MyFloat VelDisp, Vx, Vy, Vz;
  int Ngb;
}
 *DensDataResult, *DensDataOut;


static MyFloat *DM_Vx, *DM_Vy, *DM_Vz;
static long long Ntotal;

void subfind_density(int j_in)
{
  subfind_density_modern(j_in);
  return;
}

void subfind_setup_smoothinglengths(int j)
{
#ifdef FOF_DENSITY_SPLIT_TYPES
  if(ThisTask == 0)
    printf("FOF_DENSITY_SPLIT_TYPES is not yet supported by the modern SUBFIND neighbor-list path.\n");
  endrun(990505);
#endif
  int i, no, p;

  for(i = 0; i < NumPart; i++)
    {
#ifdef FOF_DENSITY_SPLIT_TYPES
      if(P[i].Type == j)
#else
      if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)) || ((1 << P[i].Type) & (FOF_SECONDARY_LINK_TYPES)))
#endif
	{
	  no = Father[i];

	  /* Not a good guess for gas/stars component, need more thought ! */
	  while(10 * All.DesLinkNgb * P[i].Mass > Nodes[no].u.d.mass)
	    {
	      p = Nodes[no].u.d.father;

	      if(p < 0)
		break;

	      no = p;
	    }
#ifdef FOF_DENSITY_SPLIT_TYPES
	  if(P[i].Type == 0) {P[i].DM_KernelRadius = P[i].KernelRadius * pow( 1.*All.DesLinkNgb / All.DesNumNgb, 1./3.);}
	  else {P[i].DM_KernelRadius = pow(3.0 / (4 * M_PI) * All.DesLinkNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;}
#else
	  P[i].DM_KernelRadius = pow(3.0 / (4 * M_PI) * All.DesLinkNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
#endif

	}
    }
}


static int Nrkern;

static struct rkern_data
{
  float KernelRadius;
  float Density;
  float VelDisp;
  MyIDType ID;
}
 *KernelRadius_list;

int subfind_compare_rkern_data(const void *a, const void *b)
{
  if(((struct rkern_data *) a)->ID < ((struct rkern_data *) b)->ID) {return -1;}
  if(((struct rkern_data *) a)->ID > ((struct rkern_data *) b)->ID) {return +1;}
  return 0;
}


void subfind_save_densities(int num)
{
  int i, nprocgroup, primaryTask, groupTask;
  char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
  double t0, t1;

  if(ThisTask == 0)
    {
      printf("start saving smoothing lengths and densities\n");
      fflush(stdout);
    }

  for(i = 0, Nrkern = 0; i < NumPart; i++)
#ifdef FOF_DENSITY_SPLIT_TYPES
    if(((1 << P[i].Type) & (FOF_DENSITY_SPLIT_TYPES)))
#else
    if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif
      Nrkern++;

  MPI_Allgather(&Nrkern, 1, MPI_INT, Send_count, 1, MPI_INT, MPI_COMM_WORLD);
  for(i = 1, Send_offset[0] = 0; i < NTask; i++)
    Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];

  sumup_large_ints(1, &Nrkern, &Ntotal);

  KernelRadius_list = (struct rkern_data *)mymalloc("KernelRadius_list", Nrkern * sizeof(struct rkern_data));

  for(i = 0, Nrkern = 0; i < NumPart; i++)
#ifdef FOF_DENSITY_SPLIT_TYPES
    if(((1 << P[i].Type) & (FOF_DENSITY_SPLIT_TYPES)))
#else
    if(((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif
      {
	KernelRadius_list[Nrkern].KernelRadius = P[i].DM_KernelRadius;
	KernelRadius_list[Nrkern].Density = P[i].u.DM_Density;
	KernelRadius_list[Nrkern].VelDisp = P[i].v.DM_VelDisp;
	KernelRadius_list[Nrkern].ID = P[i].ID;
	Nrkern++;
      }

  t0 = my_second();
  parallel_sort(KernelRadius_list, Nrkern, sizeof(struct rkern_data), subfind_compare_rkern_data);
  t1 = my_second();

  if(ThisTask == 0)
    {
      printf("Sorting of densities in ID sequence took = %g sec\n", timediff(t0, t1));
      fflush(stdout);
    }

  if(ThisTask == 0)
    {
      snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/rkerndir_%03d", All.OutputDir, num);
      mkdir(buf, 02755);
    }
  MPI_Barrier(MPI_COMM_WORLD);

  if(NTask < All.NumFilesWrittenInParallel)
    {
      printf
	("Fatal error.\nNumber of processors must be a smaller or equal than `NumFilesWrittenInParallel'.\n");
      endrun(241931);
    }

  nprocgroup = NTask / All.NumFilesWrittenInParallel;
  if((NTask % All.NumFilesWrittenInParallel))
    nprocgroup++;
  primaryTask = (ThisTask / nprocgroup) * nprocgroup;
  for(groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
      if(ThisTask == (primaryTask + groupTask))	/* ok, it's this processor's turn */
	subfind_save_local_densities(num);
      MPI_Barrier(MPI_COMM_WORLD);	/* wait inside the group */
    }

  myfree(KernelRadius_list);

}

void subfind_save_local_densities(int num)
{
  char fname[DEFAULT_PATH_BUFFERSIZE_TOUSE];
  int i;
  float *tmp;
  FILE *fd;


  snprintf(fname, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/rkerndir_%03d/%s_%03d.%d", All.OutputDir, num, "rkern", num, ThisTask);
  if(!(fd = fopen(fname, "w")))
    {
      printf("can't open file `%s`\n", fname);
      endrun(1183);
    }

  my_fwrite(&Nrkern, sizeof(int), 1, fd);
  my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd);	/* this is the number of IDs in previous files */
  my_fwrite(&Ntotal, sizeof(long long), 1, fd);
  my_fwrite(&NTask, sizeof(int), 1, fd);

  tmp = (float *)mymalloc("tmp", Nrkern * sizeof(float));

  for(i = 0; i < Nrkern; i++)
    tmp[i] = KernelRadius_list[i].KernelRadius;
  my_fwrite(tmp, sizeof(float), Nrkern, fd);

  for(i = 0; i < Nrkern; i++)
    tmp[i] = KernelRadius_list[i].Density;
  my_fwrite(tmp, sizeof(float), Nrkern, fd);

  for(i = 0; i < Nrkern; i++)
    tmp[i] = KernelRadius_list[i].VelDisp;
  my_fwrite(tmp, sizeof(float), Nrkern, fd);

  myfree(tmp);

  fclose(fd);
}


#endif
