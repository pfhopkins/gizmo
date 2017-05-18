#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../allvars.h"
#include "../proto.h"

/*
 *  This code was originally written for GADGET3 by Andreas Bauer; it has been
 *   modified slightly by Phil Hopkins for GIZMO, but is largely intact.
 */

#if defined(POWERSPEC_GRID) && defined(PERIODIC) && (defined(TURB_DRIVING))


#ifdef NOTYPEPREFIX_FFTW
#include        <rfftw_mpi.h>
#else
#ifdef DOUBLEPRECISION_FFTW
#include     <drfftw_mpi.h>	/* double precision FFTW */
#else
#include     <srfftw_mpi.h>
#endif
#endif

#define  POWERSPEC_GRID2 (2*(POWERSPEC_GRID/2 + 1))

#if (POWERSPEC_GRID > 1024)
typedef long long large_array_offset;
#else
typedef unsigned int large_array_offset;
#endif

static rfftwnd_mpi_plan fft_forward_plan;
static int slabstart_x, nslab_x, slabstart_y, nslab_y;

static int fftsize, maxfftsize;
static fftw_real *velfield[3];
static fftw_real *smoothedvelfield[3];
static fftw_real *vorticityfield[3];
static fftw_real *velrhofield[3];
static fftw_real *dis1field;
static fftw_real *dis2field;
static fftw_real *densityfield;

static fftw_real *randomfield;
static fftw_real *workspace;

static float    *RandomValue;

static fftw_complex *fft_of_field;

static float *powerspec_turb_nearest_distance, *powerspec_turb_nearest_hsml;

void powerspec_turb_calc_and_bin_spectrum(fftw_real *field, int flag);

static struct data_in
{
  MyDouble Pos[3];
  MyFloat Hsml;
  int NodeList[NODELISTLENGTH];
}
 *DataIn, *DataGet;

static struct data_out
{
  MyFloat Distance;
  MyDouble Vel[3];
  MyDouble SmoothedVel[3];
  MyDouble Vorticity[3];
  MyDouble Density;
  MyDouble DuDt_diss;
  MyDouble DuDt_drive;
  MyDouble RandomValue;
}
 *DataResult, *DataOut;



#define BINS_PS  2000	                 	/* number of bins for power spectrum computation */

static long long CountModes[BINS_PS];
static double    SumPower[BINS_PS];
static double    Power[BINS_PS];
static double    Kbin[BINS_PS];
static double    K0, K1;
static double    binfac;
static double    vel_disp[3];
static double    velrho_disp[3];
static double    empty_disp[3] = {0, 0, 0};




void powerspec_turb(int filenr)
{
  int i;
  char fname[1000];

  if(ThisTask == 0)
    printf("Start turbulent powerspec computation\n");

  double tstart = my_second();

  /* Set up the FFTW plan  */
  fft_forward_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD, POWERSPEC_GRID, POWERSPEC_GRID, POWERSPEC_GRID,
					     FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE);

  /* Workspace out the ranges on each processor. */
  rfftwnd_mpi_local_sizes(fft_forward_plan, &nslab_x, &slabstart_x, &nslab_y, &slabstart_y, &fftsize);
  MPI_Allreduce(&fftsize, &maxfftsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  /* allocate the memory to hold the FFT fields */



  velfield[0] = (fftw_real *) mymalloc("velfield[0]", maxfftsize * sizeof(fftw_real));
  velfield[1] = (fftw_real *) mymalloc("velfield[1]", maxfftsize * sizeof(fftw_real));
  velfield[2] = (fftw_real *) mymalloc("velfield[2]", maxfftsize * sizeof(fftw_real));

  smoothedvelfield[0] = (fftw_real *) mymalloc("smoothedvelfield[0]", maxfftsize * sizeof(fftw_real));
  smoothedvelfield[1] = (fftw_real *) mymalloc("smoothedvelfield[1]", maxfftsize * sizeof(fftw_real));
  smoothedvelfield[2] = (fftw_real *) mymalloc("smoothedvelfield[2]", maxfftsize * sizeof(fftw_real));

  velrhofield[0] = (fftw_real *) mymalloc("velrhofield[0]", maxfftsize * sizeof(fftw_real));
  velrhofield[1] = (fftw_real *) mymalloc("velrhofield[1]", maxfftsize * sizeof(fftw_real));
  velrhofield[2] = (fftw_real *) mymalloc("velrhofield[2]", maxfftsize * sizeof(fftw_real));

  vorticityfield[0] = (fftw_real *) mymalloc("vorticityfield[0]", maxfftsize * sizeof(fftw_real));
  vorticityfield[1] = (fftw_real *) mymalloc("vorticityfield[1]", maxfftsize * sizeof(fftw_real));
  vorticityfield[2] = (fftw_real *) mymalloc("vorticityfield[2]", maxfftsize * sizeof(fftw_real));

  dis1field = (fftw_real *) mymalloc("dis1field", maxfftsize * sizeof(fftw_real));
  dis2field = (fftw_real *) mymalloc("dis2field", maxfftsize * sizeof(fftw_real));
  randomfield = (fftw_real *) mymalloc("randomfield", maxfftsize * sizeof(fftw_real));

  densityfield = (fftw_real *) mymalloc("densityfield", maxfftsize * sizeof(fftw_real));

  workspace = (fftw_real *) mymalloc("workspace", maxfftsize * sizeof(fftw_real));

  memset(velfield[0], 0, maxfftsize * sizeof(fftw_real));
  memset(velfield[1], 0, maxfftsize * sizeof(fftw_real));
  memset(velfield[2], 0, maxfftsize * sizeof(fftw_real));

  memset(smoothedvelfield[0], 0, maxfftsize * sizeof(fftw_real));
  memset(smoothedvelfield[1], 0, maxfftsize * sizeof(fftw_real));
  memset(smoothedvelfield[2], 0, maxfftsize * sizeof(fftw_real));

  memset(velrhofield[0], 0, maxfftsize * sizeof(fftw_real));
  memset(velrhofield[1], 0, maxfftsize * sizeof(fftw_real));
  memset(velrhofield[2], 0, maxfftsize * sizeof(fftw_real));

  memset(vorticityfield[0], 0, maxfftsize * sizeof(fftw_real));
  memset(vorticityfield[1], 0, maxfftsize * sizeof(fftw_real));
  memset(vorticityfield[2], 0, maxfftsize * sizeof(fftw_real));

  memset(dis1field, 0, maxfftsize * sizeof(fftw_real));
  memset(dis2field, 0, maxfftsize * sizeof(fftw_real));
  memset(randomfield, 0, maxfftsize * sizeof(fftw_real));

  memset(densityfield, 0, maxfftsize * sizeof(fftw_real));

  RandomValue = (float *) mymalloc("RndField", N_gas * sizeof(float));

  gsl_rng *random_gen = gsl_rng_alloc(gsl_rng_ranlxd1);
  gsl_rng_set(random_gen, 42 + ThisTask);	/* start-up seed */

  for(i=0; i < N_gas; i++)
    RandomValue[i] = gsl_ran_gaussian (random_gen, 1.0);

  powerspec_turb_obtain_fields();
 
  powerspec_turb_calc_dispersion();



  /* Now compute the power spectrum of the velocities */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(velfield[0], 1);   /* only here the modes are counted */
  powerspec_turb_calc_and_bin_spectrum(velfield[1], 0);
  powerspec_turb_calc_and_bin_spectrum(velfield[2], 0);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_vel_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, vel_disp);


  /* Now compute the power spectrum of the smoothed velocities */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(smoothedvelfield[0], 1);   /* only here the modes are counted */
  powerspec_turb_calc_and_bin_spectrum(smoothedvelfield[1], 0);
  powerspec_turb_calc_and_bin_spectrum(smoothedvelfield[2], 0);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_smoothedvel_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, vel_disp);





  /* now compute the power spectrum of the sqrt(rho)-weighted veloicty */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(velrhofield[0], 1);
  powerspec_turb_calc_and_bin_spectrum(velrhofield[1], 0);
  powerspec_turb_calc_and_bin_spectrum(velrhofield[2], 0);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_velrho_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, velrho_disp);



  /* now compute the power spectrum of the vorticity */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(vorticityfield[0], 1);
  powerspec_turb_calc_and_bin_spectrum(vorticityfield[1], 0);
  powerspec_turb_calc_and_bin_spectrum(vorticityfield[2], 0);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_vorticity_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, velrho_disp);




  /* Now compute the power spectrum of the dissipation1 */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(dis1field, 1);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_dis1_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, empty_disp);




  /* Now compute the power spectrum of the dissipation2 */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(dis2field, 1);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_dis2_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, empty_disp);


  /* Now compute the power spectrum of the random field */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(randomfield, 1);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_random_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, empty_disp);

  /* Now compute the power spectrum of the density field */

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      CountModes[i] = 0;
    }

  powerspec_turb_calc_and_bin_spectrum(densityfield, 1);

  powerspec_turb_collect();

  sprintf(fname, "%s/powerspec_density_%03d.txt", All.OutputDir, filenr);
  powerspec_turb_save(fname, empty_disp);

  myfree(RandomValue);

  myfree(workspace);
  myfree(densityfield);
  myfree(randomfield);
  myfree(dis2field);
  myfree(dis1field);
  myfree(vorticityfield[2]);
  myfree(vorticityfield[1]);
  myfree(vorticityfield[0]);
  myfree(velrhofield[2]);
  myfree(velrhofield[1]);
  myfree(velrhofield[0]);
  myfree(smoothedvelfield[2]);
  myfree(smoothedvelfield[1]);
  myfree(smoothedvelfield[0]);
  myfree(velfield[2]);
  myfree(velfield[1]);
  myfree(velfield[0]);

  rfftwnd_mpi_destroy_plan(fft_forward_plan);

  double tend = my_second();
  
#ifndef IO_REDUCED_MODE
  if(ThisTask == 0)
    {
      printf("end turbulent power spectra  took %g seconds\n", timediff(tstart, tend));
    }
#endif
}



void powerspec_turb_calc_and_bin_spectrum(fftw_real *field, int flag)
{
  double k2, kx, ky, kz;
  int x, y, z, zz, ip;
  
  K0 = 2 * M_PI / All.BoxSize;	                        /* minimum k */
  K1 = K0 * POWERSPEC_GRID / 2;	                                /* maximum k */
  binfac = BINS_PS / (log(K1) - log(K0));

  /* Do the FFT of the velocity_field */  /* rhogrid -> velfield */
  
  rfftwnd_mpi(fft_forward_plan, 1, field, workspace, FFTW_TRANSPOSED_ORDER);
  
  fft_of_field = (fftw_complex *) field;

  for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
    for(x = 0; x < POWERSPEC_GRID; x++)
      for(z = 0; z < POWERSPEC_GRID; z++)
	{
	  zz = z;
	  if(z >= POWERSPEC_GRID / 2 + 1)
	    zz = POWERSPEC_GRID - z;
	  
	  if(x > POWERSPEC_GRID / 2)
	    kx = x - POWERSPEC_GRID;
	  else
	    kx = x;
	  if(y > POWERSPEC_GRID / 2)
	    ky = y - POWERSPEC_GRID;
	  else
	    ky = y;
	  if(z > POWERSPEC_GRID / 2)
	    kz = z - POWERSPEC_GRID;
	  else
	    kz = z;

	  k2 = kx * kx + ky * ky + kz * kz;
	  
	  ip = POWERSPEC_GRID * (POWERSPEC_GRID / 2 + 1) * (y - slabstart_y) + (POWERSPEC_GRID / 2 + 1) * x + zz;
	  
	  double po = (fft_of_field[ip].re * fft_of_field[ip].re
		       + fft_of_field[ip].im * fft_of_field[ip].im) / pow(POWERSPEC_GRID, 6);
	  	  
	  if(k2 > 0)
	    {
	      if(k2 < (POWERSPEC_GRID / 2.0) * (POWERSPEC_GRID / 2.0))
		{
		  double k = sqrt(k2) * 2 * M_PI / All.BoxSize;
		  
		  if(k >= K0 && k < K1)
		    {
		      int bin = log(k / K0) * binfac;
		      
		      SumPower[bin] += po;
		      
		      if(flag)
			CountModes[bin] += 1;
		    }
		}
	    }
	}
}



void powerspec_turb_collect(void)
{
  int i, n;
  long long int *countbuf = (long long int *) mymalloc("countbuf", NTask * BINS_PS * sizeof(long long));
  double *powerbuf = (double *) mymalloc("powerbuf", NTask * BINS_PS * sizeof(double));

  MPI_Allgather(CountModes, BINS_PS * sizeof(long long), MPI_BYTE,
		countbuf, BINS_PS * sizeof(long long), MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0; i < BINS_PS; i++)
    {
      CountModes[i] = 0;
      for(n = 0; n < NTask; n++)
	CountModes[i] += countbuf[n * BINS_PS + i];
    }

  MPI_Allgather(SumPower, BINS_PS * sizeof(double), MPI_BYTE,
		powerbuf, BINS_PS * sizeof(double), MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0; i < BINS_PS; i++)
    {
      SumPower[i] = 0;
      for(n = 0; n < NTask; n++)
	SumPower[i] += powerbuf[n * BINS_PS + i];
    }

  myfree(powerbuf);
  myfree(countbuf);

  for(i = 0; i < BINS_PS; i++)
    {
      Kbin[i] = exp((i + 0.5) / binfac + log(K0));

      if(CountModes[i] > 0)
	Power[i] = SumPower[i] / CountModes[i];
      else
	Power[i] = 0;
    }
}



void powerspec_turb_save(char *fname, double *disp)
{
  FILE *fd;
  char buf[500];
  int i;

  if(ThisTask == 0)
    {
      if(!(fd = fopen(fname, "w")))
	{
	  sprintf(buf, "can't open file `%s`\n", fname);
	  terminate(buf);
	}

      fprintf(fd, "%g\n", All.Time);
      i = POWERSPEC_GRID;
      fprintf(fd, "%d\n", i);
      i = BINS_PS;
      fprintf(fd, "%d\n", i);

      fprintf(fd, "%g\n", disp[0]);
      fprintf(fd, "%g\n", disp[1]);
      fprintf(fd, "%g\n", disp[2]);

      for(i = 0; i < BINS_PS; i++)
	{
	  fprintf(fd, "%g %g %g %g\n", 
		  Kbin[i], Power[i], (double) CountModes[i], SumPower[i]);
	}

      fclose(fd);
    }
}



/* this function determines the velocity fields by using the nearest cell's values 
 */ 
double powerspec_turb_obtain_fields(void)
{
  int j, dummy;
  long long ntot, npleft;
  int ndone, ndone_flag, ngrp, sendTask, recvTask, place, nexport, nimport, iter;

  double tstart = my_second();

#ifndef IO_REDUCED_MODE
  if(ThisTask == 0)
    {
      printf("Start finding nearest gas-particle for mesh-cell centers (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
    }
#endif
    
  large_array_offset i, n, Ncount = ((large_array_offset)nslab_x) * (POWERSPEC_GRID * POWERSPEC_GRID);  /* number of grid points on the local slab */

  powerspec_turb_nearest_distance = (float *) mymalloc("powerspec_turb_nearest_distance", sizeof(float) * Ncount);
  powerspec_turb_nearest_hsml = (float *) mymalloc("powerspec_turb_nearest_hsml", sizeof(float) * Ncount);

  for(n = 0; n < Ncount; n++)
    {
      powerspec_turb_nearest_distance[n] = 1.0e30;
      powerspec_turb_nearest_hsml[n] = All.BoxSize / pow(All.TotN_gas, 1.0/3);
    }

  /* allocate buffers to arrange communication */

  Ngblist = (int *) mymalloc("Ngblist", Ncount * sizeof(int));

    size_t MyBufferSize = All.BufferSize;
    All.BunchSize = (int) ((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
					     sizeof(struct data_in) + sizeof(struct data_out) +
					     sizemax(sizeof(struct data_in), sizeof(struct data_out))));
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

  report_memory_usage(&HighMark_turbpower, "TURBPOWER");

  iter = 0;
  /* we will repeat the whole thing for those points where we didn't find enough neighbours */
  do
    {
      i = 0;			/* beginn with this index */

      do
	{
	  for(j = 0; j < NTask; j++)
	    {
	      Send_count[j] = 0;
	      Exportflag[j] = -1;
	    }

	  /* do local particles and prepare export list */
	  for(nexport = 0; i < Ncount; i++)
	      {
		if(powerspec_turb_nearest_distance[i] > 1.0e29)
		  {
		    if(powerspec_turb_find_nearest_evaluate(i, 0, &nexport, Send_count) < 0)
		      break;
		  }
	      }

	  mysort_dataindex(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);

	  MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);

	  for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
	    {
	      nimport += Recv_count[j];

	      if(j > 0)
		{
		  Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
		  Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
		}
	    }

	  DataGet = (struct data_in *) mymalloc("DataGet", nimport * sizeof(struct data_in));
	  DataIn = (struct data_in *) mymalloc("DataIn", nexport * sizeof(struct data_in));

#ifndef IO_REDUCED_MODE
	  if(ThisTask == 0)
	    {
	      printf("still finding nearest... (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));
	    }
#endif
	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      int xx = place / (POWERSPEC_GRID * POWERSPEC_GRID);
	      int yy = (place - xx * POWERSPEC_GRID * POWERSPEC_GRID) / POWERSPEC_GRID;
	      int zz = (place - xx * POWERSPEC_GRID * POWERSPEC_GRID - yy * POWERSPEC_GRID); 
	      xx += slabstart_x;
	      
	      double x = (xx + 0.5) / POWERSPEC_GRID * boxSize_X;
	      double y = (yy + 0.5) / POWERSPEC_GRID * boxSize_Y;
	      double z = (zz + 0.5) / POWERSPEC_GRID * boxSize_Z;
	      
	      DataIn[j].Pos[0] = x;
	      DataIn[j].Pos[1] = y;
	      DataIn[j].Pos[2] = z;
	      DataIn[j].Hsml = powerspec_turb_nearest_hsml[place];

	      memcpy(DataIn[j].NodeList,
		     DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
	    }

	  /* exchange particle data */
	  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	    {
	      sendTask = ThisTask;
	      recvTask = ThisTask ^ ngrp;

	      if(recvTask < NTask)
		{
		  if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		    {
		      /* get the particles */
		      MPI_Sendrecv(&DataIn[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct data_in), MPI_BYTE,
				   recvTask, TAG_DENS_A,
				   &DataGet[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct data_in), MPI_BYTE,
				   recvTask, TAG_DENS_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}
	    }

	  myfree(DataIn);
	  DataResult =
	    (struct data_out *) mymalloc("DataResult", nimport * sizeof(struct data_out));
	  DataOut = (struct data_out *) mymalloc("DataOut", nexport * sizeof(struct data_out));

	  for(j = 0; j < nimport; j++)
	    powerspec_turb_find_nearest_evaluate(j, 1, &dummy, &dummy);
     
	  for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
	    {
	      sendTask = ThisTask;
	      recvTask = ThisTask ^ ngrp;
	      if(recvTask < NTask)
		{
		  if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
		    {
		      /* send the results */
		      MPI_Sendrecv(&DataResult[Recv_offset[recvTask]],
				   Recv_count[recvTask] * sizeof(struct data_out),
				   MPI_BYTE, recvTask, TAG_DENS_B,
				   &DataOut[Send_offset[recvTask]],
				   Send_count[recvTask] * sizeof(struct data_out),
				   MPI_BYTE, recvTask, TAG_DENS_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    }
		}

	    }

	  for(j = 0; j < nexport; j++)
	    {
	      place = DataIndexTable[j].Index;

	      if(DataOut[j].Distance < powerspec_turb_nearest_distance[place])
		{
		  powerspec_turb_nearest_distance[place] = DataOut[j].Distance;

		  int ii = place / (POWERSPEC_GRID * POWERSPEC_GRID);
		  int jj = (place - ii * POWERSPEC_GRID * POWERSPEC_GRID) / POWERSPEC_GRID;
		  int kk = (place - ii * POWERSPEC_GRID * POWERSPEC_GRID - jj * POWERSPEC_GRID); 
		  int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * ii + jj) + kk;

		  velfield[0][ip] = DataOut[j].Vel[0];
		  velfield[1][ip] = DataOut[j].Vel[1];
		  velfield[2][ip] = DataOut[j].Vel[2];

		  smoothedvelfield[0][ip] = DataOut[j].SmoothedVel[0];
		  smoothedvelfield[1][ip] = DataOut[j].SmoothedVel[1];
		  smoothedvelfield[2][ip] = DataOut[j].SmoothedVel[2];

		  velrhofield[0][ip] = sqrt(DataOut[j].Density) * DataOut[j].Vel[0];
		  velrhofield[1][ip] = sqrt(DataOut[j].Density) * DataOut[j].Vel[1];
		  velrhofield[2][ip] = sqrt(DataOut[j].Density) * DataOut[j].Vel[2];

		  vorticityfield[0][ip] = DataOut[j].Vorticity[0];
		  vorticityfield[1][ip] = DataOut[j].Vorticity[1];
		  vorticityfield[2][ip] = DataOut[j].Vorticity[2];


		  if(DataOut[j].DuDt_diss >= 0)
		    {
		      dis1field[ip] = sqrt(DataOut[j].DuDt_diss);
		      dis2field[ip] = 0;
		    }
		  else
		    {
		      dis1field[ip] = 0;
		      dis2field[ip] = sqrt(-DataOut[j].DuDt_diss);
		    }

		  randomfield[ip] = DataOut[j].RandomValue;

		  densityfield[ip] = DataOut[j].Density;
		}
	    }

	  if(i >= Ncount)
	    ndone_flag = 1;
	  else
	    ndone_flag = 0;

	  MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	  myfree(DataOut);
	  myfree(DataResult);
	  myfree(DataGet);
	}
      while(ndone < NTask);

      /* do final operations on results */
      for(i = 0, npleft = 0; i < Ncount; i++)
	{
	  if(powerspec_turb_nearest_distance[i] > 1.0e29)
	    {
	      /* need to redo this particle */
	      npleft++;
	      powerspec_turb_nearest_hsml[i] *= 2.0;
	      if(iter >= MAXITER - 10)
		{
		  int xx = i / (POWERSPEC_GRID * POWERSPEC_GRID);
		  int yy = (i - xx * POWERSPEC_GRID * POWERSPEC_GRID) / POWERSPEC_GRID;
		  int zz = (i - xx * POWERSPEC_GRID * POWERSPEC_GRID - yy * POWERSPEC_GRID); 
		  xx += slabstart_x;
		  
            double x = (xx + 0.5) / POWERSPEC_GRID * boxSize_X;
            double y = (yy + 0.5) / POWERSPEC_GRID * boxSize_Y;
            double z = (zz + 0.5) / POWERSPEC_GRID * boxSize_Z;
		}
	    }
	  else
	    {
	      powerspec_turb_nearest_distance[i] = 0;	/* we not continue to search for this particle */
	    }
	}

      sumup_longs(1, &npleft, &ntot);
      if(ntot > 0)
	{
	  iter++;
#ifndef IO_REDUCED_MODE
	  if(iter > 0 && ThisTask == 0)
	    {
	      printf("powespec_vel nearest iteration %d: need to repeat for %lld particles.\n", iter, ntot);
	    }
#endif
	  if(iter > MAXITER)
	    terminate("failed to converge");
	}
    }
  while(ntot > 0);

  myfree(DataNodeList);
  myfree(DataIndexTable);
  myfree(Ngblist);

  myfree(powerspec_turb_nearest_hsml);
  myfree(powerspec_turb_nearest_distance);

    if(ThisTask == 0) {printf("done finding velocity field\n");}

  double tend = my_second();
  return timediff(tstart, tend);
}


void powerspec_turb_calc_dispersion(void)
{
  int dim, i, j, k;

  for(dim = 0; dim < 3; dim++)
    {
      double vsum = 0, vsum_all, vmean, vdisp = 0, vdisp_all;

      for(i=0; i < nslab_x;i++)
	for(j=0; j< POWERSPEC_GRID; j++)
	  for(k=0; k< POWERSPEC_GRID; k++)
	    {
	      int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * i + j) + k;

	      vsum += velfield[dim][ip];
	    }

      MPI_Allreduce(&vsum, &vsum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      vmean = vsum_all / pow(POWERSPEC_GRID, 3);
      
      for(i=0; i < nslab_x;i++)
	for(j=0; j< POWERSPEC_GRID; j++)
	  for(k=0; k< POWERSPEC_GRID; k++)
	    {
	      int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * i + j) + k;
	      
	      velfield[dim][ip] -= vmean;
	    }

      for(i=0; i < nslab_x;i++)
	for(j=0; j< POWERSPEC_GRID; j++)
	  for(k=0; k< POWERSPEC_GRID; k++)
	    {
	      int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * i + j) + k;
	      
	      vdisp += velfield[dim][ip] * velfield[dim][ip];
	    }

      MPI_Allreduce(&vdisp, &vdisp_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      vel_disp[dim] = vdisp_all / pow(POWERSPEC_GRID, 3);      
    }


  for(dim = 0; dim < 3; dim++)
    {
      double vsum = 0, vsum_all, vmean, vdisp = 0, vdisp_all;

      for(i=0; i < nslab_x;i++)
	for(j=0; j< POWERSPEC_GRID; j++)
	  for(k=0; k< POWERSPEC_GRID; k++)
	    {
	      int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * i + j) + k;

	      vsum += velrhofield[dim][ip];
	    }

      MPI_Allreduce(&vsum, &vsum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      vmean = vsum_all / pow(POWERSPEC_GRID, 3);
      
      for(i=0; i < nslab_x;i++)
	for(j=0; j< POWERSPEC_GRID; j++)
	  for(k=0; k< POWERSPEC_GRID; k++)
	    {
	      int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * i + j) + k;
	      
	      velrhofield[dim][ip] -= vmean;
	    }

      for(i=0; i < nslab_x;i++)
	for(j=0; j< POWERSPEC_GRID; j++)
	  for(k=0; k< POWERSPEC_GRID; k++)
	    {
	      int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * i + j) + k;
	      
	      vdisp += velrhofield[dim][ip] * velrhofield[dim][ip];
	    }

      MPI_Allreduce(&vdisp, &vdisp_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      velrho_disp[dim] = vdisp_all / pow(POWERSPEC_GRID, 3);      
    }
}



int powerspec_turb_find_nearest_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
  int j, n, index, listindex = 0;
  int startnode, numngb_inbox;
  double h, r2max;
  double dx, dy, dz, r2;
  MyDouble pos[3];

  if(mode == 0)
    {
      int xx = target / (POWERSPEC_GRID * POWERSPEC_GRID);
      int yy = (target - xx * POWERSPEC_GRID * POWERSPEC_GRID) / POWERSPEC_GRID;
      int zz = (target - xx * POWERSPEC_GRID * POWERSPEC_GRID - yy * POWERSPEC_GRID); 
      xx += slabstart_x;

        double x = (xx + 0.5) / POWERSPEC_GRID * boxSize_X;
        double y = (yy + 0.5) / POWERSPEC_GRID * boxSize_Y;
        double z = (zz + 0.5) / POWERSPEC_GRID * boxSize_Z;

      pos[0] = x;
      pos[1] = y;
      pos[2] = z;
      h = powerspec_turb_nearest_hsml[target];
    }
  else
    {
      pos[0] = DataGet[target].Pos[0];
      pos[1] = DataGet[target].Pos[1];
      pos[2] = DataGet[target].Pos[2];
      h = DataGet[target].Hsml;
    }

  index = -1;
  r2max = 1.0e60;

  if(mode == 0)
    {
      startnode = All.MaxPart;	/* root node */
    }
  else
    {
      startnode = DataGet[target].NodeList[0];
      startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }

  while(startnode >= 0)
    {
      while(startnode >= 0)
	{
	  numngb_inbox = ngb_treefind_variable_targeted(pos, h, target, &startnode, mode, nexport, nsend_local,1); // search for gas: 2^0=1

	  if(numngb_inbox < 0)
	    return -1;

	  for(n = 0; n < numngb_inbox; n++)
	    {
	      j = Ngblist[n];
	      dx = pos[0] - P[j].Pos[0];
	      dy = pos[1] - P[j].Pos[1];
	      dz = pos[2] - P[j].Pos[2];

	      /*  now find the closest image in the given box size  */
            NEAREST_XYZ(dx,dy,dz,1);

	      r2 = dx * dx + dy * dy + dz * dz;
	      if(r2 < r2max && r2 < h * h)
		{
		  index = j;
		  r2max = r2;
		}
	    }
	}

      if(mode == 1)
	{
	  listindex++;
	  if(listindex < NODELISTLENGTH)
	    {
	      startnode = DataGet[target].NodeList[listindex];
	      if(startnode >= 0)
		startnode = Nodes[startnode].u.d.nextnode;	/* open it */
	    }
	}
    }

  if(mode == 0)
    {
      if(index >= 0)
	{
	  if(index >= N_gas)
	    terminate("index >= N_gas");

	  powerspec_turb_nearest_distance[target] = sqrt(r2max);
	  
	  int i = target / (POWERSPEC_GRID * POWERSPEC_GRID);
	  int j = (target - i * POWERSPEC_GRID * POWERSPEC_GRID) / POWERSPEC_GRID;
	  int k = (target - i * POWERSPEC_GRID * POWERSPEC_GRID - j * POWERSPEC_GRID); 
	  int ip = POWERSPEC_GRID2 * (POWERSPEC_GRID * i + j) + k;

	  velfield[0][ip] = P[index].Vel[0];
	  velfield[1][ip] = P[index].Vel[1];
	  velfield[2][ip] = P[index].Vel[2];

	  smoothedvelfield[0][ip] = SphP[index].SmoothedVel[0];
	  smoothedvelfield[1][ip] = SphP[index].SmoothedVel[1];
	  smoothedvelfield[2][ip] = SphP[index].SmoothedVel[2];

	  velrhofield[0][ip] = sqrt(SphP[index].Density) * P[index].Vel[0];
	  velrhofield[1][ip] = sqrt(SphP[index].Density) * P[index].Vel[1];
	  velrhofield[2][ip] = sqrt(SphP[index].Density) * P[index].Vel[2];

	  vorticityfield[0][ip] = SphP[index].Vorticity[0];
	  vorticityfield[1][ip] = SphP[index].Vorticity[1];
	  vorticityfield[2][ip] = SphP[index].Vorticity[2];


	  if(SphP[index].DuDt_diss >= 0)
	    {
	      dis1field[ip] = sqrt(SphP[index].DuDt_diss);
	      dis2field[ip] = 0;
	    }
	  else
	    {
	      dis1field[ip] = 0;
	      dis2field[ip] = sqrt(-SphP[index].DuDt_diss);
	    }

	  randomfield[ip] = RandomValue[index];
	  densityfield[ip] = SphP[index].Density;
	}
    }
  else
    {
      if(index >= 0)
	{
	  if(index >= N_gas)
	    terminate("index >= N_gas");

	  DataResult[target].Distance = sqrt(r2max);
	  DataResult[target].Vel[0] = P[index].Vel[0];
	  DataResult[target].Vel[1] = P[index].Vel[1];
	  DataResult[target].Vel[2] = P[index].Vel[2];
	  DataResult[target].SmoothedVel[0] = SphP[index].SmoothedVel[0];
	  DataResult[target].SmoothedVel[1] = SphP[index].SmoothedVel[1];
	  DataResult[target].SmoothedVel[2] = SphP[index].SmoothedVel[2];
	  DataResult[target].Vorticity[0] = SphP[index].Vorticity[0];
	  DataResult[target].Vorticity[1] = SphP[index].Vorticity[1];
	  DataResult[target].Vorticity[2] = SphP[index].Vorticity[2];
	  DataResult[target].Density = SphP[index].Density;
	  DataResult[target].DuDt_diss = SphP[index].DuDt_diss;
	  DataResult[target].DuDt_drive = SphP[index].DuDt_drive;
	  DataResult[target].RandomValue = RandomValue[index];
	}
      else
	DataResult[target].Distance = 2.0e30;
    }
  return 0;
}






#endif




