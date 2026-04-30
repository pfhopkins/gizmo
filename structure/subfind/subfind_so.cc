#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
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
#include "../group_search.h"
#include "subfind.h"

static MyOutputFloat *R200, *M200;
struct Subfind_DensityOtherPropsEval_data_out *Subfind_DensityOtherPropsEval_DataResult, *Subfind_DensityOtherPropsEval_DataOut, *Subfind_DensityOtherPropsEval_GlobalPasser;



/*!
 The following functions are designed to flexibly allow adding/removing additional properties computed in groups of varying defintiions/radii/overdensities, etc.
 To add similar computations, follow their template (e.g. the "SUBFIND_ADDIO_..." options). This first subroutine is the core computation of the relevant properties
 in the group. compute them/add them into the "out" structure and the code here and other two scripts below should take care of the rest
*/
/*!   -- this subroutine is not openmp parallelized at present, so there's not any issue about conflicts over shared memory. if you make it openmp, make sure you protect the writes to shared memory here! -- */
/*! first define a short structure needed to pass in the group info here */

/*! do any final opertations on the data after it comes back from  Subfind_DensityOtherProps_evaluate  */
void Subfind_DensityOtherProps_finaloperations(struct Subfind_DensityOtherPropsEval_data_out *in)
{
    if(in->M200 <= 0) return; /* no members found */
#ifdef SUBFIND_ADDIO_VELDISP
    in->Disp200/=in->M200; int k; for(k=0;k<3;k++) {in->V200[k]/=in->M200; in->Disp200-=(in->V200[k])*(in->V200[k]);}
    in->Disp200=sqrt(in->Disp200/3.); /* convert to 1D velocity dispersion */
#endif
#ifdef SUBFIND_ADDIO_BARYONS
    if(in->gas_mass>0) {in->temp/=in->gas_mass;}
#endif
}

static void subfind_density_other_props_loop_modern(void);

static void subfind_so_accumulate_particle(const int gr, const particle_data &pj, const gas_cell_data *cell,
                                           struct Subfind_DensityOtherPropsEval_data_out *out)
{
  int k;
  Vec3<double> dr;
  for(k = 0; k < 3; k++) dr[k] = pj.Pos[k] - Group[gr].Pos[k];
  nearest_xyz(dr, -1);
  out->M200 += pj.Mass;

#ifdef SUBFIND_ADDIO_VELDISP
  for(k = 0; k < 3; k++)
    {
      double dv = pj.Vel[k] / All.cf_atime + All.cf_hubble_a * All.cf_atime * dr[k];
      out->V200[k] += pj.Mass * dv;
      out->Disp200 += pj.Mass * dv * dv;
    }
#endif
#ifdef SUBFIND_ADDIO_BARYONS
  if(pj.Type == 0)
    {
      if(cell == NULL)
        {
          printf("SUBFIND modern SO missing gas cell payload for group=%d task=%d\n", gr, ThisTask);
          fflush(stdout);
          endrun(13012);
        }
      double temp_keV = 6.14e-16 * (cell->InternalEnergy / UNIT_SPECEGY_IN_CGS);
      out->gas_mass += pj.Mass;
      out->temp += pj.Mass * temp_keV;
      out->xlum += 1.52e-20 * (pj.Mass * UNIT_MASS_IN_CGS) * (cell->Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS) * sqrt(temp_keV);
    }
  else if(pj.Type == 4)
    {
      out->star_mass += pj.Mass;
    }
#endif
}

static void subfind_so_compute_modern(const std::vector<int> &groups, const MyOutputFloat *radii,
                                      struct Subfind_DensityOtherPropsEval_data_out *results)
{
  int nsources = groups.size();
  std::vector<double> centers(3 * nsources), query_radii(nsources);
  for(int a = 0; a < nsources; a++)
    {
      int gr = groups[a];
      query_radii[a] = radii[gr];
      for(int k = 0; k < 3; k++) centers[3 * a + k] = Group[gr].Pos[k];
      memset(&results[gr], 0, sizeof(struct Subfind_DensityOtherPropsEval_data_out));
    }

  group_search_import_pool_t pool;
#ifdef SUBFIND_ADDIO_BARYONS
  const bool import_cells = true;
#else
  const bool import_cells = false;
#endif
  group_search_import_particles_around_points(63, centers, query_radii, pool, import_cells);
  if(nsources <= 0) return;

  int query_start = pool.particles.size();
  pool.particles.resize(query_start + nsources);
  if(import_cells) pool.cells.resize(query_start + nsources);

  std::vector<int> sources(nsources);
  for(int a = 0; a < nsources; a++)
    {
      particle_data q;
      memset(&q, 0, sizeof(particle_data));
      int gr = groups[a];
      q.Type = 6;
      q.Mass = 1;
      q.KernelRadius = query_radii[a];
      for(int k = 0; k < 3; k++) q.Pos[k] = Group[gr].Pos[k];
      pool.particles[query_start + a] = q;
      sources[a] = query_start + a;
    }

  neighbor_list_t nl;
  group_search_build_cross_type_nl(pool.particles, sources.data(), nsources, query_radii.data(), 63, &nl);

  for(int a = 0; a < nsources; a++)
    {
      int gr = groups[a];
      for(int n = nl.offsets[a]; n < nl.offsets[a + 1]; n++)
        {
          int j = nl.neighbors[n];
          const gas_cell_data *cell = (import_cells && j < (int)pool.cells.size()) ? &pool.cells[j] : NULL;
          subfind_so_accumulate_particle(gr, pool.particles[j], cell, &results[gr]);
        }
    }

  free_neighbor_list(&nl);
}

static void subfind_density_other_props_loop_modern(void)
{
  long long ntot;
  int i, j, npleft, rep, iter;
  MyFloat *Left, *Right;
  char *Todo;
  double t0, t1, t2, t3, rguess, overdensity, Deltas[SUBFIND_ADDIO_NUMOVERDEN], z;

  Left = (MyFloat *) mymalloc("Left", sizeof(MyFloat) * Ngroups);
  Right = (MyFloat *) mymalloc("Right", sizeof(MyFloat) * Ngroups);
  R200 = (MyOutputFloat *) mymalloc("R200", sizeof(MyOutputFloat) * Ngroups);
  M200 = (MyOutputFloat *) mymalloc("M200", sizeof(MyOutputFloat) * Ngroups);
  Subfind_DensityOtherPropsEval_GlobalPasser =
    (struct Subfind_DensityOtherPropsEval_data_out *) mymalloc("Subfind_DensityOtherPropsEval_GlobalPasser",
                                                               Ngroups * sizeof(struct Subfind_DensityOtherPropsEval_data_out));
  Todo = (char *)mymalloc("Todo", sizeof(char) * Ngroups);

  if(All.ComovingIntegrationOn) {z = 1 / All.Time - 1;} else {z = 0;}
  double rhoback = 3 * All.OmegaMatter * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits / (8 * M_PI * All.G), zplusone=1.+z;
  double omegaz = All.OmegaMatter * pow(zplusone,3) / (All.OmegaRadiation * pow(zplusone,4) + All.OmegaMatter * pow(zplusone,3) + (1 - All.OmegaMatter - All.OmegaLambda - All.OmegaRadiation) * pow(zplusone,2) + All.OmegaLambda);
  double x = omegaz - 1, DeltaTopHat = (18 * M_PI * M_PI + 82 * x - 39 * x * x) / omegaz;

  double Delta_ToEvalList[10];
  Delta_ToEvalList[0] = 200;
  Delta_ToEvalList[1] = DeltaTopHat;
  Delta_ToEvalList[2] = 200/omegaz;
  Delta_ToEvalList[3] = 500/omegaz;
  Delta_ToEvalList[4] = 1000/omegaz;
  Delta_ToEvalList[5] = 2500/omegaz;
  Delta_ToEvalList[6] = 500;
  Delta_ToEvalList[7] = 1000;
  Delta_ToEvalList[8] = 2500;
  Delta_ToEvalList[9] = 5000;
  for(j = 0; j < SUBFIND_ADDIO_NUMOVERDEN; j++) Deltas[j] = Delta_ToEvalList[j];

  for(rep = 0; rep < SUBFIND_ADDIO_NUMOVERDEN; rep++)
    {
      t2 = my_second();
      for(i = 0; i < Ngroups; i++)
        {
          R200[i] = M200[i] = 0;
          memset(&Subfind_DensityOtherPropsEval_GlobalPasser[i], 0, sizeof(struct Subfind_DensityOtherPropsEval_data_out));
          if(Group[i].Nsubs > 0)
            {
              rguess = pow(All.G * Group[i].Mass / (100 * All.Hubble_H0_CodeUnits * All.Hubble_H0_CodeUnits), 1.0 / 3);
              Right[i] = 3 * rguess;
              Left[i] = 0;
              Todo[i] = 1;
            }
          else
            {
              Left[i] = Right[i] = 0;
              Todo[i] = 0;
            }
        }
      iter = 0;

      do
        {
          t0 = my_second();
          std::vector<int> active;
          active.reserve(Ngroups);
          for(i = 0; i < Ngroups; i++)
            if(Todo[i])
              {
                R200[i] = 0.5 * (Left[i] + Right[i]);
                active.push_back(i);
              }

          subfind_so_compute_modern(active, R200, Subfind_DensityOtherPropsEval_GlobalPasser);
          for(size_t a = 0; a < active.size(); a++)
            {
              int gr = active[a];
              M200[gr] = Subfind_DensityOtherPropsEval_GlobalPasser[gr].M200;
            }

          for(i = 0, npleft = 0; i < Ngroups; i++)
            {
              if(Todo[i])
                {
                  overdensity = M200[i] / (4.0 * M_PI / 3.0 * R200[i] * R200[i] * R200[i]) / rhoback;
                  if((Right[i] - Left[i]) > 1.0e-4 * Left[i])
                    {
                      npleft++;
                      if(overdensity > Deltas[rep]) {Left[i] = R200[i];} else {Right[i] = R200[i];}
                      if(iter >= MAXITER - 10)
                        {
                          printf("gr=%d task=%d  R200=%g Left=%g Right=%g Menclosed=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                                 i, ThisTask, R200[i], Left[i], Right[i], M200[i], Right[i] - Left[i],
                                 Group[i].Pos[0], Group[i].Pos[1], Group[i].Pos[2]);
                          fflush(stdout);
                        }
                    }
                  else
                    {
                      Todo[i] = 0;
                    }
                }
            }

          sumup_large_ints(1, &npleft, &ntot);
          t1 = my_second();
          if(ntot > 0)
            {
              iter++;
              if(iter > 0 && ThisTask == 0)
                {
                  printf("SO iteration %d: need to repeat for %d%09d groups. (took %g sec)\n", iter,
                         (int) (ntot / 1000000000), (int) (ntot % 1000000000), timediff(t0, t1));
                  fflush(stdout);
                }
              if(iter > MAXITER)
                {
                  printf("failed to converge in modern SUBFIND SO radius iteration\n");
                  fflush(stdout);
                  endrun(1155);
                }
            }
        }
      while(ntot > 0);

      MPI_Barrier(MPI_COMM_WORLD);
      t3 = my_second();
      if(ThisTask == 0)
        {
          printf("SO Radius calculation took %g sec\n", timediff(t2, t3));
          fflush(stdout);
        }

      t0 = my_second();
      memset(Subfind_DensityOtherPropsEval_GlobalPasser, 0, Ngroups * sizeof(struct Subfind_DensityOtherPropsEval_data_out));
      std::vector<int> active_props;
      active_props.reserve(Ngroups);
      for(i = 0; i < Ngroups; i++)
        if(Group[i].Nsubs > 0 && R200[i] > 0)
          active_props.push_back(i);
      subfind_so_compute_modern(active_props, R200, Subfind_DensityOtherPropsEval_GlobalPasser);

      MPI_Barrier(MPI_COMM_WORLD);
      t1 = my_second();
      if(ThisTask == 0)
        {
          printf("secondary subfind loop for additional information took %g sec\n", timediff(t0, t1));
          fflush(stdout);
        }

      t0 = my_second();
      for(i = 0; i < Ngroups; i++)
        {
          if(Group[i].Nsubs > 0)
            {
              M200[i] = Subfind_DensityOtherPropsEval_GlobalPasser[i].M200;
              overdensity = M200[i] / (4.0 * M_PI / 3.0 * R200[i] * R200[i] * R200[i]) / rhoback;

              if((overdensity - Deltas[rep]) > 0.1 * Deltas[rep] || M200[i] < 5 * Group[i].Mass / Group[i].Len)
                {
                  R200[i] = M200[i] = 0;
                  memset(&Subfind_DensityOtherPropsEval_GlobalPasser[i], 0, sizeof(struct Subfind_DensityOtherPropsEval_data_out));
                }
            }
          else
            {
              R200[i] = M200[i] = 0;
              memset(&Subfind_DensityOtherPropsEval_GlobalPasser[i], 0, sizeof(struct Subfind_DensityOtherPropsEval_data_out));
            }

          Subfind_DensityOtherPropsEval_GlobalPasser[i].R200 = R200[i];
          Subfind_DensityOtherProps_finaloperations(&Subfind_DensityOtherPropsEval_GlobalPasser[i]);
          memcpy(&Group[i].SubHaloProps_vsDelta[rep], &Subfind_DensityOtherPropsEval_GlobalPasser[i],
                 sizeof(struct Subfind_DensityOtherPropsEval_data_out));
        }
    }
  t1 = my_second();
  if(ThisTask == 0) {printf("Saving data took %g sec\n", timediff(t0, t1)); fflush(stdout);}

  myfree(Todo);
  myfree(Subfind_DensityOtherPropsEval_GlobalPasser);
  myfree(M200);
  myfree(R200);
  myfree(Right);
  myfree(Left);
}

    
    
    

/*! ok everything below is the 'generic' part of this set of routines */



void Subfind_DensityOtherProps_Loop(void)
{
    subfind_density_other_props_loop_modern();
    return;
}


/*! This function represents the core of the density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */

#endif
