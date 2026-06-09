#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cstring>
#include <cstdint>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* Phase 7.c: GPU replacement for force_update_tree. */
extern "C" void gpu_force_update_tree(void);

/* Atomic max for doubles using integer CAS (clang doesn't support __atomic on floats) */
static inline void atomic_max_double(double* addr, double val) {
    uint64_t val_bits; memcpy(&val_bits, &val, sizeof(double));
    uint64_t old_bits; memcpy(&old_bits, addr, sizeof(double));
    double old_val; memcpy(&old_val, &old_bits, sizeof(double));
    while(val > old_val) {
        if(__atomic_compare_exchange_n((uint64_t*)addr, &old_bits, val_bits, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {break;}
        memcpy(&old_val, &old_bits, sizeof(double));
    }
}
static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit for atomic CAS");



/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * substantially (condensed, new feedback routines added, many different
 * types of walk and calculations added, structures in memory changed,
 * switched options for nodes, optimizations, new physics modules and
 * calcutions, and new variable/memory conventions added)
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 * Mike Grudic has also made major revisions to code the Hermitian calculations and binary timestepping.
 */

void force_update_tree(void)
{
    /* Phase 7.d: GPU path is the only path — CPU fallback retired. */
    gpu_force_update_tree();
    return;
}








void force_finish_kick_nodes(void)
{
  int i, j, no, ta, totDomainNumChanged;
  int *domainList_all;
  int *counts, *counts_dp, *offset_list, *offset_dp, *offset_vmax;
  MyDouble *domainDp_loc, *domainDp_all;

#ifdef RT_SEPARATELY_TRACK_LUMPOS
    MyDouble *domainDp_stellarlum_loc, *domainDp_stellarlum_all;
#endif
#ifdef DM_SCALARFIELD_SCREENING
  MyDouble *domainDp_dm_loc, *domainDp_dm_all;
#endif
  MyFloat *domainVmax_loc, *domainVmax_all;

  /* share the momentum-data of the pseudo-particles accross CPUs */

  counts = (int *) mymalloc("counts", sizeof(int) * NTask);
  counts_dp = (int *) mymalloc("counts_dp", sizeof(int) * NTask);
  offset_list = (int *) mymalloc("offset_list", sizeof(int) * NTask);
  offset_dp = (int *) mymalloc("offset_dp", sizeof(int) * NTask);
  offset_vmax = (int *) mymalloc("offset_vmax", sizeof(int) * NTask);

  domainDp_loc = (MyDouble *) mymalloc("domainDp_loc", DomainNumChanged * 3 * sizeof(MyDouble));
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    domainDp_stellarlum_loc = (MyDouble *) mymalloc("domainDp_stellarlum_loc", DomainNumChanged * 3 * sizeof(MyDouble));
#endif
#ifdef DM_SCALARFIELD_SCREENING
  domainDp_dm_loc = (MyDouble *) mymalloc("domainDp_dm_loc", DomainNumChanged * 3 * sizeof(MyDouble));
#endif
  domainVmax_loc = (MyFloat *) mymalloc("domainVmax_loc", DomainNumChanged * sizeof(MyFloat));

  for(i = 0; i < DomainNumChanged; i++)
    {
      for(j = 0; j < 3; j++)
	{
	  domainDp_loc[i * 3 + j] = Extnodes[DomainList[i]].dp[j];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        domainDp_stellarlum_loc[i * 3 + j] = Extnodes[DomainList[i]].rt_source_lum_dp[j];
#endif
#ifdef DM_SCALARFIELD_SCREENING
	  domainDp_dm_loc[i * 3 + j] = Extnodes[DomainList[i]].dp_dm[j];
#endif
	}
      domainVmax_loc[i] = Extnodes[DomainList[i]].vmax;
    }

  MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

  for(ta = 0, totDomainNumChanged = 0, offset_list[0] = 0, offset_dp[0] = 0, offset_vmax[0] = 0; ta < NTask;
      ta++)
    {
      totDomainNumChanged += counts[ta];
      if(ta > 0)
	{
	  offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
	  offset_dp[ta] = offset_dp[ta - 1] + counts[ta - 1] * 3 * sizeof(MyDouble);
	  offset_vmax[ta] = offset_vmax[ta - 1] + counts[ta - 1] * sizeof(MyFloat);
	}
    }

  PRINT_STATUS(" ..exchanged kick momenta for %d top-level nodes out of %d", totDomainNumChanged, NTopleaves);
  domainDp_all = (MyDouble *) mymalloc("domainDp_all", totDomainNumChanged * 3 * sizeof(MyDouble));
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    domainDp_stellarlum_all = (MyDouble *) mymalloc("domainDp_stellarlum_all", totDomainNumChanged * 3 * sizeof(MyDouble));
#endif
#ifdef DM_SCALARFIELD_SCREENING
  domainDp_dm_all =
    (MyDouble *) mymalloc("domainDp_dm_all", totDomainNumChanged * 3 * sizeof(MyDouble));
#endif
  domainVmax_all = (MyFloat *) mymalloc("domainVmax_all", totDomainNumChanged * sizeof(MyFloat));

  domainList_all = (int *) mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

  MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT,
		 domainList_all, counts, offset_list, MPI_INT, MPI_COMM_WORLD);

  for(ta = 0; ta < NTask; ta++)
    {
      counts_dp[ta] = counts[ta] * 3 * sizeof(MyDouble);
      counts[ta] *= sizeof(MyFloat);
    }


  MPI_Allgatherv(domainDp_loc, DomainNumChanged * 3 * sizeof(MyDouble), MPI_BYTE,
		 domainDp_all, counts_dp, offset_dp, MPI_BYTE, MPI_COMM_WORLD);

#ifdef RT_SEPARATELY_TRACK_LUMPOS
    MPI_Allgatherv(domainDp_stellarlum_loc, DomainNumChanged * 3 * sizeof(MyDouble), MPI_BYTE,
                   domainDp_stellarlum_all, counts_dp, offset_dp, MPI_BYTE, MPI_COMM_WORLD);
#endif
#ifdef DM_SCALARFIELD_SCREENING
  MPI_Allgatherv(domainDp_dm_loc, DomainNumChanged * 3 * sizeof(MyDouble), MPI_BYTE,
		 domainDp_dm_all, counts_dp, offset_dp, MPI_BYTE, MPI_COMM_WORLD);
#endif

  MPI_Allgatherv(domainVmax_loc, DomainNumChanged * sizeof(MyFloat), MPI_BYTE,
		 domainVmax_all, counts, offset_vmax, MPI_BYTE, MPI_COMM_WORLD);


  /* DIAGNOSTIC: after MPI exchange, print totDomainNumChanged and sum of received dp */
  /* construct momentum kicks in top-level tree */
  for(i = 0; i < totDomainNumChanged; i++)
    {
      no = domainList_all[i];

      if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))	/* to avoid that the local one is kicked twice */
	no = Nodes[no].u.d.father;

      while(no >= 0)
	{
	  force_drift_node(no, All.Ti_Current);

	  Extnodes[no].dp[0] += domainDp_all[3 * i + 0];
	  Extnodes[no].dp[1] += domainDp_all[3 * i + 1];
	  Extnodes[no].dp[2] += domainDp_all[3 * i + 2];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
          Extnodes[no].rt_source_lum_dp[0] += domainDp_stellarlum_all[3 * i + 0];
          Extnodes[no].rt_source_lum_dp[1] += domainDp_stellarlum_all[3 * i + 1];
          Extnodes[no].rt_source_lum_dp[2] += domainDp_stellarlum_all[3 * i + 2];
#endif
#ifdef DM_SCALARFIELD_SCREENING
	  Extnodes[no].dp_dm[0] += domainDp_dm_all[3 * i + 0];
	  Extnodes[no].dp_dm[1] += domainDp_dm_all[3 * i + 1];
	  Extnodes[no].dp_dm[2] += domainDp_dm_all[3 * i + 2];
#endif

	  if(Extnodes[no].vmax < domainVmax_all[i])
	    Extnodes[no].vmax = domainVmax_all[i];

	  Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
	  Extnodes[no].Ti_lastkicked = All.Ti_Current;

	  no = Nodes[no].u.d.father;
	}
    }

  myfree(domainList_all);
  myfree(domainVmax_all);
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    myfree(domainDp_stellarlum_all);
#endif
#ifdef DM_SCALARFIELD_SCREENING
  myfree(domainDp_dm_all);
#endif
  myfree(domainDp_all);
  myfree(domainVmax_loc);
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    myfree(domainDp_stellarlum_loc);
#endif
#ifdef DM_SCALARFIELD_SCREENING
  myfree(domainDp_dm_loc);
#endif
  myfree(domainDp_loc);
  myfree(offset_vmax);
  myfree(offset_dp);
  myfree(offset_list);
  myfree(counts_dp);
  myfree(counts);
}



void force_drift_node(int no, integertime time1)
{
  int j;
  integertime time0;
  double dt_drift, dt_drift_hmax, fac;

  if(time1 == Nodes[no].Ti_current)
    return;

  time0 = Extnodes[no].Ti_lastkicked;

  if(Nodes[no].u.d.bitflags & (1 << BITFLAG_NODEHASBEENKICKED))
    {
      if(Extnodes[no].Ti_lastkicked != Nodes[no].Ti_current)
	{
	  printf("Task=%d Extnodes[no].Ti_lastkicked=%lld  Nodes[no].Ti_current=%lld\n",ThisTask, (long long)Extnodes[no].Ti_lastkicked, (long long)Nodes[no].Ti_current);
	  printf("inconsistency in drift node\n"); fflush(stdout); endrun(90001007); return;   /* graceful: skip node drift; bad-stop drains at the next gravity-walk poll */
	}

      if(Nodes[no].u.d.mass) {fac = 1 / Nodes[no].u.d.mass;} else {fac = 0;}

#ifdef RT_SEPARATELY_TRACK_LUMPOS
        double fac_stellar_lum;
        double l_tot=0; for(j=0;j<N_RT_FREQ_BINS;j++) {l_tot += (Nodes[no].stellar_lum[j]);}
        if(l_tot>0) {fac_stellar_lum = 1 / l_tot;} else {fac_stellar_lum = 0;}
#endif

#ifdef DM_SCALARFIELD_SCREENING
      double fac_dm;
      if(Nodes[no].mass_dm) {fac_dm = 1 / Nodes[no].mass_dm;} else {fac_dm = 0;}
#endif

      Extnodes[no].vs += fac * Extnodes[no].dp;
      Extnodes[no].dp = {};
#ifdef RT_SEPARATELY_TRACK_LUMPOS
      Extnodes[no].rt_source_lum_vs += fac_stellar_lum * Extnodes[no].rt_source_lum_dp;
      Extnodes[no].rt_source_lum_dp = {};
#endif
#ifdef DM_SCALARFIELD_SCREENING
      Extnodes[no].vs_dm += fac_dm * Extnodes[no].dp_dm;
      Extnodes[no].dp_dm = {};
#endif
      Nodes[no].u.d.bitflags &= (~(1 << BITFLAG_NODEHASBEENKICKED));
    }

    dt_drift = dt_drift_hmax = get_drift_factor(Nodes[no].Ti_current, time1, no, 1);
    

    Nodes[no].u.d.s += Extnodes[no].vs * dt_drift;
  Nodes[no].len += 2 * Extnodes[no].vmax * dt_drift;

#ifdef DM_SCALARFIELD_SCREENING
    Nodes[no].s_dm += Extnodes[no].vs_dm * dt_drift;
#endif


#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Nodes[no].rt_source_lum_s += Extnodes[no].rt_source_lum_vs * dt_drift;
#endif

    if(Extnodes[no].hmax > 0) {Extnodes[no].hmax *= exp(DMAX(-1.,DMIN(1.,Extnodes[no].divVmax * dt_drift_hmax / NUMDIMS)));}
    /* Mode B per-type bands: upward-only inflate (codex 2026-06-07). The bands
     * include static-ish sources like P[j].ForceSoftening (per
     * force_hmax_per_type_particle_radius), so decaying the band below the
     * actual FS value would under-bound the node-prune. force_update_hmax()
     * re-grows the bands per-particle each call; we just must not shrink
     * them under drift. (Scalar `hmax` retains its legacy bidirectional decay
     * — its semantics are unchanged.) */
    {
        double decay_fac = exp(DMAX(-1., DMIN(1., Extnodes[no].divVmax * dt_drift_hmax / NUMDIMS)));
        if(decay_fac > 1.0) {
            for(int t = 0; t < 6; t++) {
                if(Extnodes[no].hmax_per_type[t] > 0) {
                    Extnodes[no].hmax_per_type[t] *= (MyFloat)decay_fac;
                }
            }
        }
    }
    Nodes[no].Ti_current = time1;
}





/*! This function updates the hmax-values in tree nodes that hold gas cells. These values are needed to find all neighbors in the
 *  hydro-force computation.  Since the KernelRadius-values are potentially changed in the fluid-density computation, force_update_hmax() should be carried
 *  out just before the hydrodynamical forces are computed, i.e. after density(). */
void force_update_hmax(void)
{
  int i, no, ta, totDomainNumChanged;
  int *domainList_all;
  int *counts, *offset_list, *offset_hmax;
  MyFloat *domainHmax_loc, *domainHmax_all;
  int OffsetSIZE = 2;
  double divVel;

  GlobFlag++;

  DomainNumChanged = 0;
  DomainList = (int *) mymalloc("DomainList", NTopleaves * sizeof(int));

  /* Phase 1: drift all ancestor nodes (serial — force_drift_node is not thread-safe).
   * Mode B per-type bands now cover every type's leaf-policy-selectable radius
   * (gas via KernelRadius+ForceSoftening; non-gas adds AGS_KernelRadius when defined),
   * so non-gas particles must reach this update path in every build, not only
   * ADAPTIVE_GRAVSOFT_FORALL ones. Scalar Extnodes[no].hmax retains its legacy
   * semantics; it is only grown for the AGS-tracked path below. */
  for (int i : ActiveParticleList)
  {
    if(P[i].Mass > 0)
      {
        no = Father[i];
        while(no >= 0)
        {
            if(Nodes[no].Ti_current == All.Ti_Current) {break;}
            force_drift_node(no, All.Ti_Current);
            no = Nodes[no].u.d.father;
        }
      }
  }
  /* Phase 2: update hmax/divVmax/per-type bands with atomics (parallel). */
#pragma omp parallel for schedule(dynamic)
  for (int idx = 0; idx < (int)ActiveParticleList.size(); idx++)
  {
    int i = ActiveParticleList[idx];
    if(P[i].Mass > 0)
      {
        int no = Father[i];
        double divVel = P[i].Particle_DivVel;

        /* Mode B per-type band: conservative across every leaf-policy-selectable
         * source. Helper covers KernelRadius / ForceSoftening / AGS_KernelRadius
         * (when defined) uniformly per type. */
        int per_type_band = (int)P[i].Type;
        double per_type_htmp = force_hmax_per_type_particle_radius(i);

        /* Scalar `hmax`/`divVmax` eligibility (legacy semantics, codex 2026-06-07):
         * non-AGS-FORALL builds → gas only; AGS-FORALL builds → any Mass>0 type.
         * Non-eligible particles still update per-type bands above, but MUST NOT
         * leak their divVel / KernelRadius into the scalar band (which feeds
         * downstream cross-rank exchange and legacy walkers). */
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
        const int scalar_eligible = 1;
#else
        const int scalar_eligible = (P[i].Type == 0);
#endif

        while(no >= 0)
        {
            /* Scalar `hmax` source: legacy gas-KR or AGS-KR per ADAPTIVE_GRAVSOFT_FORALL. */
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
            double kernrad_temp = P[i].AGS_KernelRadius;
            if(P[i].Type == 0) {kernrad_temp = P[i].KernelRadius;}
            double htmp = DMIN(kernrad_temp, All.MaxKernelRadius);
#else
            double htmp = (P[i].Type == 0) ? DMIN(P[i].KernelRadius, All.MaxKernelRadius) : 0.0;
#endif
            int per_type_grew = 0;
            if(per_type_htmp > Extnodes[no].hmax_per_type[per_type_band]) {
                atomic_max_double(&Extnodes[no].hmax_per_type[per_type_band], per_type_htmp);
                per_type_grew = 1;
            }
            int scalar_grew = 0;
            if(scalar_eligible && (htmp > Extnodes[no].hmax || divVel > Extnodes[no].divVmax))
            {
                atomic_max_double(&Extnodes[no].hmax, htmp);
                atomic_max_double(&Extnodes[no].divVmax, divVel);
                scalar_grew = 1;
            }
            if(scalar_grew || per_type_grew)
            {
                if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))
                {
                    #pragma omp critical(DomainListAppendHmax)
                    {
                        if(Extnodes[no].Flag != GlobFlag)
                        {
                            Extnodes[no].Flag = GlobFlag;
                            DomainList[DomainNumChanged++] = no;
                        }
                    }
                    break;
                }
            }
            else
                break;

            no = Nodes[no].u.d.father;
        }
      }
  }

  /* share the hmax-data of the pseudo-particles accross CPUs */

  counts = (int *) mymalloc("counts", sizeof(int) * NTask);
  offset_list = (int *) mymalloc("offset_list", sizeof(int) * NTask);
  offset_hmax = (int *) mymalloc("offset_hmax", sizeof(int) * NTask);

  domainHmax_loc = (MyFloat *) mymalloc("domainHmax_loc", DomainNumChanged * OffsetSIZE * sizeof(MyFloat));

  for(i = 0; i < DomainNumChanged; i++)
    {
      domainHmax_loc[OffsetSIZE * i] = Extnodes[DomainList[i]].hmax;
      domainHmax_loc[OffsetSIZE * i + 1] = Extnodes[DomainList[i]].divVmax;
    }


  MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

  for(ta = 0, totDomainNumChanged = 0, offset_list[0] = 0, offset_hmax[0] = 0; ta < NTask; ta++)
    {
      totDomainNumChanged += counts[ta];
      if(ta > 0)
	{
	  offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
	  offset_hmax[ta] = offset_hmax[ta - 1] + counts[ta - 1] * OffsetSIZE * sizeof(MyFloat);
	}
    }

  PRINT_STATUS(" ..Hmax exchange: %d topleaves out of %d", totDomainNumChanged, NTopleaves);
  domainHmax_all = (MyFloat *) mymalloc("domainHmax_all", totDomainNumChanged * OffsetSIZE * sizeof(MyFloat));
  domainList_all = (int *) mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

  MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT,
		 domainList_all, counts, offset_list, MPI_INT, MPI_COMM_WORLD);

  for(ta = 0; ta < NTask; ta++)
    {counts[ta] *= OffsetSIZE * sizeof(MyFloat);}

  MPI_Allgatherv(domainHmax_loc, OffsetSIZE * DomainNumChanged * sizeof(MyFloat), MPI_BYTE,
		 domainHmax_all, counts, offset_hmax, MPI_BYTE, MPI_COMM_WORLD);


  for(i = 0; i < totDomainNumChanged; i++)
    {
        no = domainList_all[i];
        if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))    {no = Nodes[no].u.d.father;} /* to avoid that the hmax is updated twice */
        
        while(no >= 0)
        {
            force_drift_node(no, All.Ti_Current);
            
            if(domainHmax_all[OffsetSIZE * i] > Extnodes[no].hmax || domainHmax_all[OffsetSIZE * i + 1] > Extnodes[no].divVmax)
            {
                if(domainHmax_all[OffsetSIZE * i] > Extnodes[no].hmax) {Extnodes[no].hmax = domainHmax_all[OffsetSIZE * i];}
                if(domainHmax_all[OffsetSIZE * i + 1] > Extnodes[no].divVmax) {Extnodes[no].divVmax = domainHmax_all[OffsetSIZE * i + 1];}
            }
            else
            {break;}
            
            no = Nodes[no].u.d.father;
        }
    }


  myfree(domainList_all);
  myfree(domainHmax_all);
  myfree(domainHmax_loc);
  myfree(offset_hmax);
  myfree(offset_list);
  myfree(counts);
  myfree(DomainList);

  CPU_Step[CPU_TREEHMAXUPDATE] += measure_time();
}
