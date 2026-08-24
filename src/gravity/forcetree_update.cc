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
#include "force_node_drift_sync.h"

/* GPU replacement for force_update_tree. */
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
    /* GPU path is the only path — CPU fallback retired. */
    gpu_force_update_tree();
    return;
}








/* Packed per-changed-node kick record for the single fused Allgatherv in
 * force_finish_kick_nodes (replaces the 2-5 separate field Allgathervs). Fixed
 * layout is identical on every rank, so an MPI_BYTE exchange is field-equivalent
 * (any padding bytes are transmitted but ignored). TU-local. */
namespace {
struct DomainKickPacked
{
  int node;
  MyDouble dp[3];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
  MyDouble rt_dp[3];
#endif
#ifdef DM_SCALARFIELD_SCREENING
  MyDouble dp_dm[3];
#endif
#ifdef SINK_NODE_MOTION_TRACKED
  MyDouble sink_dp[3];
#endif
  MyFloat vmax;
};
}  /* anonymous namespace */

void force_finish_kick_nodes(void)
{
  int i, no, ta, totDomainNumChanged;
  int *counts, *counts_dp, *offset_dp;

  /* share the momentum-data of the pseudo-particles accross CPUs */

  counts = (int *) mymalloc("counts", sizeof(int) * NTask);
  counts_dp = (int *) mymalloc("counts_dp", sizeof(int) * NTask);
  offset_dp = (int *) mymalloc("offset_dp", sizeof(int) * NTask);

  /* Exchange per-rank changed-node counts FIRST (all ranks participate in this
   * collective), so a globally-empty update can early-out BEFORE any local
   * payload malloc/pack, the packed Allgatherv, and the apply loop. When
   * totDomainNumChanged==0 every rank's payload is empty, so all that work is
   * mathematically a no-op; the early-out is exact, and symmetric across ranks
   * (totDomainNumChanged is the same global sum on every rank). */
  MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

  for(ta = 0, totDomainNumChanged = 0; ta < NTask; ta++)
    totDomainNumChanged += counts[ta];

  if(totDomainNumChanged == 0)
    {   /* no rank changed a top-level node: nothing to pack/exchange/apply */
      myfree(offset_dp);
      myfree(counts_dp);
      myfree(counts);
      return;
    }

  /* Packed single-Allgatherv path: pack each changed node's fields into one
   * contiguous record and exchange in ONE collective instead of 2-5 separate
   * field Allgathervs (fewer collective latencies / skew-absorption points).
   * Field-equivalent apply. */
  {
    struct DomainKickPacked *rec_loc = (struct DomainKickPacked *)
        mymalloc("fut_rec_loc", DomainNumChanged * sizeof(struct DomainKickPacked));
    for(i = 0; i < DomainNumChanged; i++)
      {
        no = DomainList[i];
        rec_loc[i].node = no;
        rec_loc[i].dp[0] = Extnodes[no].dp[0];
        rec_loc[i].dp[1] = Extnodes[no].dp[1];
        rec_loc[i].dp[2] = Extnodes[no].dp[2];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        rec_loc[i].rt_dp[0] = Extnodes[no].rt_source_lum_dp[0];
        rec_loc[i].rt_dp[1] = Extnodes[no].rt_source_lum_dp[1];
        rec_loc[i].rt_dp[2] = Extnodes[no].rt_source_lum_dp[2];
#endif
#ifdef DM_SCALARFIELD_SCREENING
        rec_loc[i].dp_dm[0] = Extnodes[no].dp_dm[0];
        rec_loc[i].dp_dm[1] = Extnodes[no].dp_dm[1];
        rec_loc[i].dp_dm[2] = Extnodes[no].dp_dm[2];
#endif
#ifdef SINK_NODE_MOTION_TRACKED
        rec_loc[i].sink_dp[0] = Extnodes[no].sink_dp[0];
        rec_loc[i].sink_dp[1] = Extnodes[no].sink_dp[1];
        rec_loc[i].sink_dp[2] = Extnodes[no].sink_dp[2];
#endif
        rec_loc[i].vmax = Extnodes[no].vmax;
      }
    /* byte counts/offsets for the fixed-size records (reuse counts_dp/offset_dp;
     * counts[] is still the raw per-rank node count here) */
    for(ta = 0; ta < NTask; ta++)
      {
        counts_dp[ta] = counts[ta] * (int) sizeof(struct DomainKickPacked);
        offset_dp[ta] = (ta == 0) ? 0 : offset_dp[ta - 1] + counts[ta - 1] * (int) sizeof(struct DomainKickPacked);
      }
    PRINT_STATUS(" ..exchanged kick momenta for %d top-level nodes out of %d", totDomainNumChanged, NTopleaves);
    struct DomainKickPacked *rec_all = (struct DomainKickPacked *)
        mymalloc("fut_rec_all", totDomainNumChanged * sizeof(struct DomainKickPacked));
    MPI_Allgatherv(rec_loc, DomainNumChanged * (int) sizeof(struct DomainKickPacked), MPI_BYTE,
                   rec_all, counts_dp, offset_dp, MPI_BYTE, MPI_COMM_WORLD);
    for(i = 0; i < totDomainNumChanged; i++)
      {
        no = rec_all[i].node;
        if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))
          no = Nodes[no].u.d.father;
        while(no >= 0)
          {
            force_drift_node(no, All.Ti_Current);
            Extnodes[no].dp[0] += rec_all[i].dp[0];
            Extnodes[no].dp[1] += rec_all[i].dp[1];
            Extnodes[no].dp[2] += rec_all[i].dp[2];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
            Extnodes[no].rt_source_lum_dp[0] += rec_all[i].rt_dp[0];
            Extnodes[no].rt_source_lum_dp[1] += rec_all[i].rt_dp[1];
            Extnodes[no].rt_source_lum_dp[2] += rec_all[i].rt_dp[2];
#endif
#ifdef DM_SCALARFIELD_SCREENING
            Extnodes[no].dp_dm[0] += rec_all[i].dp_dm[0];
            Extnodes[no].dp_dm[1] += rec_all[i].dp_dm[1];
            Extnodes[no].dp_dm[2] += rec_all[i].dp_dm[2];
#endif
#ifdef SINK_NODE_MOTION_TRACKED
            Extnodes[no].sink_dp[0] += rec_all[i].sink_dp[0];
            Extnodes[no].sink_dp[1] += rec_all[i].sink_dp[1];
            Extnodes[no].sink_dp[2] += rec_all[i].sink_dp[2];
#endif
            if(Extnodes[no].vmax < rec_all[i].vmax)
              Extnodes[no].vmax = rec_all[i].vmax;
            Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
            Extnodes[no].Ti_lastkicked = All.Ti_Current;
            no = Nodes[no].u.d.father;
          }
      }
    myfree(rec_all);
    myfree(rec_loc);
  }
  myfree(offset_dp);
  myfree(counts_dp);
  myfree(counts);
}



void force_drift_node(int no, integertime time1)
{
  int j;
  integertime time0;
  double dt_drift, dt_drift_hmax, fac;

  /* Acquire-load: if another thread already drifted this node to time1, we both
   * skip AND observe its published geometry (paired with the release store below). */
  if(time1 == modeb_node_ti_current_acquire(no))
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
#ifdef SINK_NODE_MOTION_TRACKED
      /* sink_vel lives in Nodes rather than Extnodes, but is updated exactly as vs/vs_dm are.
         Normalised by sink_mass, not mass: it is the mass-weighted mean velocity of the sinks alone. */
      {
          double fac_sink = (Nodes[no].sink_mass > 0) ? (1.0 / Nodes[no].sink_mass) : 0.0;
          Nodes[no].sink_vel += fac_sink * Extnodes[no].sink_dp;
          Extnodes[no].sink_dp = {};
      }
#endif
      Nodes[no].u.d.bitflags &= (~(1 << BITFLAG_NODEHASBEENKICKED));
    }

    dt_drift = dt_drift_hmax = get_drift_factor(Nodes[no].Ti_current, time1, no, 1);
    

    Nodes[no].u.d.s += Extnodes[no].vs * dt_drift;
  Nodes[no].len += 2 * Extnodes[no].vmax * dt_drift;

#ifdef DM_SCALARFIELD_SCREENING
    Nodes[no].s_dm += Extnodes[no].vs_dm * dt_drift;
#endif
#ifdef SINK_NODE_MOTION_TRACKED
    /* else the sink COM stays frozen at its last-treebuild value while the sinks move, and the
       nearest-sink distance, sink timestep criteria, and (under SINGLE_STAR_DIRECT_GRAVITY) the
       monopole subtraction all read a stale position on a different clock from u.d.s. */
    Nodes[no].sink_pos += Nodes[no].sink_vel * dt_drift;
#endif


#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Nodes[no].rt_source_lum_s += Extnodes[no].rt_source_lum_vs * dt_drift;
#endif

    if(Extnodes[no].hmax > 0) {Extnodes[no].hmax *= exp(DMAX(-1.,DMIN(1.,Extnodes[no].divVmax * dt_drift_hmax / NUMDIMS)));}
    /* Mode B per-type bands: upward-only inflate. The bands
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
    /* Release store: publishes Ti_current after all geometry/Extnodes writes so a
     * threaded walk's acquire-load fast path sees fresh Ti => fresh geometry. */
    force_drift_node_publish_current(no, time1);
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
  /* Per-changed-topleaf exchange record: scalar hmax + divVmax + the 6 Mode-B
   * per-type bands.  The per-type slots ride this SAME post-density exchange so
   * remote topleaf/ancestor per-type bands are as fresh as the scalar hmax (they
   * were locally grown above but, without this, were only cross-rank-fresh at the
   * last full tree build/refresh).  Required for the Mode-B SYMMETRIC targeted
   * export band, which prunes remote topleaves by these per-type bands. */
  enum { HMAX_EXCH_HMAX = 0, HMAX_EXCH_DIVVMAX = 1, HMAX_EXCH_PTYPE0 = 2, HMAX_EXCH_SIZE = 8 };
  int OffsetSIZE = HMAX_EXCH_SIZE;
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

        /* Scalar `hmax`/`divVmax` eligibility (legacy semantics):
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
      domainHmax_loc[OffsetSIZE * i + HMAX_EXCH_HMAX]    = Extnodes[DomainList[i]].hmax;
      domainHmax_loc[OffsetSIZE * i + HMAX_EXCH_DIVVMAX] = Extnodes[DomainList[i]].divVmax;
      for(int t = 0; t < 6; t++)
          domainHmax_loc[OffsetSIZE * i + HMAX_EXCH_PTYPE0 + t] = Extnodes[DomainList[i]].hmax_per_type[t];
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

            /* Grow scalar hmax/divVmax + every per-type band; keep walking ancestors
             * while ANY of the 8 fields grew (a remote update that only grows one
             * per-type band must still propagate up the tree, just like scalar hmax). */
            int any_grew = 0;
            if(domainHmax_all[OffsetSIZE * i + HMAX_EXCH_HMAX] > Extnodes[no].hmax)
                {Extnodes[no].hmax = domainHmax_all[OffsetSIZE * i + HMAX_EXCH_HMAX]; any_grew = 1;}
            if(domainHmax_all[OffsetSIZE * i + HMAX_EXCH_DIVVMAX] > Extnodes[no].divVmax)
                {Extnodes[no].divVmax = domainHmax_all[OffsetSIZE * i + HMAX_EXCH_DIVVMAX]; any_grew = 1;}
            for(int t = 0; t < 6; t++)
            {
                MyFloat v = domainHmax_all[OffsetSIZE * i + HMAX_EXCH_PTYPE0 + t];
                if(v > Extnodes[no].hmax_per_type[t]) {Extnodes[no].hmax_per_type[t] = v; any_grew = 1;}
            }
            if(!any_grew) {break;}

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

  force_bump_hmax_refresh_generation();   /* ancestor boxes re-drifted + per-type bands re-seeded */
  CPU_Step[CPU_TREEHMAXUPDATE] += measure_time();
}
