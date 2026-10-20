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

#ifdef OPENMP_TREE_UPDATE
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
#endif



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
    force_validate_tree_links("update_tree"); /* tier-0: no-op unless a maintained rearrange changed the threading */
    PRINT_STATUS("Kick-subroutine will prepare for dynamic update of tree");
    int i, j; GlobFlag++; DomainNumChanged = 0; DomainList = (int *) mymalloc("DomainList", NTopleaves * sizeof(int));
    /* note: the current list of active particles still refers to that synchronized at the previous time. */

#ifdef OPENMP_TREE_UPDATE
    /* Phase 1: drift all ancestor nodes to Ti_Current (serial — force_drift_node has complex read-modify-write) */
    for (int i : ActiveParticleList)
    {
        MyFloat vmax_i = 0; for(int k = 0; k < 3; k++) {vmax_i = DMAX(vmax_i, fabs(P[i].Vel[k]));}
        int no = Father[i];
        while(no >= 0)
        {
            if(Nodes[no].Ti_current == All.Ti_Current) {break;} /* already drifted; ancestors above will be too since they were drifted first */
            if(Extnodes[no].vmax < vmax_i) {Extnodes[no].vmax = vmax_i;} /* before the drift; phase 2 repeats it harmlessly */
            force_drift_node(no, All.Ti_Current);
            no = Nodes[no].u.d.father;
        }
    }
    /* Phase 2: accumulate kicks in parallel (all nodes already drifted, so only atomic accumulation needed) */
#pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < (int)ActiveParticleList.size(); idx++)
    {
        int i = ActiveParticleList[idx];
        force_kick_node(i, P[i].dp);
        P[i].dp = {};
    }
#else
    for (int i : ActiveParticleList)
    {
        force_kick_node(i, P[i].dp);
        P[i].dp = {};
    }
#endif
    force_finish_kick_nodes();
    myfree(DomainList);
    PRINT_STATUS(" ..Tree has been updated dynamically");
}


void force_kick_node(int i, Vec3<MyDouble>& dp)
{
  int j, no; MyFloat v, vmax;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyDouble> rt_source_lum_dp;
    {double lum[N_RT_FREQ_BINS]; int active_check = rt_get_source_luminosity(i,-1,lum, P, CellP); rt_source_lum_dp = active_check ? dp : Vec3<MyDouble>{};}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Vec3<MyDouble> dp_dm = (P[i].Type != 0) ? dp : Vec3<MyDouble>{};
#endif
#ifdef SINK_NODE_MOTION_TRACKED
    Vec3<MyDouble> sink_dp = (P[i].Type == 5) ? dp : Vec3<MyDouble>{}; /* only sinks contribute to the sink moments */
#endif

  for(j = 0, vmax = 0; j < 3; j++)
    if((v = fabs(P[i].Vel[j])) > vmax)
      vmax = v;

  no = Father[i];

  while(no >= 0)
    {
#ifndef OPENMP_TREE_UPDATE
      /* Raise before the drift, not after it: the drift below grows len by 2*vmax*dt over the
         interval this particle has just been drifted across at its post-kick speed, so a bound
         raised afterwards leaves that one interval covered at the old speed -- and len never
         shrinks, so the deficit it leaves behind is permanent. */
      if(Extnodes[no].vmax < vmax) {Extnodes[no].vmax = vmax;}
      force_drift_node(no, All.Ti_Current);
#endif

#ifdef OPENMP_TREE_UPDATE
      for(int k = 0; k < 3; k++) {
          #pragma omp atomic
          Extnodes[no].dp[k] += dp[k];
      }
#ifdef RT_SEPARATELY_TRACK_LUMPOS
      for(int k = 0; k < 3; k++) {
          #pragma omp atomic
          Extnodes[no].rt_source_lum_dp[k] += rt_source_lum_dp[k];
      }
#endif
#ifdef DM_SCALARFIELD_SCREENING
      for(int k = 0; k < 3; k++) {
          #pragma omp atomic
          Extnodes[no].dp_dm[k] += dp_dm[k];
      }
#endif
#ifdef SINK_NODE_MOTION_TRACKED
      for(int k = 0; k < 3; k++) {
          #pragma omp atomic
          Extnodes[no].sink_dp[k] += sink_dp[k];
      }
#endif
      atomic_max_double(&Extnodes[no].vmax, vmax);
      Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
      Extnodes[no].Ti_lastkicked = All.Ti_Current;
      if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))
      {
          #pragma omp critical(DomainListAppend)
          {
              if(Extnodes[no].Flag != GlobFlag)
              {
                  Extnodes[no].Flag = GlobFlag;
                  DomainList[DomainNumChanged++] = no;
              }
          }
          break;
      }
#else
        Extnodes[no].dp += dp;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        Extnodes[no].rt_source_lum_dp += rt_source_lum_dp;
#endif
#ifdef DM_SCALARFIELD_SCREENING
        Extnodes[no].dp_dm += dp_dm;
#endif
#ifdef SINK_NODE_MOTION_TRACKED
        Extnodes[no].sink_dp += sink_dp;
#endif
      Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
      Extnodes[no].Ti_lastkicked = All.Ti_Current;
      if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))
	{
	  if(Extnodes[no].Flag != GlobFlag)
	    {
	      Extnodes[no].Flag = GlobFlag;
	      DomainList[DomainNumChanged++] = no;
	    }
	  break;
	}
#endif

      no = Nodes[no].u.d.father;
    }
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
#ifdef SINK_NODE_MOTION_TRACKED
  MyDouble *domainDp_sink_loc, *domainDp_sink_all;
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
#ifdef SINK_NODE_MOTION_TRACKED
  domainDp_sink_loc = (MyDouble *) mymalloc("domainDp_sink_loc", DomainNumChanged * 3 * sizeof(MyDouble)); /* last in its group, so freed first below: keeps the stack allocator LIFO whatever else is enabled */
#endif

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
#ifdef SINK_NODE_MOTION_TRACKED
	  domainDp_sink_loc[i * 3 + j] = Extnodes[DomainList[i]].sink_dp[j];
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
#ifdef SINK_NODE_MOTION_TRACKED
  domainDp_sink_all = (MyDouble *) mymalloc("domainDp_sink_all", totDomainNumChanged * 3 * sizeof(MyDouble)); /* likewise last, freed first */
#endif

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
#ifdef SINK_NODE_MOTION_TRACKED
  MPI_Allgatherv(domainDp_sink_loc, DomainNumChanged * 3 * sizeof(MyDouble), MPI_BYTE,
		 domainDp_sink_all, counts_dp, offset_dp, MPI_BYTE, MPI_COMM_WORLD); /* same 3*MyDouble shape as dp, so it reuses counts_dp/offset_dp */
#endif

  MPI_Allgatherv(domainVmax_loc, DomainNumChanged * sizeof(MyFloat), MPI_BYTE,
		 domainVmax_all, counts, offset_vmax, MPI_BYTE, MPI_COMM_WORLD);


  /* construct momentum kicks in top-level tree */
  for(i = 0; i < totDomainNumChanged; i++)
    {
      no = domainList_all[i];

      if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))	/* to avoid that the local one is kicked twice */
	no = Nodes[no].u.d.father;

      while(no >= 0)
	{
	  if(Extnodes[no].vmax < domainVmax_all[i]) {Extnodes[no].vmax = domainVmax_all[i];} /* before the drift, as in force_kick_node */
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
#ifdef SINK_NODE_MOTION_TRACKED
	  Extnodes[no].sink_dp[0] += domainDp_sink_all[3 * i + 0];
	  Extnodes[no].sink_dp[1] += domainDp_sink_all[3 * i + 1];
	  Extnodes[no].sink_dp[2] += domainDp_sink_all[3 * i + 2];
#endif

	  if(Extnodes[no].vmax < domainVmax_all[i])
	    Extnodes[no].vmax = domainVmax_all[i];

	  Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
	  Extnodes[no].Ti_lastkicked = All.Ti_Current;

	  no = Nodes[no].u.d.father;
	}
    }

#ifdef SINK_NODE_MOTION_TRACKED
  myfree(domainDp_sink_all);
#endif
  myfree(domainList_all);
  myfree(domainVmax_all);
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    myfree(domainDp_stellarlum_all);
#endif
#ifdef DM_SCALARFIELD_SCREENING
  myfree(domainDp_dm_all);
#endif
  myfree(domainDp_all);
#ifdef SINK_NODE_MOTION_TRACKED
  myfree(domainDp_sink_loc);
#endif
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



/* ---- Motion bounds raised outside the sync-point kick ----
 * force_drift_node grows a node's box by 2*vmax*dt, so a box holds its members only while none of
 * them moves faster than vmax. The sync-point kick raises vmax for the particles it kicks and for
 * nothing else, so a velocity changed anywhere else -- a neighbour loop's kick, accretion, a
 * merger, a newly inserted element -- must raise the bound along that particle's whole ancestor
 * chain before any of those nodes drifts past the change. Raising early is safe: it only
 * over-covers the part of a node's pending drift that precedes the change.
 * Top-level nodes are held by every rank, and the kick's own exchange cannot carry these raises
 * (it runs before the loops that make them), so the top-level nodes reached are listed and reach
 * the other ranks at the next force_flush_motion_bounds. A tree build voids the list. */
static std::vector<unsigned char> MotionChangeMark;  /* per particle: velocity changed in a threaded loop, chain not raised yet */
static int MotionChangeMarkAny = 0;
static std::vector<int> MotionTopStamp;              /* per node: == MotionTopGen while listed below */
static std::vector<int> MotionTopPending;            /* top-level nodes raised since the last flush */
static int MotionTopGen = 1;
static int MotionBoundTreeLive = 0;

/*! Called at every tree build (live=1) and teardown (live=0): a build sets every bound afresh and
 *  renumbers the nodes, so anything pending from the old tree is void. */
void force_motion_bounds_reset(int tree_live)
{
    MotionBoundTreeLive = tree_live;
    MotionChangeMark.assign(tree_live ? All.MaxPart : 0, 0); MotionChangeMarkAny = 0;
    MotionTopStamp.assign(tree_live ? Numnodestree : 0, 0); MotionTopPending.clear(); MotionTopGen = 1;
}

/*! Raise vmax (and, for a gas member, hmax) from node no up to the root, stopping at a node that
 *  already holds both: every ancestor of such a node holds them too. */
static void force_raise_chain_bounds(int no, MyFloat vmax, MyFloat hmax, int list_toplevel)
{
    for(; no >= All.MaxPart && no < All.MaxPart + Numnodestree; no = Nodes[no].u.d.father)
    {
        if(Extnodes[no].vmax >= vmax && Extnodes[no].hmax >= hmax) {break;}
        if(Extnodes[no].vmax < vmax) {Extnodes[no].vmax = vmax;}
        if(Extnodes[no].hmax < hmax) {Extnodes[no].hmax = hmax;}
        if(list_toplevel && (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
        {
            int t = no - All.MaxPart;
            if(t < (int) MotionTopStamp.size() && MotionTopStamp[t] != MotionTopGen) {MotionTopStamp[t] = MotionTopGen; MotionTopPending.push_back(no);}
        }
    }
}

/*! Particle i's velocity (or, for gas, its kernel) changed outside the sync-point kick: raise the
 *  bounds above it now. Serial callers only; from inside a threaded loop use
 *  force_note_motion_change, which defers the chain walk. */
void force_raise_motion_bound(int i)
{
    if(!MotionBoundTreeLive || i < 0 || i >= All.MaxPart) {return;}
    MyFloat vmax = 0; int k; for(k = 0; k < 3; k++) {vmax = DMAX(vmax, fabs(P[i].Vel[k]));}
    MyFloat hmax = (P[i].Type == 0) ? DMIN(P[i].KernelRadius, All.MaxKernelRadius) : 0;
    force_raise_chain_bounds(Father[i], vmax, hmax, 1); /* a particle inserted since the build has no node above it; the range check in there covers that */
}

/*! Thread-safe form: marks i, whose chain is then raised by force_raise_noted_motion_bounds. The
 *  loop that marked it must call that before anything reorders the particle arrays. */
void force_note_motion_change(int i)
{
    if(i < 0 || i >= (int) MotionChangeMark.size()) {return;}
    __atomic_store_n(&MotionChangeMark[i], (unsigned char) 1, __ATOMIC_RELAXED);
    __atomic_store_n(&MotionChangeMarkAny, 1, __ATOMIC_RELAXED);
}

void force_raise_noted_motion_bounds(void)
{
    if(!MotionChangeMarkAny) {return;}
    int i, n = IMIN(NumPart, (int) MotionChangeMark.size());
    for(i = 0; i < n; i++) {if(MotionChangeMark[i]) {MotionChangeMark[i] = 0; force_raise_motion_bound(i);}}
    MotionChangeMarkAny = 0;
    if((int) MotionChangeMark.size() > n) {memset(MotionChangeMark.data() + n, 0, MotionChangeMark.size() - n);} /* slots vacated by a rearrange */
}

/*! Collective, entered by every rank unconditionally: every top-level node any rank raised since
 *  the last flush has its (vmax, hmax) applied as a maximum up every rank's copy of the chain.
 *  Nothing else in the node moves -- no momentum, no kick flag, no timestamp. Runs before the
 *  step's first node drift (see run()); a raise made later in a step reaches this rank's own tree
 *  at once and the other ranks' copies at the next flush. */
void force_flush_motion_bounds(void)
{
    force_raise_noted_motion_bounds();
    struct MotionBoundRecord {int no; MyFloat vmax, hmax;};
    std::vector<MotionBoundRecord> loc;
    loc.reserve(MotionTopPending.size());
    for(int no : MotionTopPending) {loc.push_back({no, Extnodes[no].vmax, Extnodes[no].hmax});}
    MotionTopPending.clear(); MotionTopGen++;
    if(MotionTopGen == 0) {MotionTopGen = 1; MotionTopStamp.assign(MotionTopStamp.size(), 0);}
    int nbytes = (int) (loc.size() * sizeof(MotionBoundRecord)), nbytes_tot = 0, ta;
    std::vector<int> counts(NTask), offsets(NTask);
    MPI_Allgather(&nbytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for(ta = 0; ta < NTask; ta++) {offsets[ta] = nbytes_tot; nbytes_tot += counts[ta];}
    if(nbytes_tot == 0) {return;}
    std::vector<MotionBoundRecord> all(nbytes_tot / sizeof(MotionBoundRecord));
    MPI_Allgatherv(loc.data(), nbytes, MPI_BYTE, all.data(), counts.data(), offsets.data(), MPI_BYTE, MPI_COMM_WORLD);
    for(const MotionBoundRecord &r : all)
    {
        /* a record names a top-level node, which every rank holds at the same index; validated
           rather than trusted, as the kick exchange does */
        if(r.no < All.MaxPart || r.no >= All.MaxPart + Numnodestree) {continue;}
        if(!(Nodes[r.no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))) {continue;}
        force_raise_chain_bounds(r.no, r.vmax, r.hmax, 0);
    }
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
	  terminate("inconsistency in drift node");
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

#ifdef SINK_NODE_MOTION_TRACKED
      double fac_sink;
      if(Nodes[no].sink_mass) {fac_sink = 1 / Nodes[no].sink_mass;} else {fac_sink = 0;}
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
      Nodes[no].sink_vel += fac_sink * Extnodes[no].sink_dp; /* sink_vel lives in Nodes rather than Extnodes, but is updated exactly as vs/vs_dm are */
      Extnodes[no].sink_dp = {};
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
    Nodes[no].sink_pos += Nodes[no].sink_vel * dt_drift; /* else the sink COM stays frozen at its last-treebuild value while the sinks move, and the nearest-sink distance and sink timestep criteria read a stale position */
#endif


#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Nodes[no].rt_source_lum_s += Extnodes[no].rt_source_lum_vs * dt_drift;
#endif

    if(Extnodes[no].hmax > 0) {Extnodes[no].hmax *= exp(DMAX(-1.,DMIN(1.,Extnodes[no].divVmax * dt_drift_hmax / NUMDIMS)));}
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

#ifdef OPENMP_TREE_UPDATE
  /* Phase 1: drift all ancestor nodes (serial — force_drift_node is not thread-safe) */
  for (int i : ActiveParticleList)
  {
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if(P[i].Mass > 0)
#else
    if(P[i].Type == 0 && P[i].Mass > 0)
#endif
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
  /* Phase 2: update hmax/divVmax with atomics (parallel) */
#pragma omp parallel for schedule(dynamic)
  for (int idx = 0; idx < (int)ActiveParticleList.size(); idx++)
  {
    int i = ActiveParticleList[idx];
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if(P[i].Mass > 0)
#else
    if(P[i].Type == 0 && P[i].Mass > 0)
#endif
      {
        int no = Father[i];
        double divVel = P[i].Particle_DivVel;

        while(no >= 0)
        {
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
            double htmp = DMIN(P[i].AGS_KernelRadius, All.MaxKernelRadius);
#else
            double htmp = DMIN(P[i].KernelRadius, All.MaxKernelRadius);
#endif
            if(htmp > Extnodes[no].hmax || divVel > Extnodes[no].divVmax)
            {
                atomic_max_double(&Extnodes[no].hmax, htmp);
                atomic_max_double(&Extnodes[no].divVmax, divVel);

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
#else
  for (int i : ActiveParticleList)
  {
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if(P[i].Mass > 0)
#else
    if(P[i].Type == 0 && P[i].Mass > 0)
#endif
      {
        no = Father[i];
        divVel = P[i].Particle_DivVel;

        while(no >= 0)
        {
            force_drift_node(no, All.Ti_Current);
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
            double htmp = DMIN(P[i].AGS_KernelRadius, All.MaxKernelRadius);
#else
            double htmp = DMIN(P[i].KernelRadius, All.MaxKernelRadius);
#endif
            if(htmp > Extnodes[no].hmax || divVel > Extnodes[no].divVmax)
            {
                if(htmp > Extnodes[no].hmax) {Extnodes[no].hmax = htmp;}
                if(divVel > Extnodes[no].divVmax) {Extnodes[no].divVmax = divVel;}

                if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))
                {
                    if(Extnodes[no].Flag != GlobFlag)
                    {
                        Extnodes[no].Flag = GlobFlag;
                        DomainList[DomainNumChanged++] = no;
                    }
                    break;
                }
            }
            else
                break;

            no = Nodes[no].u.d.father;
        }
      }
  } // for (int i : ActiveParticleList)
#endif

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
