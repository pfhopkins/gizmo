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

#ifdef TREE_MAXSOFT_COVERS_HMAX
/* The largest gas softening that kernel bound hmax admits. Gas softening is the kernel radius, so a node whose
   maxsoft holds this covers every gas member's softening, at its last drift and predicted to now, without
   re-deriving: the incremental refresh carries nodes between builds while their members' kernels change. */
static inline double gas_softening_bound(double hmax)
{
#ifdef ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT
    return DMIN(hmax, ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT/All.cf_atime);
#else
    return hmax;
#endif
}
#endif

/*! Raise node no's maxsoft to cover its kernel bound (TREE_MAXSOFT_COVERS_HMAX; else nothing). Called wherever
 *  hmax can grow past it: drift, a chain raise, a re-derivation from predicted kernels. */
void force_node_cover_soft(int no)
{
#ifdef TREE_MAXSOFT_COVERS_HMAX
    double s = gas_softening_bound(Extnodes[no].hmax);
    if(Nodes[no].maxsoft < s) {Nodes[no].maxsoft = s;}
#else
    (void) no;
#endif
}



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

    /* Raise every kicked member's motion bound before any node below is drifted: the drift widens len by
       2*vmax over the interval the members have just crossed at their post-kick speeds, and the kick loop
       drifts each ancestor when it first reaches it -- a node holding several kicked members would be
       widened at the speed of whichever came first, and len never shrinks to recover. */
    for(int i : ActiveParticleList)
    {
        MyFloat vmax_i = particle_motion_speed_bound(i);
        for(int no = Father[i]; no >= 0; no = Nodes[no].u.d.father)
        {
            if(Extnodes[no].vmax < vmax_i) {Extnodes[no].vmax = vmax_i;}
            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {break;}
        }
    }

#ifdef OPENMP_TREE_UPDATE
    /* Phase 1: drift all ancestor nodes to Ti_Current (serial — force_drift_node has complex read-modify-write) */
    for (int i : ActiveParticleList)
    {
        MyFloat vmax_i = particle_motion_speed_bound(i);
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
  int j, no; MyFloat vmax;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyDouble> rt_source_lum_dp;
    {double lum[N_RT_FREQ_BINS]; int active_check = rt_get_source_luminosity(i,-1,lum, P, CellP); rt_source_lum_dp = active_check ? dp : Vec3<MyDouble>{};}
#endif
#ifdef DM_SCALARFIELD_SCREENING
    Vec3<MyDouble> dp_dm = (P[i].Type != 0) ? dp : Vec3<MyDouble>{};
#endif
#ifdef SINK_NODE_MOTION_TRACKED
#if defined(STARS_INTEGRATED_BY_RUMI) || defined(STARS_HERMITE_DIRECT_GRAVITY)
    Vec3<MyDouble> sink_dp = (P[i].Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES) ? dp : Vec3<MyDouble>{}; /* the node moments track Type 4 here */
#else
    Vec3<MyDouble> sink_dp = (P[i].Type == 5) ? dp : Vec3<MyDouble>{}; /* only sinks contribute to the sink moments */
#endif
#endif

  vmax = particle_motion_speed_bound(i);

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
  /* share the momentum-data of the pseudo-particles accross CPUs: one record per changed top-level node, all
     fields in one Allgatherv */
  struct kick_node_rec
  {
    MyDouble dp[3];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    MyDouble rt_source_lum_dp[3];
#endif
#ifdef DM_SCALARFIELD_SCREENING
    MyDouble dp_dm[3];
#endif
#ifdef SINK_NODE_MOTION_TRACKED
    MyDouble sink_dp[3];
#endif
    MyFloat vmax;
    int no;
  };
  int i, j, no, ta, totDomainNumChanged;
  int *counts = (int *) mymalloc("counts", sizeof(int) * NTask), *offsets = (int *) mymalloc("offsets", sizeof(int) * NTask);
  struct kick_node_rec *loc = (struct kick_node_rec *) mymalloc("kick_node_loc", DomainNumChanged * sizeof(struct kick_node_rec));
  memset(loc, 0, DomainNumChanged * sizeof(struct kick_node_rec)); /* padding bytes travel too */
  for(i = 0; i < DomainNumChanged; i++)
    {
      no = DomainList[i]; loc[i].no = no; loc[i].vmax = Extnodes[no].vmax;
      for(j = 0; j < 3; j++)
	{
	  loc[i].dp[j] = Extnodes[no].dp[j];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
	  loc[i].rt_source_lum_dp[j] = Extnodes[no].rt_source_lum_dp[j];
#endif
#ifdef DM_SCALARFIELD_SCREENING
	  loc[i].dp_dm[j] = Extnodes[no].dp_dm[j];
#endif
#ifdef SINK_NODE_MOTION_TRACKED
	  loc[i].sink_dp[j] = Extnodes[no].sink_dp[j];
#endif
	}
    }

  MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
  for(ta = 0, totDomainNumChanged = 0; ta < NTask; ta++)
    {
      offsets[ta] = totDomainNumChanged * (int) sizeof(struct kick_node_rec);
      totDomainNumChanged += counts[ta];
      counts[ta] *= (int) sizeof(struct kick_node_rec);
    }

  PRINT_STATUS(" ..exchanged kick momenta for %d top-level nodes out of %d", totDomainNumChanged, NTopleaves);
  struct kick_node_rec *all = (struct kick_node_rec *) mymalloc("kick_node_all", totDomainNumChanged * sizeof(struct kick_node_rec));
  MPI_Allgatherv(loc, DomainNumChanged * (int) sizeof(struct kick_node_rec), MPI_BYTE, all, counts, offsets, MPI_BYTE, MPI_COMM_WORLD);

  /* raise the bounds of every changed chain before any of its nodes is drifted, as force_update_tree does locally */
  for(i = 0; i < totDomainNumChanged; i++)
    {
      no = all[i].no;
      if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT)) {no = Nodes[no].u.d.father;}
      for(; no >= 0; no = Nodes[no].u.d.father) {if(Extnodes[no].vmax < all[i].vmax) {Extnodes[no].vmax = all[i].vmax;}}
    }

  /* construct momentum kicks in top-level tree */
  for(i = 0; i < totDomainNumChanged; i++)
    {
      no = all[i].no;

      if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))	/* to avoid that the local one is kicked twice */
	no = Nodes[no].u.d.father;

      while(no >= 0)
	{
	  if(Extnodes[no].vmax < all[i].vmax) {Extnodes[no].vmax = all[i].vmax;} /* before the drift, as in force_kick_node */
	  force_drift_node(no, All.Ti_Current);

	  Extnodes[no].dp[0] += all[i].dp[0];
	  Extnodes[no].dp[1] += all[i].dp[1];
	  Extnodes[no].dp[2] += all[i].dp[2];
#ifdef RT_SEPARATELY_TRACK_LUMPOS
          Extnodes[no].rt_source_lum_dp[0] += all[i].rt_source_lum_dp[0];
          Extnodes[no].rt_source_lum_dp[1] += all[i].rt_source_lum_dp[1];
          Extnodes[no].rt_source_lum_dp[2] += all[i].rt_source_lum_dp[2];
#endif
#ifdef DM_SCALARFIELD_SCREENING
	  Extnodes[no].dp_dm[0] += all[i].dp_dm[0];
	  Extnodes[no].dp_dm[1] += all[i].dp_dm[1];
	  Extnodes[no].dp_dm[2] += all[i].dp_dm[2];
#endif
#ifdef SINK_NODE_MOTION_TRACKED
	  Extnodes[no].sink_dp[0] += all[i].sink_dp[0];
	  Extnodes[no].sink_dp[1] += all[i].sink_dp[1];
	  Extnodes[no].sink_dp[2] += all[i].sink_dp[2];
#endif

	  if(Extnodes[no].vmax < all[i].vmax)
	    Extnodes[no].vmax = all[i].vmax;

	  Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
	  Extnodes[no].Ti_lastkicked = All.Ti_Current;

	  no = Nodes[no].u.d.father;
	}
    }

  myfree(all);
  myfree(loc);
  myfree(offsets);
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

/*! Called at every tree build or restart-file tree load (live=1) and teardown (live=0): a build sets
 *  every bound afresh and renumbers the nodes, so anything pending from the old tree is void. */
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
#ifdef TREE_REFRESH_INCREMENTAL
        /* hmax is a bound at the node's Ti_current, grown from there by drift: the same value raised on a lagging copy
           is a larger bound than on a current one. Raise it on the node drifted to now, the time the exchanges ship it
           at, so every rank's copy holds the same bound (vmax before the drift, as in force_kick_node). */
        int vmax_held = (Extnodes[no].vmax >= vmax);
        if(!vmax_held) {Extnodes[no].vmax = vmax;}
        force_drift_node(no, All.Ti_Current);
        if(vmax_held && Extnodes[no].hmax >= hmax) {break;}
#else
        if(Extnodes[no].vmax >= vmax && Extnodes[no].hmax >= hmax) {break;}
        if(Extnodes[no].vmax < vmax) {Extnodes[no].vmax = vmax;}
#endif
        if(Extnodes[no].hmax < hmax) {Extnodes[no].hmax = hmax;}
#ifdef TREE_MAXSOFT_COVERS_HMAX
        force_node_cover_soft(no);
#endif
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
    MyFloat vmax = particle_motion_speed_bound(i);
    MyFloat hmax = (P[i].Type == 0) ? DMIN(P[i].KernelRadius, All.MaxKernelRadius) : 0;
    force_raise_chain_bounds(Father[i], vmax, hmax, 1); /* a particle inserted since the build has no node above it; the range check in there covers that */
}

/*! Thread-safe form: marks i, whose chain is then raised by force_raise_noted_motion_bounds. The
 *  loop that marked it must call that before anything reorders the particle arrays. */
void force_note_motion_change(int i)
{
    if(i < 0 || i >= (int) MotionChangeMark.size()) {return;}
#ifdef TREE_REFRESH_INCREMENTAL
    __atomic_fetch_or(&MotionChangeMark[i], (unsigned char) 1, __ATOMIC_RELAXED); /* bit 2 is force_note_mass_change's */
#else
    __atomic_store_n(&MotionChangeMark[i], (unsigned char) 1, __ATOMIC_RELAXED);
#endif
    __atomic_store_n(&MotionChangeMarkAny, 1, __ATOMIC_RELAXED);
}

/*! Thread-safe: particle i's mass changed where nothing requests a refresh (feedback coupling, partial accretion).
 *  The full refresh re-derives every node, so on the default path the next one accounts for it whatever triggers
 *  it; the incremental refresh re-derives only marked chains, so here i's chain is marked (and a refresh
 *  requested) by force_raise_noted_motion_bounds, which the loop that changed the mass must call before anything
 *  reorders the particle arrays. */
void force_note_mass_change(int i)
{
#ifdef TREE_REFRESH_INCREMENTAL
    if(i < 0 || i >= (int) MotionChangeMark.size()) {return;}
    __atomic_fetch_or(&MotionChangeMark[i], (unsigned char) 2, __ATOMIC_RELAXED);
    __atomic_store_n(&MotionChangeMarkAny, 1, __ATOMIC_RELAXED);
#else
    (void) i;
#endif
}

void force_raise_noted_motion_bounds(void)
{
    if(!MotionChangeMarkAny) {return;}
    int i, n = IMIN(NumPart, (int) MotionChangeMark.size());
#ifdef TREE_REFRESH_INCREMENTAL
    for(i = 0; i < n; i++)
    {
        unsigned char m = MotionChangeMark[i];
        if(!m) {continue;}
        MotionChangeMark[i] = 0;
        if(m & 1) {force_raise_motion_bound(i);}
        if(m & 2) {force_mark_moments_dirty(i);}
    }
#else
    for(i = 0; i < n; i++) {if(MotionChangeMark[i]) {MotionChangeMark[i] = 0; force_raise_motion_bound(i);}}
#endif
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
#ifdef TREE_REFRESH_INCREMENTAL
    for(int no : MotionTopPending) {force_drift_node(no, All.Ti_Current);} /* ship the bounds as of now: receivers apply them to their copies drifted to now */
#endif
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


/*! Node no's kernel bound hmax grown over a drift of dt. Its members may each have been drifted several
 *  times within that one interval, each drift capping only its own change (drift_divv_factor), so the
 *  product of their factors can exceed one capped factor over the whole interval: the node's growth by
 *  its largest member divergence divVmax (a maximum taken from zero, so it never shrinks the bound) is
 *  therefore uncapped. No drift takes a kernel past MaxKernelRadius, so max(hmax, MaxKernelRadius) is
 *  still a bound and keeps it finite. */
static double force_node_hmax_drift(int no, double dt)
{
    double hmax = Extnodes[no].hmax, x = Extnodes[no].divVmax * dt;
    if(!(hmax >= 0) || !(x >= 0))
    {
        printf("Task=%d node %d: kernel bound cannot be drifted (hmax=%g divVmax=%g dt=%g)\n", ThisTask, no, hmax, (double) Extnodes[no].divVmax, dt);
        fflush(stdout); endrun(90000103);
    }
    if(hmax <= 0) {return hmax;}
    double ceiling = DMAX(hmax, All.MaxKernelRadius), grown = hmax * exp(x / NUMDIMS);
    return (grown < ceiling) ? grown : ceiling;
}


void force_drift_node(int no, integertime time1)
{
  int j;
  integertime time0;
  double dt_drift, dt_widen, fac;

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

    dt_drift = get_drift_factor(Nodes[no].Ti_current, time1, no, 1);
    /* The bounds widen on the undilated clock: vmax carries each member's own dilation, so the node's
       dilated clock, right for its centre of mass, would under-grow them for a member less dilated than
       the node. The same interval when no dilation is active. */
    dt_widen = get_drift_factor(Nodes[no].Ti_current, time1, -1, 0);
    

    Nodes[no].u.d.s += Extnodes[no].vs * dt_drift;
  Nodes[no].len += 2 * Extnodes[no].vmax * dt_widen;

#ifdef DM_SCALARFIELD_SCREENING
    Nodes[no].s_dm += Extnodes[no].vs_dm * dt_drift;
#endif

#ifdef SINK_NODE_MOTION_TRACKED
    Nodes[no].sink_pos += Nodes[no].sink_vel * dt_drift; /* else the sink COM stays frozen at its last-treebuild value while the sinks move, and the nearest-sink distance and sink timestep criteria read a stale position */
#endif


#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Nodes[no].rt_source_lum_s += Extnodes[no].rt_source_lum_vs * dt_drift;
#endif

    Extnodes[no].hmax = force_node_hmax_drift(no, dt_widen);
#ifdef TREE_MAXSOFT_COVERS_HMAX
    force_node_cover_soft(no); /* the walk reads maxsoft right after drifting the node, and a drifted leaf member's grown kernel as its softening */
#endif
    Nodes[no].Ti_current = time1;
}


#ifdef TREE_INTEGRITY_AUDITS
/*! The drifted fields of node no as force_drift_node would leave them at time1, without changing the
 *  node: lets an audit read lazily drifted nodes at a common time without perturbing the run. */
void force_node_predict(int no, integertime time1, struct node_prediction *np)
{
    np->s = Nodes[no].u.d.s; np->vs = Extnodes[no].vs; np->len = Nodes[no].len; np->hmax = Extnodes[no].hmax; np->maxsoft = Nodes[no].maxsoft;
    np->sink_pos = {}; np->sink_vel = {};
#ifdef SINK_CALC_DISTANCES
    np->sink_pos = Nodes[no].sink_pos;
#ifdef SINK_NODE_MOTION_TRACKED
    np->sink_vel = Nodes[no].sink_vel;
#endif
#endif
    if(Nodes[no].u.d.bitflags & (1 << BITFLAG_NODEHASBEENKICKED)) /* a kick received at Ti_current is pending until the next drift, but it is already the velocity the node moves with */
    {
        double fac = Nodes[no].u.d.mass ? 1 / Nodes[no].u.d.mass : 0;
        np->vs += fac * Extnodes[no].dp;
#ifdef SINK_NODE_MOTION_TRACKED
        double fac_sink = Nodes[no].sink_mass ? 1 / Nodes[no].sink_mass : 0;
        np->sink_vel += fac_sink * Extnodes[no].sink_dp;
#endif
    }
    if(time1 == Nodes[no].Ti_current) {return;}
    double dt_drift = get_drift_factor(Nodes[no].Ti_current, time1, no, 1), dt_widen = get_drift_factor(Nodes[no].Ti_current, time1, -1, 0);
    np->s += np->vs * dt_drift;
    np->len += 2 * Extnodes[no].vmax * dt_widen;
#ifdef SINK_NODE_MOTION_TRACKED
    np->sink_pos += np->sink_vel * dt_drift;
#endif
    np->hmax = force_node_hmax_drift(no, dt_widen);
#ifdef TREE_MAXSOFT_COVERS_HMAX
    np->maxsoft = DMAX(np->maxsoft, gas_softening_bound(np->hmax));
#endif
}
#endif





/* members whose kernel or softening force_update_hmax carries up the tree: under TREE_REFRESH_INCREMENTAL every
   member with mass, since carried nodes keep maxsoft between builds and a sink's kernel is part of its softening */
#if defined(TREE_REFRESH_INCREMENTAL) || defined(ADAPTIVE_GRAVSOFT_FORALL)
#define HMAX_UPDATE_MEMBER(i) (P[i].Mass > 0)
#else
#define HMAX_UPDATE_MEMBER(i) (P[i].Type == 0 && P[i].Mass > 0)
#endif

/*! This function updates the hmax-values in tree nodes that hold gas cells. These values are needed to find all neighbors in the
 *  hydro-force computation.  Since the KernelRadius-values are potentially changed in the fluid-density computation, force_update_hmax() should be carried
 *  out just before the hydrodynamical forces are computed, i.e. after density(). */
void force_update_hmax(void)
{
  int i, no, ta, totDomainNumChanged;
  int *counts, *offsets;
  double divVel;

  GlobFlag++;

  DomainNumChanged = 0;
  DomainList = (int *) mymalloc("DomainList", NTopleaves * sizeof(int));

#ifdef OPENMP_TREE_UPDATE
  /* Phase 1: drift all ancestor nodes (serial — force_drift_node is not thread-safe) */
  for (int i : ActiveParticleList)
  {
    if(HMAX_UPDATE_MEMBER(i))
      {
        no = Father[i];
        while(no >= 0)
        {
#ifdef TREE_REFRESH_INCREMENTAL
            /* up to the top-level node, whose bounds phase 2 raises and ships as of now: a current node does not
               imply current ancestors */
            if(Nodes[no].Ti_current != All.Ti_Current) {force_drift_node(no, All.Ti_Current);}
            if(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {break;}
#else
            if(Nodes[no].Ti_current == All.Ti_Current) {break;}
            force_drift_node(no, All.Ti_Current);
#endif
            no = Nodes[no].u.d.father;
        }
      }
  }
  /* Phase 2: update hmax/divVmax with atomics (parallel) */
#pragma omp parallel for schedule(dynamic)
  for (int idx = 0; idx < (int)ActiveParticleList.size(); idx++)
  {
    int i = ActiveParticleList[idx];
    if(HMAX_UPDATE_MEMBER(i))
      {
        int no = Father[i];
        double divVel = P[i].Particle_DivVel;
#ifdef TREE_REFRESH_INCREMENTAL
        double soft_i = force_member_softening(i);
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
        if(P[i].Type != 0) {divVel = 0;}
#endif
#endif

        while(no >= 0)
        {
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
            double htmp = DMIN(P[i].AGS_KernelRadius, All.MaxKernelRadius);
#else
            double htmp = DMIN(P[i].KernelRadius, All.MaxKernelRadius);
#endif
#ifdef TREE_REFRESH_INCREMENTAL
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
            if(P[i].Type != 0) {htmp = 0;}
#endif
            if(htmp > Extnodes[no].hmax || divVel > Extnodes[no].divVmax || soft_i > Nodes[no].maxsoft)
#else
            if(htmp > Extnodes[no].hmax || divVel > Extnodes[no].divVmax)
#endif
            {
                atomic_max_double(&Extnodes[no].hmax, htmp);
                atomic_max_double(&Extnodes[no].divVmax, divVel);
#ifdef TREE_REFRESH_INCREMENTAL
                atomic_max_double(&Nodes[no].maxsoft, soft_i); /* covers the raised hmax too: a gas member's softening is its kernel */
#endif

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
    if(HMAX_UPDATE_MEMBER(i))
      {
        no = Father[i];
        divVel = P[i].Particle_DivVel;
#ifdef TREE_REFRESH_INCREMENTAL
        double soft_i = force_member_softening(i);
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
        if(P[i].Type != 0) {divVel = 0;}
#endif
#endif

        while(no >= 0)
        {
            force_drift_node(no, All.Ti_Current);
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
            double htmp = DMIN(P[i].AGS_KernelRadius, All.MaxKernelRadius);
#else
            double htmp = DMIN(P[i].KernelRadius, All.MaxKernelRadius);
#endif
#ifdef TREE_REFRESH_INCREMENTAL
#if !defined(ADAPTIVE_GRAVSOFT_FORALL)
            if(P[i].Type != 0) {htmp = 0;}
#endif
            if(htmp > Extnodes[no].hmax || divVel > Extnodes[no].divVmax || soft_i > Nodes[no].maxsoft)
#else
            if(htmp > Extnodes[no].hmax || divVel > Extnodes[no].divVmax)
#endif
            {
                if(htmp > Extnodes[no].hmax) {Extnodes[no].hmax = htmp;}
                if(divVel > Extnodes[no].divVmax) {Extnodes[no].divVmax = divVel;}
#ifdef TREE_REFRESH_INCREMENTAL
                if(soft_i > Nodes[no].maxsoft) {Nodes[no].maxsoft = soft_i;} /* covers the raised hmax too: a gas member's softening is its kernel */
#endif

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

  /* share the hmax-data of the pseudo-particles accross CPUs: node index and both bounds in one Allgatherv */
#ifdef TREE_REFRESH_INCREMENTAL
  struct hmax_rec {MyFloat hmax, divVmax, maxsoft; int no;};
#else
  struct hmax_rec {MyFloat hmax, divVmax; int no;};
#endif
  counts = (int *) mymalloc("counts", sizeof(int) * NTask);
  offsets = (int *) mymalloc("offsets", sizeof(int) * NTask);
  struct hmax_rec *loc = (struct hmax_rec *) mymalloc("hmax_loc", DomainNumChanged * sizeof(struct hmax_rec));
  memset(loc, 0, DomainNumChanged * sizeof(struct hmax_rec));
  for(i = 0; i < DomainNumChanged; i++) {loc[i].no = DomainList[i]; loc[i].hmax = Extnodes[DomainList[i]].hmax; loc[i].divVmax = Extnodes[DomainList[i]].divVmax;}
#ifdef TREE_REFRESH_INCREMENTAL
  for(i = 0; i < DomainNumChanged; i++) {loc[i].maxsoft = Nodes[DomainList[i]].maxsoft;}
#endif

  MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
  for(ta = 0, totDomainNumChanged = 0; ta < NTask; ta++)
    {
      offsets[ta] = totDomainNumChanged * (int) sizeof(struct hmax_rec);
      totDomainNumChanged += counts[ta];
      counts[ta] *= (int) sizeof(struct hmax_rec);
    }

  PRINT_STATUS(" ..Hmax exchange: %d topleaves out of %d", totDomainNumChanged, NTopleaves);
  struct hmax_rec *all = (struct hmax_rec *) mymalloc("hmax_all", totDomainNumChanged * sizeof(struct hmax_rec));
  MPI_Allgatherv(loc, DomainNumChanged * (int) sizeof(struct hmax_rec), MPI_BYTE, all, counts, offsets, MPI_BYTE, MPI_COMM_WORLD);

  for(i = 0; i < totDomainNumChanged; i++)
    {
        no = all[i].no;
        if(Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))    {no = Nodes[no].u.d.father;} /* to avoid that the hmax is updated twice */
        
        while(no >= 0)
        {
            force_drift_node(no, All.Ti_Current);
            
#ifdef TREE_REFRESH_INCREMENTAL
            if(all[i].hmax > Extnodes[no].hmax || all[i].divVmax > Extnodes[no].divVmax || all[i].maxsoft > Nodes[no].maxsoft)
#else
            if(all[i].hmax > Extnodes[no].hmax || all[i].divVmax > Extnodes[no].divVmax)
#endif
            {
                if(all[i].hmax > Extnodes[no].hmax) {Extnodes[no].hmax = all[i].hmax;}
                if(all[i].divVmax > Extnodes[no].divVmax) {Extnodes[no].divVmax = all[i].divVmax;}
#ifdef TREE_REFRESH_INCREMENTAL
                if(all[i].maxsoft > Nodes[no].maxsoft) {Nodes[no].maxsoft = all[i].maxsoft;} /* the owner's leaf covers its raised hmax, so this covers the copy's */
#endif
            }
            else
            {break;}
            
            no = Nodes[no].u.d.father;
        }
    }


  myfree(all);
  myfree(loc);
  myfree(offsets);
  myfree(counts);
  myfree(DomainList);

  CPU_Step[CPU_TREEHMAXUPDATE] += measure_time();
}
