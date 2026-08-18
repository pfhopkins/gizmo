/** \file
    Runtime memory ledger -- node-scoped reporting of per-family memory use, so the
    true per-node footprint is visible rather than only the central Base arena.
    Node-scoped sums use the persistent node-local communicator (GizmoNodeComm);
    each node's lead task prints one line.
*/

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* Per-family byte counters, updated at each family's own allocation seam. These are
   rank-local running totals of logical (requested) bytes; the report sums them across
   the node. Allocation seams are outside OpenMP-parallel regions, so plain integer
   updates are sufficient. */
static long long g_family_bytes[GIZMO_MEM_NFAMILY] = {0};
static int g_family_neg_warned[GIZMO_MEM_NFAMILY] = {0};
static int g_badid_warned = 0;

/* A bad family id or a negative running total is an instrumentation bug (an alloc
   seam left unaccounted, or a free double-counted): surface it loudly once rather
   than let the ledger report a plausible wrong number. The total is clamped to 0 so
   the printed value is never negative. */
static void mem_ledger_flag_bad_family(int family)
{
    if(!g_badid_warned) {g_badid_warned = 1;
        printf("MEMORY LEDGER WARNING: bad allocation-family id %d (task=%d); update ignored.\n", family, ThisTask);
        fflush(stdout);}
}

static void mem_ledger_clamp_if_negative(int family)
{
    if(g_family_bytes[family] < 0) {
        if(!g_family_neg_warned[family]) {g_family_neg_warned[family] = 1;
            printf("MEMORY LEDGER WARNING: family %d total went negative (%lld B) on task=%d -- freed more than tracked; clamping to 0.\n",
                   family, g_family_bytes[family], ThisTask);
            fflush(stdout);}
        g_family_bytes[family] = 0;
    }
}

void gizmo_mem_account_add(int family, long long delta_bytes)
{
    if(family < 0 || family >= GIZMO_MEM_NFAMILY) {mem_ledger_flag_bad_family(family); return;}
    g_family_bytes[family] += delta_bytes;
    mem_ledger_clamp_if_negative(family);
}

void gizmo_mem_account_set(int family, long long value_bytes)
{
    if(family < 0 || family >= GIZMO_MEM_NFAMILY) {mem_ledger_flag_bad_family(family); return;}
    g_family_bytes[family] = value_bytes;
    mem_ledger_clamp_if_negative(family);
}

/* LET wire transport buffers are transient (allocated during an exchange, freed at
   its end), so they are tracked as a HIGH-WATER rather than a persistent current-byte
   family: grow() raises the running total and the peak; reset() zeroes the running
   total after the whole exchange's buffers are freed; note_failed() records bytes a
   realloc could not satisfy. LET packing is serial per rank, so plain counters suffice. */
static long long g_let_wire_current = 0, g_let_wire_highwater = 0, g_let_wire_failed = 0;

void gizmo_let_wire_grow(long long delta_bytes)
{
    g_let_wire_current += delta_bytes;
    if(g_let_wire_current > g_let_wire_highwater) {g_let_wire_highwater = g_let_wire_current;}
}
void gizmo_let_wire_reset(void)            {g_let_wire_current = 0;}
void gizmo_let_wire_note_failed(long long bytes) {g_let_wire_failed += bytes;}

/* Per-process virtual and resident size from /proc/self/status (Linux; 0 elsewhere).
   VIRTUAL/committed (VmSize) is what mmap reserved -- the quantity strict-overcommit
   nodes (Frontera) fail on -- and can far exceed RESIDENT (VmRSS), the pages actually
   touched. The family counters above are LOGICAL requested bytes; these two are the
   commit/physical categories the Base-reserved-vs-used divergence lives in. */
static void read_self_vm_kb(long long *vmsize_kb, long long *vmrss_kb)
{
    *vmsize_kb = 0; *vmrss_kb = 0;
    FILE *fd = fopen("/proc/self/status", "r");
    if(!fd) {return;}
    char buf[256];
    while(fgets(buf, sizeof(buf), fd)) {
        if(strncmp(buf, "VmSize:", 7) == 0) {*vmsize_kb = atoll(buf + 7);}
        else if(strncmp(buf, "VmRSS:", 6) == 0) {*vmrss_kb = atoll(buf + 6);}
    }
    fclose(fd);
}

/* Shared body. This is an extra-run-info diagnostic: without OUTPUT_ADDITIONAL_RUNINFO it does
   nothing at all -- no collectives, no output, no cost -- so a production run never pays for it
   and never has its stdout diluted by it.

   With the flag on, the node-scoped reduces are collective and ALWAYS run on every rank (safe
   only at symmetric all-rank call points); when `always` is 0 the report is emitted only if some
   node's memory footprint has grown since the last one, so a per-domain-decomposition call does
   not spam. Growth is decided AFTER the reduce -- never gate the collective itself.

   ONE report is emitted per call, describing the node using the most memory, because node memory
   is the binding pool and the worst node is the one that decides whether the run fits. Every node
   lead still tracks its own history so the growth test stays per-node; the spread across nodes
   rides the header rather than costing a block each.

   Reading the figures: every high-water is the SUM of per-rank peaks, so it is a conservative
   upper bound rather than a time-coincident node peak. The pool figures are logical requested
   bytes; "commit" is what the OS sees. "kokkos" is a SUPERSET of the pools -- never add it to
   them; its residual against them is neighbour/ghost/scratch plus Kokkos internal caching. */
static void report_memory_ledger_impl(const char *when, int always)
{
#ifndef OUTPUT_ADDITIONAL_RUNINFO
    (void) when; (void) always;
#else
    gizmo_node_comm_init();

    /* Base arena: size reserved per task vs. the central allocator's high-water use. */
    double base_reserved_mb  = (double) All.MaxMemSize;
    double base_highwater_mb = HighMarkBytes / (1024.0 * 1024.0);

    /* Sum the Base arena high-water and the per-family byte counters across the tasks
       sharing this node. Collective on GizmoNodeComm: every task must call
       report_memory_ledger(). */
    double node_base_highwater_mb = 0.0;
    MPI_Reduce(&base_highwater_mb, &node_base_highwater_mb, 1, MPI_DOUBLE, MPI_SUM, 0, GizmoNodeComm);
    long long node_family_bytes[GIZMO_MEM_NFAMILY];
    MPI_Reduce(g_family_bytes, node_family_bytes, GIZMO_MEM_NFAMILY, MPI_LONG_LONG, MPI_SUM, 0, GizmoNodeComm);

    /* Kokkos allocation telemetry: a SUPERSET of the UVM families above (all
       Kokkos-managed memory), reported separately -- never summed with the families. */
    long long kok_cur = gizmo_kokkos_mem_current_bytes(), kok_hw = gizmo_kokkos_mem_highwater_bytes();
    long long node_kok_cur = 0, node_kok_hw = 0;
    MPI_Reduce(&kok_cur, &node_kok_cur, 1, MPI_LONG_LONG, MPI_SUM, 0, GizmoNodeComm);
    MPI_Reduce(&kok_hw,  &node_kok_hw,  1, MPI_LONG_LONG, MPI_SUM, 0, GizmoNodeComm);

    /* LET wire transient buffers: report the high-water (and failed bytes). */
    long long let_in[3] = {g_let_wire_highwater, g_let_wire_current, g_let_wire_failed}, node_let[3] = {0, 0, 0};
    MPI_Reduce(let_in, node_let, 3, MPI_LONG_LONG, MPI_SUM, 0, GizmoNodeComm);

    /* Label-classified Kokkos buckets (current, bucket x space) -> node sums, so the
       "Kokkos observed" superset is decomposed into SoA / tree / tree-build scratch /
       moment scratch / NGL / unclassified, split by memory-space class. */
    const int NCELL = GIZMO_KOKCELL_COUNT;
    long long bkt_cur[NCELL], bkt_hw[NCELL], node_bkt[NCELL];
    gizmo_kokkos_mem_buckets(bkt_cur, bkt_hw);
    MPI_Reduce(bkt_cur, node_bkt, NCELL, MPI_LONG_LONG, MPI_SUM, 0, GizmoNodeComm);

    /* Tree byte breakdown (local vs foreign-LET vs aux) + foreign capacity/used-HW/floor.
       MB totals sum over the node; the node-count fields take the per-rank MAX (the
       biggest single rank -- the one that trips force_treeallocate first). */
    double tb_local = 0, tb_foreign = 0, tb_aux = 0;
    long long tb_falloc = 0, tb_fused = 0, tb_ffloor = 0, tb_fceil = 0;
    gizmo_tree_mem_breakdown(&tb_local, &tb_foreign, &tb_aux, &tb_falloc, &tb_fused, &tb_ffloor, &tb_fceil);
    double tb_mb_in[3] = {tb_local, tb_foreign, tb_aux}, node_tb_mb[3] = {0, 0, 0};
    long long tb_n_in[4] = {tb_falloc, tb_fused, tb_ffloor, tb_fceil}, node_tb_n[4] = {0, 0, 0, 0};
    MPI_Reduce(tb_mb_in, node_tb_mb, 3, MPI_DOUBLE,   MPI_SUM, 0, GizmoNodeComm);
    MPI_Reduce(tb_n_in,  node_tb_n,  4, MPI_LONG_LONG, MPI_MAX, 0, GizmoNodeComm);
    /* Rank-PAIRED foreign-LET stat: find the node-comm rank with the peak foreign
       used high-water, then read THAT rank's capacity+floor -- so slack is a single
       coherent rank, not node-max-capacity vs node-max-used mixed across ranks. */
    struct { double val; int rank; } fu_in = {(double) tb_fused, GizmoNodeRankOfTask}, fu_out = {0, 0};
    MPI_Allreduce(&fu_in, &fu_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, GizmoNodeComm);
    long long wf_mask[2] = { (GizmoNodeRankOfTask == fu_out.rank) ? tb_falloc : -1LL,
                             (GizmoNodeRankOfTask == fu_out.rank) ? tb_ffloor : -1LL };
    long long wf_pair[2] = {0, 0};
    MPI_Reduce(wf_mask, wf_pair, 2, MPI_LONG_LONG, MPI_MAX, 0, GizmoNodeComm);
    long long wf_used = (long long) fu_out.val;   /* the peak rank's used high-water */

    /* Byte categories: per-process VIRTUAL/committed and RESIDENT, summed over the node,
       so the commit-vs-physical gap (e.g. Base reserved 56 GB / resident 33 GB) is visible. */
    long long self_vmsize_kb = 0, self_vmrss_kb = 0;
    read_self_vm_kb(&self_vmsize_kb, &self_vmrss_kb);
    long long vm_in[2] = {self_vmsize_kb, self_vmrss_kb}, node_vm[2] = {0, 0};
    MPI_Reduce(vm_in, node_vm, 2, MPI_LONG_LONG, MPI_SUM, 0, GizmoNodeComm);

    /* Growth test, evaluated by every node lead on its OWN node so the history stays continuous
       whichever node ends up being the one reported. Each DISTINCT pool is tracked separately and
       any >10% rise counts: a single summed metric would hide growth in a small pool (the
       transient LET-wire buffers) behind a large one (Base arena / Kokkos). */
    int grew_local = 0;
    double node_resident_mb = 0.0;
    if(GizmoNodeRankOfTask == 0)
    {
        static double last_base = -1.0, last_mid = -1.0, last_let = -1.0;
        double base_m = node_base_highwater_mb;
        double mid_m  = (gizmo_kokkos_mem_available()
                         ? node_kok_hw / (1024.0 * 1024.0)
                         : (double)(node_family_bytes[GIZMO_MEM_PARTICLE_SOA]
                                    + node_family_bytes[GIZMO_MEM_TREE_NODES]
                                    + node_family_bytes[GIZMO_MEM_STL_TIMEBIN]) / (1024.0 * 1024.0));
        double let_m  = node_let[0] / (1024.0 * 1024.0);
        grew_local = (last_base < 0.0) || (base_m > 1.1 * last_base)
                     || (mid_m > 1.1 * last_mid) || (let_m > 1.1 * last_let);
        last_base = base_m; last_mid = mid_m; last_let = let_m;
        node_resident_mb = node_vm[1] / 1024.0;
    }

    /* Pick the node to report: the one holding the most resident memory, since node memory is the
       binding pool and the worst node decides whether the run fits. ONE all-rank reduce, which
       also carries that node's resident total back, so nothing else has to be gathered -- the node
       COUNT is already cached by gizmo_node_comm_init, and whether to report at all is decided by
       the chosen node from its own history (every lead updates that history on every call, so
       whichever one wins holds a current one). Non-leads sit it out with a sentinel.

       A consequence worth knowing: growth on a node that is NOT the busiest passes unreported, since
       the busiest node's own history is what decides. That is deliberate -- the busiest node is the
       one that decides whether the run fits -- but if a reason ever appears to want "report whenever
       ANY node grows, showing the busiest one that did", it costs nothing and needs no extra reduce:
       have a lead enter the selection only when it has something to say,
           sel_in.val = (GizmoNodeRankOfTask == 0 && (always || grew_local)) ? node_resident_mb : -1.0;
       and then report on a winner that actually entered,
           if(ThisTask == sel_out.rank && sel_out.val >= 0.0)
       so that a round where nobody grew elects nobody. */
    struct { double val; int rank; } sel_in, sel_out;
    sel_in.val  = (GizmoNodeRankOfTask == 0) ? node_resident_mb : -1.0;
    sel_in.rank = ThisTask;
    MPI_Allreduce(&sel_in, &sel_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if(ThisTask == sel_out.rank && (always || grew_local))
    {

        /* Node physical memory from /proc/meminfo. Absent on platforms without it
           (e.g. macOS); report "unavailable" rather than a misleading 0. */
        long long mem_total_kb = 0, committed_kb = 0, swaptot_kb = 0, swapfree_kb = 0;
        (void) report_comittable_memory(&mem_total_kb, &committed_kb, &swaptot_kb, &swapfree_kb);
        char node_phys[32];
        if(mem_total_kb > 0) {snprintf(node_phys, sizeof(node_phys), "%.1f MB", mem_total_kb / 1024.0);}
        else {snprintf(node_phys, sizeof(node_phys), "unavailable");}

        /* Base reservation accumulates per rank on the node -- MaxMemSize is the same
           on every rank, so the node total is simply the per-rank size times the count. */
        double node_base_reserved_mb = base_reserved_mb * GizmoRanksThisNode;

        /* Build the whole report into one buffer and emit it with a single write so it cannot
           interleave with other output. Figures are node totals in MB unless labelled otherwise.
           What each pool IS belongs in this file's comments, not in prose reprinted every time. */
        char buf[4096];
        int n = 0;
        n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                      "MEMORY LEDGER [%s] busiest of %d nodes (task=%d, %d ranks/node, physical %s)\n"
                      "  pools: base %.0f res / %.0f hw | SoA %.0f | tree %.0f | timebin %.0f |"
                      " LET-wire hw %.0f (cur %.0f, fail %.0f)\n",
                      when, GizmoNodeCount, ThisTask, GizmoRanksThisNode, node_phys,
                      node_base_reserved_mb, node_base_highwater_mb,
                      node_family_bytes[GIZMO_MEM_PARTICLE_SOA] / (1024.0 * 1024.0),
                      node_family_bytes[GIZMO_MEM_TREE_NODES] / (1024.0 * 1024.0),
                      node_family_bytes[GIZMO_MEM_STL_TIMEBIN] / (1024.0 * 1024.0),
                      node_let[0] / (1024.0 * 1024.0), node_let[1] / (1024.0 * 1024.0), node_let[2] / (1024.0 * 1024.0));
        /* Tree-node breakdown: split the "Tree nodes" total into local vs foreign-LET vs
           Father/Nextnode aux, and report the foreign storage that exists vs the since-start
           used high-water (rank-max). "used" is NOT current-at-print: force_treeallocate resets
           Numforeignnodes=0 before a controlled-stop ledger, so current would read 0.
           "index ceiling" is the range the storage sits in, not memory: it buys only its
           Nextnode ints, already inside the aux term. Allocated far below it is the design
           working, not a shortfall. */
        /* Tree detail: the node MB split, then the per-rank-max foreign-node counts. The counts are
           what force_treeallocate and the LET grow act on; "used" is the since-start high-water,
           because force_treeallocate resets the live count before a controlled-stop report. The
           index ceiling is range, not memory (it buys only its Nextnode ints, already in aux), so
           allocated sitting far below it is the sizing working rather than a shortfall. */
        n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                      "  tree: local %.0f | foreign-LET %.0f | aux %.0f;  foreign nodes rank-max:"
                      " alloc %lld used-hw %lld floor %lld ceiling %lld",
                      node_tb_mb[0], node_tb_mb[1], node_tb_mb[2], node_tb_n[0], node_tb_n[1], node_tb_n[2], node_tb_n[3]);
        if(wf_used > 0)
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  [peak-used rank: alloc %lld used %lld, alloc/used %.2f]",
                          wf_pair[0], wf_used, (double) wf_pair[0] / (double) wf_used);
        n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0, "\n");
        /* The pool figures above are LOGICAL requested bytes; these are what the OS sees, which is
           where the reserved-vs-touched gap shows up. */
        if(node_vm[0] > 0 || node_vm[1] > 0)
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  commit: virtual %.0f | resident %.0f | Committed_AS %.0f of %.0f physical\n",
                          node_vm[0] / 1024.0, node_vm[1] / 1024.0, committed_kb / 1024.0, mem_total_kb / 1024.0);
        /* Kokkos observed is a SUPERSET of the pools above and is never summed with them; the
           residual against them is neighbour/ghost/scratch plus Kokkos internal caching. */
        if(!gizmo_kokkos_mem_available())
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  kokkos: unavailable (callback API not compiled on this backend)\n");
        if(gizmo_kokkos_mem_available()) {
            /* Per-bucket totals (sum over space class) and per-space totals (sum over
               buckets) from the classified node array. */
            double bkt_mb[GIZMO_KOKBUCKET_COUNT] = {0}, spc_mb[GIZMO_KOKSPACE_COUNT] = {0};
            for(int b = 0; b < GIZMO_KOKBUCKET_COUNT; b++)
                for(int s = 0; s < GIZMO_KOKSPACE_COUNT; s++) {
                    double mb = node_bkt[b * GIZMO_KOKSPACE_COUNT + s] / (1024.0 * 1024.0);
                    bkt_mb[b] += mb; spc_mb[s] += mb;
                }
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  kokkos %.0f cur / %.0f hw: SoA %.0f tree-AoS %.0f tree-SoA %.0f walk %.0f runner %.0f sidecar %.0f"
                          " tscratch-b %.0f tscratch-m %.0f ngl-sidx %.0f ngl-pairs %.0f ngl-other %.0f unclass %.0f\n",
                          node_kok_cur / (1024.0 * 1024.0), node_kok_hw / (1024.0 * 1024.0),
                          bkt_mb[GIZMO_KOKBUCKET_PARTICLE_SOA], bkt_mb[GIZMO_KOKBUCKET_TREE_ARRAYS],
                          bkt_mb[GIZMO_KOKBUCKET_GRAVITY_TREE_SOA], bkt_mb[GIZMO_KOKBUCKET_GRAVITY_WALK],
                          bkt_mb[GIZMO_KOKBUCKET_MODEA_RUNNER], bkt_mb[GIZMO_KOKBUCKET_FINE_SIDECAR],
                          bkt_mb[GIZMO_KOKBUCKET_TREESCRATCH_BUILD], bkt_mb[GIZMO_KOKBUCKET_TREESCRATCH_MOMENT],
                          bkt_mb[GIZMO_KOKBUCKET_NGL_SIDX], bkt_mb[GIZMO_KOKBUCKET_NGL_PAIRS],
                          bkt_mb[GIZMO_KOKBUCKET_NGL], bkt_mb[GIZMO_KOKBUCKET_UNCLASSIFIED]);
            /* Space split only when more than one space class is nonzero (on a CPU/OpenMP
               backend everything is host/shared, so this line is suppressed). */
            int nspace_nz = 0;
            for(int s = 0; s < GIZMO_KOKSPACE_COUNT; s++) if(spc_mb[s] > 0.0) nspace_nz++;
            if(nspace_nz > 1)
                n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                              "  by space: host %.0f | shared/UVM %.0f | device %.0f | unknown %.0f\n",
                              spc_mb[GIZMO_KOKSPACE_HOST], spc_mb[GIZMO_KOKSPACE_SHARED],
                              spc_mb[GIZMO_KOKSPACE_DEVICE], spc_mb[GIZMO_KOKSPACE_UNKNOWN]);
            const char *unk = gizmo_kokkos_mem_unknown_space_name();
            if(unk)
                (void) snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                                "  unrecognized Kokkos space '%s' -> counted as unknown\n", unk);
        }
        fputs(buf, stdout);
        fflush(stdout);
    }
#endif
}

/* Always print (sparse, high-value call points: startup, controlled stop). */
void report_memory_ledger(const char *when) {report_memory_ledger_impl(when, 1);}

/* Print only on memory growth. For collective points reached every so often (domain
   decomposition) where an unconditional print would be too chatty. */
void report_memory_ledger_on_growth(const char *when) {report_memory_ledger_impl(when, 0);}

/* Startup persistent-memory preflight -- a user-info aid, run before the big
   allocations. Projects the deterministic per-node persistent reserve (Base arena +
   P/CellP/WakeupDirty + STL timebin + a conservative tree estimate) and compares it to
   detected node physical memory. If the projection PROVABLY exceeds node memory the run
   cannot load, so it requests a graceful controlled-stop with actionable advice (a clean
   stop instead of a part-allocated crash); if it is merely tight it only warns; if node
   memory is unknown (no /proc/meminfo) or there is headroom it is silent. The tree term
   is deliberately a conservative under-estimate so the hard-stop fires only when the run
   is infeasible beyond doubt. Collective on GizmoNodeComm -- call only on the all-rank
   allocation path. Returns nonzero iff a stop was requested. */
int gizmo_memory_preflight(void)
{
    gizmo_node_comm_init();
    long long per_rank = (long long) All.MaxMemSize * 1024 * 1024                      /* Base arena reserve */
                       + (long long) All.MaxPart    * (long long) sizeof(struct particle_data)   /* P */
                       + (long long) All.MaxPart    * (long long) sizeof(unsigned char)          /* WakeupDirty */
                       + (long long) All.MaxPartGas * (long long) sizeof(struct gas_cell_data)   /* CellP */
                       + 3LL * (long long) All.MaxPart * (long long) sizeof(int)                 /* STL timebin lists */
                       + (long long) All.MaxPart    * (long long) sizeof(unsigned char)          /* ProcessedFlag (host, outside the Base arena) */
#ifdef CHIMES
                       + (long long) All.MaxPartGas * (long long) sizeof(struct gasVariables)    /* ChimesGasVars (host, outside the Base arena) */
#endif
                       + (long long)(All.TreeAllocFactor * All.MaxPart)
                             * ((long long) sizeof(struct NODE) + (long long) sizeof(struct extNODE)); /* local tree (est.) */
    long long node_persistent = 0;
    MPI_Allreduce(&per_rank, &node_persistent, 1, MPI_LONG_LONG, MPI_SUM, GizmoNodeComm);

    long long mem_total_kb = 0, committed_kb = 0, swaptot_kb = 0, swapfree_kb = 0;
    (void) report_comittable_memory(&mem_total_kb, &committed_kb, &swaptot_kb, &swapfree_kb);
    long long node_phys = mem_total_kb * 1024;
    if(node_phys <= 0) {return 0;}   /* node memory unknown (e.g. no /proc/meminfo) -- no preflight */

    if(node_persistent > node_phys)
    {
        if(GizmoNodeRankOfTask == 0) {
            printf("MEMORY PREFLIGHT: projected persistent reserve %.1f GB exceeds node physical %.1f GB "
                   "(%d ranks/node) -- cannot load. Stopping cleanly. Feasible: fewer ranks/node, "
                   "lower PartAllocFactor, or more nodes.\n",
                   node_persistent / 1.0e9, node_phys / 1.0e9, GizmoRanksThisNode);
            fflush(stdout);
        }
        gizmo_request_controlled_stop(830, "memory preflight: projected persistent reserve exceeds node physical memory",
                                      __FILE__, __LINE__, __FUNCTION__);
        return 830;
    }
    if((double) node_persistent > 0.85 * (double) node_phys && GizmoNodeRankOfTask == 0)
    {
        printf("MEMORY PREFLIGHT WARNING: projected persistent reserve %.1f GB is %.0f%% of node physical %.1f GB "
               "(%d ranks/node) -- little headroom left for transient/ghost memory.\n",
               node_persistent / 1.0e9, 100.0 * (double) node_persistent / (double) node_phys,
               node_phys / 1.0e9, GizmoRanksThisNode);
        fflush(stdout);
    }
    return 0;
}
