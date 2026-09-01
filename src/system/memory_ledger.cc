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
#include <limits.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* Size of the working memory pool when the run's particle count cannot be read up front, which
   normally means the input is missing or unreadable and the run is about to stop and say so. Large
   enough to reach that message on any machine, and to serve a modest run whose input simply could
   not be inspected early. */
#define ARENA_MEGABYTES_WHEN_INPUT_UNREADABLE 1024

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
    double base_reserved_mb  = (double) All.WorkingMemoryPoolSize;
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

        /* Base reservation accumulates per rank on the node -- WorkingMemoryPoolSize is the same
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

/* How large a chunk to send particle data in, when the parameter file does not say.

   Everything that ships data between tasks in pieces -- reading and writing snapshots, handing
   particles over during a decomposition, and the request-driven neighbour loops -- works through
   as many passes as the chunk size requires. The bytes moved are the same either way, so the
   chunk only has to be big enough that the per-pass costs stop mattering, which a hundred
   megabytes has been on every run to date.

   The one thing that can change that is running a neighbour loop in the request-driven mode with
   far more active particles than it is allowed today. That mode sends a query and receives a
   reply per active particle per neighbouring rank, and the count below is what the heaviest loop
   costs, measured on a full-size galaxy run: about 2.3 kB per query-and-reply pair and about four
   and a half pairs per active particle. So if a run raises the limit on how many active particles
   may take that path, the chunk is raised with it -- enough for a couple of passes -- rather than
   leaving it to discover the cost as a great many of them. Below roughly twenty thousand active
   particles per task this makes no difference and the long-standing size is used.

   Both limits have to be raised for that path to widen, since a loop takes it only when the total
   active count and the largest single task's are each under their own limit. The smaller of the
   two is therefore what a task can actually be asked to carry, and either one left alone keeps the
   run where it is today. The result is held to a ceiling as well: past it the loop simply takes
   more passes, which costs little, whereas a chunk sized for an enormous limit would be a
   correspondingly enormous block to find room for -- and a run that wants one can say so. */
double comm_chunk_megabytes_default(void)
{
    const double customary_megabytes = 100.0;
    const double most_worth_taking   = 400.0;     /* the largest that has ever been worth having */
    const double bytes_per_active    = 10400.0;   /* see above; the heaviest loop, measured */
    const double passes_to_aim_for   = 2.0;

    /* A limit left unset carries whatever each loop was built with, which is small; the run is
       then where it has always been and there is nothing to raise.

       "Unset" is deliberately read as ANY value that is not a positive count, not as the -1 the
       parameter reader writes for a missing tag.  That is what makes this independent of when it
       is called: the reader marks a missing limit in a later pass than the one that asks for this
       size, so at that moment a missing limit is still zero.  Zero is also what a user writes to
       turn Mode B off outright, and all three want the same answer -- there is no widening to do
       -- so they share the one branch.  A limit that ever gained a positive "unset" marker would
       break that, and this is the line to fix if it does. */
    const int limit_total   = All.NeighborLoopModeBThresholdSum;
    const int limit_largest = All.NeighborLoopModeBThresholdMax;
    if(limit_total <= 0 || limit_largest <= 0) {return customary_megabytes;}

    const double per_task = (limit_total < limit_largest) ? (double) limit_total
                                                          : (double) limit_largest;
    double raised = (per_task * bytes_per_active / passes_to_aim_for) / (1024.0 * 1024.0);
    if(raised > most_worth_taking) {raised = most_worth_taking;}
    return (raised > customary_megabytes) ? raised : customary_megabytes;
}

/* How big to make the memory arena when the parameter file does not say. Called once, before the
   arena is created, which is the only moment early enough: the arena is the first thing built and
   everything else is allocated out of it.

   This is a sensible default, not a guarantee. The arena holds working space, not the run's bulk
   storage -- particles, cells and tree nodes live outside it and size themselves -- and most of
   what is left adapts to whatever the arena turns out to be. The transport of tree nodes between
   ranks, and the handing of particles from rank to rank during a decomposition, both work through
   as many rounds as it takes and simply use shorter rounds when there is less room. For those,
   more arena buys speed, not correctness, and sizing the arena to the largest they could ever want
   would reserve enormous amounts of memory on every node to avoid a few extra rounds -- which is
   the very thing the round-based transport was written to avoid.

   So only what genuinely has to fit is counted. Anything that adapts gets a fixed allowance to
   work in, and anything that is both rare and enormous -- group finding, a chemistry step in which
   every cell is active -- is left out entirely: if such a run does not fit it stops cleanly and
   says so, and the parameter file is there to give it more.

   Returns megabytes, matching the units the arena and the parameter file use. */
static int arena_megabytes_from_tenants(long long total_particles)
{
    /* What a rank may be asked to hold. The same expression the particle storage itself uses, so
       the working space that scales with it is sized against the same number. */
    long long maxpart = (long long) (All.PartAllocFactor * ((double) total_particles / (double) NTask));
    if(maxpart < 1) {maxpart = 1;}

    /* Held from setup until the run ends, so it sits underneath everything below. */
    long long always_held = 0;

    /* The largest that any one thing which must fit whole ever gets. These do not overlap: each is
       taken and given back before the next begins. */
    long long must_fit = 0;

#ifdef PMGRID
    {
        /* The long-range mesh. Its grids and per-particle working space are taken in one piece and
           there is no smaller way to do it, so the whole of it has to fit. */
        size_t mesh_always = 0, mesh_per_force = 0;
        long_range_estimated_arena_bytes(maxpart, &mesh_always, &mesh_per_force);
        always_held += (long long) mesh_always;
        if((long long) mesh_per_force > must_fit) {must_fit = (long long) mesh_per_force;}
    }
#endif

    /* A step: the communication buffer, the gravity walk's record of anything the local tree
       could not supply, and a per-particle array or two. The buffer belongs to reading and
       writing files and so is never held at the same time as the walk's tables, but the tables
       are small enough now that counting both costs nothing and saves an argument. */
    long long step = (long long) All.CommChunkSize * 1024 * 1024
                   + (long long) GRAVITY_LET_DETECTOR_ENTRIES
                     * (long long) (sizeof(struct data_index) + sizeof(struct data_nodelist))
                   + maxpart * (long long) (2 * sizeof(MyFloat));
    if(step > must_fit) {must_fit = step;}

    /* Sorting particles into their new owners: a key and a sort record for each, two cost arrays,
       and per-rank bookkeeping. The buffers that then carry them across are not counted here --
       those are sized from whatever the arena has free and take more rounds when it is less. */
    long long decomposition = maxpart * (long long) (sizeof(peanokey)
                                                     + sizeof(peanokey) + sizeof(int)
                                                     + 2 * sizeof(float))
                            + 16LL * (long long) NTask * (long long) sizeof(long long);
    if(decomposition > must_fit) {must_fit = decomposition;}

#if defined(FOF)
    /* Finding groups, where that is compiled in. Half a dozen lists the length of the particle
       count are held together while groups are assembled, and there is no smaller way to do it.
       Modest per particle, and only present in runs that look for groups at all. */
    long long group_finding = maxpart * (long long) (3 * sizeof(MyIDType) + 10 * sizeof(int));
    if(group_finding > must_fit) {must_fit = group_finding;}
#endif

    /* Room for everything that works in rounds, so that it can use long ones. This is the whole
       performance side of the number: more would mean fewer rounds, less would mean more of them,
       and neither is a question of the run working. */
    const long long room_to_work_in = 512LL * 1024 * 1024;

    /* And a margin for the many small things not worth naming. */
    long long total = always_held + must_fit + room_to_work_in;
    total += total / 4;

    long long mb = total / (1024 * 1024);
    if(mb < 256) {mb = 256;}   /* never so small that ordinary working space cannot be served */

    return (int) ((mb > (long long) INT_MAX) ? (long long) INT_MAX : mb);
}


/* Set the arena size for this run, unless the user asked for a specific one.

   A value in the parameter file is taken as given: it is how an expert overrides this, and how a
   run that has been tuned by hand keeps its tuning. Otherwise the size comes from the tenants,
   which needs the particle count, and the only place to get it this early is the header of the
   file the run is about to read. If that cannot be read -- the file is missing, or a restart set
   is incomplete -- nothing is reported here, because the reader will report it properly a moment
   later; a conservative size is used so that the run gets far enough to do so. */
void gizmo_size_memory_arena(void)
{
    long long total_particles = -1;
    int user_asked = (All.WorkingMemoryPoolSize > 0);

    if(user_asked)
    {
        if(ThisTask == 0)
        {
            printf("Memory: using the requested %d MB per task for the working memory pool.\n",
                   All.WorkingMemoryPoolSize);
            fflush(stdout);
        }
    }
    else
    {
        if(RestartFlag == 1)
        {
            total_particles = peek_total_particles_in_restart();
        }
        else
        {
            char fname[MAX_PATH_BUFFERSIZE_TOUSE];
            input_source_filename(fname, MAX_PATH_BUFFERSIZE_TOUSE);
            total_particles = peek_total_particles_in_input(fname);
        }

        if(total_particles > 0)
        {
            All.WorkingMemoryPoolSize = arena_megabytes_from_tenants(total_particles);
            if(ThisTask == 0)
            {
                printf("Memory: sized the working memory pool at %d MB per task, for %lld particles "
                       "over %d tasks.\n", All.WorkingMemoryPoolSize, total_particles, NTask);
                fflush(stdout);
            }
        }
        else
        {
            All.WorkingMemoryPoolSize = ARENA_MEGABYTES_WHEN_INPUT_UNREADABLE;
            if(ThisTask == 0)
            {
                printf("Memory: could not read the particle count from the input, so the working "
                       "memory pool is set to %d MB per task. If the input is missing or unreadable "
                       "the message about that follows shortly.\n", All.WorkingMemoryPoolSize);
                fflush(stdout);
            }
        }
    }

    /* Whatever the size came from, hold it against what this machine can actually give a task,
       found by asking the system how much it can allocate and dividing it among the tasks sharing
       a node. This is the only place that comparison is made, for a size from either source.

       A size the code chose itself is brought down to what the machine can back, and says so: the
       pool is working space that adapts to how much of it there is, so a smaller one costs extra
       rounds rather than correctness, whereas reserving more than the node has invites the run to
       be killed partway through with nothing to explain it. A size the user asked for is never
       overridden -- it is their machine and their judgement -- but it is still reported. */
    {
        double safe_per_task = mpi_report_comittable_memory(0, 0);
        if(safe_per_task > 0 && (double) All.WorkingMemoryPoolSize > safe_per_task)
        {
            if(user_asked)
            {
                if(ThisTask == 0)
                {
                    printf("WARNING: the requested working memory pool (%d MB per task) is larger "
                           "than this machine can safely give each task (%g MB). The run may still "
                           "work, but it will fail if enough tasks on a node use their full share "
                           "at once. Lower Working_Mem_Pool_Per_Task_in_MB, or run fewer tasks per "
                           "node.\n", All.WorkingMemoryPoolSize, safe_per_task);
                    fflush(stdout);
                }
            }
            else
            {
                int wanted = All.WorkingMemoryPoolSize;
                /* Not the whole share: the particle arrays and the tree are allocated OUTSIDE this
                   pool, so handing it everything a task has leaves them nothing and the rank is
                   killed rather than stopped.  Leave a fraction of the share for them and for the
                   system, as the advice this replaced did. */
                All.WorkingMemoryPoolSize = (int) (ARENA_SHARE_OF_TASK_MEMORY * safe_per_task);
                if(ThisTask == 0)
                {
                    printf("Memory: this run would have taken %d MB per task for its working "
                           "memory pool, which is more than this machine can safely give each task "
                           "(%g MB), so it is using %d MB instead. If it then stops for lack of "
                           "working memory, run fewer tasks per node or more nodes, or set "
                           "Working_Mem_Pool_Per_Task_in_MB explicitly.\n",
                           wanted, safe_per_task, All.WorkingMemoryPoolSize);
                    fflush(stdout);
                }
            }
        }
    }
}


/* Startup persistent-memory preflight -- a user-info aid, run before the big allocations.
   Projects the per-node reserve that is fixed once the run starts -- the Base arena, the
   particle and cell storage, and the gravity tree with the node mirror the walk reads -- and
   compares it to detected node memory. If the projection PROVABLY exceeds node memory the run
   cannot load, so it requests a graceful controlled-stop and prints what would fit instead (a
   clean stop rather than a part-allocated crash); if it is merely tight it only warns; if node
   memory is unknown (no /proc/meminfo) or there is headroom it is silent. Every term is a
   conservative under-estimate -- the tree factor is the pre-ratchet one, the mirror counts only
   its always-present fields, and memory a run needs only while running is left out entirely --
   so the stop fires only when the run is infeasible beyond doubt. Collective on GizmoNodeComm --
   call only on the all-rank allocation path. Returns nonzero iff a stop was requested. */
int gizmo_memory_preflight(void)
{
    gizmo_node_comm_init();

    /* The reserve a rank makes regardless of how many particles it holds. */
    long long arena_per_rank = (long long) All.WorkingMemoryPoolSize * 1024 * 1024;

    /* Storage for the particles and cells a rank may be given. */
    long long particles_per_rank =
                         (long long) All.MaxPart    * (long long) sizeof(struct particle_data)   /* P */
                       + (long long) All.MaxPart    * (long long) sizeof(unsigned char)          /* WakeupDirty */
                       + (long long) All.MaxPartGas * (long long) sizeof(struct gas_cell_data)   /* CellP */
                       + 3LL * (long long) All.MaxPart * (long long) sizeof(int)                 /* STL timebin lists */
                       + (long long) All.MaxPart    * (long long) sizeof(unsigned char)          /* ProcessedFlag (host, outside the Base arena) */
#ifdef CHIMES
                       + (long long) All.MaxPartGas * (long long) sizeof(struct gasVariables)    /* ChimesGasVars (host, outside the Base arena) */
#endif
                       ;
    /* The gravity tree built over them: the node arrays, and the mirror of those nodes the
     * gravity walk reads. Both are sized from the same node count. The foreign nodes imported
     * from other ranks are deliberately absent: that storage is sized to each rank's actual
     * import when the import arrives, so nothing is reserved for it here.
     * The node count uses the startup tree factor rather than All.TreeAllocFactor, which init()
     * does not assign until after the read this runs inside; reading it here would multiply the
     * whole term by zero. It is the value the run is about to start from, and the run ratchets
     * it upward from there, so this stays an under-estimate. */
    long long tree_per_rank = (long long)(TREE_ALLOC_FACTOR_START * All.MaxPart)
                            * ((long long) sizeof(struct NODE) + (long long) sizeof(struct extNODE)
                               + (long long) gpu_gravity_tree_bytes_per_node());

    long long per_rank = arena_per_rank + particles_per_rank + tree_per_rank;
    long long node_persistent = 0;
    MPI_Allreduce(&per_rank, &node_persistent, 1, MPI_LONG_LONG, MPI_SUM, GizmoNodeComm);

    long long mem_total_kb = 0, committed_kb = 0, swaptot_kb = 0, swapfree_kb = 0;
    (void) report_comittable_memory(&mem_total_kb, &committed_kb, &swaptot_kb, &swapfree_kb);
    long long node_phys = mem_total_kb * 1024;
    if(node_phys <= 0) {return 0;}   /* node memory unknown (e.g. no /proc/meminfo) -- no preflight */

    /* How much of a node this projection may fill before the run is in trouble. It is well below
     * all of it because the projection covers only what is reserved up front: a run also needs
     * memory while it is running -- imported neighbours, tree nodes from other ranks, message
     * buffers -- and none of that is counted here. Both the advice and the warning use it, so a
     * configuration this reports as fitting is not one it would immediately warn about. */
    const double safe_fraction = 0.85;
    const long long node_safe = (long long)(safe_fraction * (double) node_phys);

    if(node_persistent > node_phys)
    {
        if(GizmoNodeRankOfTask == 0) {
            /* Say what would fit, computed from the same figures rather than offered as a guess.
             *
             * THE RULE FOR EVERY OPTION BELOW, and for any option added later: a value is offered
             * only if it would survive every OTHER limit the code applies to it, not merely the
             * memory arithmetic here. Fitting node memory is one constraint among several, and a
             * suggestion that satisfies this one while failing another just moves the user from
             * this stop to the next. Each option therefore carries the test for the limit it
             * would meet next: the settings are offered only BELOW the value in use (raising
             * either would not be a fix); a particle factor must stay above what the domain
             * decomposition can accept at all, since its own margin means a factor near one can
             * never hold even a perfectly balanced share; and a workspace size must still hold
             * the communication buffer, which is drawn from that same workspace as the run
             * starts. Printed values round the way that keeps them satisfying the test they were
             * chosen for. */
            char options[320]; int no = 0;
            const int olen = (int) sizeof(options);
            const int maxopt = 4;                  /* one slot per option offered below */
            char optbuf[maxopt][64]; int nopt = 0;
            long long ranks_that_fit = (per_rank > 0) ? node_safe / per_rank : 0;
            long long share = node_safe / (GizmoRanksThisNode > 0 ? GizmoRanksThisNode : 1);
            if(ranks_that_fit >= 1 && nopt + 1 < maxopt)
            {
                snprintf(optbuf[nopt], sizeof(optbuf[0]), "%lld rank%s/node", ranks_that_fit,
                         (ranks_that_fit == 1) ? "" : "s"); nopt++;
                long long nodes_that_fit = (NTask + ranks_that_fit - 1) / ranks_that_fit;
                snprintf(optbuf[nopt], sizeof(optbuf[0]), "%lld node%s", nodes_that_fit,
                         (nodes_that_fit == 1) ? "" : "s"); nopt++;
            }
            if(All.PartAllocFactor > 0 && share > arena_per_rank && nopt < maxopt)
            {
                double per_unit = (double)(particles_per_rank + tree_per_rank) / All.PartAllocFactor;
                double fits = (per_unit > 0) ? (double)(share - arena_per_rank) / per_unit : 0;
                fits = (double)(long long)(fits * 100.0) / 100.0;   /* print no more than what was shown to fit */
                if(fits > 1.0 / REDUC_FAC_FOR_MEMORY_IN_DOMAIN && fits < All.PartAllocFactor)
                    {snprintf(optbuf[nopt], sizeof(optbuf[0]), "PartAllocFactor %.2f", fits); nopt++;}
            }
            if(share > particles_per_rank + tree_per_rank && nopt < maxopt)
            {
                long long fits_mb = (share - particles_per_rank - tree_per_rank) / (1024 * 1024);
                if(fits_mb > (long long) All.CommChunkSize && fits_mb < All.WorkingMemoryPoolSize)
                    {snprintf(optbuf[nopt], sizeof(optbuf[0]), "WorkingMemoryPoolSize %lld", fits_mb); nopt++;}
            }
            for(int k = 0; k < nopt; k++)
                no += snprintf(options + no, (no < olen) ? olen - no : 0, "%s%s",
                               (k == 0) ? "" : ((k == nopt - 1) ? ", or " : ", "), optbuf[k]);

            printf("MEMORY PREFLIGHT: this run cannot load. Node memory %.1f GB; %d ranks/node need %.1f GB\n"
                   "  (per rank: %.2f GB particles and cells, %.2f GB workspace, %.2f GB gravity tree).\n",
                   node_phys / 1.0e9, GizmoRanksThisNode, node_persistent / 1.0e9,
                   particles_per_rank / 1.0e9, arena_per_rank / 1.0e9, tree_per_rank / 1.0e9);
            if(no > 0) {printf("  It would fit with any of: %s.\n", options);}
            else       {printf("  One rank alone does not fit this node; use nodes with more memory.\n");}
            fflush(stdout);
        }
        gizmo_request_controlled_stop(830, "memory preflight: projected persistent reserve exceeds node physical memory",
                                      __FILE__, __LINE__, __FUNCTION__);
        return 830;
    }
    if(node_persistent > node_safe && GizmoNodeRankOfTask == 0)
    {
        printf("MEMORY PREFLIGHT: %d ranks/node need %.1f GB of %.1f GB node memory (%.0f%%), leaving little\n"
               "  room for the memory a run needs while running. Consider fewer ranks/node or more nodes.\n",
               GizmoRanksThisNode, node_persistent / 1.0e9, node_phys / 1.0e9,
               100.0 * (double) node_persistent / (double) node_phys);
        fflush(stdout);
    }
    return 0;
}
