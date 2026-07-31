/** \file
    Runtime memory ledger -- node-scoped reporting of per-family memory use, so the
    true per-node footprint is visible rather than only the central Base arena.
    Node-scoped sums use the persistent node-local communicator (GizmoNodeComm);
    each node's lead task prints one line.
*/

#include <mpi.h>
#include <stdio.h>
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

/* Shared body. The node-scoped reduces are collective and ALWAYS run on every rank
   (safe only at symmetric all-rank call points); when `always` is 0 the node lead
   prints only if the node memory footprint has grown since the last print, so a
   per-domain-decomposition call does not spam. Growth is decided AFTER the reduce, by
   the node lead alone -- never gate the collective itself. */
static void report_memory_ledger_impl(const char *when, int always)
{
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

    if(GizmoNodeRankOfTask == 0)
    {
        /* Growth gate for non-"always" call points. Track each DISTINCT pool separately
           and print when ANY grows >10%: a single summed metric would hide growth in a
           small pool (e.g. the transient LET-wire buffers) behind a large one (Base
           arena / Kokkos). Pools: Base high-water (libc), Kokkos-observed high-water
           (all UVM/device; or the persistent-family total where telemetry is off), and
           the transient LET-wire high-water (libc, neither Base nor Kokkos). */
        static double last_base = -1.0, last_mid = -1.0, last_let = -1.0;
        double base_m = node_base_highwater_mb;
        double mid_m  = (gizmo_kokkos_mem_available()
                         ? node_kok_hw / (1024.0 * 1024.0)
                         : (double)(node_family_bytes[GIZMO_MEM_PARTICLE_SOA]
                                    + node_family_bytes[GIZMO_MEM_TREE_NODES]
                                    + node_family_bytes[GIZMO_MEM_STL_TIMEBIN]) / (1024.0 * 1024.0));
        double let_m  = node_let[0] / (1024.0 * 1024.0);
        int grew = (last_base < 0.0)
                   || (base_m > 1.1 * last_base)
                   || (mid_m  > 1.1 * last_mid)
                   || (let_m  > 1.1 * last_let);
        if(!always && !grew) {return;}   /* nothing grew: skip the print (the collective reduce already ran on all ranks) */
        last_base = base_m; last_mid = mid_m; last_let = let_m;

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

        /* Build the whole block into one buffer and emit it with a single write, so the
           blocks printed concurrently by each node's lead task do not interleave. */
        char buf[1600];
        int n = 0;
        n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                      "MEMORY LEDGER [%s] task=%d: %d ranks/node, node physical %s\n"
                      "  Base arena: reserved %.1f MB/rank (node %.1f MB), high-water %.1f MB/rank (node %.1f MB)\n"
                      "  Particle SoA (P/CellP/WakeupDirty): node %.1f MB\n"
                      "  Tree nodes (local+foreign, UVM): node %.1f MB\n"
                      "  Timebin lists (ActiveParticleList/Next/Prev, STL): node %.1f MB\n"
                      "  LET wire buffers (transient, libc): node high-water %.1f MB (current %.1f MB, failed %.1f MB)\n",
                      when, ThisTask, GizmoRanksThisNode, node_phys,
                      base_reserved_mb, node_base_reserved_mb, base_highwater_mb, node_base_highwater_mb,
                      node_family_bytes[GIZMO_MEM_PARTICLE_SOA] / (1024.0 * 1024.0),
                      node_family_bytes[GIZMO_MEM_TREE_NODES] / (1024.0 * 1024.0),
                      node_family_bytes[GIZMO_MEM_STL_TIMEBIN] / (1024.0 * 1024.0),
                      node_let[0] / (1024.0 * 1024.0), node_let[1] / (1024.0 * 1024.0), node_let[2] / (1024.0 * 1024.0));
        if(gizmo_kokkos_mem_available())
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  Kokkos allocations observed: current node %.1f MB, high-water node %.1f MB\n"
                          "    (includes Particle SoA and Tree nodes above; NOT summed with family totals;\n"
                          "     residual vs explicit families is neighbour/ghost/scratch + Kokkos internal/caching + unclassified)\n",
                          node_kok_cur / (1024.0 * 1024.0), node_kok_hw / (1024.0 * 1024.0));
        else
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  Kokkos allocations observed: unavailable (callback API not compiled on this backend)\n");
        (void) snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                        "  (node high-water values are the SUM of per-rank peaks -- a conservative upper bound, not a time-coincident node peak)\n");
        fputs(buf, stdout);
        fflush(stdout);
    }
}

/* Always print (sparse, high-value call points: startup, controlled stop). */
void report_memory_ledger(const char *when) {report_memory_ledger_impl(when, 1);}

/* Print only on memory growth. For collective points reached every so often (domain
   decomposition) where an unconditional print would be too chatty. */
void report_memory_ledger_on_growth(const char *when) {report_memory_ledger_impl(when, 0);}
