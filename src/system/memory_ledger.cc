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
    long long tb_fcap = 0, tb_fused = 0, tb_ffloor = 0;
    gizmo_tree_mem_breakdown(&tb_local, &tb_foreign, &tb_aux, &tb_fcap, &tb_fused, &tb_ffloor);
    double tb_mb_in[3] = {tb_local, tb_foreign, tb_aux}, node_tb_mb[3] = {0, 0, 0};
    long long tb_n_in[3] = {tb_fcap, tb_fused, tb_ffloor}, node_tb_n[3] = {0, 0, 0};
    MPI_Reduce(tb_mb_in, node_tb_mb, 3, MPI_DOUBLE,   MPI_SUM, 0, GizmoNodeComm);
    MPI_Reduce(tb_n_in,  node_tb_n,  3, MPI_LONG_LONG, MPI_MAX, 0, GizmoNodeComm);
    /* Rank-PAIRED foreign-LET stat: find the node-comm rank with the peak foreign
       used high-water, then read THAT rank's capacity+floor -- so slack is a single
       coherent rank, not node-max-capacity vs node-max-used mixed across ranks. */
    struct { double val; int rank; } fu_in = {(double) tb_fused, GizmoNodeRankOfTask}, fu_out = {0, 0};
    MPI_Allreduce(&fu_in, &fu_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, GizmoNodeComm);
    long long wf_mask[2] = { (GizmoNodeRankOfTask == fu_out.rank) ? tb_fcap : -1LL,
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
        char buf[4096];
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
        /* Tree-node breakdown: split the "Tree nodes" total into local vs foreign-LET vs
           Father/Nextnode aux, and report foreign capacity vs since-start used high-water
           (rank-max). "used" is NOT current-at-print: force_treeallocate resets
           Numforeignnodes=0 before a controlled-stop ledger, so current would read 0. */
        n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                      "    tree split: local node arrays node %.1f MB | foreign-LET arrays node %.1f MB | aux (Father+Nextnode) node %.1f MB\n"
                      "    foreign-LET nodes (rank-max): capacity %lld | used high-water %lld (since start) | adaptive floor %lld\n",
                      node_tb_mb[0], node_tb_mb[1], node_tb_mb[2], node_tb_n[0], node_tb_n[1], node_tb_n[2]);
        if(wf_used > 0)
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "    foreign-LET peak-used rank (paired): capacity %lld | used %lld | slack %lld | cap/used %.2f\n",
                          wf_pair[0], wf_used, wf_pair[0] - wf_used, (double) wf_pair[0] / (double) wf_used);
        /* Byte categories: the family lines above are LOGICAL requested bytes; these are
           the commit vs physical categories (the Base reserved-vs-used gap lives here). */
        if(node_vm[0] > 0 || node_vm[1] > 0)
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  Byte categories (node): GIZMO virtual-commit %.1f MB, resident %.1f MB"
                          " (node Committed_AS %.1f MB of %.1f MB physical)\n",
                          node_vm[0] / 1024.0, node_vm[1] / 1024.0, committed_kb / 1024.0, mem_total_kb / 1024.0);
        if(gizmo_kokkos_mem_available())
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  Kokkos allocations observed: current node %.1f MB, high-water node %.1f MB\n"
                          "    (includes Particle SoA and Tree nodes above; NOT summed with family totals;\n"
                          "     residual vs explicit families is neighbour/ghost/scratch + Kokkos internal/caching + unclassified)\n",
                          node_kok_cur / (1024.0 * 1024.0), node_kok_hw / (1024.0 * 1024.0));
        else
            n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                          "  Kokkos allocations observed: unavailable (callback API not compiled on this backend)\n");
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
                          "  Kokkos buckets (node current MB): SoA %.1f | tree-AoS %.1f | tree-SoA %.1f | gravity-walk %.1f | modea-runner %.1f | fine-sidecar %.1f | treescratch-build %.1f | treescratch-moment %.1f | ngl %.1f | unclassified %.1f\n",
                          bkt_mb[GIZMO_KOKBUCKET_PARTICLE_SOA], bkt_mb[GIZMO_KOKBUCKET_TREE_ARRAYS],
                          bkt_mb[GIZMO_KOKBUCKET_GRAVITY_TREE_SOA], bkt_mb[GIZMO_KOKBUCKET_GRAVITY_WALK],
                          bkt_mb[GIZMO_KOKBUCKET_MODEA_RUNNER], bkt_mb[GIZMO_KOKBUCKET_FINE_SIDECAR],
                          bkt_mb[GIZMO_KOKBUCKET_TREESCRATCH_BUILD], bkt_mb[GIZMO_KOKBUCKET_TREESCRATCH_MOMENT],
                          bkt_mb[GIZMO_KOKBUCKET_NGL], bkt_mb[GIZMO_KOKBUCKET_UNCLASSIFIED]);
            /* Space split only when more than one space class is nonzero (on a CPU/OpenMP
               backend everything is host/shared, so this line is suppressed). */
            int nspace_nz = 0;
            for(int s = 0; s < GIZMO_KOKSPACE_COUNT; s++) if(spc_mb[s] > 0.0) nspace_nz++;
            if(nspace_nz > 1)
                n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                              "    by space (node MB): host %.1f | shared/UVM %.1f | device %.1f | unknown %.1f\n",
                              spc_mb[GIZMO_KOKSPACE_HOST], spc_mb[GIZMO_KOKSPACE_SHARED],
                              spc_mb[GIZMO_KOKSPACE_DEVICE], spc_mb[GIZMO_KOKSPACE_UNKNOWN]);
            const char *unk = gizmo_kokkos_mem_unknown_space_name();
            if(unk)
                n += snprintf(buf + n, (n < (int) sizeof(buf)) ? sizeof(buf) - n : 0,
                              "    (unrecognized Kokkos space name seen: '%s' -> bucketed as unknown)\n", unk);
        }
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
