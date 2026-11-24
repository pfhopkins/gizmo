/* OUTPUT_TIMING_BREAKDOWN: what the wall time of each cpu.txt bucket was spent on.
 *
 * cpu.txt charges every second of a task's wall time to exactly one bucket (system.cc: measure_time, the
 * bracket helpers and cpu_charge / cpu_charge_child).  This file records, without adding any synchronization,
 * the events that say how that time was used, and hands them to the bucket that is charged with the span in
 * which they happened:
 *   - each kernel launched on a device execution space (one launch; its device run time, measured by a pair of
 *     timed events placed around it in its own queue, so the host never waits for it);
 *   - each Kokkos fence of a device execution space (one fence; the host time spent in the outermost fence or
 *     deep copy is time the host waited on the device; a host space's fences wait for nothing and are not counted);
 *   - each OpenMP parallel region entered by the main thread (host time of the region: threaded host work,
 *     including the time the main thread waits at its closing barrier).
 * What is left of the wall time is single-thread host work.  Launches, fences and copies are seen through the
 * Kokkos Tools callbacks registered in code; OpenMP regions through the OpenMP tools interface (OMPT), when the
 * OpenMP runtime offers it, else only the regions of Kokkos' own OpenMP backend are seen.
 *
 * Attribution.  Events are kept in launch order with their host time stamp.  A charge to a bucket takes every
 * pending event whose stamp lies inside the span it charges; events older than the span stay pending for the
 * enclosing span (a measure_time chain interval or an outer bracket), the same way the wall time of a child
 * bucket is subtracted from its parent.  The span starts at the previous chain point for measure_time(), at
 * the start cpu_minus_children() derived for a bracket, and otherwise dt before the charge (so a bucket is
 * charged when its span ends).  A kernel's device time is added to the bucket it was launched in once its
 * end event is complete, at the latest at the next cpu.txt step, so it can land in the step after its launch.
 *
 * Kokkos fences the device around every launch for a tool that does not say otherwise (it assumes a tool that
 * times kernels on the host); this tool times kernels on the device and asks for no fences, before Kokkos is
 * initialized.  Fences Kokkos adds anyway (a tool library loaded through Kokkos' own options can ask for them)
 * are counted and reported, since they change the timings being measured.
 *
 * Memory: the pending events of one step (emptied at the end of every step) and a fixed pool of
 * TB_SLOTS event pairs; kernels launched while the pool is full are counted but not timed (reported). */

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <mutex>

#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_dispatch_templates.h"

#ifdef OUTPUT_TIMING_BREAKDOWN

#if defined(_OPENMP) && defined(__has_include)
#if __has_include(<omp-tools.h>)
#include <omp-tools.h>
#define TB_HAVE_OMPT
#endif
#endif

namespace KT = Kokkos::Tools::Experimental;

enum {TB_KIND_KERNEL = 0, TB_KIND_FENCE = 1, TB_KIND_COPY = 2, TB_KIND_OMP = 3};
struct TbRecord {double t; double dur; int kind; int slot;};   /* host stamp [s], host duration [s], kind, event-pair slot or -1 */

#define TB_SLOTS 4096                       /* event pairs of kernels whose device time is not yet read */
#define TB_NO_SLOT        ((uint64_t) -1)   /* kernel id handed to Kokkos: no event pair */
#define TB_HOST_PARALLEL  ((uint64_t) -2)   /* kernel id handed to Kokkos: an OpenMP-backend dispatch timed on the host */
struct TbSlot {GizmoGpuEvent start, stop; int made; int inst; int bucket; int stop_recorded; int refused;};
struct TbInstance {uint32_t devid; Kokkos::DefaultExecutionSpace space;};

static std::mutex tb_mutex;                 /* callbacks can come from any host thread */
static std::vector<TbRecord> tb_pending;
static std::vector<TbInstance> tb_instances;
static TbSlot tb_slot[TB_SLOTS];
static int tb_slot_head = 0, tb_slot_count = 0;
static double tb_step[CPU_PARTS][TIMING_BREAKDOWN_FIELDS];
static double tb_span_start = -1;           /* start of the span the next charge closes; <0: not given */
static long long tb_untimed = 0, tb_tool_fences = 0;
static int tb_active = 0, tb_ompt_active = 0;
static thread_local int tb_is_main = 0, tb_omp_depth = 0, tb_wait_depth = 0, tb_host_copy_depth = 0;
static thread_local double tb_wait_t0 = 0, tb_omp_t0 = 0;

static bool tb_type_is_host(KT::DeviceType type)
{
    return type == KT::DeviceType::Serial || type == KT::DeviceType::OpenMP || type == KT::DeviceType::Threads || type == KT::DeviceType::HPX;
}

static void tb_append(double t, double dur, int kind, int slot)
{
    std::lock_guard<std::mutex> guard(tb_mutex);
    tb_pending.push_back({t, dur, kind, slot});
}

/* Read the device time of completed kernels, oldest first; stop at the first one not yet charged to a bucket,
 * still being launched, or still running.  Called with tb_mutex held. */
static void tb_harvest_locked(void)
{
    if(GizmoGpuEvent::error_pending()) {return;}   /* the error is the code's own checks' to report: read nothing, clear nothing */
    while(tb_slot_count > 0)
    {
        TbSlot &s = tb_slot[tb_slot_head];
        if(s.bucket < 0 || !s.stop_recorded) {break;}
        const int status = s.stop.query();
        if(status == GIZMO_GPU_EVENT_NOT_READY && !s.refused) {break;}
        if(status == GIZMO_GPU_EVENT_COMPLETE && !s.refused)
        {
            const double dt = GizmoGpuEvent::elapsed_seconds(s.start, s.stop);
            if(dt >= 0) {tb_step[s.bucket][TIMING_BREAKDOWN_GPU] += dt;} else {tb_untimed++; GizmoGpuEvent::clear_error();}   /* this read's own error */
        } else {tb_untimed++;}   /* refused, or a device fault: it surfaces at the code's own next check; the kernel stays untimed */
        tb_slot_head = (tb_slot_head + 1) % TB_SLOTS; tb_slot_count--;
    }
}

static void tb_begin_kernel(const char *, const uint32_t devid, uint64_t *kernel_id)
{
    *kernel_id = TB_NO_SLOT;
    if(!tb_active) {return;}
    const KT::DeviceType type = KT::identifier_from_devid(devid).type;
    if(tb_type_is_host(type))
    {   /* host-backend dispatch: threaded host work, seen here only when OMPT does not see its parallel region */
        if(!tb_ompt_active && type == KT::DeviceType::OpenMP && tb_is_main && tb_omp_depth == 0) {tb_omp_t0 = my_second(); *kernel_id = TB_HOST_PARALLEL;}
        return;
    }
    const double t = my_second();
    std::lock_guard<std::mutex> guard(tb_mutex);
    if(tb_slot_count > 3 * TB_SLOTS / 4) {tb_harvest_locked();}
    int inst = -1;
    for(size_t k = 0; k < tb_instances.size(); k++) {if(tb_instances[k].devid == devid) {inst = (int) k; break;}}
    int slot = -1;
    /* An error already pending belongs to an earlier call and to that call's own check: this kernel stays untimed and
     * the error is left alone.  An error raised by the calls below is theirs, so it is cleared: a refused timing
     * event must not read as a refused launch. */
    if(inst >= 0 && tb_slot_count < TB_SLOTS && !GizmoGpuEvent::error_pending())
    {
        slot = (tb_slot_head + tb_slot_count) % TB_SLOTS;
        TbSlot &s = tb_slot[slot];
        if(!s.made)
        {
            if(s.start.create_timed() == 0) {
                if(s.stop.create_timed() == 0) {s.made = 1;} else {s.start.destroy();}
            }
        }
        if(s.made && s.start.record(tb_instances[inst].space) == 0) {s.inst = inst; s.bucket = -1; s.stop_recorded = 0; s.refused = 0; tb_slot_count++;}
        else {slot = -1; GizmoGpuEvent::clear_error();}
    }
    if(slot < 0) {tb_untimed++;}
    tb_pending.push_back({t, 0.0, TB_KIND_KERNEL, slot});
    if(slot >= 0) {*kernel_id = (uint64_t) slot;}
}

static void tb_end_kernel(const uint64_t kernel_id)
{
    if(kernel_id == TB_NO_SLOT) {return;}
    if(kernel_id == TB_HOST_PARALLEL) {const double t = my_second(); tb_append(tb_omp_t0, t - tb_omp_t0, TB_KIND_OMP, -1); return;}
    std::lock_guard<std::mutex> guard(tb_mutex);
    TbSlot &s = tb_slot[kernel_id];
    /* This runs inside the launch, before the launch's own refusal check: a pending error is that launch's, so it is left
     * for the check and the kernel stays untimed; only a refusal of this record is cleared. */
    if(GizmoGpuEvent::error_pending()) {s.refused = 1;}
    else {s.refused = (s.stop.record(tb_instances[s.inst].space) != 0); if(s.refused) {GizmoGpuEvent::clear_error();}}
    s.stop_recorded = 1;
}

/* The host waits on the device in a fence or a deep copy.  Only the outermost one on the main thread outside
 * an OpenMP region is timed (a copy fences internally; time inside a region is already threaded host time). */
static void tb_wait_begin(void)
{
    if(tb_is_main && tb_omp_depth == 0 && tb_wait_depth++ == 0) {tb_wait_t0 = my_second();}
}
static double tb_wait_end(void)   /* the outermost wait's duration, or -1 */
{
    if(!(tb_is_main && tb_omp_depth == 0) || tb_wait_depth <= 0) {return -1;}
    if(--tb_wait_depth > 0) {return -1;}
    return my_second() - tb_wait_t0;
}

static void tb_begin_fence(const char *name, const uint32_t devid, uint64_t *handle)
{
    *handle = 0;
    if(!tb_active) {return;}
    if(tb_type_is_host(KT::identifier_from_devid(devid).type)) {return;}   /* a host space's fence (also one leg of a global fence): no device wait */
    if(name && strstr(name, "Kokkos Profile Tool Fence")) {std::lock_guard<std::mutex> guard(tb_mutex); tb_tool_fences++;}
    *handle = 1;
    tb_wait_begin();
}
static void tb_end_fence(const uint64_t handle)
{
    if(!handle) {return;}
    const double dur = tb_wait_end();
    tb_append(my_second() - (dur > 0 ? dur : 0), (dur > 0 ? dur : 0), TB_KIND_FENCE, -1);
}

/* A memory space on the host side of the device: host memory and the device runtimes' page-locked host memory. */
static int tb_space_is_host(const Kokkos::Tools::SpaceHandle &h)
{
    return strcmp(h.name, Kokkos::HostSpace::name()) == 0 || strstr(h.name, "HostPinned") != NULL;
}
/* A copy with both ends on the host is host work, not a wait on the device: it stays in the row's host time. */
static void tb_begin_copy(Kokkos::Tools::SpaceHandle dst, const char *, const void *, Kokkos::Tools::SpaceHandle src, const char *, const void *, uint64_t)
{
    if(!tb_active) {return;}
    if(tb_space_is_host(dst) && tb_space_is_host(src)) {tb_host_copy_depth++; return;}
    tb_wait_begin();
}
static void tb_end_copy(void)
{
    if(!tb_active) {return;}
    if(tb_host_copy_depth > 0) {tb_host_copy_depth--; return;}
    const double dur = tb_wait_end();
    if(dur >= 0) {tb_append(my_second() - dur, dur, TB_KIND_COPY, -1);}
}

#ifdef TB_HAVE_OMPT
/* OpenMP regions opened by the main thread; nested ones are inside the outer region's time. */
static void tb_ompt_parallel_begin(ompt_data_t *, const ompt_frame_t *, ompt_data_t *, unsigned int, int, const void *)
{
    if(tb_is_main && tb_omp_depth++ == 0 && tb_active) {tb_omp_t0 = my_second();}
}
static void tb_ompt_parallel_end(ompt_data_t *, ompt_data_t *, int, const void *)
{
    if(!tb_is_main || tb_omp_depth <= 0) {return;}
    if(--tb_omp_depth == 0 && tb_active && tb_omp_t0 > 0) {const double t = my_second(); tb_append(tb_omp_t0, t - tb_omp_t0, TB_KIND_OMP, -1);}
    if(tb_omp_depth == 0) {tb_omp_t0 = 0;}
}
static int tb_ompt_initialize(ompt_function_lookup_t lookup, int, ompt_data_t *)
{
    ompt_set_callback_t set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
    if(!set_callback) {return 0;}
    const int ok_begin = set_callback(ompt_callback_parallel_begin, (ompt_callback_t) tb_ompt_parallel_begin);
    const int ok_end   = set_callback(ompt_callback_parallel_end,   (ompt_callback_t) tb_ompt_parallel_end);
    tb_ompt_active = (ok_begin == ompt_set_always && ok_end == ompt_set_always);
    return 1;
}
static void tb_ompt_finalize(ompt_data_t *) {tb_ompt_active = 0;}
/* The OpenMP runtime looks this symbol up when it starts (OpenMP 5.0, section 4.2.1). */
extern "C" ompt_start_tool_result_t *ompt_start_tool(unsigned int, const char *)
{
    static ompt_start_tool_result_t result = {tb_ompt_initialize, tb_ompt_finalize, {0}};
    return &result;
}
#endif

static void tb_request_settings(const uint32_t, Kokkos::Tools::Experimental::ToolSettings *settings)
{
    settings->requires_global_fencing = false;   /* kernels are timed on the device: no fence around launches */
}

/* Before Kokkos::initialize, which asks the tool for its settings once; on the main thread. */
void timing_breakdown_register_settings(void)
{
    tb_is_main = 1;
    KT::set_request_tool_settings_callback(tb_request_settings);
}

/* After Kokkos::initialize. */
void timing_breakdown_register(void)
{
    tb_instances.push_back({KT::device_id(Kokkos::DefaultExecutionSpace()), Kokkos::DefaultExecutionSpace()});
    KT::set_begin_parallel_for_callback(tb_begin_kernel);    KT::set_end_parallel_for_callback(tb_end_kernel);
    KT::set_begin_parallel_reduce_callback(tb_begin_kernel); KT::set_end_parallel_reduce_callback(tb_end_kernel);
    KT::set_begin_parallel_scan_callback(tb_begin_kernel);   KT::set_end_parallel_scan_callback(tb_end_kernel);
    KT::set_begin_fence_callback(tb_begin_fence);            KT::set_end_fence_callback(tb_end_fence);
    KT::set_begin_deep_copy_callback(tb_begin_copy);         KT::set_end_deep_copy_callback(tb_end_copy);
    tb_active = 1;
}

void timing_breakdown_register_instance(const Kokkos::DefaultExecutionSpace &space)
{
    std::lock_guard<std::mutex> guard(tb_mutex);
    const uint32_t devid = KT::device_id(space);
    for(size_t k = 0; k < tb_instances.size(); k++) {if(tb_instances[k].devid == devid) {return;}}
    tb_instances.push_back({devid, space});
}

/* Before Kokkos is finalized: no more callbacks, events destroyed (kernels still running are not timed). */
void timing_breakdown_release(void)
{
    std::lock_guard<std::mutex> guard(tb_mutex);
    tb_active = 0;
    KT::set_begin_parallel_for_callback(nullptr);    KT::set_end_parallel_for_callback(nullptr);
    KT::set_begin_parallel_reduce_callback(nullptr); KT::set_end_parallel_reduce_callback(nullptr);
    KT::set_begin_parallel_scan_callback(nullptr);   KT::set_end_parallel_scan_callback(nullptr);
    KT::set_begin_fence_callback(nullptr);           KT::set_end_fence_callback(nullptr);
    KT::set_begin_deep_copy_callback(nullptr);       KT::set_end_deep_copy_callback(nullptr);
    for(int k = 0; k < TB_SLOTS; k++) {if(tb_slot[k].made) {tb_slot[k].start.destroy(); tb_slot[k].stop.destroy(); tb_slot[k].made = 0;}}
    tb_slot_count = 0; tb_pending.clear(); tb_instances.clear();
}

void timing_breakdown_note_span_start(double t) {tb_span_start = t;}

/* Hand the pending events of the span ending now to `bucket` (see the attribution rule above). */
void timing_breakdown_charge(int bucket, double dt)
{
    const double t_start = (tb_span_start >= 0) ? tb_span_start : my_second() - dt;
    tb_span_start = -1;
    if(!tb_active) {return;}
    std::lock_guard<std::mutex> guard(tb_mutex);
    size_t keep = 0;
    for(size_t k = 0; k < tb_pending.size(); k++)
    {
        const TbRecord &r = tb_pending[k];
        if(r.t < t_start) {tb_pending[keep++] = r; continue;}
        double *acc = tb_step[bucket];
        switch(r.kind)
        {
            case TB_KIND_KERNEL: acc[TIMING_BREAKDOWN_LAUNCHES] += 1; if(r.slot >= 0) {tb_slot[r.slot].bucket = bucket;} break;
            case TB_KIND_FENCE:  acc[TIMING_BREAKDOWN_FENCES] += 1;   acc[TIMING_BREAKDOWN_WAIT] += r.dur; break;
            case TB_KIND_COPY:   acc[TIMING_BREAKDOWN_WAIT] += r.dur; break;
            case TB_KIND_OMP:    acc[TIMING_BREAKDOWN_OMP] += r.dur;  break;
        }
    }
    tb_pending.resize(keep);
}

/* End of a cpu.txt step: events no span claimed go to `misc`, completed kernels are read, and this step's
 * values (CPU_PARTS x TIMING_BREAKDOWN_FIELDS) are returned and cleared. */
void timing_breakdown_end_step(double *step)
{
    tb_span_start = 0;   /* the oldest possible start: everything still pending */
    timing_breakdown_charge(CPU_MISC, 0);
    std::lock_guard<std::mutex> guard(tb_mutex);
    tb_harvest_locked();
    memcpy(step, tb_step, sizeof(tb_step));
    memset(tb_step, 0, sizeof(tb_step));
    static long long untimed_reported = 0, tool_fences_reported = 0;
    static int source_reported = 0;   /* the OpenMP runtime starts at the first parallel region, so say this at the first step */
    if(ThisTask == 0 && !source_reported)
    {
        printf("OUTPUT_TIMING_BREAKDOWN: device kernels timed with events in their queues; threaded host time from %s\n",
               tb_ompt_active ? "OpenMP regions (OMPT)" : "Kokkos OpenMP-backend dispatches only (the OpenMP runtime offers no OMPT)");
        source_reported = 1;
    }
    if(ThisTask == 0 && tb_untimed > 2 * untimed_reported + 100)
    {
        printf("OUTPUT_TIMING_BREAKDOWN: %lld device kernels so far on task 0 were not timed (queue not known or event pool full); "
               "the gpu column misses their time\n", tb_untimed);
        untimed_reported = tb_untimed;
    }
    if(ThisTask == 0 && tb_tool_fences > tool_fences_reported)
    {
        printf("OUTPUT_TIMING_BREAKDOWN: WARNING: Kokkos added %lld fences for a profiling tool on task 0 (a Kokkos tool library asked "
               "for them); the timings include their waits\n", tb_tool_fences);
        tool_fences_reported = tb_tool_fences;
    }
}

#else  /* OUTPUT_TIMING_BREAKDOWN */

void timing_breakdown_register_instance(const Kokkos::DefaultExecutionSpace &) {}

#endif /* OUTPUT_TIMING_BREAKDOWN */
