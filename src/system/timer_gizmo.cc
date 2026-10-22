#include "timer.h"
#include "timer_mpi.h"
#include "timer_gizmo.h"

#include <cerrno>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>

#include "../declarations/allvars.h"

namespace gizmo { namespace timing { RegionHooks TimerKokkosHooks(); } }

namespace {
struct OutputStreams {
    FILE *tree = nullptr, *tsv = nullptr;
    bool attempted = false;
};
OutputStreams cumulative_output, step_output;
enum class ReportMode { Cumulative = 0, Step = 1, Both = 2 };
ReportMode report_mode = ReportMode::Cumulative;
long long report_interval = 1;
gizmo::timing::TimerTreeSort report_sort = gizmo::timing::TimerTreeSort::InclusiveMax;
std::vector<gizmo::timing::TimerRecord> step_baseline;
bool baseline_valid = false, context_valid = false;
gizmo::timing::TimerReportContext step_context;

bool env_enabled(const char *name)
{
    const char *value = std::getenv(name);
    return value && (!std::strcmp(value, "1") || !std::strcmp(value, "true"));
}

FILE *open_output(const char *name)
{
    std::string path(All.OutputDir);
    if(!path.empty() && path.back() != '/') {path += '/';}
    path += name;
    FILE *file = std::fopen(path.c_str(), "a");
    if(!file) {std::fprintf(stderr, "GIZMO timers: cannot open %s: %s\n", path.c_str(), std::strerror(errno));}
    return file;
}

gizmo::timing::TimerReportContext current_context()
{
    gizmo::timing::TimerReportContext result;
    result.simulation_time = All.Time;
    result.simulation_dt = All.TimeStep;
    result.particles_total = All.TotNumPart;
    result.highest_active_bin = All.HighestActiveTimeBin;
    return result;
}

std::vector<gizmo::timing::TimerRecord> step_difference(
    const std::vector<gizmo::timing::TimerRecord> &current)
{
    using Key = std::pair<std::size_t, std::string>;
    std::map<Key, const gizmo::timing::TimerRecord *> previous;
    for(const auto &record : step_baseline) {previous.emplace(Key(record.thread, record.path), &record);}
    std::vector<gizmo::timing::TimerRecord> result;
    for(auto record : current) {
        const auto found = previous.find(Key(record.thread, record.path));
        if(found != previous.end()) {
            const auto &old = *found->second;
            record.calls -= old.calls;
            record.completed -= old.completed;
            record.inclusive = std::max(0.0, record.inclusive - old.inclusive);
            record.exclusive = std::max(0.0, record.exclusive - old.exclusive);
        }
        // Active is an end-of-window gauge, not a difference. Calls and completed
        // count their respective events even for a scope crossing the boundary.
        if(record.calls || record.completed || record.active || record.inclusive > 0 || record.exclusive > 0) {
            result.push_back(std::move(record));
        }
    }
    return result;
}

void write_report(OutputStreams &output, const char *stem, const char *label, long long step,
                  const std::vector<gizmo::timing::TimerRecord> &snapshot,
                  const gizmo::timing::TimerReportContext &context)
{
    if(ThisTask == 0 && !output.attempted) {
        output.attempted = true;
        output.tree = open_output((std::string(stem) + ".txt").c_str());
        output.tsv = open_output((std::string(stem) + ".tsv").c_str());
        if(output.tree) {std::fprintf(output.tree, "\n=== New GIZMO timer session (totals reset at process startup) ===\n");}
        if(output.tsv) {
            std::fprintf(output.tsv, "# New GIZMO timer session; mode=%s; seconds\n", context.mode);
            gizmo::timing::TimerWriteTSVHeader(output.tsv);
        }
    }
    if(!gizmo::timing::TimerReportMPI(MPI_COMM_WORLD, output.tree, output.tsv, label, step,
                                    report_sort, &snapshot, &context) && ThisTask == 0) {
        std::fprintf(stderr, "GIZMO timers: skipped report exceeding MPI count limits\n");
    }
}
} // namespace

void gizmo_timers_initialize()
{
    int options[4] = {0, 0, 0, 0};
    if(ThisTask == 0) {
        options[0] = env_enabled("GIZMO_TIMERS");
        options[1] = env_enabled("GIZMO_TIMERS_KOKKOS");
        const char *sort = std::getenv("GIZMO_TIMERS_SORT");
        if(sort && !std::strcmp(sort, "mean")) {options[2] = 1;}
        else if(sort && std::strcmp(sort, "max")) {
            std::fprintf(stderr, "GIZMO timers: invalid GIZMO_TIMERS_SORT; using max\n");
        }
        const char *mode = std::getenv("GIZMO_TIMERS_MODE");
        if(mode && !std::strcmp(mode, "step")) {options[3] = 1;}
        else if(mode && !std::strcmp(mode, "both")) {options[3] = 2;}
        else if(mode && std::strcmp(mode, "cumulative")) {
            std::fprintf(stderr, "GIZMO timers: invalid GIZMO_TIMERS_MODE; using cumulative\n");
        }
        const char *interval = std::getenv("GIZMO_TIMERS_INTERVAL");
        if(interval) {
            char *end = nullptr;
            errno = 0;
            const long long value = std::strtoll(interval, &end, 10);
            if(!errno && end != interval && !*end && value > 0) {report_interval = value;}
            else {std::fprintf(stderr, "GIZMO timers: invalid GIZMO_TIMERS_INTERVAL; using 1\n");}
        }
    }
    // All ranks use rank zero's configuration so reporting remains collective
    // even when a launcher propagates environment variables inconsistently.
    MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&report_interval, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    report_sort = static_cast<gizmo::timing::TimerTreeSort>(options[2]);
    report_mode = static_cast<ReportMode>(options[3]);
    gizmo::timing::RegionHooks hooks;
    if(options[0] && options[1]) {hooks = gizmo::timing::TimerKokkosHooks();}
    gizmo::timing::TimerInitialize(options[0] != 0, hooks);
    if(ThisTask == 0 && options[0]) {
        const char *mode = report_mode == ReportMode::Both ? "both" :
                           (report_mode == ReportMode::Step ? "step" : "cumulative");
        std::printf("GIZMO hierarchical timers enabled (host elapsed time, mode=%s, interval=%lld, sort=%s, Kokkos regions=%s).\n",
                    mode, report_interval, options[2] ? "incl_mean" : "incl_max", options[1] ? "on; tool may add fences" : "off");
    }
}

void gizmo_timers_step_begin()
{
    if(!gizmo::timing::TimerEnabled()) {return;}
    step_context = current_context();
    context_valid = false;
    if(report_mode != ReportMode::Cumulative) {
        step_baseline = gizmo::timing::TimerSnapshot();
        baseline_valid = true;
    }
}

void gizmo_timers_step_context()
{
    if(!gizmo::timing::TimerEnabled()) {return;}
    step_context = current_context();
    step_context.active_particles_local = static_cast<long long>(ActiveParticleList.size());
    step_context.active_gas_local = 0;
    for(int bin = 0; bin < TIMEBINS; ++bin) {
        if(TimeBinActive[bin]) {step_context.active_gas_local += TimeBinCountGas[bin];}
    }
    context_valid = true;
}

void gizmo_timers_report(const char *label, long long step, bool force)
{
    if(!gizmo::timing::TimerEnabled() || (!force && step % report_interval != 0)) {return;}
    const auto current = gizmo::timing::TimerSnapshot();
    auto context = context_valid ? step_context : current_context();
    if(report_mode != ReportMode::Step) {
        write_report(cumulative_output, "timers", label, step, current, context);
    }
    if(report_mode != ReportMode::Cumulative && (baseline_valid || force)) {
        if(!baseline_valid) {
            context.mode = "startup";
            write_report(step_output, "timers-step", label, step, current, context);
        } else if(context_valid || force) {
            context.mode = !std::strcmp(label, "step") ? "step" : "partial";
            const auto delta = step_difference(current);
            write_report(step_output, "timers-step", label, step, delta, context);
        }
    }
}

void gizmo_timers_shutdown()
{
    gizmo::timing::TimerShutdown();
    for(auto *output : {&cumulative_output, &step_output}) {
        if(output->tree) {std::fclose(output->tree); output->tree = nullptr;}
        if(output->tsv) {std::fclose(output->tsv); output->tsv = nullptr;}
    }
    step_baseline.clear();
}
