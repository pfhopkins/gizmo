#include "timer.h"
#include "timer_mpi.h"
#include "timer_gizmo.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../declarations/allvars.h"

namespace gizmo { namespace timing { RegionHooks TimerKokkosHooks(); } }

namespace {
FILE *tree_file = nullptr, *tsv_file = nullptr;
bool output_attempted = false;
long long report_interval = 1;

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
} // namespace

void gizmo_timers_initialize()
{
    int options[2] = {0, 0};
    if(ThisTask == 0) {
        options[0] = env_enabled("GIZMO_TIMERS");
        options[1] = env_enabled("GIZMO_TIMERS_KOKKOS");
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
    MPI_Bcast(options, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&report_interval, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    gizmo::timing::RegionHooks hooks;
    if(options[0] && options[1]) {hooks = gizmo::timing::TimerKokkosHooks();}
    gizmo::timing::TimerInitialize(options[0] != 0, hooks);
    if(ThisTask == 0 && options[0]) {
        std::printf("GIZMO hierarchical timers enabled (host elapsed time, interval=%lld, Kokkos regions=%s).\n",
                    report_interval, options[1] ? "on; tool may add fences" : "off");
    }
}

void gizmo_timers_report(const char *label, long long step, bool force)
{
    if(!gizmo::timing::TimerEnabled() || (!force && step % report_interval != 0)) {return;}
    if(ThisTask == 0 && !output_attempted) {
        output_attempted = true;
        tree_file = open_output("timers.txt");
        tsv_file = open_output("timers.tsv");
        if(tree_file) {std::fprintf(tree_file, "\n=== New GIZMO timer session (totals reset at process startup) ===\n");}
        if(tsv_file) {
            std::fprintf(tsv_file, "# New GIZMO timer session; cumulative values, seconds\n");
            gizmo::timing::TimerWriteTSVHeader(tsv_file);
        }
    }
    if(!gizmo::timing::TimerReportMPI(MPI_COMM_WORLD, tree_file, tsv_file, label, step) && ThisTask == 0) {
        std::fprintf(stderr, "GIZMO timers: skipped report exceeding MPI count limits\n");
    }
}

void gizmo_timers_shutdown()
{
    gizmo::timing::TimerShutdown();
    if(tree_file) {std::fclose(tree_file); tree_file = nullptr;}
    if(tsv_file) {std::fclose(tsv_file); tsv_file = nullptr;}
}
