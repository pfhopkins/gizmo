#include "timer_mpi.h"
#include "timer.h"

#include <algorithm>
#include <climits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace gizmo {
namespace timing {
namespace {
struct Totals {
    std::uint64_t calls = 0, completed = 0, active = 0;
    double inclusive = 0, exclusive = 0;
};

double imbalance(double sum, double maximum, int ranks)
{
    return sum > 0 ? maximum / (sum / ranks) : 1.0;
}

/* Compare path components so a sibling named "A!" cannot appear between "A"
 * and "A::child" in the readable tree. The lane is part of the first component. */
struct PathLess {
    bool operator()(const std::string &a, const std::string &b) const
    {
        std::size_t pa = 0, pb = 0;
        for(;;) {
            const auto ea = a.find("::", pa), eb = b.find("::", pb);
            const std::string_view ca(a.data() + pa, (ea == std::string::npos ? a.size() : ea) - pa);
            const std::string_view cb(b.data() + pb, (eb == std::string::npos ? b.size() : eb) - pb);
            if(ca != cb) {return ca < cb;}
            if(ea == std::string::npos || eb == std::string::npos) {
                return ea == std::string::npos && eb != std::string::npos;
            }
            pa = ea + 2; pb = eb + 2;
        }
    }
};

std::vector<std::string> lines(const std::string &text)
{
    std::vector<std::string> result;
    std::size_t begin = 0;
    while(begin < text.size()) {
        const auto end = text.find('\n', begin);
        if(end == std::string::npos) {break;}
        result.push_back(text.substr(begin, end - begin));
        begin = end + 1;
    }
    return result;
}

struct TreeRow {
    int index;
    std::string prefix;
};

std::vector<TreeRow> sorted_tree_rows(const std::vector<std::string> &keys,
                                      const std::vector<double> &metric)
{
    std::map<std::string, int> indices;
    for(int i = 0; i < static_cast<int>(keys.size()); ++i) {indices.emplace(keys[i], i);}
    std::vector<std::vector<int>> children(keys.size());
    std::map<std::string, std::vector<int>> roots;
    for(int i = 0; i < static_cast<int>(keys.size()); ++i) {
        const auto parent_end = keys[i].rfind("::");
        if(parent_end != std::string::npos) {
            const auto parent = indices.find(keys[i].substr(0, parent_end));
            if(parent != indices.end()) {
                children[parent->second].push_back(i);
                continue;
            }
        }
        roots[keys[i].substr(0, keys[i].find('\t'))].push_back(i);
    }
    const auto hotter = [&](int a, int b) {
        if(metric[2 * a] != metric[2 * b]) {return metric[2 * a] > metric[2 * b];}
        return keys[a] < keys[b];
    };
    for(auto &group : children) {std::sort(group.begin(), group.end(), hotter);}
    struct PendingRow { int index; std::string parent_prefix; bool last; };
    std::vector<TreeRow> result;
    std::vector<PendingRow> pending;
    result.reserve(keys.size());
    // Keep the main/worker lanes separate and emit each parent before its
    // complete subtree. Use an explicit stack so deeply nested scopes are safe.
    for(auto &lane : roots) {
        auto &group = lane.second;
        std::sort(group.begin(), group.end(), hotter);
        for(int j = static_cast<int>(group.size()) - 1; j >= 0; --j) {
            pending.push_back({group[j], "", j == static_cast<int>(group.size()) - 1});
        }
        while(!pending.empty()) {
            const auto row = pending.back();
            pending.pop_back();
            result.push_back({row.index, row.parent_prefix + (row.last ? "`-- " : "|-- ")});
            const auto child_prefix = row.parent_prefix + (row.last ? "    " : "|   ");
            const auto &nested = children[row.index];
            for(int j = static_cast<int>(nested.size()) - 1; j >= 0; --j) {
                pending.push_back({nested[j], child_prefix, j == static_cast<int>(nested.size()) - 1});
            }
        }
    }
    return result;
}
} // namespace

void TimerWriteTSVHeader(FILE *stream)
{
    if(stream) {
        std::fprintf(stream, "label\tsync_point\tlane\tpath\tranks_present\tcalls_sum\tcompleted_sum\tactive_sum"
                     "\tinclusive_min_s\tinclusive_mean_s\tinclusive_max_s"
                     "\texclusive_min_s\texclusive_mean_s\texclusive_max_s"
                     "\tinclusive_imbalance\texclusive_imbalance\tmode"
                     "\tsimulation_time\tsimulation_dt\tparticles_total"
                     "\tactive_particles_sum\tactive_particles_mean\tactive_particles_max"
                     "\tactive_gas_sum\tactive_gas_mean\tactive_gas_max\thighest_active_bin\tfrom_sync_point"
                     "\tprevious_active_particles_sum\tprevious_active_particles_mean\tprevious_active_particles_max"
                     "\tprevious_active_gas_sum\tprevious_active_gas_mean\tprevious_active_gas_max\n");
    }
}

bool TimerReportMPI(MPI_Comm comm, FILE *tree, FILE *tsv, const char *label, long long step,
                    TimerTreeSort sort, const std::vector<TimerRecord> *snapshot,
                    const TimerReportContext *context)
{
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    std::map<std::string, Totals> local;
    const auto captured = snapshot ? std::vector<TimerRecord>() : TimerSnapshot();
    for(const auto &record : snapshot ? *snapshot : captured) {
        auto &sum = local[std::string(record.thread == 0 ? "main\t" : "workers\t") + record.path];
        sum.calls += record.calls;
        sum.completed += record.completed;
        sum.active += record.active;
        sum.inclusive += record.inclusive;
        sum.exclusive += record.exclusive;
    }
    std::string local_keys;
    for(const auto &entry : local) {local_keys += entry.first + '\n';}

    // MPI's counts/displacements are ints. Refuse oversized reports collectively
    // rather than truncate lengths or let only some ranks enter a collective.
    int oversized = local_keys.size() > INT_MAX ? 1 : 0, any_oversized = 0;
    MPI_Allreduce(&oversized, &any_oversized, 1, MPI_INT, MPI_MAX, comm);
    if(any_oversized) {return false;}
    const int local_size = static_cast<int>(local_keys.size());
    std::vector<int> sizes(rank == 0 ? ranks : 0), offsets(rank == 0 ? ranks : 0);
    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, comm);
    long long total = 0;
    if(rank == 0) {
        for(int r = 0; r < ranks; ++r) {
            if(total <= INT_MAX) {offsets[r] = static_cast<int>(total);}
            total += sizes[r];
        }
        oversized = total > INT_MAX ? 1 : 0;
    }
    MPI_Bcast(&oversized, 1, MPI_INT, 0, comm);
    if(oversized) {return false;}
    std::vector<char> gathered(rank == 0 ? static_cast<std::size_t>(total) : 0);
    MPI_Gatherv(local_keys.data(), local_size, MPI_CHAR, gathered.data(), sizes.data(),
                offsets.data(), MPI_CHAR, 0, comm);

    std::string canonical;
    if(rank == 0) {
        std::set<std::string, PathLess> keys;
        if(!gathered.empty()) {
            for(const auto &key : lines(std::string(gathered.data(), gathered.size()))) {keys.insert(key);}
        }
        for(const auto &key : keys) {canonical += key + '\n';}
    }
    int canonical_size = static_cast<int>(canonical.size());
    MPI_Bcast(&canonical_size, 1, MPI_INT, 0, comm);
    canonical.resize(canonical_size);
    MPI_Bcast(canonical.data(), canonical_size, MPI_CHAR, 0, comm);
    const auto keys = lines(canonical);
    // Each key consumes four uint64 values in the largest reduction.
    if(keys.size() > static_cast<std::size_t>(INT_MAX / 4)) {return false;}
    const int n = static_cast<int>(keys.size());
    std::vector<double> values(2 * n, 0), sums(2 * n), minima(2 * n), maxima(2 * n);
    std::vector<std::uint64_t> counts(4 * n, 0), count_sums(4 * n);
    for(int i = 0; i < n; ++i) {
        const auto entry = local.find(keys[i]);
        if(entry == local.end()) {continue;}
        const auto &sum = entry->second;
        values[2 * i] = sum.inclusive;
        values[2 * i + 1] = sum.exclusive;
        counts[4 * i] = sum.calls;
        counts[4 * i + 1] = sum.completed;
        counts[4 * i + 2] = sum.active;
        counts[4 * i + 3] = 1;
    }
    MPI_Reduce(values.data(), sums.data(), 2 * n, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(values.data(), minima.data(), 2 * n, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(values.data(), maxima.data(), 2 * n, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(counts.data(), count_sums.data(), 4 * n, MPI_UINT64_T, MPI_SUM, 0, comm);
    const std::uint64_t errors = TimerErrorCount();
    std::uint64_t error_sum = 0;
    MPI_Reduce(&errors, &error_sum, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
    const TimerReportContext empty_context;
    const auto &info = context ? *context : empty_context;
    const long long work[4] = {info.active_particles_local, info.active_gas_local,
                              info.previous_active_particles_local, info.previous_active_gas_local};
    long long work_sum[4], work_max[4];
    MPI_Reduce(work, work_sum, 4, MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(work, work_max, 4, MPI_LONG_LONG, MPI_MAX, 0, comm);
    if(rank != 0) {return true;}
    for(int i = 0; i < 4; ++i) {if(work_sum[i] < 0) {work_sum[i] = work_max[i] = -1;}}
    const double active_mean = work_sum[0] >= 0 ? static_cast<double>(work_sum[0]) / ranks : -1;
    const double gas_mean = work_sum[1] >= 0 ? static_cast<double>(work_sum[1]) / ranks : -1;
    const double previous_active_mean = work_sum[2] >= 0 ? static_cast<double>(work_sum[2]) / ranks : -1;
    const double previous_gas_mean = work_sum[3] >= 0 ? static_cast<double>(work_sum[3]) / ranks : -1;

    if(tree) {
        const std::string mode(info.mode);
        const char *description = mode == "step" ? "evolution interval" :
                                  mode == "initial" ? "initial sync-point work" :
                                  mode == "partial" ? "partial interval" : info.mode;
        std::fprintf(tree, "\nTimer report: %s | Sync-Point %lld | ranks=%d | errors(total)=%llu",
                     description, step, ranks, static_cast<unsigned long long>(error_sum));
        if(std::string(label) != "sync-point" && std::string(label) != "initial-sync-point") {
            std::fprintf(tree, " | event=%s", label);
        }
        std::fprintf(tree, "\n");
        if(info.from_sync_point >= 0) {
            std::fprintf(tree, "Window: Sync-Point %lld -> %lld\n", info.from_sync_point, step);
        } else if(mode == "cumulative") {
            std::fprintf(tree, "Window: process startup through this sync-point (includes initialization).\n");
        }
        std::fprintf(tree, "Means include absent ranks as zero; row maxima need not add up. Imbalance = max/mean.\n"
                     "Siblings sorted by descending %s; branches preserve each subtree.\n",
                     sort == TimerTreeSort::InclusiveMean ? "incl_mean" : "incl_max");
        std::fprintf(tree, "Workload: time=%.17g dt=%.17g particles=%lld active=%lld gas_active=%lld "
                     "active_rank_mean=%.6g active_rank_max=%lld highest_active_bin=%d (-1 = unavailable)\n",
                     info.simulation_time, info.simulation_dt, info.particles_total,
                     work_sum[0], work_sum[1], active_mean, work_max[0], info.highest_active_bin);
        std::fprintf(tree, "Interval-entry workload: active=%lld gas_active=%lld active_rank_mean=%.6g "
                     "active_rank_max=%lld (before timestep selection and first half-kick)\n",
                     work_sum[2], work_sum[3], previous_active_mean, work_max[2]);
        // Ranking by sums is equivalent to ranking by means: the divisor is
        // the same number of MPI ranks for every row.
        const auto rows = sorted_tree_rows(keys, sort == TimerTreeSort::InclusiveMean ? sums : maxima);
        std::string displayed_lane;
        for(const auto &row : rows) {
            const int i = row.index;
            const auto tab = keys[i].find('\t');
            const auto lane = keys[i].substr(0, tab);
            const auto path = keys[i].substr(tab + 1);
            if(lane != displayed_lane) {
                displayed_lane = lane;
                std::fprintf(tree, "\n%s\n", lane == "main" ?
                             "Main threads: host elapsed seconds per MPI rank" :
                             "Worker threads: summed thread-seconds per MPI rank");
                std::fprintf(tree, "%12s %12s %12s %12s %10s %10s %12s %8s %11s  %s\n",
                             "incl_mean", "incl_max", "self_mean", "self_max", "incl_imb", "self_imb", "calls_sum",
                             "ranks", "open_scopes", "region");
            }
            const auto last = path.rfind("::");
            const char *leaf = path.c_str() + (last == std::string::npos ? 0 : last + 2);
            const auto *c = &count_sums[4 * i];
            std::fprintf(tree, "%12.6g %12.6g %12.6g %12.6g %10.4g %10.4g %12llu %8llu %11llu  %s%s\n",
                         sums[2 * i] / ranks, maxima[2 * i],
                         sums[2 * i + 1] / ranks, maxima[2 * i + 1],
                         imbalance(sums[2 * i], maxima[2 * i], ranks),
                         imbalance(sums[2 * i + 1], maxima[2 * i + 1], ranks),
                         static_cast<unsigned long long>(c[0]), static_cast<unsigned long long>(c[3]),
                         static_cast<unsigned long long>(c[2]), row.prefix.c_str(), leaf);
        }
    }
    for(int i = 0; i < n; ++i) {
        const auto tab = keys[i].find('\t');
        const auto lane = keys[i].substr(0, tab);
        const auto path = keys[i].substr(tab + 1);
        const auto *c = &count_sums[4 * i];
        if(tsv) {
            std::fprintf(tsv, "%s\t%lld\t%s\t%s\t%llu\t%llu\t%llu\t%llu"
                         "\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g"
                         "\t%.17g\t%.17g\t%s\t%.17g\t%.17g\t%lld"
                         "\t%lld\t%.17g\t%lld\t%lld\t%.17g\t%lld\t%d\t%lld"
                         "\t%lld\t%.17g\t%lld\t%lld\t%.17g\t%lld\n",
                         label, step, lane.c_str(), path.c_str(),
                         static_cast<unsigned long long>(c[3]), static_cast<unsigned long long>(c[0]),
                         static_cast<unsigned long long>(c[1]), static_cast<unsigned long long>(c[2]),
                         minima[2 * i], sums[2 * i] / ranks, maxima[2 * i],
                         minima[2 * i + 1], sums[2 * i + 1] / ranks, maxima[2 * i + 1],
                         imbalance(sums[2 * i], maxima[2 * i], ranks),
                         imbalance(sums[2 * i + 1], maxima[2 * i + 1], ranks), info.mode,
                         info.simulation_time, info.simulation_dt, info.particles_total,
                         work_sum[0], active_mean, work_max[0], work_sum[1], gas_mean,
                         work_max[1], info.highest_active_bin, info.from_sync_point,
                         work_sum[2], previous_active_mean, work_max[2], work_sum[3],
                         previous_gas_mean, work_max[3]);
        }
    }
    if(tree) {std::fflush(tree);}
    if(tsv) {std::fflush(tsv);}
    return true;
}
} // namespace timing
} // namespace gizmo
