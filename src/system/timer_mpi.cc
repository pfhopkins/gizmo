#include "timer_mpi.h"
#include "timer.h"

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
} // namespace

void TimerWriteTSVHeader(FILE *stream)
{
    if(stream) {
        std::fprintf(stream, "label\tstep\tlane\tpath\tranks_present\tcalls_sum\tcompleted_sum\tactive_sum"
                     "\tinclusive_min_s\tinclusive_mean_s\tinclusive_max_s"
                     "\texclusive_min_s\texclusive_mean_s\texclusive_max_s\n");
    }
}

bool TimerReportMPI(MPI_Comm comm, FILE *tree, FILE *tsv, const char *label, long long step)
{
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    std::map<std::string, Totals> local;
    for(const auto &record : TimerSnapshot()) {
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
    if(rank != 0) {return true;}

    if(tree) {
        std::fprintf(tree, "\nTimer snapshot: %s step=%lld ranks=%d errors=%llu (cumulative)\n"
                     "Main rows: host elapsed seconds. Worker rows: summed thread seconds.\n"
                     "Rank means include zero for absent regions; row maxima need not add up.\n"
                     "%-8s %12s %12s %12s %12s %12s %8s %8s  %s\n",
                     label, step, ranks, static_cast<unsigned long long>(error_sum),
                     "lane", "incl_mean", "incl_max", "self_mean", "self_max", "calls_sum",
                     "ranks", "active", "region");
    }
    for(int i = 0; i < n; ++i) {
        const auto tab = keys[i].find('\t');
        const auto lane = keys[i].substr(0, tab);
        const auto path = keys[i].substr(tab + 1);
        int depth = 0;
        for(std::size_t p = 0; (p = path.find("::", p)) != std::string::npos; p += 2) {++depth;}
        const auto last = path.rfind("::");
        const char *leaf = path.c_str() + (last == std::string::npos ? 0 : last + 2);
        const auto *c = &count_sums[4 * i];
        if(tree) {
            std::fprintf(tree, "%-8s %12.6g %12.6g %12.6g %12.6g %12llu %8llu %8llu  %*s%s\n",
                         lane.c_str(), sums[2 * i] / ranks, maxima[2 * i],
                         sums[2 * i + 1] / ranks, maxima[2 * i + 1],
                         static_cast<unsigned long long>(c[0]), static_cast<unsigned long long>(c[3]),
                         static_cast<unsigned long long>(c[2]), 2 * depth, "", leaf);
        }
        if(tsv) {
            std::fprintf(tsv, "%s\t%lld\t%s\t%s\t%llu\t%llu\t%llu\t%llu"
                         "\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",
                         label, step, lane.c_str(), path.c_str(),
                         static_cast<unsigned long long>(c[3]), static_cast<unsigned long long>(c[0]),
                         static_cast<unsigned long long>(c[1]), static_cast<unsigned long long>(c[2]),
                         minima[2 * i], sums[2 * i] / ranks, maxima[2 * i],
                         minima[2 * i + 1], sums[2 * i + 1] / ranks, maxima[2 * i + 1]);
        }
    }
    if(tree) {std::fflush(tree);}
    if(tsv) {std::fflush(tsv);}
    return true;
}
} // namespace timing
} // namespace gizmo
