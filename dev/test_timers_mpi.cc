#include "../src/system/timer.h"
#include "../src/system/timer_mpi.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace gizmo::timing;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    TimerInitialize(true);
    FILE *output = rank == 0 ? std::tmpfile() : nullptr;
    assert(rank != 0 || output);
    if(rank == 0) {TimerWriteTSVHeader(output);}
    // Report an empty registry too, including zero-length collective buffers.
    assert(TimerReportMPI(MPI_COMM_WORLD, nullptr, output, "empty", 0));

    // Rank 0 registers A before Shared; rank 1 registers them in reverse order.
    // Remaining ranks never enter the main-thread Shared region.
    if(rank == 0) {TimerStart("A"); TimerStop();}
    if(rank < 2) {
        for(int i = 0; i <= rank; ++i) {GIZMO_TIMER_SCOPE("Shared");}
    }
    // Same name on workers must be reported in a separate lane.
    std::thread worker([] {
        for(int i = 0; i < 2; ++i) {GIZMO_TIMER_SCOPE("Shared");}
    });
    worker.join();
    TimerStart("A"); TimerStart("Child"); TimerStop(); TimerStop();
    TimerStart("A!"); TimerStop();
    if(rank == 1) {
        TimerStart("A"); TimerStop();
        TimerStart("OnlyRank1"); // deliberately active at the snapshot
    }
    assert(TimerReportMPI(MPI_COMM_WORLD, nullptr, output, "union", 7));
    if(rank == 1) {assert(TimerStop());}

    if(rank == 0) {
        std::rewind(output);
        char buffer[4096];
        bool seen_shared = false, seen_unique = false, seen_workers = false;
        int parent_position = -1, child_position = -1, sibling_position = -1, position = 0;
        while(std::fgets(buffer, sizeof(buffer), output)) {
            std::istringstream stream(buffer);
            std::vector<std::string> fields;
            std::string field;
            while(std::getline(stream, field, '\t')) {fields.push_back(field);}
            if(fields.size() != 14 || fields[0] != "union") {continue;}
            ++position;
            if(fields[2] == "main" && fields[3] == "A") {parent_position = position;}
            if(fields[2] == "main" && fields[3] == "A::Child") {child_position = position;}
            if(fields[2] == "main" && fields[3] == "A!") {sibling_position = position;}
            if(fields[2] == "workers" && fields[3] == "Shared") {
                seen_workers = true;
                assert(std::stoi(fields[4]) == ranks);
                assert(std::stoi(fields[5]) == 2 * ranks && std::stoi(fields[6]) == 2 * ranks);
                continue;
            }
            if(fields[3] == "Shared") {
                seen_shared = true;
                assert(std::stoi(fields[4]) == (ranks > 1 ? 2 : 1));
                assert(std::stoi(fields[5]) == (ranks > 1 ? 3 : 1));
                assert(std::stoi(fields[6]) == (ranks > 1 ? 3 : 1));
                assert(std::stod(fields[10]) >= std::stod(fields[9]));
                if(ranks > 2) {assert(std::stod(fields[8]) == 0);}
            }
            if(fields[3] == "OnlyRank1") {
                seen_unique = true;
                assert(fields[4] == "1" && fields[5] == "1" && fields[6] == "0" && fields[7] == "1");
                assert(std::stod(fields[8]) == 0); // absent on rank 0
            }
        }
        assert(seen_shared && seen_workers && seen_unique == (ranks > 1));
        assert(parent_position < child_position && child_position < sibling_position);
        std::fclose(output);
        std::printf("PASS: %d-rank reports reconcile different registration orders, missing regions and live calls\n", ranks);
    }
    TimerShutdown();
    MPI_Finalize();
}
