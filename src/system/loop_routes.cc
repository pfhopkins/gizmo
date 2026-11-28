/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* OUTPUT_LOOP_ROUTES: rank-local counters of the route each named loop took (system/loop_routes.h), gathered to task 0
   and written to loop_routes.txt at every cpu.txt entry (one gather of a few hundred bytes per rank, test builds only). */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <string>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "loop_routes.h"

#ifdef OUTPUT_LOOP_ROUTES
static const char *loop_route_names[LOOP_ROUTE_N] = {"device_mode_A", "device_mode_D", "host_mode_B", "host_mode_D", "device_kernel", "host_kernel", "host_plain"};
struct loop_route_row {long long calls[LOOP_ROUTE_N], elements[LOOP_ROUTE_N];};
static std::map<std::string, loop_route_row> loop_routes_local;

void loop_route_count(const char *name, int route, long long n_elements)
{
    if(route < 0 || route >= LOOP_ROUTE_N) {return;}
    loop_route_row &r = loop_routes_local[name]; /* value-initialised on first use: zeros */
    r.calls[route]++; r.elements[route] += n_elements;
}

void loop_routes_write(void)
{
    std::string text; char buf[64]; /* one line per loop: name, then calls and elements per route */
    for(const auto &kv : loop_routes_local)
    {
        text += kv.first;
        for(int k = 0; k < LOOP_ROUTE_N; k++) {snprintf(buf, sizeof(buf), " %lld %lld", kv.second.calls[k], kv.second.elements[k]); text += buf;}
        text += "\n";
    }
    int len = (int) text.size();
    std::vector<int> lens(NTask), offs(NTask);
    MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    long long total = 0;
    if(ThisTask == 0) {for(int t = 0; t < NTask; t++) {offs[t] = (int) total; total += lens[t];}}
    std::vector<char> all(ThisTask == 0 ? (size_t) total + 1 : 1);
    MPI_Gatherv(text.data(), len, MPI_CHAR, all.data(), lens.data(), offs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    all[total] = 0;
    std::map<std::string, loop_route_row> sum; /* summed over ranks */
    char *save = NULL;
    for(char *line = strtok_r(all.data(), "\n", &save); line; line = strtok_r(NULL, "\n", &save))
    {
        char name[256]; int used = 0;
        if(sscanf(line, "%255s%n", name, &used) != 1) {continue;}
        loop_route_row &r = sum[name]; const char *p = line + used;
        for(int k = 0; k < LOOP_ROUTE_N; k++)
        {
            long long c = 0, e = 0; int m = 0;
            if(sscanf(p, " %lld %lld%n", &c, &e, &m) != 2) {break;}
            r.calls[k] += c; r.elements[k] += e; p += m;
        }
    }
    char fn[DEFAULT_PATH_BUFFERSIZE_TOUSE]; snprintf(fn, sizeof(fn), "%sloop_routes.txt", All.OutputDir);
    FILE *fd = fopen(fn, "w"); if(!fd) {printf("OUTPUT_LOOP_ROUTES: cannot open %s, routes not written\n", fn); return;}
    fprintf(fd, "# step %lld time %.10g ranks %d: calls and rank-local elements summed over ranks since the start; Mode D on one rank counts per evaluation batch, and a Mode D call that falls back to the host walker counts once more under host_mode_B\n# loop", (long long) All.NumCurrentTiStep, All.Time, NTask);
    for(int k = 0; k < LOOP_ROUTE_N; k++) {fprintf(fd, " %s_calls %s_elements", loop_route_names[k], loop_route_names[k]);}
    fprintf(fd, "\n");
    for(const auto &kv : sum)
    {
        fprintf(fd, "%s", kv.first.c_str());
        for(int k = 0; k < LOOP_ROUTE_N; k++) {fprintf(fd, " %lld %lld", kv.second.calls[k], kv.second.elements[k]);}
        fprintf(fd, "\n");
    }
    fclose(fd);
}
#endif
