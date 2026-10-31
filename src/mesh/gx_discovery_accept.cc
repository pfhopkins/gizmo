/* mesh/gx_discovery_accept.cc -- the exact accept of request-driven ghost discovery, for the host and the device.
 *
 * Which supply particles a rank sends is decided by gx_pair_accept_wrap_and_test (ghost_exchange_functions.h).
 * This translation unit is compiled without contraction to fused multiply-add on both sides (target-specific flags
 * in the Makefile), and both instantiations that decide a send set live here, so the host accept
 * (gx_send_set_accept_rows) and the device one give the same bits.  Contract: mesh/gpu_neighbor_list.h. */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h (device-pass All redirect) */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "neighbor_list.h"
#include "ghost_exchange_functions.h"
#include "gpu_neighbor_list.h"

int gx_discovery_accept_host(double dx0, double dy0, double dz0, double h_q, double h_j, int search_mode)
{
    return gx_pair_accept_wrap_and_test(dx0, dy0, dz0, h_q, h_j, search_mode);
}

int gx_discovery_device_accept_on(void)
{
    static int on = -1;
    if(on < 0) {
        const char *env = getenv("GIZMO_GPU_GHOST_ACCEPT");
        on = (env && env[0]) ? (atoi(env) != 0) : 1;
        if(ThisTask == 0) {printf("Ghost discovery: exact accept on the device when every candidate is current (GIZMO_GPU_GHOST_ACCEPT) is %s.\n", on ? "on" : "off"); fflush(stdout);}
    }
    return on;
}

void gx_discovery_accept_rows_on_device(const struct gx_export_envelope_t *env, const int64_t *offsets, int r0, int r1,
                                        int64_t sub_base, int *pairs, const struct gx_recv_leaf_t *leaves, int search_mode)
{
    if(r1 <= r0) {return;}
    GIZMO_GPU_ENSURE_ALL_FRESH();   /* the wrap reads the box size through this unit's All mirror */
    Kokkos::parallel_for("gx_discovery_accept", Kokkos::RangePolicy<>(r0, r1), KOKKOS_LAMBDA(int b) {
        const struct gx_export_envelope_t &e = env[b];
        const int64_t c1 = offsets[b + 1] - sub_base;
        for(int64_t c = offsets[b] - sub_base; c < c1; c++) {
            const struct gx_recv_leaf_t &L = leaves[pairs[c]];
            pairs[c] = gx_pair_accept_wrap_and_test(e.pos[0] - L.pos[0], e.pos[1] - L.pos[1], e.pos[2] - L.pos[2],
                                                    e.h, L.hj_accept, search_mode) ? L.pool : -1;
        }
    });
}
