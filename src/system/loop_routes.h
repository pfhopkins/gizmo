/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* OUTPUT_LOOP_ROUTES: where each named loop ran -- device neighbour-loop Mode A or D, the host walker (Mode B), a Kokkos
   kernel on the device or on the host execution space, or a plain host loop -- counted per call and per rank-local
   element, summed over ranks into loop_routes.txt (rewritten at every cpu.txt entry), so a test can assert that a loop
   ran on the device. Without the flag the calls compile to nothing. */
#ifndef GIZMO_LOOP_ROUTES_H
#define GIZMO_LOOP_ROUTES_H

enum loop_route
{
    LOOP_ROUTE_DEVICE_MODE_A = 0,  /* neighbour loop, device NGL */
    LOOP_ROUTE_DEVICE_MODE_D = 1,  /* neighbour loop, fused device walk */
    LOOP_ROUTE_HOST_MODE_B   = 2,  /* neighbour loop, host walker */
    LOOP_ROUTE_DEVICE_KERNEL = 3,  /* Kokkos kernel on the device execution space */
    LOOP_ROUTE_HOST_KERNEL   = 4,  /* the same Kokkos kernel on the host execution space */
    LOOP_ROUTE_HOST_PLAIN    = 5,  /* a plain host loop */
    LOOP_ROUTE_N             = 6
};

#ifdef OUTPUT_LOOP_ROUTES
void loop_route_count(const char *name, int route, long long n_elements); /* rank-local, no communication */
void loop_routes_write(void);                                              /* collective: every rank, from write_cpu_log */
#else
static inline void loop_route_count(const char *, int, long long) {}
static inline void loop_routes_write(void) {}
#endif

#endif
