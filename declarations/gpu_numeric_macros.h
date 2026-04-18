/* gpu_numeric_macros.h — GPU-safe overrides of host-only math macros.
 *
 * glibc's isfinite / isnan expand to functions tagged host-only.  nvcc stubs
 * them to return 0 on device, so every device-side value appears non-finite.
 * The arithmetic replacements below are identical on host and device and
 * compile to cheap PTX/HIP comparison intrinsics.
 *
 * Include AFTER <math.h> (or the TU's stdlib preamble) and BEFORE any
 * _functions.h header that uses isfinite/isnan on device.
 *
 * No-op on pure-CPU builds (GIZMO_GPU_COMPILER undefined).
 */
#ifndef GIZMO_GPU_NUMERIC_MACROS_H
#define GIZMO_GPU_NUMERIC_MACROS_H

#ifdef GIZMO_GPU_COMPILER
#undef isfinite
#undef isnan
#define isfinite(x) (((double)(x) == (double)(x)) && ((double)(x) - (double)(x) == 0.0))
#define isnan(x) ((double)(x) != (double)(x))
#endif

#endif /* GIZMO_GPU_NUMERIC_MACROS_H */
