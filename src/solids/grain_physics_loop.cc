/* solids/grain_physics_loop.cc — host hooks + ghost-writeback bundle for
 * GrainBackrxSpec (GRAIN_BACKREACTION) and GrainRTGasSpec + GrainRTGrainSpec
 * (RT_OPACITY_FROM_EXPLICIT_GRAINS), plus toplevel callers
 * grain_backrx_calc() and grain_rt_opacity_calc().
 *
 * Device-callable hooks (pair_kernel, zero_accum, load_active, load_neighbor)
 * live in grain_physics_loop.h so the runner instantiates them from GPU TUs.
 * This file owns: is_active, search_radius, populate_call_scalars,
 * apply_active_writeback, merge_accum, ghost_writeback_begin/end
 * (GrainBackrxSpec), and the two toplevels.
 *
 * Replaces grain_backrx_evaluate_gpu + ghost_writeback_{zero_,}grainbackrx
 * and interpolate_fluxes_opacities_gasgrains_evaluate_gpu from
 * solids/grain_physics_gpu.cc (retired in the cleanup commit).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../declarations/multifluid_helpers.h"  /* IS_PARTICLE_DRAGVALID + FLUID_* IDs for GrainBackrxSpec::is_active */
#include "../core/proto.h"
#include "../mesh/kernel.h"            /* MUST precede grain_physics_loop.h (no include guards) */
#include "grain_physics_loop.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_writeback_ops.h"   /* GHOST_WRITEBACK_BUNDLE_* manifest macros — explicit include, not pulled transitively (matches thermal_fb_loop.cc) */

#ifdef DO_FLUID_ALTSPECIES_DRAG_CALCULATION


/* ============================================================================
 * GrainBackrxSpec host hooks.
 * ========================================================================== */
#if defined(GRAIN_BACKREACTION)

bool GrainBackrxSpec::is_active(int i)
{
    return (IS_PARTICLE_DRAGVALID(P[i].Type, P[i].FluidType)
            && P[i].TimeBin >= 0
            && P[i].Mass > 0
            && P[i].KernelRadius > 0);
}

double GrainBackrxSpec::search_radius(const neighbor_loop_args& args,
                                      int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

GrainBackrxSpec::CallScalars
GrainBackrxSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    return nlr_common_scalars_from_all();
}

/* No i-side accumulation — apply_active_writeback is a no-op. */
void GrainBackrxSpec::apply_active_writeback(const neighbor_loop_args& /*args*/,
                                             int /*active_slot*/, int /*i*/,
                                             const AccumData& /*accum*/)
{}


/* Ghost-writeback bundle for GrainBackrxSpec.
 *
 * MANIFEST — one macro per ghost-written field.
 * Coverage matches the legacy ghost_writeback_{zero_,}grainbackrx in
 * mesh/ghost_writeback.cc (retired in the cleanup commit):
 *   P[j].Vel[k]                        (always, Vec3)
 *   P[j].dp[k]                         (always, Vec3)
 *   P[j].Grain_AccelTimeMin            (always, min-reduce)
 *   CellP[j].VelPred[k]               (always, Vec3)
 *   CellP[j].VolatileSpecies[kv]       (GRAIN_EVOLUTION & (32|64), array)
 *   CellP[j].InternalEnergy            (GRAIN_EVOLUTION & (32|64))
 *   CellP[j].InternalEnergyPred        (GRAIN_EVOLUTION & (32|64))
 *
 * All ops already exist in mesh/ghost_writeback_ops.h; no new ops required.
 * PARTICLE_MIN uses post<snap predicate — correct for atomic_min semantics. */
GHOST_WRITEBACK_BUNDLE_BEGIN(grainbackrx)
    GHOST_WRITEBACK_PARTICLE_ADD_VEC3(Vel)
    GHOST_WRITEBACK_PARTICLE_ADD_VEC3(dp)
    GHOST_WRITEBACK_PARTICLE_MIN(Grain_AccelTimeMin)
    GHOST_WRITEBACK_GAS_ADD_VEC3(VelPred)
#if defined(GRAIN_EVOLUTION) && (GRAIN_EVOLUTION & (32|64))
    GHOST_WRITEBACK_GAS_ADD_ARRAY(VolatileSpecies, GRAIN_NUM_VOLATILE_SPECIES)
    GHOST_WRITEBACK_GAS_ADD(InternalEnergy)
    GHOST_WRITEBACK_GAS_ADD(InternalEnergyPred)
#endif
GHOST_WRITEBACK_BUNDLE_END(grainbackrx)

void GrainBackrxSpec::ghost_writeback_begin(const neighbor_loop_args& /*args*/,
                                            const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_begin_bundle(grainbackrx_ghost_writeback_bundle_ptr());
}

void GrainBackrxSpec::ghost_writeback_end(const neighbor_loop_args& /*args*/,
                                          const NeighborLoopPlan& /*plan*/)
{
    ghost_writeback_end_bundle(grainbackrx_ghost_writeback_bundle_ptr());
}

/* Toplevel: grain_backrx_calc — replaces grain_backrx_evaluate_gpu +
 * ghost_writeback_{zero_,}grainbackrx.
 *
 * Hard invariant: active gas Grain_AccelTimeMin is reset to
 * MAX_REAL_NUMBER in apply_grain_dragforce() BEFORE this call. PARTICLE_MIN
 * only reduces the field; stale low values would permanently collapse gas
 * timesteps. This reset lives in grain_physics.cc, not here. */
void grain_backrx_calc(void)
{
    PRINT_STATUS(" ..assigning grain back-reaction to gas [runner]");

    /* Build the active-grain list via the engine helper (filters by
     * GrainBackrxSpec::is_active, MPI_Allreduce-gates on the global count).
     * Without this the runner sees num_active=0 and does nothing. */
    int *active_list = nullptr;
    int  num_active = 0, num_global_active = 0;
    if(!nlr_build_active_list(GrainBackrxSpec::is_active,
                               &active_list, &num_active, &num_global_active,
                               "grain_backrx_active_list")) {
        CPU_Step[CPU_DRAGFORCE] += measure_time();
        return;
    }

    neighbor_loop_args args = nlr_default_args();
    args.active_list = active_list;
    args.num_active  = num_active;
    run_neighbor_loop<GrainBackrxSpec>(args);

    nlr_free_active_list(active_list);
    CPU_Step[CPU_DRAGFORCE] += measure_time();
}

#endif /* GRAIN_BACKREACTION */


/* ============================================================================
 * GrainRTGasSpec host hooks.
 * ========================================================================== */
#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)

bool GrainRTGasSpec::is_active(int i)
{
    return (P[i].Type == 0
            && P[i].TimeBin >= 0
            && P[i].Mass > 0
            && P[i].KernelRadius > 0);
}

double GrainRTGasSpec::search_radius(const neighbor_loop_args& args,
                                     int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

GrainRTGasSpec::CallScalars
GrainRTGasSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    return nlr_common_scalars_from_all();
}

/* Additive i-side scatter: += matches legacy ASSIGN_ADD(mode=0) semantics.
 * The toplevel grain_rt_opacity_calc pre-zeroes J_dust_cell under
 * MHD_BATTERY_MECHANISMS & 8 before the runner call (design §G invariant 2). */
void GrainRTGasSpec::apply_active_writeback(const neighbor_loop_args& args,
                                            int /*active_slot*/, int i,
                                            const AccumData& accum)
{
    args.CellP[i].InterpolatedGeometricDustCrossSection +=
        accum.InterpolatedGeometricDustCrossSection;
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {
        args.CellP[i].Interpolated_Opacity[k] += accum.Interpolated_Opacity[k];
    }
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
    for(int k = 0; k < 3; k++) {
        args.CellP[i].J_dust_cell[k] += (MyDouble)accum.J_dust_contribution[k];
    }
#endif
}


/* ============================================================================
 * GrainRTGrainSpec host hooks.
 * ========================================================================== */

bool GrainRTGrainSpec::is_active(int i)
{
    return (((1 << P[i].Type) & (GRAIN_PTYPES))
            && P[i].TimeBin >= 0
            && P[i].Mass > 0
            && P[i].KernelRadius > 0);
}

double GrainRTGrainSpec::search_radius(const neighbor_loop_args& args,
                                       int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

GrainRTGrainSpec::CallScalars
GrainRTGrainSpec::populate_call_scalars(const neighbor_loop_args& /*args*/)
{
    return nlr_common_scalars_from_all();
}

/* Scatter radiation acceleration onto the grain source.
 * Division by All.cf_a2inv matches the legacy scatter in grain_physics_gpu.cc. */
void GrainRTGrainSpec::apply_active_writeback(const neighbor_loop_args& args,
                                              int /*active_slot*/, int i,
                                              const AccumData& accum)
{
    args.P[i].GravAccel += accum.Interpolated_Radiation_Acceleration / All.cf_a2inv;
}


/* ============================================================================
 * grain_rt_opacity_calc — toplevel for both RT Specs.
 *
 * Hard invariant: under MHD_BATTERY_MECHANISMS & 8, active gas
 * CellP[i].J_dust_cell is zeroed here before run_neighbor_loop<GrainRTGasSpec>.
 * GrainRTGasSpec::apply_active_writeback uses += so without this pre-zero the
 * field would accumulate stale values from previous steps.
 * ========================================================================== */
void grain_rt_opacity_calc(void)
{
    PRINT_STATUS(" ..interpolating RT fluxes/opacities from grain distribution [runner]");

    /* Direction 1: gas sources → grain neighbors (opacity accumulation).
     * Pre-zero all fields that apply_active_writeback accumulates with +=.
     * InterpolatedGeometricDustCrossSection and Interpolated_Opacity are reset
     * unconditionally (legacy only zeroed J_dust_cell; this fixes the stale
     * cross-section/opacity accumulation bug where prior-step values persisted). */
    {
        for(int i : ActiveParticleList) {
            if(P[i].Type == 0 && P[i].Mass > 0) {
                CellP[i].InterpolatedGeometricDustCrossSection = 0;
                for(int k = 0; k < N_RT_FREQ_BINS; k++) CellP[i].Interpolated_Opacity[k] = 0;
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 8)
                CellP[i].J_dust_cell = {};
#endif
            }
        }
        int *active_list = nullptr;
        int  num_active = 0, num_global_active = 0;
        if(nlr_build_active_list(GrainRTGasSpec::is_active,
                                  &active_list, &num_active, &num_global_active,
                                  "grain_rt_gas_active_list")) {
            neighbor_loop_args args = nlr_default_args();
            args.active_list = active_list;
            args.num_active  = num_active;
            run_neighbor_loop<GrainRTGasSpec>(args);
            nlr_free_active_list(active_list);
        }
    }

    /* Direction 2: grain sources → gas neighbors (radiation acceleration). */
    {
        int *active_list = nullptr;
        int  num_active = 0, num_global_active = 0;
        if(nlr_build_active_list(GrainRTGrainSpec::is_active,
                                  &active_list, &num_active, &num_global_active,
                                  "grain_rt_grain_active_list")) {
            neighbor_loop_args args = nlr_default_args();
            args.active_list = active_list;
            args.num_active  = num_active;
            run_neighbor_loop<GrainRTGrainSpec>(args);
            nlr_free_active_list(active_list);
        }
    }

    CPU_Step[CPU_DRAGFORCE] += measure_time();
}

#endif /* RT_OPACITY_FROM_EXPLICIT_GRAINS */

#endif /* DO_FLUID_ALTSPECIES_DRAG_CALCULATION */
