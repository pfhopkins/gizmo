/* grain_promotion.h — Phase 17g: Type 3 grain super-particle → Type 0 solid body conversion.
 * Header-only helpers; grain_promotion.cc provides grain_promotion_parent_routine().
 * Requires includer to have pulled in allvars.h.
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO. */

#ifndef GRAIN_PROMOTION_H
#define GRAIN_PROMOTION_H

#if defined(GRAIN_FLUID) && defined(GRAIN_FLUID_PROMOTION)

#include <math.h>
#include "../declarations/allvars.h"
#include "../eos/composition_registry.h"

/* Map Composition[0..GRAIN_NUM_SPECIES-1] argmax → Tillotson CompositionType integer.
 *   0 silicate → MATERIAL_TILLOTSON_OLIVINE (=5)
 *   1 carbon   → MATERIAL_TILLOTSON_BASALT  (=2, closest graphite proxy)
 *   2 iron     → MATERIAL_TILLOTSON_IRON    (=3)
 *   3..5 ice   → MATERIAL_TILLOTSON_ICE     (=4)
 * Falls back to MATERIAL_TILLOTSON_OLIVINE if all zeros.
 * Only available when GRAIN_EVOLUTION is defined (provides Composition[] field). */
#ifdef GRAIN_EVOLUTION
static inline int grain_composition_to_eos_type(MyFloat *Composition)
{
    int best = 0;
    MyFloat best_val = Composition[0];
    for(int s = 1; s < GRAIN_NUM_SPECIES; s++) {
        if(Composition[s] > best_val) { best_val = Composition[s]; best = s; }
    }
    if(best == 0) return MATERIAL_TILLOTSON_OLIVINE;
    if(best == 1) return MATERIAL_TILLOTSON_BASALT;
    if(best == 2) return MATERIAL_TILLOTSON_IRON;
    return MATERIAL_TILLOTSON_ICE;
}
#endif /* GRAIN_EVOLUTION */

/* Initialize CellP[i] for a freshly promoted solid body.
 * Call this BEFORE changing P[i].Type to 0. */
static inline void grain_promotion_init_cellp(int i)
{
#ifdef GRAIN_EVOLUTION
    int ct = grain_composition_to_eos_type(P[i].Composition);
#else
    int ct = MATERIAL_TILLOTSON_OLIVINE;
#endif
    CellP[i].CompositionType = ct;

    double rho0   = All.Tillotson_EOS_params[ct][3];
    double A_mod  = All.Tillotson_EOS_params[ct][4];
    double u_s    = All.Tillotson_EOS_params[ct][6];

    CellP[i].Density               = rho0;
#ifdef HYDRO_EXPLICITLY_INTEGRATE_VOLUME
    CellP[i].Density_ExplicitInt   = rho0;
#endif
    CellP[i].InternalEnergy        = 0.01 * u_s;
    CellP[i].InternalEnergyPred    = 0.01 * u_s;
    CellP[i].Pressure              = 0.0;
    CellP[i].SoundSpeed            = (rho0 > 0.0) ? sqrt(A_mod / rho0) : 0.0;
    CellP[i].DtInternalEnergy      = 0.0;

    CellP[i].Elastic_Stress_Tensor      = {};
    CellP[i].Elastic_Stress_Tensor_Pred = {};
    CellP[i].Dt_Elastic_Stress_Tensor   = {};

#ifdef EOS_DAMAGE_POROSITY
    CellP[i].Damage       = 0.0;
    CellP[i].ActiveCracks = 0.0;
    CellP[i].Distention   = All.Tillotson_EOS_params[ct][15]; /* alpha_0 */
#endif

    CellP[i].Mass = P[i].Mass;
}

#endif /* GRAIN_FLUID && GRAIN_FLUID_PROMOTION */
#endif /* GRAIN_PROMOTION_H */
