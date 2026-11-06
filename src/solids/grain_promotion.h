/* grain_promotion.h — Type 3 grain super-particle → Type 0 solid body conversion.
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
    /* the grain's cell row has never belonged to this element: start it from zero, then set the material state and the fields whose
       fresh value is not zero; gas-only state (chemistry abundances, turbulent-diffusion coefficients) is not given a startup value */
    CellP[i] = {};
    CellP[i].VelPred = P[i].Vel;
    CellP[i].ConditionNumber = 1;
    CellP[i].Gamma = GAMMA_DEFAULT; CellP[i].MeanMolecularWeight = MEAN_MOLECULAR_WEIGHT_DEFAULT; /* refreshed by its first equation-of-state call */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    CellP[i].MassTrue = P[i].Mass;
#endif
#if defined(COOLING) && !defined(COOLING_OPERATOR_SPLIT)
    CellP[i].CoolingIsOperatorSplitThisTimestep = 1;
#endif
#if defined(COOLING) && !defined(CHIMES)
    CellP[i].Ne = 1.0; CellP[i].HI = DMAX(0, DMIN(1, 1. - CellP[i].Ne / 1.2)); /* the chemistry solver's own first guess, as for a new cell at startup */
#endif
#ifdef EOS_ELASTIC_MELT
    CellP[i].StrengthFraction = 1; CellP[i].StrengthFractionPred = 1; /* its (zero) stress is at the cold modulus */
#endif
    CellP[i].CompositionType = ct;

    double rho0   = All.Tillotson_EOS_params[ct][3];
    double u_s    = All.Tillotson_EOS_params[ct][6];
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 4)
    rho0 /= DMAX(1., solid_material_of(ct).InitialDistention); /* porous: the matrix (at distention alpha_0 times the bulk density) starts at its reference density */
#endif
    CellP[i].Density               = rho0;
#ifdef HYDRO_EXPLICITLY_INTEGRATE_VOLUME
    CellP[i].Density_ExplicitInt   = rho0; /* KernelDensityForVolumeRelaxation is set by the next density pass */
#endif
    CellP[i].InternalEnergy        = 0.01 * u_s;
    CellP[i].InternalEnergyPred    = 0.01 * u_s;
#ifdef EOS_ANCHOR_INTERNALENERGY_IN_DRIFTS
    CellP[i].u_anchor              = CellP[i].InternalEnergyPred;
#endif
    /* no hydro force (HydroAccel 0) and no stress yet: its first half-step kick comes before its first force evaluation */
#ifdef EOS_DAMAGE_POROSITY
    CellP[i].Distention   = solid_material_of(ct).InitialDistention; /* alpha_0; damage and active flaws start at 0 */
#if ((EOS_DAMAGE_POROSITY) & 1)
    CellP[i].TensileFlawsPerMass = 0; /* a body of its own, of one element: exactly one tensile flaw */
#endif
#endif
    CellP[i].Mass = P[i].Mass;
}

#endif /* GRAIN_FLUID && GRAIN_FLUID_PROMOTION */
#endif /* GRAIN_PROMOTION_H */
