#ifndef EOS_COMPOSITION_REGISTRY_H
#define EOS_COMPOSITION_REGISTRY_H

/* Per-particle material/composition registry shared by the solid-EOS
   branches (Tillotson, ANEOS, future Chabrier-Mazevet H/He, etc.).

   The framework was already in place pre-17c: every gas cell carries
   `CompositionType` (cell_data.h) and each EOS branch dispatches off it
   internally (Tillotson_EOS_params[CompositionType], aneos_compute(...)).
   What 17c adds is a single header documenting the ID layout and a
   branch resolver, so that when a second solid-EOS flag is enabled
   simultaneously the dispatch in eos.cc selects exactly one branch
   (instead of running serial #ifdef blocks that would double-stomp). */

#ifndef GIZMO_config_H
#include "../GIZMO_config.h"
#endif
/* GIZMO_GPU_FUNCTION comes from here rather than from whatever an includer
   happened to pull in first. Every current include path reaches this header
   after macros.h, so nothing changes today, but a header whose bodies only
   compile in that order is a defect its two siblings in this module were just
   hardened against. */
#include "../declarations/macros.h"

#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)

/* Branches for the per-particle solid-EOS dispatch in eos.cc. */
enum EosBranch {
    EOS_BRANCH_NONE      = 0, /* fall through; gas/cooling default applies */
    EOS_BRANCH_TILLOTSON = 1,
    EOS_BRANCH_ANEOS     = 2
    /* EOS_BRANCH_CHABRIER_HHE = 3, // 17d */
};

/* Tillotson sub-indices into All.Tillotson_EOS_params[N_TILLOTSON_MATERIALS][10] (and, for every elastic build, All.SolidMaterial[]).
   Indices 1..6 are the named presets (set by tillotson_eos_init() in
   solids/elastic_physics.cc). Slot 0 is a usable custom material whose
   parameters come from the parameterfile (Tillotson_EOS_params_*), loaded into
   slot 0 by begrun.cc -- this is the standard single-material solid setup.

   Under EOS_TYPES_DEFAULTGAS_AND_SOLIDS (hybrid gas+solid setups, e.g. grain
   promotion) slot 0 instead means "no solid EOS -- behave as the standard gas
   fluid", and MATERIAL_TILLOTSON_UNUSED becomes a sentinel (-1) that no real
   cell ever carries (init.cc stops the run on any CompositionType outside the build's materials). */
enum {
#ifdef EOS_TYPES_DEFAULTGAS_AND_SOLIDS
    MATERIAL_TILLOTSON_UNUSED  = 0,  /* 0 = gas / no solid EOS */
#else
    MATERIAL_TILLOTSON_UNUSED  = -1, /* sentinel only; never assigned to a cell */
    MATERIAL_TILLOTSON_CUSTOM  = 0,  /* 0 = custom material, parameters from the parameterfile */
#endif
    MATERIAL_TILLOTSON_GRANITE = 1,
    MATERIAL_TILLOTSON_BASALT  = 2,
    MATERIAL_TILLOTSON_IRON    = 3,
    MATERIAL_TILLOTSON_ICE     = 4,
    MATERIAL_TILLOTSON_OLIVINE = 5,
    MATERIAL_TILLOTSON_WATER   = 6
};
#define N_TILLOTSON_MATERIALS 7

/* Constitutive (strength, damage, porosity) properties of a solid material, kept apart from
   the EOS coefficients: an ANEOS table and a Tillotson row both name a material whose strength
   comes from here. All.SolidMaterial[] holds one entry per CompositionType, resolved once at
   startup (solid_material_resolve) from the named presets and the parameterfile, and every
   consumer reads it through solid_material_of(). Code units, except where noted. */
struct solid_material {
    double ShearModulus;          /* mu */
    double YieldStrength;         /* Y, the constant von Mises yield strength (radial return to sqrt(NUMDIMS J2) = Y); also the strength scale of the recent-yield memory */
    double WeibullK;              /* Weibull flaw density coefficient k [1/code length^3] */
    double WeibullM;              /* Weibull exponent m */
    double InternalFriction;      /* mu_i of the pressure-dependent intact strength (EOS_DAMAGE_POROSITY model 2; 0 = the constant von Mises YieldStrength) */
    double InitialDistention;     /* alpha_0 of the P-alpha porosity model (1 = compact) */
    double CrushPressureElastic;  /* P_e of the P-alpha model */
    double CrushPressureSolid;    /* P_s of the P-alpha model */
    double ReferenceDensity;      /* compact reference density rho_0 (Tillotson rho_0, or an ANEOS table's own value): an element's flaw volume is its mass x InitialDistention / rho_0 */
    double FractureActivation;    /* fracture law: 0 = tensile flaws (Benz & Asphaug), 1 = the shear-activated law */
    double Cohesion;              /* Y_c: the pressure-dependent strength at zero pressure (with InternalFriction > 0) */
    double YieldStrengthMax;      /* Y_M: its limit at high pressure (Collins et al. 2004); 0 = no limit (linear in pressure) */
};
#define N_SOLID_MATERIAL_PROPERTIES 12  /* the doubles above, in order (written as one row per id to the snapshot header) */
#define FRACTURE_ACTIVATION_TENSILE_FLAWS 0
#define FRACTURE_ACTIVATION_SHEAR         1
static_assert(sizeof(struct solid_material) == N_SOLID_MATERIAL_PROPERTIES * sizeof(double), "struct solid_material must stay N_SOLID_MATERIAL_PROPERTIES doubles");
#ifdef EOS_ANEOS
#include "aneos.h" /* ANEOS_MAX_MATERIALS: the one definition, shared by the tables and the global struct */
/* every CompositionType layout: Tillotson rows + ANEOS tables + the gas slot (never fewer than the 16 of the other builds) */
#define N_SOLID_MATERIAL_IDS ((N_TILLOTSON_MATERIALS + ANEOS_MAX_MATERIALS + 1 > 16) ? (N_TILLOTSON_MATERIALS + ANEOS_MAX_MATERIALS + 1) : 16)
#else
#define N_SOLID_MATERIAL_IDS 16   /* >= every CompositionType layout: Tillotson rows (+1 for the gas slot) */
#endif

#if defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
/* Counts of the times the solid / tabulated equations of state had to limit a state, per material (CompositionType), written to
   eos_limits.txt. Every count is added on the branch where the limit acts, so a step that limits nothing adds nothing; the
   active cell-updates (the first count) are the denominator. See the User Guide (eos_limits.txt) for what each count means. */
#define EOS_LIMIT_SLOTS N_SOLID_MATERIAL_IDS /* one per CompositionType of any layout */
enum {EOS_LIMIT_CELL_UPDATES, EOS_LIMIT_RHO_BELOW_TABLE, EOS_LIMIT_RHO_ABOVE_TABLE, EOS_LIMIT_U_BELOW_FLOOR, EOS_LIMIT_U_ABOVE_TABLE,
      EOS_LIMIT_PLATEAU, EOS_LIMIT_RECOVERED,
      EOS_LIMIT_KICK_HALVED, EOS_LIMIT_KICK_FLOOR, EOS_LIMIT_KICK_MIN_ENERGY, EOS_LIMIT_DRIFT_HALVED, EOS_LIMIT_DRIFT_FLOOR,
      EOS_LIMIT_DRIFT_MIN_ENERGY, EOS_LIMIT_RADIATION_FLOOR, EOS_LIMIT_N_COUNTS};
enum {EOS_LIMIT_ENERGY_KICK_HALVED, EOS_LIMIT_ENERGY_KICK_FLOOR, EOS_LIMIT_ENERGY_KICK_MIN_ENERGY, EOS_LIMIT_N_ENERGIES};
enum {EOS_LIMIT_MAX_PLATEAU_LOGT_WIDTH, EOS_LIMIT_MAX_PLATEAU_PRESSURE_SPREAD, EOS_LIMIT_N_MAXIMA};
struct eos_limit_slot {unsigned long long count[EOS_LIMIT_N_COUNTS]; double energy[EOS_LIMIT_N_ENERGIES]; double maximum[EOS_LIMIT_N_MAXIMA];};
struct eos_limit_counts {struct eos_limit_slot slot[EOS_LIMIT_SLOTS];};

GIZMO_GPU_FUNCTION static inline int eos_limit_slot_of(int composition_type)
{
    return (composition_type < 0) ? 0 : ((composition_type >= EOS_LIMIT_SLOTS) ? EOS_LIMIT_SLOTS - 1 : composition_type);
}

/* Thread-safe adds to a counter block. The device pass uses the GPU compiler's own atomics and the host pass the compiler's
   builtins (no Kokkos here, so the header stays usable from host-only files). Only the rare limiting branches call these. */
GIZMO_GPU_FUNCTION static inline void eos_limit_atomic_count(unsigned long long *c, unsigned long long n)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicAdd(c, n);
#else
    __atomic_fetch_add(c, n, __ATOMIC_RELAXED);
#endif
}
static inline void eos_limit_atomic_energy(double *e, double de) /* host only: the energies are added by the kick */
{
    double old_value, new_value;
    __atomic_load(e, &old_value, __ATOMIC_RELAXED);
    do {new_value = old_value + de;} while(!__atomic_compare_exchange(e, &old_value, &new_value, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}
GIZMO_GPU_FUNCTION static inline void eos_limit_atomic_max(double *m, double x) /* x >= 0: non-negative doubles order as their bit patterns */
{
    union {double d; unsigned long long u;} value; value.d = (x > 0) ? x : 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicMax((unsigned long long *) m, value.u);
#else
    unsigned long long *bits = (unsigned long long *) m, old_bits = __atomic_load_n(bits, __ATOMIC_RELAXED);
    while(old_bits < value.u && !__atomic_compare_exchange_n(bits, &old_bits, value.u, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
#endif
}

/* Which limits internal_energy_positivity_catch applies to the update u_old -> u_new, as bits: 1 = the halving, 2 = the material's floor
   (applied after the halving; both can act on one update), 0 = none. Computed exactly as the catch decides, without changing it. */
GIZMO_GPU_FUNCTION static inline int internal_energy_positivity_catch_event(double u_old, double u_new, double u_floor)
{
    int halved = (u_new < 0.5*u_old); double u = halved ? 0.5*u_old : u_new;
    return (halved ? 1 : 0) | ((u_floor > 0 && u < u_floor) ? 2 : 0);
}

#endif /* EOS_TILLOTSON || EOS_ANEOS */

#ifdef EOS_ELASTIC_MELT
/* Thermal softening (melt) and molten cavitation of a solid material, resolved once at
   startup (solid_material_resolve) per CompositionType. A material keeps the fraction f of its cold shear modulus and yield strength:
   f = 1 - u/u_melt (specific-energy form), or f = tanh(xi (T_m/T - 1)) with the Simon melt curve T_m = T_m0 (1 + max(P,0)/a)^(1/c)
   (temperature form, Ohnaka 1995 as in Collins et al. 2004), clipped to [0,1]. Code units, except the temperatures [K]. */
struct solid_strength_model {
    double MeltForm;              /* 0 = no thermal softening, 1 = specific-energy form, 2 = temperature form */
    double MeltSpecificEnergy;    /* u_melt (energy form) */
    double MeltTemperature0;      /* T_m0 [K] (temperature form) */
    double MeltSimonA;            /* a [pressure] (temperature form) */
    double MeltSimonC;            /* c (temperature form) */
    double MeltOhnakaXi;          /* xi (temperature form) */
    double CavitationStrength;    /* P_cav >= 0: the tension fully molten material holds */
};
#define N_SOLID_STRENGTH_MODEL_PROPERTIES 7  /* the doubles above, in order (written as one row per id to the snapshot header) */
static_assert(sizeof(struct solid_strength_model) == N_SOLID_STRENGTH_MODEL_PROPERTIES * sizeof(double), "struct solid_strength_model must stay N_SOLID_STRENGTH_MODEL_PROPERTIES doubles");
#define MELT_FORM_NONE            0
#define MELT_FORM_SPECIFIC_ENERGY 1
#define MELT_FORM_TEMPERATURE     2

/* What the parameterfile gives for one material channel (SolidMaterial_*, SolidMaterialPreset<N>_*, AneosTable<N>_*); NaN = not given */
struct solid_strength_model_input {
    double MeltModel;             /* 0 = off (not given), 1 = on */
    double MeltSpecificEnergy, MeltTemperature0, MeltSimonA, MeltSimonC, MeltOhnakaXi, CavitationStrength;
};
#define N_SOLID_STRENGTH_MODEL_INPUTS 7
static_assert(sizeof(struct solid_strength_model_input) == N_SOLID_STRENGTH_MODEL_INPUTS * sizeof(double), "struct solid_strength_model_input must stay N_SOLID_STRENGTH_MODEL_INPUTS doubles");
#endif

/* Is this composition an elastic solid? Under EOS_TYPES_DEFAULTGAS_AND_SOLIDS composition 0 is the
   ordinary gas fluid, which carries no deviatoric stress; every other id is a solid. */
GIZMO_GPU_FUNCTION static inline int solid_material_is_elastic(int composition_type)
{
    return (composition_type != MATERIAL_TILLOTSON_UNUSED);
}

/* CompositionType -> EosBranch resolver.

   With a single solid-EOS flag enabled, every valid CompositionType maps to that
   branch and is its own sub-index. With both EOS_TILLOTSON and EOS_ANEOS enabled,
   the dual-flag arm partitions the ID space:
     0 .. N_TILLOTSON_MATERIALS-1     -> Tillotson (sub-index = ID)
     N_TILLOTSON_MATERIALS .. N+M-1   -> ANEOS     (sub-index = ID - N)
   (the sub-index helpers below convert). Under EOS_TYPES_DEFAULTGAS_AND_SOLIDS slot 0 is
   the gas, the Tillotson presets keep 1..6 and ANEOS tables start at 7 (with Tillotson) or
   at 1 (ANEOS alone); the User Guide entry of that flag gives the numbering.

   The single gas / no-solid-EOS carve-out lives at the top here (SSOT). In
   standard solid builds MATERIAL_TILLOTSON_UNUSED is the sentinel -1, which no
   real cell carries, so it never fires and every valid id reaches a solid branch.
   Under EOS_TYPES_DEFAULTGAS_AND_SOLIDS it is 0, so CompositionType==0 cells use
   the standard gas fluid. */
GIZMO_GPU_FUNCTION static inline enum EosBranch eos_branch_of(int composition_type)
{
    if(composition_type == MATERIAL_TILLOTSON_UNUSED) {return EOS_BRANCH_NONE;}
#if defined(EOS_TILLOTSON) && defined(EOS_ANEOS)
    if(composition_type < N_TILLOTSON_MATERIALS) {return EOS_BRANCH_TILLOTSON;}
    return EOS_BRANCH_ANEOS;
#elif defined(EOS_TILLOTSON)
    return EOS_BRANCH_TILLOTSON;
#elif defined(EOS_ANEOS)
    return EOS_BRANCH_ANEOS;
#else
    (void)composition_type;
    return EOS_BRANCH_NONE;
#endif
}

GIZMO_GPU_FUNCTION static inline int tillotson_subindex(int composition_type)
{
    return composition_type;
}

GIZMO_GPU_FUNCTION static inline int aneos_subindex(int composition_type)
{
#if defined(EOS_TILLOTSON) && defined(EOS_ANEOS)
    return composition_type - N_TILLOTSON_MATERIALS;
#elif defined(EOS_TYPES_DEFAULTGAS_AND_SOLIDS)
    /* ANEOS-only hybrid: slot 0 is gas, so the tables start at CompositionType=1. */
    return composition_type - 1;
#else
    return composition_type;
#endif
}

#endif /* EOS_TILLOTSON || EOS_ELASTIC || EOS_ANEOS */
#endif /* EOS_COMPOSITION_REGISTRY_H */
