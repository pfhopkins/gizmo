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

#include "../GIZMO_config.h"

#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)

/* Branches for the per-particle solid-EOS dispatch in eos.cc. */
enum EosBranch {
    EOS_BRANCH_NONE      = 0, /* fall through; gas/cooling default applies */
    EOS_BRANCH_TILLOTSON = 1,
    EOS_BRANCH_ANEOS     = 2
    /* EOS_BRANCH_CHABRIER_HHE = 3, // 17d */
};

/* Tillotson sub-indices into All.Tillotson_EOS_params[N_TILLOTSON_MATERIALS][12].
   Indices 1..6 are the named presets (set by tillotson_eos_init() in
   solids/elastic_physics.cc). Slot 0 is a usable custom material whose
   parameters come from the parameterfile (Tillotson_EOS_params_*), loaded into
   slot 0 by begrun.cc -- this is the standard single-material solid setup.

   Under EOS_TYPES_DEFAULTGAS_AND_SOLIDS (hybrid gas+solid setups, e.g. grain
   promotion) slot 0 instead means "no solid EOS -- behave as the standard gas
   fluid", and MATERIAL_TILLOTSON_UNUSED becomes a sentinel (-1) that no real
   cell ever carries (the CompositionType clamp in init.cc enforces this). */
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

/* CompositionType -> EosBranch resolver.

   With a single solid-EOS flag enabled, every valid CompositionType maps to that
   branch and is its own sub-index. With both EOS_TILLOTSON and EOS_ANEOS enabled,
   the dual-flag arm partitions the ID space:
     0 .. N_TILLOTSON_MATERIALS-1     -> Tillotson (sub-index = ID)
     N_TILLOTSON_MATERIALS .. N+M-1   -> ANEOS     (sub-index = ID - N)
   (the sub-index helpers below convert). Note: the hybrid dual-flag numbering
   under EOS_TYPES_DEFAULTGAS_AND_SOLIDS (slot 0 = gas, Tillotson 1..6, ANEOS 7+)
   is not yet a supported public interface.

   The single gas / no-solid-EOS carve-out lives at the top here (SSOT). In
   standard solid builds MATERIAL_TILLOTSON_UNUSED is the sentinel -1, which no
   real cell carries, so it never fires and every valid id reaches a solid branch.
   Under EOS_TYPES_DEFAULTGAS_AND_SOLIDS it is 0, so CompositionType==0 cells use
   the standard gas fluid. */
static inline enum EosBranch eos_branch_of(int composition_type)
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

static inline int tillotson_subindex(int composition_type)
{
    return composition_type;
}

static inline int aneos_subindex(int composition_type)
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
