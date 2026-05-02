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
   Set by tillotson_eos_init() in solids/elastic_physics.cc; index 0 is
   left zero-initialized and unused, indices 1..6 are the named presets. */
enum {
    MATERIAL_TILLOTSON_UNUSED  = 0,
    MATERIAL_TILLOTSON_GRANITE = 1,
    MATERIAL_TILLOTSON_BASALT  = 2,
    MATERIAL_TILLOTSON_IRON    = 3,
    MATERIAL_TILLOTSON_ICE     = 4,
    MATERIAL_TILLOTSON_OLIVINE = 5,
    MATERIAL_TILLOTSON_WATER   = 6
};
#define N_TILLOTSON_MATERIALS 7

/* CompositionType -> EosBranch resolver.

   Today only ONE of EOS_TILLOTSON / EOS_ANEOS is enabled per build, so the
   entire CompositionType integer space maps unconditionally to that branch
   and CompositionType is the sub-index — bit-identical to pre-17c.

   When a second solid-EOS flag lands (17d Chabrier-Mazevet, etc.), the
   dual-flag arm partitions the ID space:
     0 .. N_TILLOTSON_MATERIALS-1     -> Tillotson (sub-index = ID)
     N_TILLOTSON_MATERIALS .. N+M-1   -> ANEOS     (sub-index = ID - N)
   and the corresponding sub-index helpers below convert. */
static inline enum EosBranch eos_branch_of(int composition_type)
{
#if defined(EOS_TILLOTSON) && defined(EOS_ANEOS)
    if(composition_type < N_TILLOTSON_MATERIALS) {return EOS_BRANCH_TILLOTSON;}
    return EOS_BRANCH_ANEOS;
#elif defined(EOS_TILLOTSON)
    (void)composition_type;
    return EOS_BRANCH_TILLOTSON;
#elif defined(EOS_ANEOS)
    (void)composition_type;
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
#else
    return composition_type;
#endif
}

#endif /* EOS_TILLOTSON || EOS_ELASTIC || EOS_ANEOS */
#endif /* EOS_COMPOSITION_REGISTRY_H */
