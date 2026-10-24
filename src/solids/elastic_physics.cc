#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "damage_porosity_functions.h"

/*
 
 This module contains the relevant module physics for various elastic,
   visco-elastic, and plastic body simulations.
 
 This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 
 */



#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC)
/* The named solid materials (rows 1-6; row 0 is the custom material), in cgs.
   Columns 0-9: Tillotson EOS a, b, u_0, rho_0, A, B, u_s, u_s', alpha, beta.
   Columns 10-17: shear modulus mu, yield strength Y_0, Weibull k [1/cm^3], Weibull m, Drucker-Prager mu_DP,
   initial distention alpha_0, P-alpha crush pressures P_e (elastic limit) and P_s (full compaction). */
static const double solid_material_presets[6][18] = {
    {0.5,1.3,1.60e11,2.700,1.80e11,1.80e11,3.50e10,1.80e11,5.0,5.0,2.17e11,3.8e10, 4.0e29, 9.5, 1.0, 1.0, 3.0e8, 1.5e10},  // granite
    {0.5,1.5,4.87e12,2.700,2.67e11,2.67e11,4.72e10,1.82e11,5.0,5.0,2.27e11,3.5e10, 4.0e29, 9.5, 1.0, 1.0, 1.0e8, 7.0e10},  // basalt
    {0.5,1.5,9.50e10,7.860,1.28e12,1.05e12,1.42e10,8.45e10,5.0,5.0,7.75e11,8.5e10, 1.0e25, 9.0, 1.0, 1.0, 1.0e10,1.0e12},  // iron
    {0.3,0.1,1.00e11,0.917,9.47e10,9.47e10,7.730e9,3.04e10,10.,5.0,2.80e10,1.0e10, 1.4e32, 9.1, 1.0, 1.0, 1.0e7, 1.5e9 },  // ice
    {0.5,1.4,5.50e12,3.500,1.31e12,4.90e11,4.50e10,1.50e11,5.0,5.0,8.13e11,9.0e10, 4.0e29, 9.5, 1.0, 1.0, 2.0e8, 1.0e10},  // olivine/dunite
    {0.5,0.9,2.00e10,1.000,2.00e11,1.00e11,4.000e9,2.00e10,5.0,5.0,1.000e0,1.00e0, 0.0,    1.0, 0.0, 1.0, 0.0,   1.0e10}}; // water
#endif

#ifdef EOS_TILLOTSON
/* Tillotson EOS coefficients of the named presets, in code units (row 0 comes from the parameterfile) */
void tillotson_eos_init(void)
{
    for(int j_t=1;j_t<N_TILLOTSON_MATERIALS;j_t++)
    {
        for(int k_t=0;k_t<10;k_t++)
        {
            All.Tillotson_EOS_params[j_t][k_t] = solid_material_presets[j_t-1][k_t];
            if((k_t==2)||(k_t==6)||(k_t==7)) {All.Tillotson_EOS_params[j_t][k_t] /= UNIT_SPECEGY_IN_CGS;}
            if(k_t==3) {All.Tillotson_EOS_params[j_t][k_t] /= UNIT_DENSITY_IN_CGS;}
            if((k_t==4)||(k_t==5)) {All.Tillotson_EOS_params[j_t][k_t] /= UNIT_PRESSURE_IN_CGS;}
        }
    }
}
#endif

#ifdef EOS_ELASTIC
static const char *solid_material_preset_names[N_TILLOTSON_MATERIALS] = {"custom", "granite", "basalt", "iron", "ice", "olivine/dunite", "water"};

/* constitutive properties of preset row 1-6, in code units */
static struct solid_material solid_material_preset(int row)
{
    const double *q = solid_material_presets[row-1];
    struct solid_material m;
    m.ShearModulus         = q[10] / UNIT_PRESSURE_IN_CGS;
    m.YieldStrength        = q[11] / UNIT_PRESSURE_IN_CGS;
    m.WeibullK             = q[12] * (UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS);
    m.WeibullM             = q[13];
    m.DruckerPragerMu      = q[14];
    m.InitialDistention    = q[15];
    m.CrushPressureElastic = q[16] / UNIT_PRESSURE_IN_CGS;
    m.CrushPressureSolid   = q[17] / UNIT_PRESSURE_IN_CGS;
    return m;
}

/* What is missing or invalid in a material for the models this build runs (NULL if it is complete).
   Unset parameters hold NaN, so 'not given' and an explicit zero stay distinct. */
static const char *solid_material_problem(const struct solid_material *m, int is_aneos)
{
    if(!(isfinite(m->ShearModulus) && m->ShearModulus >= 0)) {return "ShearModulus (finite, >= 0)";}
    if(!(isfinite(m->YieldStrength) && m->YieldStrength >= 0)) {return "YieldStrength (finite, >= 0)";}
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1)
    if(!(isfinite(m->WeibullK) && m->WeibullK >= 0)) {return "WeibullK (finite, >= 0; 0 = no flaws, never damaged)";}
    if(m->WeibullK > 0 && !(isfinite(m->WeibullM) && m->WeibullM > 0)) {return "WeibullM (finite, > 0)";}
#endif
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 2)
    if(!(isfinite(m->DruckerPragerMu) && m->DruckerPragerMu >= 0)) {return "DruckerPragerMu (finite, >= 0)";}
#endif
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 4)
    if(!is_aneos)
    {
        if(!(isfinite(m->InitialDistention) && m->InitialDistention >= 1)) {return "InitialDistention (finite, >= 1)";}
        if(!(isfinite(m->CrushPressureElastic) && m->CrushPressureElastic >= 0)) {return "CrushPressureElastic (finite, >= 0)";}
        if(!(isfinite(m->CrushPressureSolid) && m->CrushPressureSolid > m->CrushPressureElastic)) {return "CrushPressureSolid (finite, > CrushPressureElastic)";}
    }
#else
    (void)is_aneos;
#endif
    return NULL;
}

/* Is this CompositionType an ANEOS material? */
static int solid_material_id_is_aneos(int id)
{
#ifdef EOS_ANEOS
    return (solid_material_is_elastic(id) && eos_branch_of(id) == EOS_BRANCH_ANEOS);
#else
    (void)id; return 0;
#endif
}

/* Fill All.SolidMaterial[] for every CompositionType from the presets and the parameterfile, once at startup.
   A Tillotson row (or, without a tabulated EOS, the id itself) 1-6 is that preset and 0 is the custom
   material (SolidMaterial_* parameters); an ANEOS table takes the preset or its own values named by
   AneosTable<N>_SolidMaterial; the gas slot of EOS_TYPES_DEFAULTGAS_AND_SOLIDS has no strength.
   Completeness of the materials cells actually carry is checked in init(). */
void solid_material_resolve(void)
{
#ifdef EOS_ANEOS
    static_assert(N_TILLOTSON_MATERIALS + ANEOS_MAX_MATERIALS + 1 <= N_SOLID_MATERIAL_IDS, "N_SOLID_MATERIAL_IDS must cover every CompositionType layout");
#endif
    struct solid_material none = {0, 0, 0, 1, 0, 1, 0, 0}; /* no strength, never damaged, compact */
    for(int id = 0; id < N_SOLID_MATERIAL_IDS; id++)
    {
        struct solid_material m = none;
        char name[96] = "unused id";
        if(!solid_material_is_elastic(id)) {strcpy(name, "gas (no strength)");}
#ifdef EOS_ANEOS
        else if(solid_material_id_is_aneos(id))
        {
            int table = aneos_subindex(id);
            if(table < 0 || table >= All.AneosNumMaterials) {All.SolidMaterial[id] = none; continue;}
            int choice = All.AneosSolidMaterialChoice[table];
            if(choice >= 1 && choice < N_TILLOTSON_MATERIALS) {m = solid_material_preset(choice); snprintf(name, sizeof(name), "ANEOS table %d: %s", table, solid_material_preset_names[choice]);}
            else if(choice == 0) {m = All.AneosSolidMaterialCustom[table]; snprintf(name, sizeof(name), "ANEOS table %d: custom (AneosTable%d_*)", table, table);}
            else /* not set (or not a material): refused in init() only if cells carry this table */
            {
                double nan_value = NAN; m.ShearModulus = m.YieldStrength = nan_value;
                snprintf(name, sizeof(name), "ANEOS table %d: AneosTable%d_SolidMaterial = %d", table, table, choice);
            }
            m.InitialDistention = 1; m.CrushPressureElastic = 0; m.CrushPressureSolid = 0; /* P-alpha porosity is not available for ANEOS materials: they stay compact */
        }
#endif
        else if(id < N_TILLOTSON_MATERIALS)
        {
            if(id == 0) {m = All.SolidMaterialCustom; strcpy(name, "custom (SolidMaterial_*)");} else {m = solid_material_preset(id); strcpy(name, solid_material_preset_names[id]);}
        }
        else {All.SolidMaterial[id] = none; continue;}
        All.SolidMaterial[id] = m;
        if(ThisTask == 0 && solid_material_is_elastic(id))
        {
            const char *problem = solid_material_problem(&m, solid_material_id_is_aneos(id));
            printf("SOLID MATERIAL CompositionType %2d: %-40s mu=%g Y0=%g k=%g m=%g mu_DP=%g alpha_0=%g P_e=%g P_s=%g%s%s\n", id, name,
                   m.ShearModulus, m.YieldStrength, m.WeibullK, m.WeibullM, m.DruckerPragerMu, m.InitialDistention, m.CrushPressureElastic, m.CrushPressureSolid,
                   problem ? "  [no valid " : "", problem ? problem : "");
            if(problem) {printf(": the run stops only if cells carry this CompositionType]\n");}
        }
    }
}

/* Refuse the run if a cell carries a material that is incomplete for the models this build runs
   (called by every rank from init(), after the CompositionType of every cell is final). */
void solid_material_check_cells(void)
{
    int used_local[N_SOLID_MATERIAL_IDS] = {0}, used[N_SOLID_MATERIAL_IDS] = {0}, bad = 0;
    for(int i = 0; i < N_gas; i++) {int ct = CellP[i].CompositionType; if(ct >= 0 && ct < N_SOLID_MATERIAL_IDS) {used_local[ct] = 1;}}
    MPI_Allreduce(used_local, used, N_SOLID_MATERIAL_IDS, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    for(int id = 0; id < N_SOLID_MATERIAL_IDS; id++)
    {
        if(!used[id] || !solid_material_is_elastic(id)) {continue;}
        const char *problem = solid_material_problem(&All.SolidMaterial[id], solid_material_id_is_aneos(id));
        if(problem)
        {
            bad = 1;
            if(ThisTask == 0)
            {
#ifdef EOS_ANEOS
                if(solid_material_id_is_aneos(id)) {int table = aneos_subindex(id), choice = All.AneosSolidMaterialChoice[table];
                    if(choice < 0 || choice >= N_TILLOTSON_MATERIALS) {printf("ERROR: cells carry CompositionType %d (ANEOS table %d, %s), so AneosTable%d_SolidMaterial must name its solid material: 1-6 = preset (granite, basalt, iron, ice, olivine, water), 0 = the table's own AneosTable%d_* values; got %d.\n", id, table, All.AneosTableFiles[table], table, table, choice); continue;}}
#endif
                printf("ERROR: cells carry CompositionType %d, whose solid material has no valid %s; set it in the parameterfile (SolidMaterial_* for the custom material, AneosTable<N>_* for an ANEOS table's own values).\n", id, problem);
            }
        }
    }
    if(bad) {endrun(1152);}
}
#endif


/* elastic_body_update_driftkick: body is elastic_body_update_driftkick_P in
   elastic_physics_functions.h; the host wrapper is below the include block. */

#if defined(EOS_ELASTIC) || defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
/* get_negative_pressure_tensilecorrfac body moved to elastic_physics_functions.h
 * as KOKKOS_INLINE_FUNCTION so the hydro pair body (a __host__ __device__
 * function in the corridor hot path) can call it from the device pass without
 * the nvc++ #20011-D silent-physics warning. This block re-includes the header
 * with empty KOKKOS_INLINE_FUNCTION to emit the non-inline host external
 * symbol for any caller resolved via the core/proto.h forward declaration. */
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "elastic_physics_functions.h"
#endif


#ifdef EOS_ELASTIC
/* routine to update the deviatoric stress tensor. Sits after the include block
   above so that block stays the FIRST inclusion of the header in this file and
   still emits the non-inline host symbols; an earlier include would make it a
   #pragma once no-op. */
void elastic_body_update_driftkick(int i, double dt_entr, int mode)
{
    elastic_body_update_driftkick_P(i, dt_entr, mode, P, CellP);
}
#endif
