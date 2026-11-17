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
   Columns 10-19: shear modulus mu, constant von Mises yield strength Y, Weibull k [1/cm^3], Weibull m, internal friction mu_i,
   initial distention alpha_0, P-alpha crush pressures P_e (elastic limit) and P_s (full compaction), cohesion Y_c, limit Y_M.
   Y (the constant law; radial return to sqrt(NUMDIMS J2) = Y): granite 3.1e10 = Swegle 1989 (SAND89-1443, Table 3: fitted to the
   Hugoniot elastic limit 4.6 GPa with nu 0.25); basalt 3.5e10 = Benz & Asphaug 1999 (Icarus 142, 5, Table 1); iron 5.3e9 and ice 3.1e9 from measured
   Hugoniot elastic limits, Y = HEL (1 - 2 nu)/(1 - nu): pure iron 0.90 GPa, nu 0.292 (Thomas et al. 2018, J. Appl. Phys. 123, 175902,
   Tables V-VI), ice Ih at 100 K 0.62 GPa (Stewart & Ahrens 2005, J. Geophys. Res. 110, E03005; warmer ice is weaker, down to 0.05 GPa), nu 0.33; olivine 5.0e10 from the 7.44 GPa elastic limit
   of single-crystal forsteritic olivine (Obata et al. 2023, Phys. Earth Planet. Inter. 338, 107009), nu 0.25: the least well constrained.
   mu_i, Y_c, Y_M (the pressure-dependent law of EOS_DAMAGE_POROSITY model 2, in the published convention sqrt(J2) <= Y): granite
   Collins et al. 2004 (Meteorit. Planet. Sci. 39, 217, Table A1); basalt and olivine/dunite Miljkovic et al. 2013 (Science 342, 724,
   Table S2); ice Bray et al. 2014 (Icarus 231, 394; as tabulated by Silber et al. 2021, arXiv:2104.07909, Table 1). Iron is a metal, not a frictional material, and water has no strength: mu_i = 0, the constant law. */
static const double solid_material_presets[6][20] = {
    {0.5,1.3,1.60e11,2.700,1.80e11,1.80e11,3.50e10,1.80e11,5.0,5.0,2.17e11,3.1e10, 4.0e29, 9.5, 2.0, 1.0, 3.0e8, 1.5e10, 9.0e8, 1.5e10},  // granite
    {0.5,1.5,4.87e12,2.700,2.67e11,2.67e11,4.72e10,1.82e11,5.0,5.0,2.27e11,3.5e10, 4.0e29, 9.5, 1.4, 1.0, 1.0e8, 7.0e10, 2.0e8, 2.5e10},  // basalt
    {0.5,1.5,9.50e10,7.860,1.28e12,1.05e12,1.42e10,8.45e10,5.0,5.0,7.75e11,5.3e9,  1.0e25, 9.0, 0.0, 1.0, 1.0e10,1.0e12, 0.0,   0.0   },  // iron
    {0.3,0.1,1.00e11,0.917,9.47e10,9.47e10,7.730e9,3.04e10,10.,5.0,2.80e10,3.1e9,  1.4e32, 9.1, 2.0, 1.0, 1.0e7, 1.5e9,  1.0e8, 1.1e9 },  // ice
    {0.5,1.4,5.50e12,3.500,1.31e12,4.90e11,4.50e10,1.50e11,5.0,5.0,8.13e11,5.0e10, 4.0e29, 9.5, 1.5, 1.0, 2.0e8, 1.0e10, 5.0e8, 3.5e10},  // olivine/dunite
    {0.5,0.9,2.00e10,1.000,2.00e11,1.00e11,4.000e9,2.00e10,5.0,5.0,1.000e0,1.00e0, 0.0,    1.0, 0.0, 1.0, 0.0,   1.0e10, 0.0,   0.0   }}; // water
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
    m.InternalFriction     = q[14];
    m.InitialDistention    = q[15];
    m.CrushPressureElastic = q[16] / UNIT_PRESSURE_IN_CGS;
    m.CrushPressureSolid   = q[17] / UNIT_PRESSURE_IN_CGS;
    m.ReferenceDensity     = q[3] / UNIT_DENSITY_IN_CGS;
    m.FractureActivation   = FRACTURE_ACTIVATION_TENSILE_FLAWS;
    m.Cohesion             = q[18] / UNIT_PRESSURE_IN_CGS;
    m.YieldStrengthMax     = q[19] / UNIT_PRESSURE_IN_CGS;
    return m;
}

/* What is missing or invalid in a material for the models this build runs (NULL if it is complete).
   Unset parameters hold NaN, so 'not given' and an explicit zero stay distinct. */
static const char *solid_material_problem(const struct solid_material *m, int is_aneos)
{
    if(!(isfinite(m->ShearModulus) && m->ShearModulus >= 0)) {return "ShearModulus (finite, >= 0)";}
    if(!(isfinite(m->YieldStrength) && m->YieldStrength >= 0)) {return "YieldStrength (finite, >= 0)";}
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1)
    if(!(m->FractureActivation == FRACTURE_ACTIVATION_TENSILE_FLAWS || m->FractureActivation == FRACTURE_ACTIVATION_SHEAR)) {return "FractureActivation (0 = tensile flaws, 1 = shear-activated law)";}
    if(!(isfinite(m->WeibullK) && m->WeibullK >= 0)) {return "WeibullK (finite, >= 0; 0 = no flaws, never damaged)";}
    if(m->WeibullK > 0 && !(isfinite(m->WeibullM) && m->WeibullM > 0)) {return "WeibullM (finite, > 0)";}
    if(m->WeibullK > 0 && m->FractureActivation == FRACTURE_ACTIVATION_TENSILE_FLAWS)
    {
        if(!(isfinite(m->ReferenceDensity) && m->ReferenceDensity > 0)) {return "ReferenceDensity (finite, > 0: the compact reference density that sets each element's flaw volume)";}
#if (NUMDIMS < 3)
        if(!(isfinite(All.FractureOutOfPlaneExtent) && All.FractureOutOfPlaneExtent > 0)) {return "FractureOutOfPlaneExtent (finite, > 0: the thickness in 2D or the cross-sectional area in 1D, code units, that turns an element's area or length into the volume holding its flaws)";}
#endif
    }
#endif
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 2)
    if(!(isfinite(m->InternalFriction) && m->InternalFriction >= 0)) {return "InternalFriction (finite, >= 0; 0 = the constant von Mises YieldStrength; it replaces DruckerPragerMu, and is given in the published convention sqrt(J2) <= Y with Cohesion and YieldStrengthMax)";}
    if(m->InternalFriction > 0)
    {
        if(!(isfinite(m->Cohesion) && m->Cohesion >= 0)) {return "Cohesion (finite, >= 0: the pressure-dependent strength at zero pressure)";}
        if(!(isfinite(m->YieldStrengthMax) && (m->YieldStrengthMax == 0 || m->YieldStrengthMax > m->Cohesion))) {return "YieldStrengthMax (0 = no limit, or > Cohesion)";}
    }
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

#ifdef EOS_ELASTIC_MELT
/* Thermal-softening defaults of the presets (rows 1-6; cgs, temperatures in K); NaN = no cited value, so a material relying on it is
   refused. Column 0: the preset carries strength (water: 0; its mu = Y = 1 are placeholders, so it is never softened).
   Specific-energy form (Tillotson materials), u_melt: basalt 3.4e10 and ice 7.0e9 erg/g (Benz & Asphaug 1999, Icarus 142, 5, Table 1).
   Temperature form (an ANEOS table naming the preset), T_m0, Simon a and c, Ohnaka xi: olivine 2163 K, 8.8456e10, 3.9533 and iron 1809 K,
   1.0418e11, 3.0633 = Simon fits to the melt curves of the Stewart M-ANEOS forsterite (2019) and iron (2020) tables (from their strength
   notebooks), so the strength melts where those equations of state do; granite 1673 K, 6.0e10, 3 (Silber et al. 2021, arXiv:2104.07909,
   Table 1); basalt 1360 K, 4.5e10, 3.0 and xi 0.7 (Miljkovic et al. 2013, Science 342, 724, Table S2); xi 1.2 otherwise (Collins et al. 2004,
   Meteorit. Planet. Sci. 39, 217, Table A1). Ice Ih melts at a LOWER temperature under pressure, which this curve cannot describe: no
   temperature-form default for ice. */
static const double solid_material_preset_melt[6][6] = {
    {1, NAN,    1673., 6.0e10,    3.0,    1.2},   // granite
    {1, 3.4e10, 1360., 4.5e10,    3.0,    0.7},   // basalt
    {1, NAN,    1809., 1.0418e11, 3.0633, 1.2},   // iron
    {1, 7.0e9,  NAN,   NAN,       NAN,    NAN},   // ice
    {1, NAN,    2163., 8.8456e10, 3.9533, 1.2},   // olivine/dunite
    {0, NAN,    NAN,   NAN,       NAN,    NAN}};  // water
static char solid_strength_model_problem[N_SOLID_MATERIAL_IDS][160]; /* empty = complete (or off) */

/* one value: given on the material's own channel > given on the channel of the preset it names > its default (the preset's cited value, or the model's) > missing */
static double solid_strength_model_pick(double own, const struct solid_strength_model_input *preset_channel, int k_field, double preset_default, const char *default_label, const char **source)
{
    if(!isnan(own)) {*source = "parameterfile"; return own;}
    if(preset_channel && !isnan(((const double *)preset_channel)[k_field])) {*source = "SolidMaterialPreset parameter"; return ((const double *)preset_channel)[k_field];}
    if(!isnan(preset_default)) {*source = default_label; return preset_default;}
    *source = "MISSING"; return NAN;
}

/* resolve and report the thermal softening of one material: own = its own parameter channel, preset_row = the preset it is or names (0: none) */
static void solid_strength_model_resolve(int id, const char *name, int is_aneos, const struct solid_strength_model_input *own, int preset_row, const struct solid_material *m)
{
    struct solid_strength_model model = {MELT_FORM_NONE, 0, 0, 0, 0, 0, 0};
    const struct solid_strength_model_input *preset_channel = (preset_row > 0 && own != &All.SolidStrengthModelPreset[preset_row]) ? &All.SolidStrengthModelPreset[preset_row] : NULL;
    const double *defaults = (preset_row > 0) ? solid_material_preset_melt[preset_row-1] : NULL;
    char *problem = solid_strength_model_problem[id]; problem[0] = 0;
    double on = own->MeltModel; if(isnan(on) && preset_channel) {on = preset_channel->MeltModel;}
    const char *s_u = "", *s_T = "", *s_a = "", *s_c = "", *s_x = "", *s_p = "";
    if(on > 0)
    {
        int no_strength = (defaults && !(defaults[0] > 0)) || (!(m->ShearModulus > 0) && !(m->YieldStrength > 0));
        if(no_strength) {if(ThisTask == 0) {printf("MELT MODEL   CompositionType %2d: %-40s off (the material has no strength to soften)\n", id, name);} All.SolidStrengthModel[id] = model; return;}
        model.CavitationStrength = solid_strength_model_pick(own->CavitationStrength, preset_channel, 6, 0, "default", &s_p);
        if(!is_aneos)
        {
            model.MeltForm = MELT_FORM_SPECIFIC_ENERGY;
            model.MeltSpecificEnergy = solid_strength_model_pick(own->MeltSpecificEnergy, preset_channel, 1, defaults ? defaults[1] / UNIT_SPECEGY_IN_CGS : NAN, "preset default", &s_u);
            if(!(isfinite(model.MeltSpecificEnergy) && model.MeltSpecificEnergy > 0)) {snprintf(problem, 160, "MeltSpecificEnergy (finite, > 0; %s)", s_u);}
        } else {
            model.MeltForm = MELT_FORM_TEMPERATURE;
            model.MeltTemperature0 = solid_strength_model_pick(own->MeltTemperature0, preset_channel, 2, defaults ? defaults[2] : NAN, "preset default", &s_T);
            model.MeltSimonA = solid_strength_model_pick(own->MeltSimonA, preset_channel, 3, defaults ? defaults[3] / UNIT_PRESSURE_IN_CGS : NAN, "preset default", &s_a);
            model.MeltSimonC = solid_strength_model_pick(own->MeltSimonC, preset_channel, 4, defaults ? defaults[4] : NAN, "preset default", &s_c);
            model.MeltOhnakaXi = solid_strength_model_pick(own->MeltOhnakaXi, preset_channel, 5, defaults ? defaults[5] : NAN, "preset default", &s_x);
            if(!(isfinite(model.MeltTemperature0) && model.MeltTemperature0 > 0)) {snprintf(problem, 160, "MeltTemperature0 (finite, > 0 [K]; %s)", s_T);}
            else if(!(isfinite(model.MeltSimonA) && model.MeltSimonA > 0)) {snprintf(problem, 160, "MeltSimonA (finite, > 0; %s)", s_a);}
            else if(!(isfinite(model.MeltSimonC) && model.MeltSimonC > 0)) {snprintf(problem, 160, "MeltSimonC (finite, > 0; %s)", s_c);}
            else if(!(isfinite(model.MeltOhnakaXi) && model.MeltOhnakaXi > 0)) {snprintf(problem, 160, "MeltOhnakaXi (finite, > 0; %s)", s_x);}
        }
        if(!problem[0] && !(isfinite(model.CavitationStrength) && model.CavitationStrength >= 0)) {snprintf(problem, 160, "CavitationStrength (finite, >= 0; %s)", s_p);}
    }
    if(ThisTask == 0)
    {
        if(model.MeltForm == MELT_FORM_SPECIFIC_ENERGY) {printf("MELT MODEL   CompositionType %2d: %-40s specific-energy form: u_melt=%g [%s] P_cav=%g [%s]", id, name, model.MeltSpecificEnergy, s_u, model.CavitationStrength, s_p);}
        else if(model.MeltForm == MELT_FORM_TEMPERATURE) {printf("MELT MODEL   CompositionType %2d: %-40s temperature form: T_m0=%g K [%s] a=%g [%s] c=%g [%s] xi=%g [%s] P_cav=%g [%s]", id, name,
            model.MeltTemperature0, s_T, model.MeltSimonA, s_a, model.MeltSimonC, s_c, model.MeltOhnakaXi, s_x, model.CavitationStrength, s_p);}
        else {printf("MELT MODEL   CompositionType %2d: %-40s off", id, name);}
        if(problem[0]) {printf("  [no valid %s: the run stops only if cells carry this CompositionType]", problem);}
        printf("\n");
    }
    if(problem[0]) {model.MeltForm = MELT_FORM_NONE;} /* refused in init() if cells carry it; never half-resolved */
    All.SolidStrengthModel[id] = model;
}
#endif

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
    struct solid_material none = {0, 0, 0, 1, 0, 1, 0, 0, 0, FRACTURE_ACTIVATION_TENSILE_FLAWS, 0, 0}; /* no strength, never damaged, compact */
    for(int id = 0; id < N_SOLID_MATERIAL_IDS; id++)
    {
        struct solid_material m = none;
        char name[96] = "unused id";
#ifdef EOS_ELASTIC_MELT
        const struct solid_strength_model_input *melt_own = NULL; int melt_preset = 0; /* its thermal-softening channel and preset */
        struct solid_strength_model melt_off = {MELT_FORM_NONE, 0, 0, 0, 0, 0, 0}; All.SolidStrengthModel[id] = melt_off; solid_strength_model_problem[id][0] = 0;
#endif
        if(!solid_material_is_elastic(id)) {strcpy(name, "gas (no strength)");}
#ifdef EOS_ANEOS
        else if(solid_material_id_is_aneos(id))
        {
            int table = aneos_subindex(id);
            if(table < 0 || table >= All.AneosNumMaterials) {All.SolidMaterial[id] = none; continue;}
            int choice = All.AneosSolidMaterialChoice[table];
#ifdef EOS_ELASTIC_MELT
            melt_own = &All.AneosSolidStrengthModel[table]; melt_preset = (choice >= 1 && choice < N_TILLOTSON_MATERIALS) ? choice : 0;
#endif
            if(choice >= 1 && choice < N_TILLOTSON_MATERIALS)
            {
                m = solid_material_preset(choice); snprintf(name, sizeof(name), "ANEOS table %d: %s", table, solid_material_preset_names[choice]);
                m.FractureActivation = All.AneosSolidMaterialCustom[table].FractureActivation; /* the table's own choice of fracture law (not given: the default) */
            }
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
            if(id == 0)
            {
                m = All.SolidMaterialCustom; strcpy(name, "custom (SolidMaterial_*)");
#ifdef EOS_ELASTIC_MELT
                melt_own = &All.SolidStrengthModelCustom;
#endif
#ifdef EOS_TILLOTSON
                m.ReferenceDensity = All.Tillotson_EOS_params[0][3]; /* the custom Tillotson material's own rho_0 */
#endif
            }
            else
            {
                m = solid_material_preset(id); strcpy(name, solid_material_preset_names[id]);
#ifdef EOS_ELASTIC_MELT
                melt_own = &All.SolidStrengthModelPreset[id]; melt_preset = id;
#endif
#ifdef EOS_DAMAGE_POROSITY
                m.FractureActivation = All.SolidMaterialPresetFractureActivation[id]; /* SolidMaterialPreset<N>_FractureActivation (not given: the default) */
#if ((EOS_DAMAGE_POROSITY) & 4)
                /* SolidMaterialPreset<N>_InitialDistention / _CrushPressureElastic / _CrushPressureSolid (code units; not given: the preset's own) */
                if(!isnan(All.SolidMaterialPresetPorosity[id][0])) {m.InitialDistention = All.SolidMaterialPresetPorosity[id][0];}
                if(!isnan(All.SolidMaterialPresetPorosity[id][1])) {m.CrushPressureElastic = All.SolidMaterialPresetPorosity[id][1];}
                if(!isnan(All.SolidMaterialPresetPorosity[id][2])) {m.CrushPressureSolid = All.SolidMaterialPresetPorosity[id][2];}
#endif
#endif
            }
        }
        else {All.SolidMaterial[id] = none; continue;}
        if(isnan(m.FractureActivation)) {m.FractureActivation = FRACTURE_ACTIVATION_TENSILE_FLAWS;} /* not given: the default law */
        if(isnan(m.YieldStrengthMax)) {m.YieldStrengthMax = 0;} /* not given: no high-pressure limit */
        if(!(m.InternalFriction > 0) && isnan(m.Cohesion)) {m.Cohesion = 0;} /* the constant law: no cohesion needed */
        All.SolidMaterial[id] = m;
        if(ThisTask == 0 && solid_material_is_elastic(id))
        {
            const char *problem = solid_material_problem(&m, solid_material_id_is_aneos(id));
            printf("SOLID MATERIAL CompositionType %2d: %-40s mu=%g Y=%g k=%g m=%g mu_i=%g Y_c=%g Y_M=%g alpha_0=%g P_e=%g P_s=%g rho_0=%g fracture=%s%s%s\n", id, name,
                   m.ShearModulus, m.YieldStrength, m.WeibullK, m.WeibullM, m.InternalFriction, m.Cohesion, m.YieldStrengthMax, m.InitialDistention, m.CrushPressureElastic, m.CrushPressureSolid,
                   m.ReferenceDensity, (m.FractureActivation == FRACTURE_ACTIVATION_SHEAR) ? "shear-activated" : "tensile flaws",
                   problem ? "  [no valid " : "", problem ? problem : "");
            if(problem) {printf(": the run stops only if cells carry this CompositionType]\n");}
        }
#ifdef EOS_ELASTIC_MELT
        if(melt_own && solid_material_is_elastic(id)) {solid_strength_model_resolve(id, name, solid_material_id_is_aneos(id), melt_own, melt_preset, &m);}
#endif
    }
}

#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1)
static int solid_material_uses_tensile_flaws(int id) {return solid_material_is_elastic(id) && All.SolidMaterial[id].WeibullK > 0 && All.SolidMaterial[id].FractureActivation == FRACTURE_ACTIVATION_TENSILE_FLAWS;}
#endif

/* Refuse the run if a cell carries a material that is incomplete for the models this build runs
   (called by every rank from init(), after the CompositionType of every cell is final). */
void solid_material_check_cells(void)
{
    int used_local[N_SOLID_MATERIAL_IDS] = {0}, used[N_SOLID_MATERIAL_IDS] = {0}, bad = 0;
    for(int i = 0; i < N_gas; i++) {int ct = CellP[i].CompositionType; if(ct >= 0 && ct < N_SOLID_MATERIAL_IDS) {used_local[ct] = 1;}}
#if defined(GRAIN_FLUID) && defined(GRAIN_FLUID_PROMOTION)
    /* grain promotion creates cells of these materials during the run (solids/grain_promotion.h): check them now as well */
#ifdef GRAIN_EVOLUTION
    const int promoted_materials[] = {MATERIAL_TILLOTSON_OLIVINE, MATERIAL_TILLOTSON_BASALT, MATERIAL_TILLOTSON_IRON, MATERIAL_TILLOTSON_ICE};
#else
    const int promoted_materials[] = {MATERIAL_TILLOTSON_OLIVINE};
#endif
    for(int id : promoted_materials) {used_local[id] = 1;}
#endif
    MPI_Allreduce(used_local, used, N_SOLID_MATERIAL_IDS, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    for(int id = 0; id < N_SOLID_MATERIAL_IDS; id++)
    {
        if(!used[id] || !solid_material_is_elastic(id)) {continue;}
        const char *problem = solid_material_problem(&All.SolidMaterial[id], solid_material_id_is_aneos(id));
#ifdef EOS_ELASTIC_MELT
        if(!problem && solid_strength_model_problem[id][0])
        {
            bad = 1;
            if(ThisTask == 0) {printf("ERROR: cells carry CompositionType %d, whose thermal softening (MeltModel 1) has no valid %s; give it in the parameterfile (SolidMaterial_*, SolidMaterialPreset<N>_* or AneosTable<N>_*), or set MeltModel 0.\n", id, solid_strength_model_problem[id]);}
        }
#endif
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
#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1) && defined(HYDRO_MESHLESS_FINITE_VOLUME)
        if(!problem && solid_material_uses_tensile_flaws(id))
        {
            bad = 1;
            if(ThisTask == 0) {printf("ERROR: cells carry CompositionType %d with tensile-flaw fracture, whose flaws belong to each element's fixed mass: it needs a fixed-mass method (MFM or SPH), not MFV or a regular grid.\n", id);}
        }
#endif
    }
    if(bad) {endrun(1152);}

#if defined(EOS_DAMAGE_POROSITY) && ((EOS_DAMAGE_POROSITY) & 1)
    /* tensile flaws: Benz & Asphaug's N ln N flaws per body, shared out by mass: an element of mass m holds ~ F m of them, with F = ln(N) N / M
       of its body's N elements of this material and total mass M (every element holds at least one flaw, so a one-element body has exactly one).
       F belongs to the element (TensileFlawsPerMass): set once, carried by snapshots and restarts, never recomputed, so a body keeps its flaws
       when it is placed in a run beside another body. An element without it (negative: a fresh IC, or a file written without the dataset) gets
       its material's value over all of this run's elements, which is its body's value only if all of them form one body. */
    double census_local[4][N_SOLID_MATERIAL_IDS] = {{0}}, census[4][N_SOLID_MATERIAL_IDS]; /* per material: elements, their mass, how many are unset, how many not finite */
    for(int i = 0; i < N_gas; i++)
    {
        int ct = CellP[i].CompositionType; if(ct < 0 || ct >= N_SOLID_MATERIAL_IDS) {continue;}
        double flaws_per_mass = CellP[i].TensileFlawsPerMass;
        census_local[0][ct] += 1; census_local[1][ct] += P[i].Mass;
        if(!isfinite(flaws_per_mass)) {census_local[3][ct] += 1;} else if(flaws_per_mass < 0) {census_local[2][ct] += 1;}
    }
    MPI_Allreduce(census_local, census, 4 * N_SOLID_MATERIAL_IDS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double *count = census[0], *mass = census[1], *unset = census[2], *invalid = census[3];
    double flaws_one_body[N_SOLID_MATERIAL_IDS];
    for(int id = 0; id < N_SOLID_MATERIAL_IDS; id++)
    {
        flaws_one_body[id] = (count[id] >= 2 && mass[id] > 0) ? log(count[id]) * count[id] / mass[id] : 0;
        if(!(count[id] > 0) || !solid_material_uses_tensile_flaws(id)) {continue;} /* only tensile-flaw fracture reads the value: any other material just gets the one-body value below */
        if(invalid[id] > 0)
        {
            bad = 1;
            if(ThisTask == 0) {printf("ERROR: CompositionType %d: %g of its elements carry a tensile-flaw normalisation (TensileFlawsPerMass) that is not a finite number.\n", id, invalid[id]);}
        }
        else if(unset[id] > 0 && unset[id] < count[id])
        {
            bad = 1;
            if(ThisTask == 0) {printf("ERROR: CompositionType %d: %g of its %g elements carry a tensile-flaw normalisation (TensileFlawsPerMass) and %g do not. Give every element of the material its body's value: initial conditions that combine bodies carry each body's from that body's own snapshot.\n", id, count[id] - unset[id], count[id], unset[id]);}
        }
        else if(unset[id] > 0 && ThisTask == 0)
        {
            printf("\n**********************************************************************\n"
                   "NOTE: CompositionType %d: its %g elements carry no tensile-flaw normalisation (no TensileFlawsPerMass in the input), so they get\n"
                   "ln(N) N / M over all of them, %g per unit mass. This ASSUMES every element of this material belongs to ONE body. If these initial\n"
                   "conditions combine several bodies, stop: carry each body's TensileFlawsPerMass from that body's own snapshot instead.\n"
                   "**********************************************************************\n\n", id, count[id], flaws_one_body[id]);
        }
    }
    if(bad) {endrun(1152); return;}
    double flaws_max_local[N_SOLID_MATERIAL_IDS] = {0}, flaws_max[N_SOLID_MATERIAL_IDS];
    for(int i = 0; i < N_gas; i++)
    {
        int ct = CellP[i].CompositionType; if(ct < 0 || ct >= N_SOLID_MATERIAL_IDS) {continue;}
        if(!(CellP[i].TensileFlawsPerMass >= 0)) {CellP[i].TensileFlawsPerMass = flaws_one_body[ct];}
        flaws_max_local[ct] = DMAX(flaws_max_local[ct], CellP[i].TensileFlawsPerMass * P[i].Mass);
    }
    MPI_Allreduce(flaws_max_local, flaws_max, N_SOLID_MATERIAL_IDS, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for(int id = 0; id < N_SOLID_MATERIAL_IDS; id++)
    {
        if(count[id] > 0 && solid_material_uses_tensile_flaws(id) && flaws_max[id] > TENSILE_FLAWS_MAX_PER_ELEMENT)
        {
            bad = 1;
            if(ThisTask == 0) {printf("ERROR: CompositionType %d uses tensile-flaw fracture, and its most flawed element would hold %g flaws (TensileFlawsPerMass x mass), above the %d one element may hold: its element masses are too unequal within a body for this model.\n",
                                      id, flaws_max[id], TENSILE_FLAWS_MAX_PER_ELEMENT);}
        }
    }
#if defined(GRAIN_FLUID) && defined(GRAIN_FLUID_PROMOTION)
    {
        int any_tensile = 0;
        for(int id : promoted_materials) {if(solid_material_uses_tensile_flaws(id)) {any_tensile = 1;}}
        if(any_tensile && ThisTask == 0) {printf("Grain promotion with tensile-flaw fracture: a promoted element is a body of its own (one element), so it holds exactly one tensile flaw.\n");}
    }
#endif
    if(bad) {endrun(1152);}
#endif
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
void elastic_stress_rate(int i, const double velocity_gradient[3][3])
{
    elastic_stress_rate_P(i, velocity_gradient, CellP);
}
#endif
#ifdef ELASTIC_STATE_DEPENDENT_STRENGTH
void elastic_project_after_eos(int i, int state_copies)
{
    elastic_project_after_eos_P(i, CellP, state_copies);
}
#endif
