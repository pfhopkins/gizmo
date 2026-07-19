"""Generate a customized citation/acknowledgement block for GIZMO simulations,
based on which physics modules are enabled in a run's Config.sh.

Ported/updated for the Kokkos-GPU trunk from the original script contributed
by Tom Wagg (Software Citation Station, https://www.tomwagg.com/software-citation-station/)
and Michael Ryan (flag/citation mapping), gizmo-public PR #11
(https://github.com/pfhopkins/gizmo-public/pull/11). Flag names below were
updated to match this codebase's current Config.sh (several BH_* flags were
renamed SINK_*, some CR/GALSF flags were consolidated or retired, and several
new physics modules -- dust chemistry, the CBE integrator, nuclear reaction
networks, ANEOS/elastic EOS, MHD battery mechanisms, multi-fluid hydro -- were
added with their own citations).

Usage:
    python scripts/get_citations.py -f path/to/Config.sh
"""
import argparse

tags_from_keywords = {
    "ADAPTIVE_GRAVSOFT_FORALL": ["Hopkins2017"],
    "ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION": ["Hopkins2023a"],
    "ADAPTIVE_TREEFORCE_UPDATE": ["Grudic2021a"],
    "CHIMES": ["Richings2014", "Richings2014a"],
    "CONDUCTION": ["Hopkins2017"],
    "CONDUCTION_SPITZER": ["Su2017"],
    "COOLING": ["Hopkins2017"],
    "COOL_LOW_TEMPERATURES": ["Hopkins2017", "Hopkins2022d"],
    "COOL_METAL_LINES_BY_SPECIES": ["Hopkins2017", "Wiersma2009"],
    "COOL_MOLECFRAC": ["Hopkins2017", "Hopkins2022d"],
    "COSMIC_RAY_FLUID": ["Chan2019", "Hopkins2019b", "Hopkins2022", "Hopkins2021"],
    "COSMIC_RAY_SUBGRID_LEBRON": ["Hopkins2023"],
    "CRFLUID_ALT_RSOL_FORM": ["Chan2019", "Hopkins2019b", "Hopkins2022", "Hopkins2021"],
    "CRFLUID_ALT_VARIABLE_RSOL": ["Hopkins2021"],
    "CRFLUID_DIFFUSION_MODEL": ["Chan2019", "Hopkins2019b", "Hopkins2020", "Hopkins2021", "Hopkins2022", "Hopkins2022b"],
    "CRFLUID_EVOLVE_SCATTERINGWAVES": ["Chan2019", "Hopkins2019b", "Hopkins2022", "Hopkins2021"],
    "CRFLUID_EVOLVE_SPECTRUM": ["Chan2019", "Hopkins2019b", "Hopkins2021", "Hopkins2022", "Hopkins2022c"],
    "DM_FUZZY": ["Hopkins2019"],
    "DM_SCALARFIELD_SCREENING": ["Hopkins2014a"],
    "DM_SIDM": ["Rocha2013", "Robles2017"],
    "EOS_ANEOS": ["Stewart2019", "Stewart2020"],
    "EOS_ELASTIC": ["Reinhardt2017"],
    "EOS_SUBSTELLAR_ISM": ["Grudic2021"],
    "EOS_TILLOTSON": ["Deng2017", "Reinhardt2017"],
    "EOS_TRUELOVE_PRESSURE": ["Robertson2008"],
    "FIRE_PHYSICS_DEFAULTS": ["Hopkins2017"],
    "FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT": ["Hopkins2017"],
    "FOF": ["Springel2001"],
    "FOF_DENSITY_SPLIT_TYPES": ["Springel2001"],
    "FOF_GROUP_MIN_SIZE": ["Springel2001"],
    "FOF_PRIMARY_LINK_TYPES": ["Springel2001"],
    "FOF_SECONDARY_LINK_TYPES": ["Springel2001"],
    "GALSF": ["Springel2003"],
    "GALSF_EFFECTIVE_EQS": ["Springel2003"],
    "GALSF_FB_FIRE_AGE_TRACERS": ["Hopkins2022d"],
    "GALSF_FB_MECHANICAL": ["Hopkins2018"],
    "GALSF_FB_TURNOFF_COOLING": ["Stinson2006"],
    "GALSF_ISMDUSTCHEM_GRAINSIZEEVO": ["Choban2022"],
    "GALSF_ISMDUSTCHEM_MODEL": ["Choban2022"],
    "GALSF_SFR_CRITERION": ["Grudic2021"],
    "GALSF_SFR_IMF_SAMPLING": ["Su2018"],
    "GALSF_SFR_IMF_VARIATION": ["Guszejnov2017"],
    "GALSF_SFR_VIRIAL_SCALING": ["Grudic2018", "Hopkins2013"],
    "GALSF_SUBGRID_WINDS": ["Springel2003"],
    "GALSF_SUBGRID_WIND_SCALING": ["Springel2003", "Oppenheimer2006", "Zhu2016"],
    "GALSF_WINDS_ORIENTATION": ["Oppenheimer2006", "Zhu2016"],
    "GRAIN_BACKREACTION": ["Hopkins2016", "Lee2017", "Moseley2019"],
    "GRAIN_COLLISIONS": ["Hopkins2016", "Lee2017"],
    "GRAIN_EPSTEIN_STOKES": ["Hopkins2016", "Lee2017"],
    "GRAIN_FLUID": ["Hopkins2016", "Lee2017", "Moseley2019"],
    "GRAIN_FLUID_AND_PIC_BOTH_DEFINED": ["Ji2022"],
    "GRAIN_LORENTZFORCE": ["Hopkins2016", "Lee2017"],
    "GRAIN_RDI_TESTPROBLEM": ["Hopkins2016", "Lee2017", "Moseley2019"],
    "GRAVITY_SPHERICAL_SYMMETRY": ["Lane2021"],
    "HERMITE_INTEGRATION": ["Grudic2021"],
    "HYDRO_MULTIFLUID_DUST_DRAG": ["Hopkins2016", "Lee2017", "Moseley2019"],
    "HYDRO_MULTIFLUID_IONNEUTRAL": ["Draine1980", "Hopkins2018b"],
    "MAGNETIC": ["Hopkins2015"],
    "MAINTAIN_TREE_IN_REARRANGE": ["Grudic2021"],
    "METALS": ["Hopkins2017"],
    "MHD_BATTERY_MECHANISMS": ["Kulsrud1997", "Harrison1973", "Durrive2015", "Soliman2025"],
    "MHD_B_SET_IN_PARAMS": ["Hopkins2015"],
    "MHD_CONSTRAINED_GRADIENT": ["Hopkins2015", "Hopkins2016b"],
    "MHD_NON_IDEAL": ["Hopkins2015", "Hopkins2016c"],
    "NUCLEAR_NETWORK": ["Timmes1999"],
    "NUCLEAR_NETWORK_SCREENING": ["Chugunov2007", "Yakovlev2006"],
    "PIC_MHD": ["Ji2022"],
    "PIC_SPEEDOFLIGHT_REDUCTION": ["Ji2022a"],
    "RANDOMIZE_GRAVTREE": ["Grudic2021"],
    "RT_CHEM_PHOTOION": ["Hopkins2019a"],
    "RT_FLUXLIMITEDDIFFUSION": ["Hopkins2018a"],
    "RT_FREEFREE": ["Hopkins2019a"],
    "RT_GENERIC_USER_FREQ": ["Hopkins2019a"],
    "RT_INFRARED": ["Hopkins2019a"],
    "RT_LEBRON": ["Hopkins2017", "Hopkins2018a", "Hopkins2012"],
    "RT_LOCALRAYGRID": ["Hopkins2018a", "Jiang2014"],
    "RT_LYMAN_WERNER": ["Hopkins2019a"],
    "RT_M1": ["Hopkins2018a", "Rosdahl2013"],
    "RT_NUV": ["Hopkins2019a"],
    "RT_OPACITY_FROM_EXPLICIT_GRAINS": ["Hopkins2019a", "Hopkins2022a"],
    "RT_OPTICAL_NIR": ["Hopkins2019a"],
    "RT_OTVET": ["Hopkins2018a", "Gnedin2001a"],
    "RT_PHOTOELECTRIC": ["Hopkins2019a"],
    "RT_REPROCESS_INJECTED_PHOTONS": ["Grudic2021"],
    "RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION": ["Grudic2021"],
    "RT_USE_TREECOL_FOR_NH": ["Clark2011"],
    "RT_XRAY": ["Hopkins2019a"],
    "SINGLE_STAR_ACCRETION": ["Grudic2021"],
    "SINGLE_STAR_FB_JETS": ["Grudic2021", "Su2021"],
    "SINGLE_STAR_FB_RAD": ["Grudic2021"],
    "SINGLE_STAR_FB_SNE": ["Grudic2021"],
    "SINGLE_STAR_FB_WINDS": ["Grudic2021"],
    "SINGLE_STAR_FIND_BINARIES": ["Grudic2021"],
    "SINGLE_STAR_SINK_FORMATION": ["Grudic2021"],
    "SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION": ["Grudic2021"],
    "SINGLE_STAR_TIMESTEPPING": ["Grudic2021"],
    "SINK_CALC_DISTANCES": ["Garrison_Kimmel2017"],
    "SINK_COMPTON_HEATING": ["Hopkins2016a"],
    "SINK_DYNFRICTION_FROMTREE": ["Ma2021", "Ma2023"],
    "SINK_GRAVACCRETION": ["Hopkins2011", "AnglesAlcazar2016"],
    "SINK_GRAVACCRETION_STELLARFBCORR": ["Hopkins2022e"],
    "SINK_GRAVCAPTURE_FIXEDSINKRADIUS": ["Hopkins2016a"],
    "SINK_GRAVCAPTURE_GAS": ["Hopkins2016a"],
    "SINK_GRAVCAPTURE_NONGAS": ["Hopkins2016a"],
    "SINK_OUTPUT_MOREINFO": ["AnglesAlcazar2017"],
    "SINK_SEED_FROM_FOF": ["AnglesAlcazar2017"],
    "SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA": ["Hopkins2022e"],
    "SINK_SUBGRIDBHVARIABILITY": ["Hopkins2011"],
    "SINK_THERMALFEEDBACK": ["Springel2005", "Wellons2023"],
    "SINK_WIND_KICK": ["AnglesAlcazar2016"],
    "SINK_WIND_SPAWN_SET_BFIELD_POLTOR": ["Su2021"],
    "SINK_WIND_SPAWN_SET_JET_PRECESSION": ["Su2021"],
    "SUBFIND": ["Springel2001", "Weinberger2020"],
    "SUBFIND_ADDIO_BARYONS": ["Springel2001", "Weinberger2020"],
    "SUBFIND_ADDIO_NUMOVERDEN": ["Springel2001", "Weinberger2020"],
    "SUBFIND_ADDIO_VELDISP": ["Springel2001", "Weinberger2020"],
    "SUBFIND_REMOVE_GAS_STRUCTURES": ["Springel2001", "Weinberger2020"],
    "SUBFIND_SAVE_PARTICLEDATA": ["Springel2001", "Weinberger2020"],
    "TURB_DIFF_DYNAMIC": ["Hopkins2017", "Rennehan2018", "Rennehan2021"],
    "TURB_DIFF_ENERGY": ["Hopkins2017", "Khurshudyan2016"],
    "TURB_DIFF_METALS": ["Hopkins2017", "Khurshudyan2016"],
    "TURB_DIFF_VELOCITY": ["Hopkins2017", "Khurshudyan2016"],
    "TURB_DRIVING": ["Bauer2012"],
    "TURB_DRIVING_SPECTRUMGRID": ["Bauer2012"],
    "VISCOSITY": ["Hopkins2016c"],
    "VISCOSITY_BRAGINSKII": ["Su2017"],
}

special_tags = {
    "SINK_SEED_FROM_LOCALGAS": "Grudic et al. (arXiv:1612.05635) and Lamberts et al. (MNRAS, 2016, 463, L31) if you use something like the default values",
    "COOLING": "either 1903.08657, 1607.04218, and 2001.02696; or 0901.4554 (depends on GIZMO version) if you use the TREECOOL file",
    "COOL_GRACKLE": "the GRACKLE software and methods papers",
    "GALSF_FB_MECHANICAL": "Hopkins et al 2014 (MNRAS 445, 581), and subsequently Kimm and Cen 2014 (ApJ 788, 121), Martizzi et al 2015 (MNRAS 450, 504), Rosdahl et al 2017 (MNRAS 466, 11) as suggested",
    "GALSF_SFR_VIRIAL_SCALING": "Padoan 2012, Federrath and Klessen 2012 ApJ 761,156; and 2013 ApJ 763,51, Hopkins MNRAS 2013, 430 1653 depending on model",
    "GRAIN_RDI_TESTPROBLEM": "Moseley et al 2019MNRAS.489..325M, Seligman et al 2019MNRAS.485.3991S, Steinwandel et al arXiv:2111.09335, Ji et al arXiv:2112.00752, Hopkins et al 2020MNRAS.496.2123H and arXiv:2107.04608, Squire et al 2022MNRAS.510..110S, Fuentes et al. 2022, Soliman et al. 2022 depending on test",
    "CBE_INTEGRATOR": "Hopkins 2026 (in prep) -- this method paper is not yet public; check with PFH for the current citation before publishing results that use this module",
}

def get_citation_tags_from_config(filepath):
    """Retrieve a list of BiBTeX citation tags based on the settings specified in a configuration file.

    Parameters
    ----------
    filepath : `str`
        Path to the configuration file.

    Returns
    -------
    citations : `list`
        List of BiBTeX citation tags that correspond to the settings in the configuration file.
    """

    # check which settings are turned on in the config file
    settings_on = set()
    with open(filepath, 'r') as file:
        # go through each line
        for line in file:
            # find any lines that aren't blank/comments
            # Need to parse inline comments too. '#' is guaranteed to introduce a comment
            setting = line.strip().split("#")[0].strip()
            if len(setting) > 0:
                # take just the initial setting name, remove anything after an equals sign
                setting = setting.split("=")[0].strip() if "=" in setting else setting
                settings_on.add(setting)

    # track which tags are relevant based on the settings
    citation_tags = set()
    special_flag = False
    for setting in settings_on:
        if setting in special_tags:
            if not special_flag:
                print()
                print("TODO List")
                print("---------")
            print(f" - You used {setting}. Please consider citing {special_tags[setting]}.")
            special_flag = True
        if setting in tags_from_keywords:
            for tag in tags_from_keywords[setting]:
                citation_tags.add(tag)

    # Add some extra spacing so the acknowledgement statement stands out
    if special_flag:
        print('\n')

    return sorted(list(citation_tags))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get citation tags based on a configuration file.")
    parser.add_argument("--filepath", "-f", required=True,
                        type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    citation_tags = get_citation_tags_from_config(args.filepath)
    acknowledgement = "This work makes use of the GIZMO code \\citep{Hopkins2014, Springel2005}."

    if len(citation_tags) > 0:
        acknowledgement += " GIZMO simulations in this study were run with additional features, which are based on a variety of other works \\citep{" + ", ".join(citation_tags) + "}."

    # print the acknowledgement
    BOLD, RESET, GREEN, BLUE = "\033[1m", "\033[0m", "\033[0;32m", "\033[0;34m"
    print("You can paste this acknowledgement into the relevant section of your manuscript:")
    print(f"{BOLD}{GREEN}{acknowledgement}{RESET}")

    # either print bibtex to terminal or save to file
    print(f"The associated bibtex can be found in {GREEN}scripts/gizmo_sources.bib{RESET}")
