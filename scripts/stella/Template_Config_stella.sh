# Stella options on top of Config_stella_default.sh / _cloud.sh / _galaxy.sh (flag names of this tree). Uncomment what a run needs.
## cloud variants of the paper runs
#CHEMISTRYNETWORK=15                    # replaces =5 (net 17: the same line with 17)
#COSMIC_RAY_FLUID                       # CR fluid: drop CR_BACKGROUND_IONRATE_ATTENUATED; zeta then comes from the fluid
#CRFLUID_DIFFUSION_MODEL=7
#CRFLUID_INJECTION_AT_SHOCKS=(0.1)
#KETJU_REGULARIZATION                   # replaces ADAPTIVE_GRAVSOFT_FORALL=1+16 by =1 (fixed star softening)
#KETJU_MERGE_STARS
#KETJU_MERGE_BH
#GALSF_SINGLE_STARS_BH_PROMOTION
## moment-method transport instead of the tree columns: drop GALSF_FB_PHOTOIONIZATION_PER_DIRECTION and the TreeCol band fluxes
#RT_M1
#RT_CHEM_PHOTOION=2
#RT_SPEEDOFLIGHT_REDUCTION=3.0e-4
#GALSF_SINGLE_STARS_RT_TABLE_BANDS
## options
#GALSF_FB_MECHANICAL_HEALPIX_WEIGHTS
#GALSF_ISMDUSTCHEM_HU_MODEL
#CR_SCALE_WITH_G0
#ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR
#OUTPUT_SHIELD_FAC
#OUTPUT_CHEM_PHOTORATES
