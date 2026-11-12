// Unit test (precompiler logic): on the Stella tree path the upstream band flags RT_NUV and RT_OPTICAL_NIR switch on the
// tree band fields without radiation pressure (needed by SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS); no
// radiation-pressure force is compiled without GALSF_RESOLVEDISM_RADPRESSURE.
#define GALSF_RESOLVEDISM
#define GALSF_RESOLVEDISM_FB
#define GALSF_RESOLVEDISM_SAMPLE_IMF
#define N_STELLAR_MASS 5
#define GALSF_RESOLVEDISM_G0_VARIABLE
#define GALSF_RESOLVEDISM_STELLAR_TABLES
#define SGCHEM
#define CHEMISTRYNETWORK 17
#define COOLING
#define METALS
#define RT_USE_TREECOL_FOR_NH 12
#define NEED_HEALPIX
#define RT_NUV
#define RT_OPTICAL_NIR
#define SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS
#include "test_harness.h"
#include "declarations/precompiler_logic.h"

TEST_CASE("bands on, radiation pressure off") {
    int nuv = 0, opt = 0, f68 = 0, radp = 0, radp_tree = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
    nuv = 1;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
    opt = 1;
#endif
#ifdef GALSF_RESOLVEDISM_CELL_FLUX_6_8EV_TOTAL
    f68 = 1;
#endif
#ifdef GALSF_RESOLVEDISM_RADPRESSURE
    radp = 1;
#endif
#ifdef GALSF_RESOLVEDISM_RADPRESSURE_TREERAD
    radp_tree = 1;
#endif
    CHECK(nuv && opt && f68);
    CHECK(!radp && !radp_tree);
    CHECK(RT_FREQ_BIN_NUV != RT_FREQ_BIN_OPTICAL_NIR && RT_FREQ_BIN_OPTICAL_NIR != RT_FREQ_BIN_PHOTOELECTRIC); // upstream bins, distinct
}

TEST_MAIN()
