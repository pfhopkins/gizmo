/* KETJU parameter tags: registered into read_parameter_file's table, with defaults for optional tags */
#include <stdio.h>
#include <string.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"

#ifdef KETJU_REGULARIZATION

int ketju_register_params(char (*tag)[50], void **addr, int *id, int nt, int maxtags, int type_real, int type_string, int type_int)
{
#define KETJU_TAG(name, ptr, type) do {if(nt >= maxtags) {printf("KETJU parameter tags exceed MAXTAGS=%d\n", maxtags); endrun(886612);} strcpy(tag[nt], name); addr[nt] = (void *) (ptr); id[nt++] = (type);} while(0)
    KETJU_TAG("KetjuRegionRadius", &All.KetjuRegionRadius, type_real);
    KETJU_TAG("KetjuMinStarMass", &All.KetjuMinStarMass, type_real);
    KETJU_TAG("KetjuIntegrationTolerance", &All.KetjuIntegrationTolerance, type_real);
    KETJU_TAG("KetjuUseStarStarSoftening", &All.KetjuUseStarStarSoftening, type_int);
    KETJU_TAG("KetjuTimestepLimitingFactor", &All.KetjuTimestepLimitingFactor, type_real);
    KETJU_TAG("KetjuMinParticlesPerTask", &All.KetjuMinParticlesPerTask, type_real);
    KETJU_TAG("KetjuMaxStepCount", &All.KetjuMaxStepCount, type_int);
    KETJU_TAG("KetjuKeplerPairs", &All.KetjuKeplerPairs, type_int);
#undef KETJU_TAG
    return nt;
}

/* 1 if tag is an optional KETJU tag (default written to addr), 0 otherwise */
int ketju_param_default(const char *tag, void *addr)
{
#define KETJU_DEF_REAL(name, val, why) if(strcmp(name, tag) == 0) {*((double *) addr) = (val); printf("Tag %s not set: defaulting to %g%s\n", tag, (double)(val), why); return 1;}
#define KETJU_DEF_INT(name, val, why) if(strcmp(name, tag) == 0) {*((int *) addr) = (val); printf("Tag %s not set: defaulting to %d%s\n", tag, (int)(val), why); return 1;}
    KETJU_DEF_REAL("KetjuMinStarMass", 0, " (Type 5 stage<7/star chain centers disabled)");
    KETJU_DEF_REAL("KetjuIntegrationTolerance", 1e-9, "");
    KETJU_DEF_INT("KetjuUseStarStarSoftening", 0, " (unsoftened chains for Type-4 stars)");
    KETJU_DEF_INT("KetjuKeplerPairs", 1, " (exact Kepler drift for isolated star pairs)");
    KETJU_DEF_REAL("KetjuTimestepLimitingFactor", 100., "");
    KETJU_DEF_REAL("KetjuMinParticlesPerTask", 10., "");
    KETJU_DEF_INT("KetjuMaxStepCount", 100000, "");
#undef KETJU_DEF_REAL
#undef KETJU_DEF_INT
    return 0;
}

#endif /* KETJU_REGULARIZATION */
