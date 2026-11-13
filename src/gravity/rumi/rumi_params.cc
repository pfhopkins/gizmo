/* RUMI++ parameter tags: registered into read_parameter_file's table, with defaults for optional tags */
#include <stdio.h>
#include <string.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"

#ifdef STARS_INTEGRATED_BY_RUMI

int rumi_register_params(char (*tag)[50], void **addr, int *id, int nt, int maxtags, int type_real, int type_string, int type_int)
{
#define RUMI_TAG(name, ptr, type) do {if(nt >= maxtags) {printf("RUMI++ parameter tags exceed MAXTAGS=%d\n", maxtags); endrun(886613);} strcpy(tag[nt], name); addr[nt] = (void *) (ptr); id[nt++] = (type);} while(0)
    RUMI_TAG("RumiConfigFile", All.RumiConfigFile, type_string);
    RUMI_TAG("RumiStarMaxTimestep", &All.RumiStarMaxTimestep, type_real);
    RUMI_TAG("RumiLengthUnitPc", &All.RumiLengthUnitPc, type_real);
    RUMI_TAG("RumiMassUnitMsun", &All.RumiMassUnitMsun, type_real);
    RUMI_TAG("RumiVelUnitKms", &All.RumiVelUnitKms, type_real);
    RUMI_TAG("RumiNumRanks", &All.RumiNumRanks, type_int);
    RUMI_TAG("RumiPrintInterval", &All.RumiPrintInterval, type_int);
#undef RUMI_TAG
    return nt;
}

/* 1 if tag is an optional RUMI++ tag (default written to addr), 0 otherwise */
int rumi_param_default(const char *tag, void *addr)
{
#define RUMI_DEF_REAL(name, val, why) if(strcmp(name, tag) == 0) {*((double *) addr) = (val); printf("Tag %s not set: defaulting to %g%s\n", tag, (double)(val), why); return 1;}
#define RUMI_DEF_INT(name, val, why) if(strcmp(name, tag) == 0) {*((int *) addr) = (val); printf("Tag %s not set: defaulting to %d%s\n", tag, (int)(val), why); return 1;}
    if(strcmp("RumiConfigFile", tag) == 0) {strcpy((char *) addr, "rumi_config.txt"); printf("Tag %s not set: defaulting to rumi_config.txt\n", tag); return 1;}
    RUMI_DEF_REAL("RumiStarMaxTimestep", 0, " (no cap beyond the stars' own criteria)");
    RUMI_DEF_REAL("RumiLengthUnitPc", 0, " (the code unit in pc)");
    RUMI_DEF_REAL("RumiMassUnitMsun", 0, " (the code unit in Msun)");
    RUMI_DEF_REAL("RumiVelUnitKms", 0, " (the code unit in km/s)");
    RUMI_DEF_INT("RumiNumRanks", 1, "");
    RUMI_DEF_INT("RumiPrintInterval", 64, "");
#undef RUMI_DEF_REAL
#undef RUMI_DEF_INT
    return 0;
}

#endif /* STARS_INTEGRATED_BY_RUMI */
