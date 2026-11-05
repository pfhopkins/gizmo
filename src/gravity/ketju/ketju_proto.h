/* KETJU function declarations; included at the end of core/proto.h */
#ifndef KETJU_PROTO_H
#define KETJU_PROTO_H
#include "ketju_member_state.h"
int ketju_register_params(char (*tag)[50], void **addr, int *id, int nt, int maxtags, int type_real, int type_string, int type_int);
int ketju_param_default(const char *tag, void *addr);
void ketju_init_particle(int i);
void ketju_find_regions(void);
struct ketju_member_state &ketju_member(int i);
#endif /* KETJU_PROTO_H */
