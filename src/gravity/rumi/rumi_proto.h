/* RUMI++ coupling function declarations; included at the end of core/proto.h */
#ifndef RUMI_PROTO_H
#define RUMI_PROTO_H
#include "rumi_member_state.h"
int rumi_register_params(char (*tag)[50], void **addr, int *id, int nt, int maxtags, int type_real, int type_string, int type_int);
int rumi_param_default(const char *tag, void *addr);
void rumi_init(void);
void rumi_finalize(void);
void rumi_limit_timesteps(void);
void rumi_integrate(void);
void rumi_finish_drift(void);
int rumi_direct_gravity_needed(void);
void rumi_swap_output_state(void);
void rumi_save_state(void);
void rumi_load_state(void);
struct rumi_member_state &rumi_member(int i);
#endif
