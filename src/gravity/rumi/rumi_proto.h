/* RUMI++ coupling function declarations; included at the end of core/proto.h */
#ifndef RUMI_PROTO_H
#define RUMI_PROTO_H
int rumi_register_params(char (*tag)[50], void **addr, int *id, int nt, int maxtags, int type_real, int type_string, int type_int);
int rumi_param_default(const char *tag, void *addr);
void rumi_init(void);
void rumi_finalize(void);
#endif
