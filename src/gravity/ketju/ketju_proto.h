/* KETJU function declarations; included at the end of core/proto.h */
#ifndef KETJU_PROTO_H
#define KETJU_PROTO_H
#include "ketju_member_state.h"
int ketju_register_params(char (*tag)[50], void **addr, int *id, int nt, int maxtags, int type_real, int type_string, int type_int);
int ketju_param_default(const char *tag, void *addr);
void ketju_init_particle(int i);
#ifdef TEST_KETJU_MSTAR_PAIR
void ketju_test_mstar_pair(void);
#endif
int ketju_active_candidates(void);
void ketju_note_candidates(int any);
void ketju_report_stats(int final);
void ketju_limit_timesteps(void);
void ketju_find_regions(void);
void ketju_run_integration(void);
void ketju_set_final_velocities(void);
void ketju_finish_step(void);
void ketju_swap_output_velocities(void);
void ketju_open_output_file(void);
void ketju_close_output_file(void);
void ketju_write_output(void);
#ifdef TEST_KETJU_TREE_MOTION_AUDIT
void ketju_tree_audit_bounds(void);
void ketju_tree_audit_nodes(void);
void ketju_tree_audit_imports(void);
void ketju_tree_audit_report(void);
#endif
struct ketju_member_state &ketju_member(int i);
void tree_pair_softening(int p, double *soft, double *zeta);
void ketju_let_gather_spheres(void);
int ketju_let_sphere_opens(int rank, double cx, double cy, double cz, double len);
/* chain member whose MSTAR step just ended, with its chain-external field set: its host step follows only that field */
#define KETJU_DT_MEMBER(p) (P[(p)].KetjuIntegrated && (P[(p)].KetjuChainID != 0) && (ketju_member(p).com_dt_tidal > 0))
#endif /* KETJU_PROTO_H */
