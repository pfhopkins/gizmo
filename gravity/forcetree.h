#ifndef FORCETREE_H
#define FORCETREE_H

#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif
#endif

/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * substantially (condensed, new feedback routines added, many different
 * types of walk and calculations added, structures in memory changed,
 * switched options for nodes, optimizations, new physics modules and
 * calcutions, and new variable/memory conventions added)
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 * Mike Grudic has also made major revisions to code the Hermitian calculations and binary timestepping.
 */


#define BITFLAG_TOPLEVEL                   0
#define BITFLAG_DEPENDS_ON_LOCAL_ELEMENT   1
#define BITFLAG_INTERNAL_TOPLEVEL          6
#define BITFLAG_MULTIPLEPARTICLES          7
#define BITFLAG_NODEHASBEENKICKED          8
#define BITFLAG_INSIDE_LINKINGLENGTH       9

void force_update_tree(void);
void force_refresh_node_moments(void);

void force_flag_localnodes(void);

void *gravity_primary_loop(void *p);
void *gravity_secondary_loop(void *p);

int force_treeevaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
int force_treeevaluate_ewald_correction(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
int force_treeevaluate_potential(int target, int type, int *nexport, int *nsend_local);
void force_drift_node(int no, integertime time1);
void force_tree_discardpartials(void);
void force_treeupdate_pseudos(int);
void force_update_pseudoparticles(void);
void force_kick_node(int i, Vec3<MyDouble>& dv);
void force_dynamic_update(void);
void force_dynamic_update_node(int no, int mode, MyFloat *minbound, MyFloat *maxbound);
void force_update_hmax(void);
void force_update_hmax_of_node(int no, int mode);
void force_finish_kick_nodes(void);
int force_create_empty_nodes(int no, int topnode, int bits, int x, int y, int z, int *nodecount, int *nextfree);
void force_exchange_pseudodata(void);
void force_insert_pseudo_particles(void);
void force_add_element_to_tree(int igas, int istar);

void   force_costevaluate(void);
int    force_getcost_single(void);
int    force_getcost_quadru(void);
void   force_resetcost(void);
void   force_setupnonrecursive(int no);
void   force_treeallocate(int maxnodes, int maxpart);  
int    force_treebuild(int npart, struct unbind_data *mp);
int    force_treebuild_single(int npart, struct unbind_data *mp);
int    force_treeevaluate_direct(int target, int mode);
void   force_treefree(void);
void   force_update_node(int no, int flag);
void   force_update_node_recursive(int no, int sib, int father);
void   force_update_size_of_parent_node(int no);

void   dump_particles(void);

MyFloat  INLINE_FUNC ngb_periodic(MyFloat x);
MyFloat  INLINE_FUNC ngb_periodic_longbox(MyFloat x);
MyFloat  ngb_select_closest(int k, int n, MyFloat *arr, int *ind);
void   ngb_treeallocate(int npart);
void   ngb_treebuild(void);


void   ngb_treefree(void);
void   ngb_treesearch(int);
void   ngb_treesearch_pairs(int);
void   ngb_update_nodes(void);
void   ngb_treesearch_notsee(int no);

int ngb_treefind_fof_primary(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode,
			    int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES);
int ngb_clear_buf(MyDouble searchcenter[3], MyFloat hguess, int numngb);
void ngb_treefind_flagexport(MyDouble searchcenter[3], MyFloat hguess);


int ngb_treefind_pairs_threads(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
		       int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);
inline int ngb_treefind_pairs_threads(const Vec3<MyDouble>& searchcenter, MyFloat rkern, int target, int *startnode, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist) { return ngb_treefind_pairs_threads(const_cast<MyDouble*>(searchcenter.data), rkern, target, startnode, mode, exportflag, exportnodecount, exportindex, ngblist); }
int ngb_treefind_variable_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode,
 			  int *nexport, int *nsend_local, int TARGET_BITMASK);
inline int ngb_treefind_variable_targeted(const Vec3<MyDouble>& searchcenter, MyFloat rkern, int target, int *startnode, int mode, int *nexport, int *nsend_local, int TARGET_BITMASK) { return ngb_treefind_variable_targeted(const_cast<MyDouble*>(searchcenter.data), rkern, target, startnode, mode, nexport, nsend_local, TARGET_BITMASK); }
int ngb_treefind_pairs_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode,
                                   int *nexport, int *nsend_local, int TARGET_BITMASK);
int ngb_treefind_variable_threads(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode,
			  int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);
inline int ngb_treefind_variable_threads(const Vec3<MyDouble>& searchcenter, MyFloat rkern, int target, int *startnode, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist) { return ngb_treefind_variable_threads(const_cast<MyDouble*>(searchcenter.data), rkern, target, startnode, mode, exportflag, exportnodecount, exportindex, ngblist); }

#endif



