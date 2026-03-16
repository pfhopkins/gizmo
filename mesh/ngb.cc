#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/vector.h"


/*!
 * This file contains routines for neighbour finding.  We use the gravity-tree and a range-searching technique to find neighbours.
 */
/*!
 * This file was originally part of the GADGET3 code developed by Volker Springel. The code has been heavily modified
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO (adding/consolidating some of the search routines as needed for different fluids).
 * the modules now are more modular and primarily run on a generic structure which was built entirely for GIZMO, as opposed to the
 * GADGET3-style neighbor finding, to allow for greater flexibility, more stable memory use, and efficient multi-threading.
 * More importantly general routines for neighbor-searching and walking trees have been completely rewritten into a new
 * set of modular files and structured portions of routines designed to be called by a high-level evaluation routine, so that all code can use
 * consistent definitions and these only need to be modified in a single place.
 */


/*! This function constructs the neighbour tree. To this end, we actually need to construct the gravitational tree, 
 *  because we use it now for the neighbour search.
 */
void ngb_treebuild(void)
{
    if(ThisTask == 0) {printf("Begin Ngb-tree construction.\n");}
    CPU_Step[CPU_MISC] += measure_time();
    force_treebuild(NumPart, NULL);
    CPU_Step[CPU_TREEBUILD] += measure_time();
    if(ThisTask == 0) {printf("Ngb-Tree contruction finished \n");}
}


/*! This routine finds all neighbours `j' that can interact with the particle `i' in the communication buffer.
 *  Note that an interaction can take place if: \f$ r_{ij} < h_i \f$  OR if  \f$ r_{ij} < h_j \f$.
 *
 *  In the range-search this is taken into account, i.e. it is guaranteed that all particles are found that fulfill this condition, 
 *  including the (more difficult) second part of it. For this purpose, each node knows the maximum h occuring among the particles it represents.
 */
int ngb_treefind_pairs_threads(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
                               int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(P[p].Type > 0) continue; // skip particles with non-gas types
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define NGB_ONLY_OPEN_NODES_CONTAINING_GAS // only want gas
#define SEARCHBOTHWAYS 1 // need neighbors that can -mutually- see one another, not just single-directional searching here
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS // must be undefined after code block inserted, or compiler will crash
#undef NGB_ONLY_OPEN_NODES_CONTAINING_GAS
}


/*! This function returns neighbours with distance <= rkern and returns them in Ngblist. Actually, particles in a box of half side length rkern are
 *  returned, i.e. the reduction to a sphere still needs to be done in the calling routine.
 */
int ngb_treefind_variable_threads(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
				  int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(P[p].Type > 0) continue; // skip particles with non-gas types
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define NGB_ONLY_OPEN_NODES_CONTAINING_GAS // only want gas
#define SEARCHBOTHWAYS 0 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS
#undef NGB_ONLY_OPEN_NODES_CONTAINING_GAS
}

/* this is the same as above, but the simpler un-threaded version, useful for historical reasons and because some sub-routines use 
    this model without threading because there is no real performance gain. note the slightly different construction of the subroutine below.
    TARGET_BITMASK should be set as a bitmask, i.e. SUM(2^n), where n are all the particle types desired for neighbor finding,
    so e.g. if you want particle types 0 and 4, set TARGET_BITMASK = 17 = 1 + 16 = 2^0 + 2^4
*/
int ngb_treefind_variable_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode, int *nexport, int *nsend_local, int TARGET_BITMASK)
{
    long nexport_save = *nexport; /* this line must be here in the un-threaded versions */
#include "../system/ngb_codeblock_before_condition.h" // call the same variable/initialization block
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 0 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_unthreaded.h" // call the main loop block as above, but this time the -unthreaded- version
#undef SEARCHBOTHWAYS
}
/* identical to above but includes 'both ways' search for interacting neighbors */
int ngb_treefind_pairs_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode, int *nexport, int *nsend_local, int TARGET_BITMASK)
{
    long nexport_save = *nexport; /* this line must be here in the un-threaded versions */
#include "../system/ngb_codeblock_before_condition.h" // call the same variable/initialization block
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 1 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_unthreaded.h" // call the main loop block as above, but this time the -unthreaded- version
#undef SEARCHBOTHWAYS
}


/*  slightly modified version of treefind that searches for one or more types of particles: 
        TARGET_BITMASK should be set as a bitmask, i.e. SUM(2^n), where n are all the particle types desired for neighbor finding,
        so e.g. if you want particle types 0 and 4, set TARGET_BITMASK = 17 = 1 + 16 = 2^0 + 2^4
 */
int ngb_treefind_variable_threads_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
                                           int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                           int *ngblist, int TARGET_BITMASK)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 0 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS
}
/* identical to above but includes 'both ways' search for interacting neighbors */
int ngb_treefind_pairs_threads_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
                                           int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                           int *ngblist, int TARGET_BITMASK)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 1 // also want particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS
}







/* 
    custom code for FOF finder -- needs to be able to deal with complications like pure node-linkages and hard-codes a
    local requirement, so we can't use our simple routines above. this is a customized version of the "ngb_treefind_variable" routine above. 
    as a result, updates to the core neighbor search routine will not alter this subroutine 
 */
int ngb_treefind_fof_primary(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode, int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    MyDouble dx, dy, dz, dist, r2;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    long bunchSize = All.BunchSize;
    MyDouble xtmp; xtmp=0;

    nexport_save = *nexport;
    
    numngb = 0;
    no = *startnode;
    
    while(no >= 0)
    {
        if(no < maxPart)		/* single particle */
        {
            p = no;
            no = Nextnode[no];
            
            if(!((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
                continue;
            
            if(mode == 0)
                continue;
            
            dist = rkern;
            dx = NGB_PERIODIC_BOX_LONG_X(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
            if(dx > dist) continue;
            dy = NGB_PERIODIC_BOX_LONG_Y(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
            if(dy > dist) continue;
            dz = NGB_PERIODIC_BOX_LONG_Z(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
            if(dz > dist) continue;
            if(dx * dx + dy * dy + dz * dz > dist * dist) continue;

            Ngblist[numngb++] = p;
        }
        else
        {
            if(no >= maxPart + maxNodes)	/* pseudo particle */
            {
                if(mode == 1)
                    endrun(123125);
                
                if(mode == 0)
                {
                    if(Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                    {
                        Exportflag[task] = target;
                        Exportnodecount[task] = NODELISTLENGTH;
                    }
                    
                    if(Exportnodecount[task] == NODELISTLENGTH)
                    {
                        if(*nexport >= bunchSize)
                        {
                            *nexport = nexport_save;
                            if(nexport_save == 0) {endrun(13005);}	/* in this case, the buffer is too small to process even a single particle */
                            for(task = 0; task < NTask; task++) {nsend_local[task] = 0;}
                            for(no = 0; no < nexport_save; no++) {nsend_local[DataIndexTable[no].Task]++;}
                            return -1; /* buffer has filled -- important that only this and other buffer-full conditions return the negative condition for the routine */
                        }
                        Exportnodecount[task] = 0;
                        Exportindex[task] = *nexport;
                        DataIndexTable[*nexport].Task = task;
                        DataIndexTable[*nexport].Index = target;
                        DataIndexTable[*nexport].IndexGet = *nexport;
                        *nexport = *nexport + 1;
                        nsend_local[task]++;
                    }
                    
                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                    DomainNodeIndex[no - (maxPart + maxNodes)];
                    
                    if(Exportnodecount[task] < NODELISTLENGTH)
                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                }
                
                if(mode == -1)
                {
                    *nexport = 1;
                }
                
                no = Nextnode[no - maxNodes];
                continue;
                
            }
            
            current = &Nodes[no];
            
            if(mode == 1)
            {
                if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }
            
            if(mode == 0)
            {
                if(!(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))	/* we have a node with only local particles, can skip branch */
                {
                    no = current->u.d.sibling;
                    continue;
                }
            }
            
            no = current->u.d.sibling;	/* in case the node can be discarded */
            
            dist = rkern + 0.5 * current->len;
            dx = NGB_PERIODIC_BOX_LONG_X(current->center[0]-searchcenter[0],current->center[1]-searchcenter[1],current->center[2]-searchcenter[2],-1);
            if(dx > dist) continue;
            dy = NGB_PERIODIC_BOX_LONG_Y(current->center[0]-searchcenter[0],current->center[1]-searchcenter[1],current->center[2]-searchcenter[2],-1);
            if(dy > dist) continue;
            dz = NGB_PERIODIC_BOX_LONG_Z(current->center[0]-searchcenter[0],current->center[1]-searchcenter[1],current->center[2]-searchcenter[2],-1);
            if(dz > dist) continue;
            /* now test against the minimal sphere enclosing everything */
            dist += CUBE_EDGEFACTOR_1 * current->len;
            if((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist) continue;
            
            if((current->u.d.bitflags & ((1 << BITFLAG_TOPLEVEL) + (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))) == 0)	/* only use fully local nodes */
            {
                /* test whether the node is contained within the sphere */
                dist = rkern - CUBE_EDGEFACTOR_2 * current->len;
                if(dist > 0)
                    if(r2 < dist * dist)
                    {
                        if(current->u.d.bitflags & (1 << BITFLAG_INSIDE_LINKINGLENGTH))	/* already flagged */
                        {
                            /* sufficient to return only one particle inside this cell */
                            
                            p = current->u.d.nextnode;
                            while(p >= 0)
                            {
                                if(p < maxPart)
                                {
                                    if(((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
                                    {
                                        dx = NGB_PERIODIC_BOX_LONG_X(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
                                        dy = NGB_PERIODIC_BOX_LONG_Y(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
                                        dz = NGB_PERIODIC_BOX_LONG_Z(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
                                        if(dx * dx + dy * dy + dz * dz > rkern * rkern) break;

                                        Ngblist[numngb++] = p;
                                        break;
                                    }
                                    p = Nextnode[p];
                                }
                                else if(p >= maxPart + maxNodes)
                                    p = Nextnode[p - maxNodes];
                                else
                                    p = Nodes[p].u.d.nextnode;
                            }
                            continue;
                        }
                        else
                        {
                            /* flag it now */
                            current->u.d.bitflags |= (1 << BITFLAG_INSIDE_LINKINGLENGTH);
                        }
                    }
            }
            
            no = current->u.d.nextnode;	/* ok, we need to open the node */
        }
    }
    
    *startnode = -1;
    return numngb;
}





