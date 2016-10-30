#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"

#define GSLWORKSIZE 100000

/*! \file sidm_routines.c
 *  \brief Fuctions and routines needed for the calculations of dark matter self interactions
 *
 *  This file contains the functions and routines necesary for the computation of
 *  the self-interaction probabilities and the velocity kicks due to the interactios.
 *  Originally written by Miguel Rocha, rocham@uci.edu. Oct 2010. Updated on 2014
 */
/*
 * This file was written by Miguel Rocha (merocha@ucsc.edu) for GIZMO
 */

/*! This function calculates the interaction probability between two particles.
 *  It checks if comoving integration is on and does the necesary change of
 *  variables and units.
 */
double prob_of_interaction(double mass, double r, double h_si, double Vtarget[3], double Vno[3], int dt_step)
{
    double dT, dloga, dV, dvx, dvy, dvz, prob,h,hubble_a,mp;
    
    dvx = Vno[0]-Vtarget[0];
    dvy = Vno[1]-Vtarget[1];
    dvz = Vno[2]-Vtarget[2];
    dV = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
    
    if(All.ComovingIntegrationOn)
    {
        r =  All.Time*r;
        h = h_si*All.Time;
        
        dloga = (dt_step)*All.Timebase_interval;
        hubble_a = (All.Omega0 / (All.Time * All.Time * All.Time)
                    + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) + All.OmegaLambda);
        hubble_a = All.Hubble_H0_CodeUnits * sqrt(hubble_a);
        dT = dloga/hubble_a;
        
        dV = dV/All.Time; /* Convert from internal velocity p = a^2 dx/dt to peculiar velocity v = a dx/dt*/
        //dV -= hubble_a*r; /* Ignore for know, should be negligible in short-range interactions */
    }
    else
    {
        dT = (dt_step)*All.Timebase_interval;
        h = h_si;
    }
    
    mp =   mass * All.UnitMass_in_g;
    prob = mp*All.InteractionCrossSection*dV*dT*g_geo(r/h)/(h*h*h*All.UnitLength_in_cm*All.UnitLength_in_cm);
    
    return prob/2; /* We need the factor of 2 because we are double counting pair interaction probabiliies */
}

/*! This function returns the value of the geometrical factor needed for
 *  the calculation of the interaction probability.
 */
double g_geo(double r)
{
    double  f, u;
    int i;
    
    u = r / 2.0 * GEOFACTOR_TABLE_LENGTH;
    i = (int) u;
    if (i >= GEOFACTOR_TABLE_LENGTH)
        i = GEOFACTOR_TABLE_LENGTH - 1;
    
    if (i <= 1)
        f = 0.992318  + (GeoFactorTable[0] - 0.992318)*u;
    else
        f = GeoFactorTable[i - 1] + (GeoFactorTable[i] - GeoFactorTable[i - 1]) * (u - i);
    
    return f;
}

/*! This routine sets the kicks for each particle after it has been decided that they will
 *  interact. It uses an algorithm tha conserves energy and momentum but picks a random direction
 *  so it does not conserves angular momentum.
 */
void  calculate_interact_kick(double Vtarget[3], double Vno[3], double kick_target[3], double kick_no[3])
{
    double dV,theta,phi,dvx,dvy,dvz;
    
    dvx = Vno[0]-Vtarget[0];
    dvy = Vno[1]-Vtarget[1];
    dvz = Vno[2]-Vtarget[2];
    dV = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
    
    theta = acos(2*gsl_rng_uniform(random_generator)-1.0);
    phi = gsl_rng_uniform(random_generator)*2.0*M_PI;
    
    kick_target[0] = (-Vtarget[0] + Vno[0] + dV*sin(theta)*cos(phi))/2.0;
    kick_no[0] = (Vtarget[0] - Vno[0] - dV*sin(theta)*cos(phi))/2.0;
    
    kick_target[1] = (-Vtarget[1] + Vno[1] + dV*sin(theta)*sin(phi))/2.0;
    kick_no[1] = (Vtarget[1] - Vno[1] - dV*sin(theta)*sin(phi))/2.0;
    
    kick_target[2] = (-Vtarget[2] + Vno[2] + dV*cos(theta))/2.0;
    kick_no[2] = (Vtarget[2] - Vno[2] - dV*cos(theta))/2.0;
}

/*! This routine initializes the table that will be used to get the geometrical factor
 *  as a function of the two particle separations. It populates a table with the results of
 *  the numerical integration.
 */
void init_geofactor_table(void)
{
    int i;
    double result, abserr,r;
    gsl_function F;
    gsl_integration_workspace *workspace;
    workspace = gsl_integration_workspace_alloc(GSLWORKSIZE);
    
    for(i = 0; i < GEOFACTOR_TABLE_LENGTH; i++)
    {
        r =  2.0/GEOFACTOR_TABLE_LENGTH * (i + 1);
        F.function = &geofactor_integ;
        F.params = &r;
        gsl_integration_qag(&F, 0.0, 1.0, 0, 1.0e-8, GSLWORKSIZE, GSL_INTEG_GAUSS41,workspace, &result, &abserr);
        GeoFactorTable[i] = 2*M_PI*result;
    }
    gsl_integration_workspace_free(workspace);
}

/*! This function returns the integrand of the numerical integration done on init_geofactor_table().
 */
double geofactor_integ(double x, void * params)
{
    double result, abserr,r;
    double newparams[2];
    
    r = *(double *) params;
    newparams[0] = r;
    newparams[1] = x;
    
    gsl_function F;
    gsl_integration_workspace *workspace;
    workspace = gsl_integration_workspace_alloc(GSLWORKSIZE);
    
    F.function = &geofactor_angle_integ;
    F.params = newparams;
    
    gsl_integration_qag(&F, -1.0, 1.0, 0, 1.0e-8, GSLWORKSIZE, GSL_INTEG_GAUSS41,workspace, &result, &abserr);
    gsl_integration_workspace_free(workspace);
    
    /*! This function returns the value W(x). The values of the density kernel as a funtion of x=r/h */
    double wk=0; if(x<1) kernel_main(x, 1, 1, &wk, &wk, -1);
    return x*x*wk*result;
}

/*! This function returns the integrand of the angular part of the integral done on
 *  init_geofactor_table().
 */
double geofactor_angle_integ(double u, void * params)
{
    double x,r,f;
    
    r = *(double *) params;
    x = *(double *) (params + sizeof(double));
    f = sqrt(x*x + r*r + 2*x*r*u);
    /*! This function returns the value W(x). The values of the density kernel as a funtion of x=r/h */
    double wk=0; if(f<1) kernel_main(f, 1, 1, &wk, &wk, -1);
    return wk;
}


/*! This routine updates the interaction table each time there is an interaction */
void update_interaction_table(MyIDType id1, MyIDType id2)
{
    int i,j,id1_index,id2_index;
    id1 += 1;
    id2 += 1;
    id1_index = -1;
    id2_index = -1;
    
    for(i = 0; i < INTERACTION_TABLE_LENGTH; i++)
    {
        if(InteractionTable[i][0] == id1)
        {
            id1_index = i;
            break;
        }
        else if(InteractionTable[i][0] == id2)
        {
            id2_index = i;
            break;
        }
    }
    
    if(id1_index == -1 && id2_index == -1)
    {
        for(i = 0; i < INTERACTION_TABLE_LENGTH; i++)
        {
            if(InteractionTable[i][0] == 0)
            {
                InteractionTable[i][0] = id1;
                InteractionTable[i][1] = id2;
                break;
            }
            if (i == INTERACTION_TABLE_LENGTH - 1)
            {
                printf("ERROR: There were more interactions in this timestep than INTERACTION_TABLE_LENGTH\n");
                endrun(1);
            }
        }
    }
    else if(id1_index >= 0)
    {
        for(j = 1; j < PARTICLE_MAX_INTERACTIONS + 1; j++)
        {
            if(InteractionTable[id1_index][j] == 0)
            {
                InteractionTable[id1_index][j] = id2;
                break;
            }
            if (j == PARTICLE_MAX_INTERACTIONS )
            {
                printf("ERROR: A particle interacted more than PARTICLE_MAX_INTERACTIONS\n");
                endrun(1);
            }
        }
    }
    else if(id2_index >= 0)
    {
        for(j = 1; j < PARTICLE_MAX_INTERACTIONS + 1; j++)
        {
            if(InteractionTable[id2_index][j] == 0)
            {
                InteractionTable[id2_index][j] = id1;
                break;
            }
            if (j == PARTICLE_MAX_INTERACTIONS )
            {
                printf("ERROR: A particle interacted more than PARTICLE_MAX_INTERACTIONS\n");
                endrun(1);
            }
        }
    }
}


/*! This function checks if a pair of particles have interacted in the current time step. It looks at the InteractionTable
 and returns 0 if the pair is not found and 1 if pair is found. This is used to avoid double counting of interactions.
 */
int check_interaction_table(MyIDType id1, MyIDType id2)
{
    int i,j,flag;
    
    flag = 0;
    id1 += 1;
    id2 += 1;
    
    for(i = 0; i < INTERACTION_TABLE_LENGTH; i++)
    {
        if(InteractionTable[i][0] == id1)
        {
            for(j = 1; j < PARTICLE_MAX_INTERACTIONS + 1; j++)
            {
                if(InteractionTable[i][j] == id2)
                {
                    flag = 1;
                    break;
                }
            }
        }
    }
    
    if(flag == 0)
    {
        for(i = 0; i < INTERACTION_TABLE_LENGTH; i++)
        {
            if(InteractionTable[i][0] == id2)
            {
                for(j = 1; j < PARTICLE_MAX_INTERACTIONS + 1; j++)
                {
                    if(InteractionTable[i][j] == id1)
                    {
                        flag = 1;
                        break;
                    }
                }
            }
        }
    }
    
    return flag;
}

void AllocateInteractionTable(int x, int y)
{
    int i,j;
    
    InteractionTable = (MyIDType**) malloc(x*sizeof(MyIDType*));
    if (InteractionTable == NULL)
    {
        free(InteractionTable);
        printf("Failed to allocate memory for the interaction table\n");
        endrun(2);
    }
    
    for (i = 0; i < x; i++)
    {   
        InteractionTable[i] = (MyIDType*) malloc(y*sizeof(MyIDType)); 
        if (InteractionTable[i] == NULL)
        {
            free(InteractionTable[i]);
            printf("Failed to allocate memory for the interaction table\n");
            endrun(2);
        }
    }
    
    for (i = 0; i < x; i++)
        for(j = 0; j < y; j++)
            InteractionTable[i][j]= 0;
}

void init_self_interactions()
{
    int i;
    for(i = 0; i < NumPart; i++)
    {
        P[i].dt_step = 0;
        P[i].dt_step_sidm = 0;
        P[i].NInteractions = 0;
    }
}

void log_self_interactions()
{
#ifndef IO_REDUCED_MODE
    if(ThisTask == 0)
    {
        fprintf(FdInfo, "\nNumber of DM self-interactions at this time-step: %lu\n", All.Ndmsi);
        printf("\nNumber of DM self-interactions at this time-step: %lu\n", All.Ndmsi);
    }
#endif
}

