#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>


#include "../declarations/allvars.h"
#include "../core/proto.h"

/*! This function reads initial conditions that are in the default file format
 * of Gadget, i.e. snapshot files can be used as input files.  However, when a
 * snapshot file is used as input, not all the information in the header is
 * used: THE STARTING TIME NEEDS TO BE SET IN THE PARAMETERFILE.
 * Alternatively, the code can be started with restartflag==2, then snapshots
 * from the code can be used as initial conditions-files without having to
 * change the parameterfile (except for changing the name of the IC file to the snapshot,
 * and ensuring the format tag matches it).  For gas particles, only the internal energy is
 * read, the density and mean molecular weight will be recomputed by the code.
 * When InitGasTemp>0 is given, the gas temperature will be initialzed to this
 * value assuming a mean colecular weight either corresponding to complete
 * neutrality, or full ionization.
 */

/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified
 * in part (adding/removing read items and changing variable units as necessary,
 * changing some parser options, adding run-time flexibility, allowing different
 * input types and structures and compression, updating for modern libraries,
 * and many other under-the-hood changes)
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

void read_ic(char *fname)
{
    long i; int num_files, rest_files, ngroups, gr, filenr, primaryTask, lastTask, groupTaskIterator;
    double u_init, molecular_weight; char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];

    CPU_Step[CPU_MISC] += measure_time();

#ifdef CBE_INTEGRATOR
    /* C7 (2026-05-30): reset per-type VlasovMoments-loaded flags before
     * any HDF5 open. Static globals are zero-initialized at process
     * start but this defensive reset covers any future case where
     * read_ic() is called more than once in-process. The flags are then
     * set per PartType in the optional-block HDF5 reader inside the
     * `if(hdf5_dataset >= 0)` branch after H5Dread succeeds. */
    for(int t = 0; t < 6; t++) { CBE_Moments_LoadedFromIC_PType[t] = 0; }
#endif

    NumPart = 0;
    N_gas = 0;
    All.TotNumPart = 0;

    num_files = find_files(fname);

    /* ===== ALL-RANK particle-storage setup (graceful-OOM phase) =====
     * find_files() above read file-0's header on rank 0 and MPI_Bcast it to EVERY rank, so
     * the global particle counts are known symmetrically here -- the one point in the IC path
     * where a collective preflight is safe (read_file() below is subset/turn-dispatched and
     * cannot host one). Compute MaxPart, preflight+allocate the particle storage all-rank,
     * then the read_file turns only read data into the pre-allocated arrays. (Moved out of
     * read_file()'s former `if(All.TotNumPart==0)` block so allocate_memory() is never
     * subset/turn-called from the IC path.) Normal-run behavior is unchanged: identical
     * MaxPart, identical allocations, just hoisted ahead of the per-file turns. */
    if(All.TotNumPart == 0)
    {
        /* find_files() above read file-0's header via the now-soft my_fread(), so a truncated/
         * corrupt header may have set a bad-stop. Validate the input precision here too (the
         * per-file copy of this check still runs inside read_file() for multi-file ICs), then
         * drain BOTH before the header is used to size or allocate anything. */
        /* precision check applies to the unformatted-binary formats only: their layout is implicit,
         * so the header flag is the only record of it. HDF5 datasets carry their own dtype and are
         * read at whatever precision they hold, so the flag cannot constrain that path. */
        if(All.ICFormat != 3)
        {
#ifdef INPUT_IN_DOUBLEPRECISION
            if(!header.flag_doubleprecision) {if(ThisTask == 0) {printf("\nProblem: Code compiled with INPUT_IN_DOUBLEPRECISION, but input files are in single precision!\n"); fflush(stdout);} endrun(11);}
#else
            if(header.flag_doubleprecision) {if(ThisTask == 0) {printf("\nProblem: Code not compiled with INPUT_IN_DOUBLEPRECISION, but input files are in double precision!\n"); fflush(stdout);} endrun(10);}
#endif
        }
        gizmo_exit_bad_stop_if_requested("read_ic:header");   /* drains corrupt-header (soft my_fread) + precision mismatch, before any header-derived sizing/alloc */

        if(header.num_files <= 1)
            for(i = 0; i < 6; i++) { header.npartTotal[i] = header.npart[i]; header.npartTotalHighWord[i] = 0; }

        All.TotN_gas = header.npartTotal[0] + (((long long) header.npartTotalHighWord[0]) << 32);
        for(i = 0, All.TotNumPart = 0; i < 6; i++)
        {
            All.TotNumPart += header.npartTotal[i];
            All.TotNumPart += (((long long) header.npartTotalHighWord[i]) << 32);
        }
        for(i = 0; i < 6; i++) {All.MassTable[i] = header.mass[i];}

        All.MaxPart = (int) (All.PartAllocFactor * (All.TotNumPart / NTask));
        All.MaxPartGas = (int) (All.PartAllocFactor * (All.TotN_gas / NTask));	/* sets the maximum number of particles that may reside on a processor */
        if(All.PartAllocFactor < 10.0 && NTask > 1 && ThisTask == 0) {
            printf("WARNING: PartAllocFactor=%.1f is low for the GPU neighbor-list build.\n", All.PartAllocFactor);
            printf("  Ghost particles from other MPI ranks are appended to P[]/CellP[] arrays and need\n");
            printf("  substantial headroom. Recommend PartAllocFactor >= 10 to avoid ghost overflow.\n");
        }
#ifdef ALLOW_IMBALANCED_GASPARTICLELOAD
        All.MaxPartGas = All.MaxPart;
#endif

        /* All-rank preflight + allocate. allocate_memory(1) sets a SOFT bad-stop (not an
         * fatal hard-exit) on a preflight/UVM/STL OOM; CommBuffer gets the same cheap arena
         * preflight (it is now in an all-rank phase). The poll below reconciles any of these
         * across ranks and finalizes cleanly BEFORE any P/CellP/CommBuffer is dereferenced. */
        (void) allocate_memory(1);

        {
            size_t cbuf_bytes = (size_t) All.BufferSize * 1024 * 1024;
            if(!gizmo_alloc_fits_all_ranks(gizmo_mymalloc_rounded_size(cbuf_bytes), 1))
                gizmo_request_controlled_stop(2, "read_ic: CommBuffer arena preflight won't fit on >=1 rank", __FILE__, __LINE__, __FUNCTION__);
            else
                CommBuffer = mymalloc("CommBuffer", cbuf_bytes);
        }

        if(RestartFlag >= 2)
        {
            All.Time = All.TimeBegin = header.time;
            set_cosmo_factors_for_current_time();
        }
        gizmo_exit_bad_stop_if_requested("read_ic:alloc");   /* drains particle-storage + CommBuffer OOM, before any P/CellP/CommBuffer use */

#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_TAG_ANCHOR
        {int i_rf; for(i_rf = 0; i_rf < All.MaxPart; i_rf++) {P[i_rf].Refinement_Flag = 0;}} /* zero the refinement tag across the whole allocated range before reads, so a particle whose IC lacks the RefinementFlag dataset defaults to untagged rather than garbage; the block read below overwrites with the true tag where present */
#endif
    }

    rest_files = num_files;

    while(rest_files > NTask)
    {
        snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.%d", fname, ThisTask + (rest_files - NTask));
        if(All.ICFormat == 3) {snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.%d.hdf5", fname, ThisTask + (rest_files - NTask));}

        ngroups = NTask / All.NumFilesWrittenInParallel;
        if((NTask % All.NumFilesWrittenInParallel)) {ngroups++;}
        groupTaskIterator = (ThisTask / ngroups) * ngroups;

        for(gr = 0; gr < ngroups; gr++)
        {
            if(ThisTask == (groupTaskIterator + gr)) {(void)read_file(buf, ThisTask, ThisTask);}	/* ok, it's this processor's turn */
            /* All-rank drain replaces the post-turn barrier (one collective, same sync): an IC
             * read failure on the turn's rank set a soft bad-stop inside read_file(); collect it
             * here and finalize cleanly BEFORE read_ic continues into myfree(CommBuffer)/mass-init/
             * P[] loops on junk state. No MPI_Abort, node releases. */
            gizmo_exit_bad_stop_if_requested("read_ic:read_file_turn");
        }
        rest_files -= NTask;
    }


    if(rest_files > 0)
    {
        distribute_file(rest_files, 0, 0, NTask - 1, &filenr, &primaryTask, &lastTask);

        if(num_files > 1)
        {
            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.%d", fname, filenr);
            if(All.ICFormat == 3)
                snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.%d.hdf5", fname, filenr);
        }
        else
        {
            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s", fname);
            if(All.ICFormat == 3)
                snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.hdf5", fname);
        }

        ngroups = rest_files / All.NumFilesWrittenInParallel;
        if((rest_files % All.NumFilesWrittenInParallel))
            ngroups++;

        for(gr = 0; gr < ngroups; gr++)
        {
            if((filenr / All.NumFilesWrittenInParallel) == gr) {(void)read_file(buf, primaryTask, lastTask);}	/* ok, it's this processor's turn */
            gizmo_exit_bad_stop_if_requested("read_ic:read_file_turn");
        }
    }


    myfree(CommBuffer);

#ifdef CBE_INTEGRATOR
    /* C7 IC reader multi-rank fix: CBE_Moments_LoadedFromIC_PType[] is set
     * inside read_file()'s H5Dread block, which only fires on the task
     * that actually reads the file (primaryTask via distribute_file for
     * the single-file case, or each task's own file when num_files>=NTask).
     * Other tasks receive their share of the loaded VlasovMoments data via
     * MPI distribution from the reading task, but their per-type flag
     * stays at 0. Without this broadcast, do_cbe_initialization would call
     * cbe_synthesize_cold_default on non-reading ranks and overwrite the
     * correctly-distributed loaded data with placeholder synth. MAX-reduce
     * the flags so any rank that loaded the dataset propagates the truth
     * to all ranks. */
    {
        int loaded_local[6];
        for(int t = 0; t < 6; t++) { loaded_local[t] = CBE_Moments_LoadedFromIC_PType[t]; }
        MPI_Allreduce(loaded_local, CBE_Moments_LoadedFromIC_PType, 6, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
#endif

    if(header.flag_ic_info != FLAG_SECOND_ORDER_ICS)
    {
        /* this makes sure that masses are initialized in the case that the mass-block is empty for this particle type */
        for(i = 0; i < NumPart; i++) {if(All.MassTable[P[i].Type] != 0) {P[i].Mass = All.MassTable[P[i].Type];}}
    }

    /* zero this out, since various operations in the code will want to change particle
     masses and keeping MassTable fixed won't allow that to happen */
    for(i=0;i<6;i++) All.MassTable[i]=0;


#if defined(SINK_PARTICLES)
#if defined(SINK_SWALLOWGAS) || defined(SINK_WIND_KICK) || defined(SINK_GRAVACCRETION) || defined(SINK_GRAVCAPTURE_GAS) || defined(SINK_SEED_FROM_LOCALGAS)
    if(RestartFlag == 0) {All.MassTable[5] = 0;}
#endif
#endif

#ifdef GALSF
    if(RestartFlag == 0)
    {
        if(All.MassTable[4] == 0 && All.MassTable[0] > 0)
        {
            All.MassTable[0] = 0;
            All.MassTable[4] = 0;
        }
    }
#endif

#if defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL)
    if(RestartFlag == 0)
    {
        All.MassTable[2] = 0;
        All.MassTable[3] = 0;
        All.MassTable[4] = 0;
    }
#endif

    u_init = All.InitGasTemp / ((GAMMA_DEFAULT-1) * U_TO_TEMP_UNITS);

    molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)); /* assume full ionization */
    if(All.InitGasTemp < 1.0e4) {molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);} /* assume neutral gas */
#if defined(COOL_LOW_TEMPERATURES) || defined(COOL_GRACKLE)
    if(All.InitGasTemp < 1.0e3 && All.ComovingIntegrationOn==0) {molecular_weight =  1. / ( HYDROGEN_MASSFRAC*0.5 + (1-HYDROGEN_MASSFRAC)/4. + 1./(16.+12.));} /* assume fully molecular [self-consistency requires cooling can handle this, and that this is intended to represent dense gas, not e.g. neutral early-universe gas] */
#endif

    u_init /= molecular_weight;

    All.InitGasU = u_init;

    if(RestartFlag == 0)
    {
        if(All.InitGasTemp > 0)
        {
            for(i = 0; i < N_gas; i++)
            {
                if(ThisTask == 0 && i == 0) // && CellP[i].InternalEnergy == 0)
                    {printf("Initializing u from InitGasTemp : InitGasTemp=%g InitGasU=%g MinEgySpec=%g CellP[0].InternalEnergy=%g\n",
                           All.InitGasTemp,All.InitGasU,All.MinEgySpec,CellP[i].InternalEnergy);}

                CellP[i].InternalEnergy = All.InitGasU;
            }
        }
    }

    for(i = 0; i < N_gas; i++) {CellP[i].InternalEnergyPred = CellP[i].InternalEnergy = DMAX(All.MinEgySpec, CellP[i].InternalEnergy);}
    gizmo_exit_bad_stop_if_requested("read_ic:done");   /* final all-rank drain (replaces the closing barrier) */
    if(ThisTask == 0) {printf("Reading done. Total number of particles :  %d%09d\n\n", (int) (All.TotNumPart / 1000000000), (int) (All.TotNumPart % 1000000000)); fflush(stdout);}

    CPU_Step[CPU_SNAPSHOT] += measure_time();
}


/*! Cursor over the float-valued elements of CommBuffer. The staged width depends on the input
 *  format (get_input_float_bytes is the SSOT): HDF5 blocks arrive as double, unformatted-binary
 *  blocks at the compile-time MyInputFloat/MyInputPosFloat width. Dereferencing yields a double
 *  either way, so the unpack below is written once and is correct for both.
 */
static_assert(sizeof(MyInputFloat) == sizeof(float) || sizeof(MyInputFloat) == sizeof(double), "MyInputFloat must be float or double: input_float_cursor treats any non-double staged width as float");
static_assert(sizeof(MyInputPosFloat) == sizeof(float) || sizeof(MyInputPosFloat) == sizeof(double), "MyInputPosFloat must be float or double: see input_float_cursor");

struct input_float_cursor
{
    const char *ptr; size_t width;
    input_float_cursor(const void *buf, size_t w) : ptr((const char *) buf), width(w) {}
    double at(long i) const {return (width == sizeof(double)) ? ((const double *) ptr)[i] : ((const float *) ptr)[i];}
    double operator*(void) const {return at(0);}
    double operator[](long i) const {return at(i);}
    input_float_cursor &operator++(void) {ptr += width; return *this;}
    input_float_cursor operator++(int) {input_float_cursor prev = *this; ptr += width; return prev;}
    input_float_cursor &operator+=(long i) {ptr += i * (long) width; return *this;}
};

/*! This function reads out the buffer that was filled with particle data.
 */
void empty_read_buffer(enum iofields blocknr, int offset, int pc, int type)
{
    long n, k; MyIDType *ip; int *ip_int;
    input_float_cursor fp(CommBuffer, get_input_float_bytes(blocknr));
    input_float_cursor fp_pos(CommBuffer, get_input_float_bytes(IO_POS));
    ip = (MyIDType *) CommBuffer;
    ip_int = (int *) CommBuffer;

    switch(blocknr)
    {
        case IO_POS:		/* positions */
            for(n = 0; n < pc; n++)
                for(k = 0; k < 3; k++)
                {
                    P[offset + n].Pos[k] = *fp_pos++;
                    // P[offset + n].Pos[k] += 0.5*All.BoxSize; /* manually turn on for some ICs */
                }

            for(n = 0; n < pc; n++) {P[offset + n].Type = type;}	/* initialize type here as well */
            break;

        case IO_VEL:		/* velocities */
            for(n = 0; n < pc; n++) {for(k = 0; k < 3; k++) {P[offset + n].Vel[k] = *fp++;}}
            break;

        case IO_ID:		/* particle ID */
            for(n = 0; n < pc; n++) {P[offset + n].ID = (MyIDType) (*ip++);}
            break;

        case IO_CHILD_ID:		// particle child ID //
            if(RestartFlag == 2) {for(n = 0; n < pc; n++) {P[offset + n].ID_child_number = *ip++;}}
            break;

        case IO_GENERATION_ID:		// particle generation ID //
            if(RestartFlag == 2) {for(n = 0; n < pc; n++) {P[offset + n].ID_generation = *ip++;}}
            break;

#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_TAG_ANCHOR
        case IO_REFINE_FLAG:		// nuclear-zoom refinement tag; read on ALL restart flags (incl. fresh IC start) so tagged ICs work as intended //
            for(n = 0; n < pc; n++) {P[offset + n].Refinement_Flag = *ip++;}
            break;
#endif

        case IO_MASS:		/* particle mass */
            for(n = 0; n < pc; n++) {P[offset + n].Mass = *fp++;}
            break;

        case IO_U:			/* temperature */
            for(n = 0; n < pc; n++) {CellP[offset + n].InternalEnergy = *fp++;}
            break;

        case IO_RHO:		/* density */
            for(n = 0; n < pc; n++) {CellP[offset + n].Density = *fp++;}
            break;

        case IO_NE:		/* electron abundance */
#if defined(COOLING) || defined(RT_CHEM_PHOTOION)
#ifndef CHIMES
            for(n = 0; n < pc; n++) {CellP[offset + n].Ne = *fp++;}
#endif
#endif
            break;


        case IO_KERNELRADIUS:		/* gas kernel length */
            for(n = 0; n < pc; n++) {P[offset + n].KernelRadius = *fp++;}
            break;

        case IO_DELAYTIME:
#ifdef GALSF_SUBGRID_WINDS
            for(n = 0; n < pc; n++) {CellP[offset + n].DelayTime = *fp++;}
#endif
            break;

        case IO_AGE:		/* Age of stars */
#ifdef GALSF
            for(n = 0; n < pc; n++) {P[offset + n].StellarAge = *fp++;}
#endif
            break;

        case IO_GRAINSIZE:
#ifdef GRAIN_FLUID
            for(n = 0; n < pc; n++) {P[offset + n].Grain_Size = *fp++;}
#endif
            break;

        case IO_GRAINTYPE:
#if defined(PIC_MHD)
            for(n = 0; n < pc; n++) {P[offset + n].MHD_PIC_SubType = *ip_int++;}
#endif
            break;

        case IO_FLUIDTYPE:
#ifdef HYDRO_MULTIFLUID
            for(n = 0; n < pc; n++) {P[offset + n].FluidType = (unsigned char) (*ip_int++);}
#endif
            break;

        case IO_Z:			/* Gas and star metallicity */
#ifdef METALS
            for(n = 0; n < pc; n++) {
                int nmax=NUM_METAL_SPECIES;
                if(RestartFlag==2 && All.ICFormat==3 && header.flag_metals<NUM_METAL_SPECIES && header.flag_metals>0) {nmax=header.flag_metals;} // special clause to catch cases where read-in snapshot did not use all the metals fields we want to read now
                for(k=0;k<nmax;k++) {P[offset + n].Metallicity[k] = *fp++;} // normal read-in
                if(nmax<NUM_METAL_SPECIES) {for(k=nmax;k<NUM_METAL_SPECIES;k++) {P[offset + n].Metallicity[k]=0;}} // any extra fields zero'd
            }
#endif
            break;

       case IO_DUSTCHEMZMET:
#if defined(GALSF_ISMDUSTCHEM_MODEL)
            for(n = 0; n < pc; n++) {
                for(k = 0; k < NUM_ISMDUSTCHEM_ELEMENTS; k++) {CellP[offset + n].ISMDustChem_Dust_Metal[k] = *fp++;} // Get dust fractions
                for(k = 0; k < NUM_ISMDUSTCHEM_SOURCES; k++) {CellP[offset + n].ISMDustChem_Dust_Source[k] = (*fp++) * CellP[offset + n].ISMDustChem_Dust_Metal[0];} // Then get the sources of dust, converting from dust mass fraction to total gas mass fraction
            }
#endif
            break;

        case IO_DUSTCHEMSPECIESMET:
#if defined(GALSF_ISMDUSTCHEM_MODEL)
            for(n = 0; n < pc; n++) {for(k = 0; k < NUM_ISMDUSTCHEM_SPECIES; k++) {CellP[offset + n].ISMDustChem_Dust_Species[k] = *fp++;}} // Get dust species fractions
#endif
            break;

        case IO_ISMDUSTCHEMMOL:    /* gas dust species following Species routines */
#if defined(GALSF_ISMDUSTCHEM_MODEL) && !defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
            for(n = 0; n < pc; n++) {CellP[offset + n].ISMDustChem_MassFractionInDenseMolecular = *fp++; CellP[offset + n].ISMDustChem_C_in_CO = *fp++;}
#endif
            break;

        case IO_DUSTCHEMGRAINBINNUMBERS:
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
            // Need to check whether the number of grain size bins is the same as the snapshot. If not zero extra bins for now and recalcuate them later in the code
            for(n = 0; n < pc; n++) {
                int nmax=NUM_ISMDUSTCHEM_SIZE_BINS;
                for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
                    int kf;
                    for(kf=0;kf<nmax;kf++) { // normal read-in
                        // Grain bin number are stored in special units
                        CellP[offset + n].ISMDustChem_Dust_NumberInBin[k][kf] = (*fp++) * UNIT_GRAIN_NUMBER;
                        if (CellP[offset + n].ISMDustChem_Dust_NumberInBin[k][kf] < 0) {CellP[offset + n].ISMDustChem_Dust_NumberInBin[k][kf] = 0;}
                    } 
                    if(nmax<NUM_ISMDUSTCHEM_SIZE_BINS) {for(kf=nmax;kf<NUM_ISMDUSTCHEM_SIZE_BINS;kf++) {CellP[offset + n].ISMDustChem_Dust_NumberInBin[k][kf]=0;}} // any extra fields zero'd
                }
            }
#endif
            break;

        case IO_DUSTCHEMGRAINBINMASS:
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
            for(n = 0; n < pc; n++) {
                // Check whether the desired number of grain size bins is the same as the snapshot. 
                // If not zero extra bins for now and recalcuate them later in the code.
                int nmax=NUM_ISMDUSTCHEM_SIZE_BINS;
                // The code outputs the bin mass but does not track it directly, instead tracking the bin slope
                // which depends on the bin number, mass, and bin edges. Bin mass is temporarily stored in 
                // the slope field and then the slope is recalculated in Initialize_ISMDustChemEvo_Particle_Variables(, P, CellP)
                for(k=0;k<NUM_ISMDUSTCHEM_SPECIES;k++) {
                    int kf;
                    for(kf=0;kf<nmax;kf++) { // normal read-in
                        CellP[offset + n].ISMDustChem_Dust_SlopeInBin[k][kf] = (*fp++) * UNIT_MASS_IN_CGS;
                        if (CellP[offset + n].ISMDustChem_Dust_SlopeInBin[k][kf] < 0) {CellP[offset + n].ISMDustChem_Dust_SlopeInBin[k][kf] = 0;}
                    } 
                    if(nmax<NUM_ISMDUSTCHEM_SIZE_BINS) {for(kf=nmax;kf<NUM_ISMDUSTCHEM_SIZE_BINS;kf++) {CellP[offset + n].ISMDustChem_Dust_SlopeInBin[k][kf]=0;}} // any extra fields zero'd
                }
            }
#endif
            break;

        case IO_BFLD:		/* Magnetic field */
#ifdef MAGNETIC
            for(n = 0; n < pc; n++)
            {
                for(k = 0; k < 3; k++) {CellP[offset + n].BPred[k] = *fp++;}
                CellP[offset + n].divB = 0;
#ifdef DIVBCLEANING_DEDNER
                CellP[offset + n].Phi = 0;
                CellP[offset + n].PhiPred = 0;
#endif
            }
#endif
            break;

        case IO_SINKMASS:
#ifdef SINK_PARTICLES
            for(n = 0; n < pc; n++) {P[offset + n].Sink_Mass = *fp++;}
#endif
            break;

        case IO_SINKDUSTMASSACC:
#if defined(SINK_PARTICLES) && defined(GRAIN_FLUID)
            for(n = 0; n < pc; n++) {P[offset + n].Sink_Dust_Mass = *fp++;}
#endif
            break;

        case IO_SINK_DIST:
            break;

        case IO_SINK_ANGMOM:
#ifdef SINK_FOLLOW_ACCRETED_ANGMOM
            for(n = 0; n < pc; n++) {for(k = 0; k < 3; k++) {P[offset + n].Sink_Specific_AngMom[k] = *fp++;}}
#endif
            break;

        case IO_SINKMASSALPHA:
#ifdef SINK_ALPHADISK_ACCRETION
            for(n = 0; n < pc; n++) {P[offset + n].Sink_Mass_Reservoir = *fp++;}
#endif
            break;

        case IO_SINKMDOT:
#ifdef SINK_PARTICLES
            for(n = 0; n < pc; n++) {P[offset + n].Sink_Mdot = *fp++;}
#endif
        case IO_R_PROTOSTAR:
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
            for(n = 0; n < pc; n++) {P[offset + n].ProtoStellarRadius_inSolar = *fp++;}
#endif
            break;

        case IO_MASS_D_PROTOSTAR:
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
            for(n = 0; n < pc; n++) {P[offset + n].Mass_D = *fp++;}
#endif
            break;

        case IO_ZAMS_MASS:
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
            for(n = 0; n < pc; n++) {P[offset + n].ZAMS_Mass = *fp++;}
#endif
            break;

        case IO_STAGE_PROTOSTAR:
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
            for(n = 0; n < pc; n++) {P[offset + n].ProtoStellarStage = *ip_int++;}
#endif
            break;
            
        case IO_AGE_PROTOSTAR:
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
            for(n = 0; n < pc; n++) {P[offset + n].ProtoStellarAge = *fp++;}
#endif
            break;

        case IO_LUM_SINGLESTAR:
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
            for(n = 0; n < pc; n++) {P[offset + n].StarLuminosity_Solar = *fp++;}

#endif
            break;

        case IO_SINKPROGS:
#ifdef SINK_COUNTPROGS
            for(n = 0; n < pc; n++) {P[offset + n].Sink_CountProgs = *ip_int++;}
#endif
            break;

        case IO_EOSTEMP:
            for(n = 0; n < pc; n++) {CellP[offset + n].Temperature = *fp++;}
            break;

        case IO_EOSABAR:
#ifdef EOS_CARRIES_ABAR
            for(n = 0; n < pc; n++) {CellP[offset + n].Abar = *fp++;}
#endif
            break;

        case IO_EOSCOMP:
#if defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
            for(n = 0; n < pc; n++) {CellP[offset + n].CompositionType = *ip_int++;}
#endif
            break;

        case IO_EOSPHASE:
            break;

        case IO_EOSYE:
#ifdef EOS_CARRIES_YE
            for(n = 0; n < pc; n++) {CellP[offset + n].Ye = *fp++;}
#endif
            break;

#ifdef NUCLEAR_NETWORK
        case IO_NUCLEAR_COMPOSITION:
            for(n = 0; n < pc; n++) {for(k = 0; k < NUM_NUCLEAR_SPECIES; k++) {P[offset + n].Metallicity[NUCLEAR_SPECIES_OFFSET_IN_METALLICITY + k] = *fp++;}}
            break;
#endif

        case IO_PARTVEL:
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==1)||(HYDRO_FIX_MESH_MOTION==2)||(HYDRO_FIX_MESH_MOTION==3))
            for(n = 0; n < pc; n++) {for(k = 0; k < 3; k++) {CellP[offset + n].ParticleVel[k] = *fp++;}}
#endif
            break;


        case IO_RADGAMMA:
#ifdef RADTRANSFER
            for(n = 0; n < pc; n++) {for(k = 0; k < N_RT_FREQ_BINS; k++) {CellP[offset + n].Rad_E_gamma[k] = *fp++;}}
#endif
            break;

        case IO_RAD_OPACITY:
#if defined(RADTRANSFER) && defined(OUTPUT_RT_RAD_OPACITY)
            for(n = 0; n < pc; n++) {for(k = 0; k < N_RT_FREQ_BINS; k++) {CellP[offset + n].Rad_Kappa[k] = *fp++;}}
#endif
            break;

        case IO_RAD_TEMP:
#if defined(RADTRANSFER) && defined(RT_INFRARED)
            for(n = 0; n < pc; n++) {CellP[offset + n].Radiation_Temperature = *fp++;}
#endif
            break;

        case IO_DUST_TEMP:
#if (defined(RADTRANSFER) && defined(RT_INFRARED)) || (defined(OUTPUT_DUST_TEMPERATURE) && (GALSF_FB_FIRE_STELLAREVOLUTION > 2))
            for(n = 0; n < pc; n++) {CellP[offset + n].Dust_Temperature = *fp++;}
#endif
            break;

        case IO_RAD_FLUX:
#if defined(RADTRANSFER) && defined(OUTPUT_RT_RAD_FLUX) && defined(RT_EVOLVE_FLUX)
            for(n = 0; n < pc; n++) {
                for(k=0;k<3;k++) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {CellP[offset + n].Rad_Flux_Pred[kf][k] = fp[N_RT_FREQ_BINS*k + kf];}} // will be corrected back into proper 'conserved variable' code units in rt_set_simple_inits subroutine later
                fp += 3*N_RT_FREQ_BINS;
            }
#endif
            break;
            
        case IO_EDDINGTON_TENSOR:
#if defined(RADTRANSFER) && defined(OUTPUT_EDDINGTON_TENSOR)
            for(n = 0; n < pc; n++) {
                for(k=0;k<6;k++) {int kf; for(kf=0;kf<N_RT_FREQ_BINS;kf++) {CellP[offset + n].ET[kf].data[k] = fp[N_RT_FREQ_BINS*k + kf];}}
                fp += 6*N_RT_FREQ_BINS;
            }
#endif
            break;

            
            
            /* adaptive softening parameters */
        case IO_AGS_HKERN:
#if defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE)
            for(n = 0; n < pc; n++) {P[offset + n].AGS_KernelRadius = *fp++;}
#endif
            break;

        case IO_AGS_ZETA:
#if defined (AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE) && defined(AGS_OUTPUTZETA)
            for(n = 0; n < pc; n++) {P[offset + n].AGS_zeta = *fp++;}
#endif
            break;

        case IO_CHIMES_ABUNDANCES:
#if defined(CHIMES) && !defined(CHIMES_INITIALISE_IN_EQM)
            for (n = 0; n < pc; n++)
            {
    	        allocate_gas_abundances_memory(&(ChimesGasVars[offset + n]), &ChimesGlobalVars);
	            for (k = 0; k < ChimesGlobalVars.totalNumberOfSpecies; k++) {ChimesGasVars[offset + n].abundances[k] = (ChimesFloat) (*fp++);}
#ifdef CHIMES_TURB_DIFF_IONS
                chimes_update_turbulent_abundances(n, 1, P, CellP);
#endif
            }
#endif
            break;

        case IO_COSMICRAY_ENERGY:
#ifdef COSMIC_RAY_FLUID
#ifdef CRFLUID_ALT_SPECTRUM_SPECIALSNAPRESTART
            for(n = 0; n < pc; n++) {CellP[offset + n].CosmicRayEnergy[0] = *fp++;}
#else
            for(n = 0; n < pc; n++) {for(k=0; k<N_CR_PARTICLE_BINS; k++) {CellP[offset + n].CosmicRayEnergy[k] = *fp++;}}
#endif
#endif
            break;

        case IO_COSMICRAY_SLOPES:
#if defined(COSMIC_RAY_FLUID) && defined(CRFLUID_EVOLVE_SPECTRUM)
#if !defined(CRFLUID_ALT_SPECTRUM_SPECIALSNAPRESTART) /* normal behavior - read the same list in that we would use */
            for(n = 0; n < pc; n++) {for(k=0; k<N_CR_PARTICLE_BINS; k++) {CellP[offset + n].CosmicRay_Number_in_Bin[k] = *fp++;}} // NOTE this still contains the SLOPE information; in init.c we convert back to number, our evolved variable!
#endif
#endif
            break;

        case IO_COSMICRAY_ALFVEN:
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            for(n = 0; n < pc; n++) {
                int k2; for(k=0;k<2;k++) {for(k2=0;k2<N_CR_PARTICLE_BINS;k2++) {
                        CellP[offset + n].CosmicRayAlfvenEnergy[k2][k] = fp[N_CR_PARTICLE_BINS*k + k2];}}
                fp += 2*N_CR_PARTICLE_BINS;
            }
#endif
            break;

        case IO_OSTAR:
#ifdef GALSF_SFR_IMF_SAMPLING
             for(n = 0; n < pc; n++) {P[offset + n].IMF_NumMassiveStars = *fp++;}
#endif
            break;

        case IO_DTOSTAR:
#ifdef GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF
            for(n = 0; n < pc; n++) {P[offset + n].TimeDistribOfStarFormation = *fp++;}
#endif
            break;

        case IO_UNSPMASS:
#if defined(SINK_WIND_SPAWN) && defined(OUTPUT_UNSPAWNED_SINKMASS)
             for(n = 0; n < pc; n++) {P[offset + n].unspawned_wind_mass = *fp++;}
#endif
            break; 
            
        case IO_TURB_DYNAMIC_COEFF:
#ifdef TURB_DIFF_DYNAMIC
            for (n = 0; n < pc; n++) {CellP[offset + n].TD_DynDiffCoeff = *fp++;}
#endif
            break;

        case IO_SINKRAD:
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            for(n = 0; n < pc; n++) {P[offset + n].SinkRadius = *fp++;}
#endif
            break;

        case IO_SINK_FORM_MASS:
#ifdef SINK_PARTICLES
            for(n = 0; n < pc; n++) {P[offset + n].Sink_Formation_Mass = *fp++;}
#endif
            break;	    
            
        case IO_MOLECULARFRACTION:
#if defined(COOL_MOLECFRAC_NONEQM) & !defined(IO_MOLECFRAC_NOT_IN_ICFILE)
            for (n = 0; n < pc; n++) {CellP[offset + n].MolecularMassFraction_perNeutralH = *fp++;}
#endif
            break;

        case IO_NH:        /* neutral hydrogen fraction */
#if defined(RT_CHEM_PHOTOION)
            for(n = 0; n < pc; n++) {CellP[offset + n].HI = *fp++;}
#endif
            break;
            
        case IO_HII:        /* ionized hydrogen abundance */
#if defined(RT_CHEM_PHOTOION)
            for(n = 0; n < pc; n++) {CellP[offset + n].HII = *fp++;}
#endif
            break;
            
        case IO_HeI:        /* neutral Helium */
#if defined(RT_CHEM_PHOTOION_HE)
            for(n = 0; n < pc; n++) {CellP[offset + n].HeI = *fp++;}
#endif
            break;
            
        case IO_HeII:        /* ionized Helium */
#if defined(RT_CHEM_PHOTOION_HE)
            for(n = 0; n < pc; n++) {CellP[offset + n].HeII = *fp++;}
#endif
            break;

        case IO_CBE_MOMENTS:   /* C7 (2026-05-30): real reader — see comment */
#ifdef CBE_INTEGRATOR
            /* CommBuffer holds the per-particle flat array of
             * NBASIS*NMOMENTS MyInputFloat values, stored basis-major:
             * index = NMOMENTS*basis + moment. Each per-basis row is
             * [m, p_x, p_y, p_z, T_xx, T_yy, T_zz, T_xy, T_xz, T_yz]
             * truncated to NMOMENTS = NUMDIMS+1 (no SECONDMOMENT) or
             * +NUMDIMS*(NUMDIMS+1)/2 (with SECONDMOMENT).
             *
             * RELATIVE-FRAME STORAGE CONVENTION (binding):
             *   basis_p_stored[α]  = m_α * (v_phys[α] − P.Vel)
             *   v_phys[α]          = basis_p_stored[α] / m_α + P.Vel
             *   Σ_α basis_p_stored = 0  (enforced by do_cbe_initialization
             *                            closure and the IC writer in
             *                            test/cbe_vlasov_common.py).
             * Per-basis stress slots (when SECONDMOMENT) are likewise stored
             * relative to P.Vel; the flux-frame helper performs the
             * relative→absolute boost when needed for the HLLC vacuum solve.
             *
             * For ranks/PartTypes whose /PartTypeX/VlasovMoments dataset
             * was absent in the HDF5 file, the upstream optional-block
             * path memset CommBuffer to zero (read_ic.cc around line
             * 1246), so this case will store zeros — that's harmless
             * because CBE_Moments_LoadedFromIC_PType[type] stays at 0
             * for those cases and do_cbe_initialization() will overwrite
             * the zeros with the cold-default synthesis. The per-type
             * flag (set up at the HDF5 H5Dread success point) is the
             * SSOT distinguishing "actually loaded" vs "absent → zeros". */
            for(n = 0; n < pc; n++) {
                for(k = 0; k < CBE_INTEGRATOR_NBASIS; k++) {
                    for(int kf = 0; kf < CBE_INTEGRATOR_NMOMENTS; kf++) {
                        P[offset + n].CBE_basis_moments[k][kf] = (MyFloat)(*fp++);
                    }
                }
            }
#endif
            break;

#ifdef EOS_DAMAGE_POROSITY
        /* damage and porosity are integrated history, not diagnostics: they cannot be recomputed
         * from the other fields, so a snapshot restart has to read them back (init.cc initializes
         * them only on a fresh start). A snapshot written without these datasets arrives as zeros;
         * the distention floor in init.cc restores the material default in that case. */
        case IO_DAMAGE_POROSITY_DAMAGE:
            for(n = 0; n < pc; n++) {CellP[offset + n].Damage = *fp++;}
            break;

        case IO_DAMAGE_POROSITY_DISTENTION:
            for(n = 0; n < pc; n++) {CellP[offset + n].Distention = *fp++;}
            break;

        case IO_DAMAGE_POROSITY_ACTVCRACKS:
            for(n = 0; n < pc; n++) {CellP[offset + n].ActiveCracks = *fp++;}
            break;
#endif


        /* the other input fields (if present) are not needed to define the
             initial conditions of the code */

        case IO_COSMICRAY_KAPPA:
        case IO_AGS_RHO:
        case IO_AGS_QPT:
        case IO_AGS_PSI_RE:
        case IO_AGS_PSI_IM:
        case IO_EOSCS:
        case IO_EOS_STRESS_TENSOR:
        case IO_SFR:
        case IO_POT:
        case IO_ACCEL:
        case IO_HYDROACCEL:
        case IO_DTENTR:
        case IO_RAD_ACCEL:
        case IO_CRATE:
        case IO_HRATE:
        case IO_NHRATE:
        case IO_HHRATE:
        case IO_MCRATE:
        case IO_PHRATE:
        case IO_DCRATE:
        case IO_TSTP:
        case IO_IMF:
        case IO_DIVB:
        case IO_ABVC:
        case IO_COOLRATE:
        case IO_AMDC:
        case IO_PHI:
        case IO_GRADPHI:
        case IO_GRADRHO:
        case IO_GRADVEL:
        case IO_GRADMAG:
        case IO_TIDALTENSORPS:
        case IO_PRESSURE:
        case IO_HSMS:
        case IO_ACRB:
        case IO_VSTURB_DISS:
        case IO_VSTURB_DRIVE:
        case IO_grHI:
        case IO_grHII:
        case IO_grHM:
        case IO_grHeI:
        case IO_grHeII:
        case IO_grHeIII:
        case IO_grH2I:
        case IO_grH2II:
        case IO_grDI:
        case IO_grDII:
        case IO_grHDI:
        case IO_TURB_DIFF_COEFF:
        case IO_DYNERROR:
        case IO_DYNERRORDEFAULT:
        case IO_VDIV:
        case IO_VORT:
        case IO_CHIMES_MU:
        case IO_CHIMES_REDUCED:
        case IO_CHIMES_NH:
        case IO_CHIMES_STAR_SIGMA:
        case IO_DENS_AROUND_STAR:
        case IO_DELAY_TIME_HII:
        case IO_CHIMES_FLUX_G0:
        case IO_CHIMES_FLUX_ION:
        case IO_MACHNUM:
        case IO_DUST_TO_GAS:
        case IO_AMBIPOLAR:
        case IO_OHMIC:
        case IO_HALL:
        case IO_SOFT:
        case IO_SHOCKMACHNUM:
            break;

        case IO_LASTENTRY:
            /* Internal "can't happen": unknown iofield. Pure-local void function (no MPI),
             * and every rank unpacking a given block hits the identical iofield, so a soft
             * bad-stop here is collective-symmetric and drains at the read_ic turn poll. */
            endrun(220);
            break;
    }
}



/*! Participant-only bad-stop status reconcile for read_file(). Point-to-point over the
 *  participating ranks ONLY (readTask..lastTask, TAG_IC_STATUS) -- deliberately NOT an
 *  MPI_COMM_WORLD collective: read_file() is dispatched on a SUBSET of ranks per driver
 *  turn (peers outside the group sit at the post-turn read_ic poll), so a COMM_WORLD
 *  reduce here would deadlock. readTask gathers each participant's local status,
 *  OR-combines (first nonzero wins), and scatters the combined status back. Returns the
 *  combined status on every participant (0 == all clear). Lets a per-rank IC failure
 *  become an all-participant graceful return BEFORE any header-derived allocation/use. */
static int read_file_sync_status(int readTask, int lastTask, int local_status)
{
    if(lastTask <= readTask) {return local_status;}   /* single participant (per-rank own-file): no peers waiting */
    MPI_Status mpistat;
    int combined = local_status;
    if(ThisTask == readTask)
    {
        int task, peer_status;
        for(task = readTask + 1; task <= lastTask; task++)
        {
            MPI_Recv(&peer_status, 1, MPI_INT, task, TAG_IC_STATUS, MPI_COMM_WORLD, &mpistat);
            if(peer_status != 0 && combined == 0) {combined = peer_status;}
        }
        for(task = readTask + 1; task <= lastTask; task++) {MPI_Ssend(&combined, 1, MPI_INT, task, TAG_IC_STATUS, MPI_COMM_WORLD);}
    }
    else
    {
        MPI_Ssend(&local_status, 1, MPI_INT, readTask, TAG_IC_STATUS, MPI_COMM_WORLD);
        MPI_Recv(&combined, 1, MPI_INT, readTask, TAG_IC_STATUS, MPI_COMM_WORLD, &mpistat);
    }
    return combined;
}


/*! This function reads a snapshot file and distributes the data it contains
 *  to tasks 'readTask' to 'lastTask'. Returns 0 on success, or a nonzero IC-error
 *  code on failure (a soft bad-stop is also requested; the caller's per-turn poll
 *  drains it to a clean finalize -- no MPI_Abort, the node releases).
 */
int read_file(char *fname, int readTask, int lastTask)
{
    int read_status = 0;   /* soft bad-stop accumulator; reconciled among participants below */
    size_t blockmaxlen;
    long long i, n_in_file, n_for_this_task, ntask, pc, offset = 0, task, nall, nread, nstart, npart;
    int blksize1, blksize2, type, bnr, bytes_per_blockelement, nextblock, typelist[6];
    MPI_Status status;
    FILE *fd = 0;
    char label[4], buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    enum iofields blocknr;


    int rank, pcsum;
    hid_t hdf5_file = 0, hdf5_grp[6], hdf5_dataspace_in_file;
    hid_t hdf5_datatype = 0, hdf5_dataspace_in_memory, hdf5_dataset;
    hsize_t dims[2], count[2], start[2];

    
#define SKIP  {my_fread(&blksize1,sizeof(int),1,fd);}
#define SKIP2  {my_fread(&blksize2,sizeof(int),1,fd);}

    if(ThisTask == readTask)
    {
        if(All.ICFormat == 1 || All.ICFormat == 2)
        {
            if(!(fd = fopen(fname, "r")))
            {
                printf("can't open file `%s' for reading initial conditions.\n", fname);
                read_status = 123;   /* soft: cannot open IC file (common user error). Reconciled at Checkpoint A. */
            }

            if(fd)   /* only touch the file if it opened -- a NULL-fd my_fread would segfault before the reconcile */
            {
                if(All.ICFormat == 2)
                {
                    SKIP;
                    my_fread(&label, sizeof(char), 4, fd);
                    my_fread(&nextblock, sizeof(int), 1, fd);
                    printf("Reading header => '%c%c%c%c' (%d byte)\n", label[0], label[1], label[2], label[3], nextblock);
                    SKIP2;
                }

                SKIP;
                my_fread(&header, sizeof(header), 1, fd);
                SKIP2;

                if(blksize1 != 256 || blksize2 != 256)
                {
                    printf("incorrect header format\n");
                    fflush(stdout);
                    read_status = 890;   /* soft: bad IC header format. Reconciled at Checkpoint A. */
                    /* Probable error is wrong size of fill[] in header file. Needs to be 256 bytes in total. */
                }
            }
        }



        if(All.ICFormat == 3)
        {
            read_header_attributes_in_hdf5(fname);
            hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
            for(type = 0; type < 6; type++)
            {
                if(header.npart[type] > 0)
                {
                    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "/PartType%d", type);
                    hdf5_grp[type] = H5Gopen(hdf5_file, buf);
                }
            }
        }

        
        for(task = readTask + 1; task <= lastTask; task++)
        {
            MPI_Ssend(&header, sizeof(header), MPI_BYTE, task, TAG_HEADER, MPI_COMM_WORLD);
        }

    }
    else
    {
        MPI_Recv(&header, sizeof(header), MPI_BYTE, readTask, TAG_HEADER, MPI_COMM_WORLD, &status);
    }

    /* CHECKPOINT A: reconcile header open/parse status among the participants BEFORE any
     * header-derived logic (precision check, TotNumPart, allocate_memory, CommBuffer). The
     * readTask may have failed fopen/format and already Ssent a junk header to its peers; all
     * participants must bail together here, before touching that junk. (readTask..lastTask
     * P2P only -- not a COMM_WORLD collective.) The per-turn read_ic poll then finalizes. */
    read_status = read_file_sync_status(readTask, lastTask, read_status);
    if(read_status) {endrun(read_status); return read_status;}

    /* unformatted-binary formats only -- see the matching note in read_ic(): HDF5 datasets carry
     * their own dtype and are read at the file's precision, so the header flag does not gate them */
    if(All.ICFormat != 3)
    {
#ifdef INPUT_IN_DOUBLEPRECISION
        if(header.flag_doubleprecision == 0)
        {
            if(ThisTask == 0) {printf("\nProblem: Code compiled with INPUT_IN_DOUBLEPRECISION, but input files are in single precision!\n"); fflush(stdout);}
            /* Symmetric across all participants (identical header + compile flag) -- soft bad-stop + return together. */
            endrun(11); return 11;
        }
#else
        if(header.flag_doubleprecision)
        {
            if(ThisTask == 0) {printf("\nProblem: Code not compiled with INPUT_IN_DOUBLEPRECISION, but input files are in double precision!\n"); fflush(stdout);}
            /* Symmetric across all participants (identical header + compile flag) -- soft bad-stop + return together. */
            endrun(10); return 10;
        }
#endif
    }
    /* Precision mismatch (above) is collective-symmetric: every participant sees the same
     * header.flag_doubleprecision and the same compile flag, so all return together -- no
     * peer is stranded, and the per-turn read_ic poll finalizes cleanly (no MPI_Abort). */

    /* NOTE: the global-count compute + MaxPart + allocate_memory() + CommBuffer + cosmo-factor
     * setup that used to live here (gated on `All.TotNumPart==0`, i.e. the first read_file call)
     * has been MOVED to the ALL-RANK setup phase in read_ic() right after find_files(), so the
     * particle storage is preflighted+allocated collectively (graceful OOM) and allocate_memory()
     * is never subset/turn-called from the IC path. By the time read_file() runs, All.TotNumPart
     * is already set and P/CellP/CommBuffer are already allocated; read_file() only reads data. */

    if(ThisTask == readTask)
    {
        for(i = 0, n_in_file = 0; i < 6; i++) {n_in_file += header.npart[i];}

        printf("\nReading file `%s' on task=%d (contains %lld particles.)\n"
               " ..distributing this file to tasks %d-%d\n"
               "Type 0 (gas):   %8d  (tot=%6d%09d) masstab=%g\n"
               "Type 1 (halo):  %8d  (tot=%6d%09d) masstab=%g\n"
               "Type 2 (alt):   %8d  (tot=%6d%09d) masstab=%g\n"
               "Type 3 (pic):   %8d  (tot=%6d%09d) masstab=%g\n"
               "Type 4 (stars): %8d  (tot=%6d%09d) masstab=%g\n"
               "Type 5 (sink):  %8d  (tot=%6d%09d) masstab=%g\n\n", fname, ThisTask, n_in_file, readTask,
               lastTask, header.npart[0], (int) (header.npartTotal[0] / 1000000000),
               (int) (header.npartTotal[0] % 1000000000), All.MassTable[0], header.npart[1],
               (int) (header.npartTotal[1] / 1000000000), (int) (header.npartTotal[1] % 1000000000),
               All.MassTable[1], header.npart[2], (int) (header.npartTotal[2] / 1000000000),
               (int) (header.npartTotal[2] % 1000000000), All.MassTable[2], header.npart[3],
               (int) (header.npartTotal[3] / 1000000000), (int) (header.npartTotal[3] % 1000000000),
               All.MassTable[3], header.npart[4], (int) (header.npartTotal[4] / 1000000000),
               (int) (header.npartTotal[4] % 1000000000), All.MassTable[4], header.npart[5],
               (int) (header.npartTotal[5] / 1000000000), (int) (header.npartTotal[5] % 1000000000),
               All.MassTable[5]);
        fflush(stdout);
    }


    ntask = lastTask - readTask + 1;


    /* to collect the gas particles all at the beginning (in case several
     snapshot files are read on the current CPU) we move the collisionless
     particles such that a gap of the right size is created */

    for(type = 0, nall = 0; type < 6; type++)
    {
        n_in_file = header.npart[type];
        n_for_this_task = n_in_file / ntask;
        if((ThisTask - readTask) < (n_in_file % ntask)) {n_for_this_task++;}


        if(type == 0)
        {
            if(N_gas + n_for_this_task > All.MaxPartGas)
            {
                printf("Not enough space on task=%d for gas/fluid cells (space for %d, need at least %lld)\n", ThisTask, All.MaxPartGas, N_gas + n_for_this_task);
                fflush(stdout);
                read_status = 172;   /* soft: insufficient gas space (low PartAllocFactor). Reconciled at Checkpoint B before memmove. */
            }
        }

        nall += n_for_this_task;
    }

    if(NumPart + nall > All.MaxPart)
    {
        printf("Not enough space on task=%d (space for %d, need at least %lld)\n", ThisTask, All.MaxPart, NumPart + nall);
        fflush(stdout);
        read_status = 173;   /* soft: insufficient particle space (low PartAllocFactor). Reconciled at Checkpoint B before memmove. */
    }

    /* CHECKPOINT B: reconcile capacity/CommBuffer/space status among participants BEFORE the
     * memmove and the per-block P2P loop. A returning rank with bad `nall` must not memmove
     * (would corrupt P[]); and a rank that bailed must not be left out of the block-loop
     * Ssend/Recv. (readTask..lastTask P2P only -- not a COMM_WORLD collective.) */
    read_status = read_file_sync_status(readTask, lastTask, read_status);
    if(read_status) {endrun(read_status); return read_status;}

    memmove(&P[N_gas + nall], &P[N_gas], (NumPart - N_gas) * sizeof(struct particle_data));
    nstart = N_gas;


    for(bnr = 0; bnr < 1000; bnr++)
    {
        blocknr = (enum iofields) bnr;
        if(blocknr == IO_LASTENTRY) {break;}
        if(RestartFlag == 5 && blocknr > IO_MASS) {continue;}	/* if we only do power spectra, we don't need to read other blocks beyond the mass */

        if(blockpresent(blocknr))
        {
                /* blocks only for restartflag == 0 */
                if(RestartFlag == 0 && blocknr > IO_U
                   && blocknr != IO_BFLD
#ifdef INPUT_READ_KERNELRADIUS
                   && blocknr != IO_KERNELRADIUS
#endif
#ifdef INPUT_READ_EOSTEMP
                   && blocknr != IO_EOSTEMP
#endif
#ifdef EOS_CARRIES_ABAR
                   && blocknr != IO_EOSABAR
#endif
#ifdef EOS_CARRIES_YE
                   && blocknr != IO_EOSYE
#endif
#ifdef NUCLEAR_NETWORK
                   && blocknr != IO_NUCLEAR_COMPOSITION
                   && blocknr != IO_Z /* need to read Metallicity for nuclear species stored within it */
#endif
#if (defined(EOS_TILLOTSON) || defined(EOS_ANEOS)) && !defined(IO_COMPOSITIONTYPE_NOT_IN_ICFILE)
                   && blocknr != IO_EOSCOMP
#endif
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==1)||(HYDRO_FIX_MESH_MOTION==2)||(HYDRO_FIX_MESH_MOTION==3))
                   && blocknr != IO_PARTVEL
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS) && defined(INPUT_READ_SINKPROPS)
                   && blocknr != IO_SINKRAD
#endif
#if defined(SINK_PARTICLES) && defined(INPUT_READ_SINKPROPS)
                   && blocknr != IO_SINK_FORM_MASS
#endif		   
#if defined(CHIMES) && !defined(CHIMES_INITIALISE_IN_EQM)
                   && blocknr != IO_CHIMES_ABUNDANCES
#endif
#ifdef PIC_MHD
                   && blocknr != IO_GRAINTYPE
#endif
#ifdef HYDRO_MULTIFLUID
                   && blocknr != IO_FLUIDTYPE
#endif
#ifdef CBE_INTEGRATOR
                   /* C7 IC reader: allow VlasovMoments through during
                    * RestartFlag==0 so the loaded values reach
                    * do_cbe_initialization. Without this, blockpresent()
                    * returns 1 but the dataset is silently skipped, and
                    * cold-default synthesis quietly overwrites the IC
                    * (CBE_Moments_LoadedFromIC_PType stays 0). */
                   && blocknr != IO_CBE_MOMENTS
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && defined(INPUT_READ_SINKPROPS)
                   && blocknr != IO_R_PROTOSTAR
                   && blocknr != IO_MASS_D_PROTOSTAR
                   && blocknr != IO_ZAMS_MASS
                   && blocknr != IO_STAGE_PROTOSTAR
                   && blocknr != IO_AGE_PROTOSTAR
                   && blocknr != IO_LUM_SINGLESTAR
                   && blocknr != IO_AGE
#endif
                   )
                                continue;	/* ignore all other blocks in initial conditions */


            if(RestartFlag == 0 && (blocknr == IO_GENERATION_ID || blocknr == IO_CHILD_ID)) {continue;}
#if defined(NO_CHILD_IDS_IN_ICS) || defined(ASSIGN_NEW_IDS)
            if(blocknr == IO_GENERATION_ID || blocknr == IO_CHILD_ID) {continue;}
#endif
            if((RestartFlag == 0) && (All.InitGasTemp > 0) && (blocknr == IO_U)) {continue;}


#ifdef MHD_B_SET_IN_PARAMS
            if(RestartFlag == 0 && blocknr == IO_BFLD) {continue;}
#endif

#ifdef SUBFIND
            if(RestartFlag == 2 && blocknr == IO_HSMS) {continue;}
#endif

#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
#ifndef AGS_OUTPUTZETA
            if(blocknr == IO_AGS_ZETA) {continue;}
#endif
#endif
            
#ifdef CRFLUID_ALT_SPECTRUM_SPECIALSNAPRESTART
            if(RestartFlag == 2 && blocknr == IO_COSMICRAY_SLOPES) {continue;}
#if (CRFLUID_ALT_SPECTRUM_SPECIALSNAPRESTART==2)
            if(RestartFlag == 2 && blocknr == IO_COSMICRAY_ENERGY) {continue;}
#endif
#endif

#if !defined(RADTRANSFER)
            if(RestartFlag == 2 && blocknr == IO_RADGAMMA) {continue;}
#endif
#if !(defined(OUTPUT_RT_RAD_OPACITY) && defined(RADTRANSFER))
            if(RestartFlag == 2 && blocknr == IO_RAD_OPACITY) {continue;}
#endif
#if !(defined(RT_INFRARED) && defined(RADTRANSFER))
            if(RestartFlag == 2 && blocknr == IO_RAD_TEMP) {continue;}
#endif
#if !(defined(RT_INFRARED) && defined(RADTRANSFER))
            if(RestartFlag == 2 && blocknr == IO_DUST_TEMP) {continue;}
#endif
#if !(defined(OUTPUT_RT_RAD_FLUX) && defined(RT_EVOLVE_FLUX) && defined(RADTRANSFER))
            if(RestartFlag == 2 && blocknr == IO_RAD_FLUX) {continue;}
#endif
#if !(defined(OUTPUT_EDDINGTON_TENSOR) && defined(RADTRANSFER))
            if(RestartFlag == 2 && blocknr == IO_EDDINGTON_TENSOR) {continue;}
#endif

#if defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL) && defined(SINGLE_STAR_RESTART_FROM_FIRESIM)
            if(RestartFlag == 2 && blocknr == IO_RADGAMMA) {continue;}
            if(RestartFlag == 2 && blocknr == IO_RAD_OPACITY) {continue;}
            if(RestartFlag == 2 && blocknr == IO_RAD_TEMP) {continue;}
            if(RestartFlag == 2 && blocknr == IO_DUST_TEMP) {continue;}
            if(RestartFlag == 2 && blocknr == IO_RAD_FLUX) {continue;}
            if(RestartFlag == 2 && blocknr == IO_EDDINGTON_TENSOR) {continue;}
            if(RestartFlag == 2 && blocknr == IO_OSTAR) {continue;}
            if(RestartFlag == 2 && blocknr == IO_DTOSTAR) {continue;}
            if(RestartFlag == 2 && blocknr == IO_HII) {continue;}
            if(RestartFlag == 2 && blocknr == IO_HeI) {continue;}
            if(RestartFlag == 2 && blocknr == IO_HeII) {continue;}
#endif

#if defined(IO_MOLECFRAC_NOT_IN_ICFILE)
            if(RestartFlag == 2 && blocknr == IO_MOLECULARFRACTION) {continue;}
#endif

#if defined(IO_DUST_NOT_IN_ICFILE)
#if defined(GALSF_ISMDUSTCHEM_MODEL)
            if(RestartFlag == 2 && blocknr == IO_DUST_TO_GAS) {continue;}
            if(RestartFlag == 2 && blocknr == IO_DUSTCHEMZMET) {continue;}
            if(RestartFlag == 2 && blocknr == IO_DUSTCHEMSPECIESMET) {continue;}
            if(RestartFlag == 2 && blocknr == IO_ISMDUSTCHEMMOL) {continue;}
            if(RestartFlag == 2 && blocknr == IO_DCRATE) {continue;}
#if defined(GALSF_ISMDUSTCHEM_GRAINSIZEEVO)
            if(RestartFlag == 2 && blocknr == IO_MACHNUM) {continue;}
            if(RestartFlag == 2 && blocknr == IO_DUSTCHEMGRAINBINNUMBERS) {continue;}
            if(RestartFlag == 2 && blocknr == IO_DUSTCHEMGRAINBINMASS) {continue;}
#endif
#endif
#endif
            
            if(blocknr == IO_HSMS) {continue;}

#ifdef TURB_DIFF_DYNAMIC
            if(RestartFlag == 0 && blocknr == IO_TURB_DYNAMIC_COEFF) {continue;}
#endif

            if(ThisTask == readTask)
            {
                get_dataset_name(blocknr, buf);
                printf("reading block %d (%s)...\n", bnr, buf);
            }
            int printed_legacy_alias = 0;   /* limit the legacy-name notice below to one line per block */

            bytes_per_blockelement = get_bytes_per_blockelement(blocknr, 1);
            /* the old unformatted fortran binary GADGET-2 format used unsigned int for IDs, so that
             * width has to be respected when reading one. All.ICFormat is what the reader actually
             * parses the file as (it selects every format branch below), so it gates this too --
             * All.SnapFormat alone described the file the run WRITES, and on a format-3 read with
             * SnapFormat=1 it mis-sized the block against the HDF5 dtype. */
            if(blocknr == IO_ID && All.ICFormat == 1 && ((RestartFlag == 0) || (RestartFlag == 2 && All.SnapFormat == 1))) {bytes_per_blockelement = sizeof(unsigned int);}
#if (CRFLUID_ALT_SPECTRUM_SPECIALSNAPRESTART==1)
            if(RestartFlag == 2 && blocknr == IO_COSMICRAY_ENERGY) {bytes_per_blockelement = (1) * get_input_float_bytes(blocknr);}
#endif
#ifdef METALS /* some trickery here to enable snapshot-restarts from runs with different numbers of metal species */
            if(blocknr==IO_Z && RestartFlag==2 && All.ICFormat==3 && header.flag_metals<NUM_METAL_SPECIES && header.flag_metals>0) {bytes_per_blockelement = (header.flag_metals) * get_input_float_bytes(blocknr);}
#endif
            size_t MyBufferSize = All.BufferSize;
            blockmaxlen = (size_t) ((MyBufferSize * 1024 * 1024) / bytes_per_blockelement);
            npart = get_particles_in_block(blocknr, &typelist[0]);

            if(npart > 0)
            {
                    if(ThisTask == readTask)
                    {
                        if(All.ICFormat == 2)
                        {
                            get_Tab_IO_Label(blocknr, label);
                            int fb_status = find_block(label, fd);   /* 0 ok; nonzero = bad format / block absent */
                            /* soft: keep the block loop running (readTask keeps Ssending stale/junk, peers keep
                             * Recv-ing) so no peer is stranded; drains at the per-turn read_ic poll. */
                            if(fb_status) {if(read_status == 0) {endrun(fb_status);} read_status = fb_status;}
                        }

                        /* read_status != 0 here means find_block failed -> fd is at EOF / a bad
                         * position; do NOT touch the stream (the SKIP/my_fread below would hit the
                         * my_fread EOF hold). The block's data is zeroed instead (do-while below). */
                        if((All.ICFormat == 1 || All.ICFormat == 2) && read_status == 0) {
                            SKIP;
                            if (blksize1 == 0) { /* workaround for MUSIC ICs */
                              SKIP2;
                              SKIP;
                            }
                        }
                    }

                for(type = 0, offset = 0, nread = 0; type < 6; type++)
                {
                    n_in_file = header.npart[type];

                    pcsum = 0;

                    if(typelist[type] == 0)
                    {
                        n_for_this_task = n_in_file / ntask;
                        if((ThisTask - readTask) < (n_in_file % ntask)) {n_for_this_task++;}

                        offset += n_for_this_task;
                    }
                    else
                    {
                        for(task = readTask; task <= lastTask; task++)
                        {
                            n_for_this_task = n_in_file / ntask;
                            if((task - readTask) < (n_in_file % ntask)) {n_for_this_task++;}

                            if(task == ThisTask)
                                if(NumPart + n_for_this_task > All.MaxPart)
                                {
                                    printf("too many particles. %d %lld %d\n", NumPart, n_for_this_task, All.MaxPart);
                                    /* soft: per-rank overflow mid-distribution. Keep the Ssend/Recv loop running (so no
                                     * peer is stranded); the out-of-bounds empty_read_buffer write below is skipped while
                                     * read_status is set. Drains at the per-turn read_ic poll. */
                                    if(read_status == 0) {endrun(1313);}
                                    read_status = 1313;
                                }


                            do
                            {
                                pc = n_for_this_task;
                                if(pc > (int)blockmaxlen) {pc = blockmaxlen;}

                                if(ThisTask == readTask)
                                {
                                    if(All.ICFormat == 1 || All.ICFormat == 2)
                                    {
                                        if(read_status == 0)
                                        {
                                            my_fread(CommBuffer, bytes_per_blockelement, pc, fd);
                                            nread += pc;
                                        }
                                        else
                                        {
                                            /* stream position invalid (find_block/earlier read failed): do NOT touch fd
                                             * (would hit the my_fread EOF hold). Zero the buffer; the Ssend below still
                                             * delivers defined data so no peer is stranded. Drains at the per-turn poll. */
                                            memset(CommBuffer, 0, bytes_per_blockelement * pc);
                                        }
                                    }


                                    if(All.ICFormat == 3 && pc > 0)
                                    {
                                        get_dataset_name(blocknr, buf);
                                        /* suppress HDF5 error messages for optional datasets that may not exist in the file */
                                        H5E_auto_t old_func; void *old_client_data;
                                        H5Eget_auto(&old_func, &old_client_data);
                                        H5Eset_auto(NULL, NULL);
                                        hdf5_dataset = H5Dopen(hdf5_grp[type], buf);
                                        if(hdf5_dataset < 0)
                                        {
                                            /* modern dataset name absent: retry with the legacy pre-rename name
                                             * (backward-compat for older HDF5 snapshots; see get_dataset_name_legacy_alias) */
                                            char altbuf[DEFAULT_PATH_BUFFERSIZE_TOUSE]; get_dataset_name_legacy_alias(blocknr, altbuf);
                                            if(altbuf[0] != '\0')
                                            {
                                                hdf5_dataset = H5Dopen(hdf5_grp[type], altbuf);
                                                if(hdf5_dataset >= 0 && ThisTask == readTask && !printed_legacy_alias) {printf("   (block %d: '%s' absent, reading legacy-named '%s')\n", bnr, buf, altbuf); printed_legacy_alias = 1;}
                                            }
                                        }
                                        H5Eset_auto(old_func, old_client_data);

                                      if(hdf5_dataset >= 0)
                                      {
                                        dims[0] = header.npart[type];
                                        dims[1] = get_values_per_blockelement(blocknr);
#if (CRFLUID_ALT_SPECTRUM_SPECIALSNAPRESTART==1)
                                        if(RestartFlag == 2 && blocknr == IO_COSMICRAY_ENERGY) {dims[1] = 1;}
#endif
#ifdef METALS /* some trickery here to enable snapshot-restarts from runs with different numbers of metal species */
                                        if(blocknr==IO_Z && RestartFlag==2 && All.ICFormat==3 && header.flag_metals<NUM_METAL_SPECIES && header.flag_metals>0) {dims[1] = header.flag_metals;}
#endif
                                        if(dims[1] == 1) {rank = 1;} else {rank = 2;}
                                        hdf5_dataspace_in_file = H5Screate_simple(rank, dims, NULL);

                                        dims[0] = pc;
                                        hdf5_dataspace_in_memory = H5Screate_simple(rank, dims, NULL);

                                        start[0] = pcsum;
                                        start[1] = 0;

                                        count[0] = pc;
                                        count[1] = get_values_per_blockelement(blocknr);
#if (CRFLUID_ALT_SPECTRUM_SPECIALSNAPRESTART==1)
                                        if(RestartFlag == 2 && blocknr == IO_COSMICRAY_ENERGY) {count[1] = 1;}
#endif
#ifdef METALS /* some trickery here to enable snapshot-restarts from runs with different numbers of metal species */
                                        if(blocknr==IO_Z && RestartFlag==2 && All.ICFormat==3 && header.flag_metals<NUM_METAL_SPECIES && header.flag_metals>0) {count[1] = header.flag_metals;}
#endif
                                        pcsum += pc;

                                        H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

                                        switch(get_datatype_in_block(blocknr))
                                        {
                                            case 0:
                                                hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT);
                                                break;

                                            case 1:
                                                /* read at the file's own precision: HDF5 converts the
                                                 * stored dtype into this memory type. Reading a
                                                 * double-precision dataset as float silently collapses
                                                 * positions onto the float grid. */
                                                hdf5_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
                                                break;

                                            case 2:
                                                hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT64);
                                                break;

                                            case 3:
                                                hdf5_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
                                                break;
                                        }

                                        if(H5Dread(hdf5_dataset, hdf5_datatype, hdf5_dataspace_in_memory, hdf5_dataspace_in_file, H5P_DEFAULT, CommBuffer) < 0)
                                          {
                                            printf("read_ic: H5Dread FAILED for dataset '%s' (particle type %d). "
                                                   "If the IC is HDF5-compressed, the linked HDF5 library is likely "
                                                   "missing the required filter (e.g. deflate). Stopping gracefully "
                                                   "rather than continuing with uninitialized particle data.\n", buf, type);
                                            fflush(stdout);
                                            /* soft: zero the buffer so the Ssend below delivers defined zeros (peers don't
                                             * hang on their Recv), set the bad-stop, finish the block loop, drain at the
                                             * per-turn read_ic poll. Same shape as the optional-dataset-absent path below. */
                                            memset(CommBuffer, 0, bytes_per_blockelement * pc);
                                            if(read_status == 0) {endrun(1888);}
                                            read_status = 1888;
                                          }
                                        H5Tclose(hdf5_datatype);
                                        H5Sclose(hdf5_dataspace_in_memory);
                                        H5Sclose(hdf5_dataspace_in_file);
                                        H5Dclose(hdf5_dataset);
#ifdef CBE_INTEGRATOR
                                        /* C7 (2026-05-30): per-type flag set ONLY here, inside the
                                         * `if(hdf5_dataset >= 0)` block after a successful H5Dread.
                                         * Setting it inside empty_read_buffer's case would mark
                                         * absent-dataset zeros as "loaded" and silently corrupt the
                                         * IC. Per-PartType (not a global flag) because VlasovMoments
                                         * lives under /PartTypeX/ in the HDF5 layout. */
                                        if(blocknr == IO_CBE_MOMENTS) {
                                            if(type >= 0 && type < 6) {
                                                CBE_Moments_LoadedFromIC_PType[type] = 1;
                                            }
                                        }
#endif
                                      }
                                      else
                                      {
                                        /* Optional HDF5 dataset absent: zero CommBuffer so empty_read_buffer's
                                         * dispatch defaults the field rather than reading stale buffer contents. */
                                        memset(CommBuffer, 0, bytes_per_blockelement * pc);
                                      }
                                    }

                                }

                                if(ThisTask == readTask && task != readTask && pc > 0) {MPI_Ssend(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, task, TAG_PDATA, MPI_COMM_WORLD);}

                                if(ThisTask != readTask && task == ThisTask && pc > 0) {MPI_Recv(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, readTask, TAG_PDATA, MPI_COMM_WORLD, &status);}

                                if(ThisTask == task && read_status == 0)   /* skip the P[] write if this rank overflowed (read_status set); Recv above still completed the Ssend */
                                {
                                    empty_read_buffer(blocknr, nstart + offset, pc, type);
                                    offset += pc;
                                }

                                n_for_this_task -= pc;
                            }
                            while(n_for_this_task > 0);
                        }
                    }
                }

                if(ThisTask == readTask)
                {
                        /* read_status guard: if the stream is already bad (find_block/earlier read failed)
                         * skip the trailing SKIP2 -- it would hit the my_fread EOF hold and the blksize
                         * mismatch check is meaningless on an invalid position. */
                        if((All.ICFormat == 1 || All.ICFormat == 2) && read_status == 0)
                        {
                            SKIP2;
                            if(blksize1 != blksize2)
                            {
                                printf("incorrect block-sizes detected!\n");
                                printf("Task=%d   blocknr=%d  blksize1=%d  blksize2=%d\n", ThisTask, bnr, blksize1, blksize2);
                                if(blocknr == IO_ID) {printf("Possible mismatch of 32bit and 64bit ID's in IC file and GIZMO compilation !\n");}
                                fflush(stdout);
                                /* soft: corrupt/misaligned binary block. This block's P2P already completed; keep the
                                 * outer block loop running (readTask keeps Ssending, peers keep Recv-ing) so no peer is
                                 * stranded; drains at the per-turn read_ic poll. */
                                if(read_status == 0) {endrun(1889);}
                                read_status = 1889;
                            }
                        }
                }
            }
        }
    }

    for(type = 0; type < 6; type++)
    {
        n_in_file = header.npart[type];
        n_for_this_task = n_in_file / ntask;
        if((ThisTask - readTask) < (n_in_file % ntask)) {n_for_this_task++;}
        NumPart += n_for_this_task;
        if(type == 0) {N_gas += n_for_this_task;}
    }

    if(ThisTask == readTask)
    {
        if(All.ICFormat == 1 || All.ICFormat == 2) {fclose(fd);}

        if(All.ICFormat == 3)
        {
            for(type = 5; type >= 0; type--) {if(header.npart[type] > 0) {H5Gclose(hdf5_grp[type]);}}
            H5Fclose(hdf5_file);
        }

    }

    return read_status;   /* 0 on success; any block-loop soft bad-stop was already requested and drains at the per-turn read_ic poll */
}



/*! This function determines on how many files a given snapshot is distributed.
 */
int find_files(char *fname)
{
    FILE *fd; char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE], buf1[DEFAULT_PATH_BUFFERSIZE_TOUSE]; int dummy;

    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.%d", fname, 0);
    snprintf(buf1, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s", fname);

    if(All.ICFormat == 3)
    {
        snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.%d.hdf5", fname, 0);
        snprintf(buf1, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s.hdf5", fname);
    }

    header.num_files = 0;

    if(ThisTask == 0)
    {
        if((fd = fopen(buf, "r")))
        {
            if(All.ICFormat == 1 || All.ICFormat == 2)
            {
                if(All.ICFormat == 2)
                {
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                }

                my_fread(&dummy, sizeof(dummy), 1, fd);
                my_fread(&header, sizeof(header), 1, fd);
                my_fread(&dummy, sizeof(dummy), 1, fd);
            }
            fclose(fd);

            if(All.ICFormat == 3) {read_header_attributes_in_hdf5(buf);}

        }
    }

    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    if(header.num_files > 0) {return header.num_files;}

    if(ThisTask == 0)
    {
        if((fd = fopen(buf1, "r")))
        {
            if(All.ICFormat == 1 || All.ICFormat == 2)
            {
                if(All.ICFormat == 2)
                {
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                    my_fread(&dummy, sizeof(dummy), 1, fd);
                }

                my_fread(&dummy, sizeof(dummy), 1, fd);
                my_fread(&header, sizeof(header), 1, fd);
                my_fread(&dummy, sizeof(dummy), 1, fd);
            }
            fclose(fd);

            if(All.ICFormat == 3) {read_header_attributes_in_hdf5(buf1);}

            header.num_files = 1;
        }
    }

    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    if(header.num_files > 0) {return header.num_files;}

    if(ThisTask == 0)
    {
        printf("\nCan't find initial conditions file.");
        printf("neither as '%s'\nnor as '%s'\n", buf, buf1);
        fflush(stdout);
    }

    endrun(0);
    return 0;
}



/*! This function assigns a certain number of files to processors, such that
 *  each processor is exactly assigned to one file, and the number of cpus per
 *  file is as homogenous as possible. The number of files may at most be
 *  equal to the number of processors.
 */
void distribute_file(int nfiles, int firstfile, int firsttask, int lasttask, int *filenr, int *primary_taskID, int *last)
{
    int ntask, filesleft, filesright, tasksleft;

    if(nfiles > 1)
    {
        ntask = lasttask - firsttask + 1;

        filesleft = (int) ((((double) (ntask / 2)) / ntask) * nfiles);
        if(filesleft <= 0)
            filesleft = 1;
        if(filesleft >= nfiles)
            filesleft = nfiles - 1;

        filesright = nfiles - filesleft;

        tasksleft = ntask / 2;

        distribute_file(filesleft, firstfile, firsttask, firsttask + tasksleft - 1, filenr, primary_taskID, last);
        distribute_file(filesright, firstfile + filesleft, firsttask + tasksleft, lasttask, filenr, primary_taskID, last);
    }
    else
    {
        if(ThisTask >= firsttask && ThisTask <= lasttask)
        {
            *filenr = firstfile;
            *primary_taskID = firsttask;
            *last = lasttask;
        }
    }
}




void read_header_attributes_in_hdf5(char *fname)
{
    hid_t hdf5_file, hdf5_headergrp, hdf5_attribute;

    hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    hdf5_headergrp = H5Gopen(hdf5_file, "/Header");

    /* Suppress HDF5 error messages for optional attributes that may not exist in the IC file.
       Many ICs lack attributes like Flag_DoublePrecision, Flag_Metals, cell merge/split limits, etc.
       HDF5 prints verbose error stacks to stderr for each missing attribute, which is harmless
       but confusing. We restore error reporting after reading. */
#if H5_VERSION_GE(1,8,0) && !defined(H5_USE_16_API)
    H5E_auto2_t old_func; void *old_client_data;
    H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);
#else
    H5E_auto_t old_func; void *old_client_data;
    H5Eget_auto(&old_func, &old_client_data);
    H5Eset_auto(NULL, NULL);
#endif

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_ThisFile");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, header.npart);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total");
    H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotal);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumPart_Total_HighWord");
    H5Aread(hdf5_attribute, H5T_NATIVE_UINT, header.npartTotalHighWord);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "MassTable");
    H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, header.mass);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Time");
    H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &header.time);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "NumFilesPerSnapshot");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.num_files);
    H5Aclose(hdf5_attribute);

    hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Flag_DoublePrecision");
    H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.flag_doubleprecision);
    H5Aclose(hdf5_attribute);

#ifdef METALS /* some trickery here to enable snapshot-restarts from runs with different numbers of metal species */
    if(RestartFlag==2)
    {
        hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Flag_Metals");
        H5Aread(hdf5_attribute, H5T_NATIVE_INT, &header.flag_metals);
        H5Aclose(hdf5_attribute);
    }
#endif
    
    /* things that are not part of the header 'structure' we define in-code, but used in the hdf5 headers and wanted for read here, can be read below */
    if(H5Aexists(hdf5_headergrp, "Minimum_Mass_For_Cell_Merge")) { /* test for existence of this field */
        hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Minimum_Mass_For_Cell_Merge"); /* open it */
        H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &All.MinMassForParticleMerger); H5Aclose(hdf5_attribute);} /* read it and close */
    
    if(H5Aexists(hdf5_headergrp, "Maximum_Mass_For_Cell_Split")) { /* test for existence of this field */
        hdf5_attribute = H5Aopen_name(hdf5_headergrp, "Maximum_Mass_For_Cell_Split"); /* open it */
        H5Aread(hdf5_attribute, H5T_NATIVE_DOUBLE, &All.MaxMassForParticleSplit); H5Aclose(hdf5_attribute);} /* read it and close */

    H5Gclose(hdf5_headergrp);
    H5Fclose(hdf5_file);
#if H5_VERSION_GE(1,8,0) && !defined(H5_USE_16_API)
    H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
#else
    H5Eset_auto(old_func, old_client_data);
#endif
}







/*---------------------- Routine find a block in a snapfile -------------------*/
/*-------- FILE *fd:      File handle -----------------------------------------*/
/*-------- char *label:   4 byte identifyer for block -------------------------*/
/*-------- returns length of block found, -------------------------------------*/
/*-------- the file fd points to starting point of block ----------------------*/
/*-----------------------------------------------------------------------------*/
/*! Returns 0 if the labelled block is located; nonzero IC-error code otherwise. The caller
 *  (read_file, readTask only) turns a nonzero return into a soft bad-stop and keeps the block
 *  loop running so peers are not stranded -- drains at the per-turn read_ic poll. */
int find_block(char *label, FILE * fd)
{
    unsigned int blocksize = 0, blksize;
    char blocklabel[5] = { "    " };

#define FBSKIP  {my_fread(&blksize,sizeof(int),1,fd);}

    rewind(fd);

    while(!feof(fd) && blocksize == 0)
    {
        FBSKIP;
        if(blksize != 8)
        {
            printf("Incorrect Format (blksize=%llu)!\n", (unsigned long long)blksize);
            fflush(stdout);
            return 1891;   /* soft: incorrect block format */
        }
        else
        {
            my_fread(blocklabel, 4 * sizeof(char), 1, fd);
            my_fread(&blocksize, sizeof(int), 1, fd);
            /*
             printf("Searching <%c%c%c%c>, found Block <%s> with %d bytes\n",
             label[0],label[1],label[2],label[3],blocklabel,blocksize);
             */
            FBSKIP;
            if(strncmp(label, blocklabel, 4) != 0)
            {
                fseek(fd, blocksize, 1);
                blocksize = 0;
            }
        }
    }
    if(feof(fd))
    {
        printf("Block '%c%c%c%c' not found !\n", label[0], label[1], label[2], label[3]);
        fflush(stdout);
        return 1890;   /* soft: labelled block not found */
    }
    return 0;   /* block located */
}
