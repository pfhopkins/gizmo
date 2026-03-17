
#-----------------------------------------------------------------
#
# You might be looking for the compile-time Makefile options of the code...
#
# They have moved to a separate file.
#
# To build the code, do the following:
#
#  (1) Copy the file "Template-Config.sh"  to  "Config.sh"
#
#        cp Template-Config.sh Config.sh 
#
#  (2) Edit "Config.sh" as needed for your application
#
#  (3) Run "make"
#
#
#  New compile-time options should be added to the 
#  file "Template-Config.sh" only. Usually, the should be added
#  there in the disabled/default version.
#
#  "Config.sh" should *not* be checked in to the repository
#
#  Note: It is possible to override the default name of the 
#  Config.sh file, if desired, as well as the name of the
#  executable. For example:
#
#   make  CONFIG=MyNewConf.sh  EXEC=GIZMO
# 
#-----------------------------------------------------------------
#
# You might also be looking for the target system SYSTYPE option
#
# It has also moved to a separate file.
#
# To build the code, do the following:
#
# (A) set the SYSTYPE variable in your .bashrc (or similar file):
#
#        e.g. export SYSTYPE=Magny
# or
#
# (B) set SYSTYPE in Makefile.systype 
#     This file has priority over your shell variable.:
#
#     Uncomment your system in  "Makefile.systype".
#
# If you add an ifeq for a new system below, also add that systype to
# Template-Makefile.systype
#
###########
#
# This file was originally part of the GADGET3 code developed by
#   Volker Springel. The code has been modified
#   substantially by Phil Hopkins (phopkins@caltech.edu) for GIZMO
#   (dealing with new files and filename conventions, libraries, parser, logic)
#
#############

CONFIG   =  Config.sh
PERL     =  /usr/bin/perl

RESULT     := $(shell CONFIG=$(CONFIG) PERL=$(PERL) make -f config-makefile)
CONFIGVARS := $(shell cat GIZMO_config.h)

HG_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null)
HG_REPO := $(shell git config --get remote.origin.url)
HG_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
BUILDINFO = "Build on $(HOSTNAME) by $(USER) from $(HG_BRANCH):$(HG_COMMIT) at $(HG_REPO)"
OPT += -DBUILDINFO='$(BUILDINFO)'


# initialize some default flags -- these will all get re-written below
CC	= mpicc		# sets the C-compiler (default, will be set for machine below)
CXX	= mpiCC		# sets the C++-compiler (default, will be set for machine below)
FC	= mpif90	# sets the fortran compiler (default, will be set for machine below)
OPTIMIZE = -Wall  -g   # optimization and warning flags (default)
MPICHLIB = -lmpich	# mpi library (arbitrary default, set for machine below)
CHIMESINCL = # default to empty, will only be used below if called
CHIMESLIBS = # default to empty, will only be used below if called




## read the systype information to use the blocks below for different machines
ifdef SYSTYPE
SYSTYPE := "$(SYSTYPE)"
-include Makefile.systype
else
include Makefile.systype
endif

ifeq ($(wildcard Makefile.systype), Makefile.systype)
INCL = Makefile.systype
else
INCL =
endif
FINCL =



#----------------------------------------------------------------------------------------------
ifeq ($(SYSTYPE),"Frontera")
CC       =  mpicc
CXX      =  mpicxx -std=c++11
FC       =  mpif90 -nofor_main
OPTIMIZE = -ggdb -O2 -xCORE-AVX2 -Wno-unknown-pragmas -Wall -Wno-format-security -qopenmp
ifeq (CHIMES,$(findstring CHIMES,$(CONFIGVARS)))
CHIMESINCL = -I$(TACC_SUNDIALS_INC)
CHIMESLIBS = -L$(TACC_SUNDIALS_LIB) -lsundials_cvode -lsundials_nvecserial
endif
MKL_INCL = -I$(TACC_MKL_INC)
MKL_LIBS = -L$(TACC_MKL_LIB) -mkl=sequential
GSL_INCL = -I$(TACC_GSL_INC)
GSL_LIBS = -L$(TACC_GSL_LIB) -lgsl -lgslcblas
FFTW_INCL= -I$(TACC_FFTW3_INC)
FFTW_LIBS= -L$(TACC_FFTW3_LIB)
HDF5INCL = -I$(TACC_HDF5_INC) -DH5_USE_16_API
HDF5LIB  = -L$(TACC_HDF5_LIB) -lhdf5 -lz
MPICHLIB = #
OPT     += -DHDF5_DISABLE_VERSION_CHECK
## compiles with module set: intel/19 impi hdf5 fftw3 gsl valgrind python3
endif


#----------------------------------------------------------------------------------------------
ifeq ($(SYSTYPE),"CaltechHPC")
CC       =  mpicc
CXX      =  mpic++
FC       =  mpif90 -nofor_main
OPTIMIZE = -O2 -xCORE-AVX2
ifeq (OPENMP,$(findstring OPENMP,$(CONFIGVARS)))
OPTIMIZE += -qopenmp
endif
MKL_INCL = -I$(CPATH)
MKL_LIBS = -L$(LIBRARY_PATH) -mkl=sequential
GSL_INCL = -I$(CPATH)
GSL_LIBS = -L$(LIBRARY_PATH)
FFTW_INCL= -I$(CPATH)
FFTW_LIBS= -L$(LIBRARY_PATH)
HDF5INCL = -I$(CPATH) -DH5_USE_16_API
HDF5LIB  = -L$(LIBRARY_PATH) -lhdf5 -lz
MPICHLIB = #
OPT     += -DHDF5_DISABLE_VERSION_CHECK
# Compiles with following modules:   1) intel/20.1    2) hdf5/1.10.1   3) gsl/2.4       4) fftw/3.3.7
endif

#----------------------------------------------------------------------------------------------
ifeq ($(SYSTYPE),"Expanse")
CC       = mpicc
CXX      = mpicxx
FC       = $(CC)
OPTIMIZE = -Ofast
ifeq (OPENMP,$(findstring OPENMP,$(CONFIGVARS)))
OPTIMIZE += -qopenmp
endif
GSL_INCL = -I$(GSLHOME)/include
GSL_LIBS = -L$(GSLHOME)/lib
FFTW_INCL= -I$(FFTWHOME)/include
FFTW_LIBS= -L$(FFTWHOME)/lib
HDF5INCL = -I$(HDF5HOME)/include -DH5_USE_16_API
HDF5LIB  = -L$(HDF5HOME)/lib -lhdf5 -lz
MPICHLIB = #
OPT     += #
## modules to load
## module load slurm intel openmpi_ib fftw/2.1.5 gsl hdf5
## run job with
## mpirun -v -x LD_LIBRARY_PATH ./GIZMO params.txt
endif

#----------------------------------------------------------------------------------------------
ifeq ($(SYSTYPE),"MacBookCellar")
CC       =  mpicc
CXX      =  mpicxx -std=c++11
FC       =  $(CC) #mpifort  ## change this to "mpifort" for packages requiring linking secondary fortran code, currently -only- the helmholtz eos modules do this, so I leave it un-linked for now to save people the compiler headaches
OPTIMIZE = -O1 -funroll-loops
OPTIMIZE += -g -Wall # compiler warnings
ifeq (CHIMES,$(findstring CHIMES,$(CONFIGVARS)))
CXX     = mpic++
CHIMESINCL = -I/usr/local/include/sundials
CHIMESLIBS = -L/usr/local/lib -lsundials_cvode -lsundials_nvecserial
endif
MKL_INCL = #
MKL_LIBS = #
GSL_INCL = -I/opt/homebrew/Cellar/gsl/2.8/include #-I$(PORTINCLUDE)
GSL_LIBS = -L/opt/homebrew/Cellar/gsl/2.8/lib #-L$(PORTLIB)
FFTW_INCL= -I/usr/local/include
FFTW_LIBS= -L/usr/local/lib
HDF5INCL = -I/opt/homebrew/Cellar/hdf5/1.14.6/include -DH5_USE_16_API  #-I$(PORTINCLUDE) -DH5_USE_16_API
HDF5LIB  = -L/opt/homebrew/Cellar/hdf5/1.14.6/lib -lhdf5 -lz  #-L$(PORTLIB)
MPICHLIB = #
OPT     += -DDISABLE_ALIGNED_ALLOC -DCHIMES_USE_DOUBLE_PRECISION #
endif

#----------------------------
ifeq ($(SYSTYPE),"github-ubuntu")
CC       =  mpicc
CXX      =  mpicxx
FC       =  $(CC)
OPTIMIZE = -g -fcommon -O1 -funroll-loops -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -fipa-cp-clone  ## optimizations for gcc compilers (1/2)
OPTIMIZE += -ftree-loop-distribute-patterns -fvect-cost-model -ftree-partial-pre   ## optimizations for gcc compilers (2/2)
OPTIMIZE += -g -Wall # compiler warnings
ifeq (CHIMES,$(findstring CHIMES,$(CONFIGVARS)))
CXX     = mpic++
CHIMESINCL = -I/usr/include/sundials
CHIMESLIBS = -L/usr/lib -lsundials_cvode -lsundials_nvecserial
endif
ifeq (OPENMP,$(findstring OPENMP,$(CONFIGVARS)))
OPTIMIZE += -fopenmp # openmp required compiler flags
FC       = $(CC)
endif
MKL_INCL = #
MKL_LIBS = #
GSL_INCL = -I/usr/include
GSL_LIBS = -L/usr/lib 
FFTW_INCL= -I/usr/include
FFTW_LIBS= -L/usr/lib
HDF5INCL = -I/usr/include/hdf5/openmpi -DH5_USE_16_API
HDF5LIB  = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ -lhdf5 -lz
MPICHLIB = #
OPT     += -DDISABLE_ALIGNED_ALLOC -DCHIMES_USE_DOUBLE_PRECISION
## to get required packages: sudo apt install libhdf5-openmpi-dev libgsl-dev libopenmpi-dev
endif


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------


#
# different code groups that need to be compiled. the groupings below are
# arbitrary (they will all be added to OBJS and compiled, and if they are
# un-used it should be through use of macro flags in the source code). But
# they are grouped below for the sake of clarity when adding/removing code
# blocks in the future
#
CORE_OBJS =	core/main.o core/accel.o core/timestep.o core/init.o file_io/restart.o file_io/io.o \
			core/predict.o declarations/global.o core/begrun.o core/run.o declarations/allvars.o \
			file_io/read_ic.o domain/domain.o core/driftfac.o core/kicks.o mesh/ngb.o \
			compile_time_info.o mesh/merge_split.o

SYSTEM_OBJS =   system/system.o \
				system/allocate.o \
				system/mymalloc.o \
				system/parallel_sort.o \
                system/peano.o \
                system/parallel_sort_special.o \
                system/mpi_util.o \
                system/pinning.o

GRAVITY_OBJS  = gravity/forcetree.o \
                gravity/forcetree_update.o \
                gravity/gravtree.o \
				gravity/cosmology.o \
				gravity/potential.o \
				gravity/pm_periodic.o \
                gravity/pm_nonperiodic.o \
                gravity/longrange.o \
                gravity/ags_rkern.o \
                gravity/binary.o

HYDRO_OBJS = 	hydro/hydro_toplevel.o \
				hydro/density.o \
				hydro/gradients.o \
				turb/dynamic_diffusion.o \
				turb/dynamic_diffusion_velocities.o \
				turb/turb_driving.o \
				turb/turb_powerspectra.o

EOSCOOL_OBJS =  cooling/cooling.o \
				cooling/grackle.o \
				cooling/simple_chemistry.o \
				eos/eos.o \
				eos/hydrogen_molecule.o \
				eos/cosmic_ray_fluid/cosmic_ray_alfven.o \
				eos/cosmic_ray_fluid/cosmic_ray_utilities.o \
				solids/elastic_physics.o \
				solids/grain_physics.o \
				solids/ism_dust_chemistry.o

STARFORM_OBJS = galaxy_sf/sfr_eff.o \
                galaxy_sf/stellar_evolution.o \
                galaxy_sf/mechanical_fb.o \
                galaxy_sf/thermal_fb.o \
                galaxy_sf/radfb_local.o \
                galaxy_sf/dm_dispersion_rkern.o

SINK_OBJS = sinks/sink.o \
            sinks/sink_util.o \
            sinks/sink_environment.o \
            sinks/sink_feed.o \
            sinks/sink_swallow_and_kick.o

RHD_OBJS =  radiation/rt_utilities.o \
			radiation/rt_CGmethod.o \
			radiation/rt_source_injection.o \
			radiation/rt_chem.o \
			radiation/rt_dust_opacity.o 

FOF_OBJS =	structure/fof.o \
			structure/subfind/subfind.o \
			structure/subfind/subfind_vars.o \
			structure/subfind/subfind_collective.o \
			structure/subfind/subfind_serial.o \
			structure/subfind/subfind_so.o \
			structure/subfind/subfind_cont.o \
			structure/subfind/subfind_distribute.o \
			structure/subfind/subfind_findlinkngb.o \
			structure/subfind/subfind_nearesttwo.o \
			structure/subfind/subfind_loctree.o \
			structure/subfind/subfind_potential.o \
			structure/subfind/subfind_density.o \
			structure/twopoint.o \
			structure/lineofsight.o

MISC_OBJS = sidm/cbe_integrator.o \
			sidm/dm_fuzzy.o \
			sidm/sidm_core.o

## name of executable and optimizations
EXEC   = GIZMO
OPTIONS = $(OPTIMIZE) $(OPT)

## combine all the objects above
OBJS  = $(CORE_OBJS) $(SYSTEM_OBJS) $(GRAVITY_OBJS) $(HYDRO_OBJS) \
		$(EOSCOOL_OBJS) $(STARFORM_OBJS) $(SINK_OBJS) $(RHD_OBJS) \
		$(FOF_OBJS) $(MISC_OBJS)

## fortran recompiler block
FOPTIONS = $(OPTIMIZE) $(FOPT)
FOBJS =

## include files needed at compile time for the above objects
INCL    += 	declarations/allvars.h \
			core/proto.h \
			gravity/forcetree.h \
			gravity/myfftw3.h \
			domain/domain.h \
			system/myqsort.h \
			mesh/kernel.h \
			eos/eos.h \
			sinks/sink.h \
			structure/fof.h \
			structure/subfind/subfind.h \
			cooling/cooling.h \
			Makefile


## now we add special cases dependent on compiler flags. normally we would
##  include the files always, and simply use the in-file compiler variables
##  to determine whether certain code is compiled [this allows us to take
##  advantage of compiler logic, and makes it easier for the user to
##  always specify what they want]. However special cases can arise, if e.g.
##  there are certain special libraries needed, or external compilers, for
##  certain features

# helmholtz eos routines need special treatment here because they are written
#  in fortran and call the additional fortran compilers and linkers. these could
#  be written to always compile and just be ignored, but then the large majority
#  of cases that -don't- need the fortran linker would always have to go
#  through these additional compilation options and steps (and this 
#  can cause additional problems on some machines). so we sandbox it here.
ifeq (EOS_HELMHOLTZ,$(findstring EOS_HELMHOLTZ,$(CONFIGVARS)))
OBJS    += eos/eos_interface.o
INCL    += eos/helmholtz/helm_wrap.h
FOBJS   += eos/helmholtz/helm_impl.o eos/helmholtz/helm_wrap.o
FINCL   += eos/helmholtz/helm_const.dek eos/helmholtz/helm_implno.dek eos/helmholtz/helm_table_storage.dek eos/helmholtz/helm_vector_eos.dek
endif

# chimes files are treated as special for now because they require special external libraries (e.g. sundials) that are otherwise not
#   used anywhere else in the code, and have not had their macro logic cleaned up to allow appropriate compilation without chimes flags enabled
ifeq (CHIMES,$(findstring CHIMES,$(CONFIGVARS)))
OBJS    += cooling/chimes/chimes.o cooling/chimes/chimes_cooling.o cooling/chimes/init_chimes.o cooling/chimes/rate_equations.o cooling/chimes/update_rates.o 
INCL    += cooling/chimes/chimes_interpol.h cooling/chimes/chimes_proto.h cooling/chimes/chimes_vars.h 
endif

# if grackle libraries are installed they must be a shared library as defined here
ifeq (COOL_GRACKLE,$(findstring COOL_GRACKLE,$(CONFIGVARS)))
OPTIONS += -DCONFIG_BFLOAT_8
GRACKLEINCL =
GRACKLELIBS = -lgrackle
else
GRACKLEINCL =
GRACKLELIBS =
endif

# linking libraries (includes machine-dependent options above)
CFLAGS = $(OPTIONS) $(GSL_INCL) $(FFTW_INCL) $(HDF5INCL) \
         $(GRACKLEINCL) $(CHIMESINCL)



# one annoying thing here is the FFTW libraries, since they are named differently depending on
#  whether they are compiled in different precision levels, or with different parallelization options, so we
#  have to have a big block here 'sorting them out'.
#
fftw_on_key = # default to 'off'
ifeq (PMGRID,$(findstring PMGRID, $(CONFIGVARS)))
  fftw_on_key = yes # needed for this module
endif
ifeq (TURB_DRIVING_SPECTRUMGRID,$(findstring TURB_DRIVING_SPECTRUMGRID, $(CONFIGVARS)))
  fftw_on_key = yes  # needed for this module
endif
FFTW_LIBNAMES = # default to 'off'
ifdef fftw_on_key
ifeq (DOUBLEPRECISION_FFTW,$(findstring DOUBLEPRECISION_FFTW,$(CONFIGVARS)))  # test for double precision libraries
  FFTW_LIBNAMES = -lfftw3_mpi -lfftw3 # double-precision libraries
else
  FFTW_LIBNAMES = -lfftw3f_mpi -lfftw3f # single-precision libraries
endif
endif


LIBS = $(HDF5LIB) -g $(MPICHLIB) $(GSL_LIBS) -lgsl -lgslcblas \
	   $(FFTW_LIBS) $(FFTW_LIBNAMES) -lm $(GRACKLELIBS) $(CHIMESLIBS)


$(EXEC): $(OBJS)
	$(CXX) $(OPTIMIZE) $(OBJS) $(LIBS) -o $(EXEC)

$(OBJS): %.o: %.cc $(INCL) $(CONFIG) compile_time_info.cc
	$(CXX) $(CFLAGS) -c $< -o $@

#$(FOBJS): %.o: %.f90
#	$(FC) $(OPTIMIZE) -c $< -o $@

compile_time_info.cc: $(CONFIG)
	$(PERL) file_io/prepare-config.perl $(CONFIG)

clean:
	rm -f $(OBJS) $(FOBJS) $(EXEC) *.oo *.c~ compile_time_info.cc GIZMO_config.h


