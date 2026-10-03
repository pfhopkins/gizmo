# Standalone test of cooling/jaco_solver.cc outside GIZMO: regenerate the jaco code for MODEL into
# build/MODEL/, compile the solver against it, replay known failure cases, sweep a (n, T0, dt, pdv, u0)
# grid, and check every answer independently. Exits non-zero on any failure.
#
#   make -C test/jaco_solver [MODEL=starforge_legacy]   # needs the jaco package importable by $(PYTHON)
#   make -C test/jaco_solver MODEL=starforge_legacy TD=T,H+,H_2   # override the model's time-dependent set
#   make -C test/jaco_solver MODEL=wind_comparison   # models outside the STARFORGE family: test_jaco_solver_generic.cc
#   make -C test/jaco_solver compare [REF=gizmo_jaco_dev]   # the solver at git revision REF (default HEAD)
#                       # against the working tree's on the same sweep: tiers, nfeval, answers
#   make -C test/jaco_solver OPTFLAGS="-O3 -ffast-math -funroll-loops -march=native"   # GIZMO's flags
#
# OPTFLAGS apply to the solver and the generated code (which also gets -fno-associative-math, as in
# GIZMO's Makefile) and to the link (-ffast-math links crtfastmath, as GIZMO's does), not to the checks.
# The reference solver must share the working tree's jaco_solver.h interface.
# The solver sources are copied into the build directory so that their #include "microphysics_func_jac.h"
# finds the freshly generated header, never a stale one left in cooling/ by a GIZMO build of another model.
# HDF5 is found via HDF5_BASE (set by `module load hdf5` on FI systems) or the default search paths.

PYTHON ?= python3
MODEL  ?= starforge
TD     ?=
REF    ?= HEAD
OPTFLAGS ?=
comma  := ,
BUILD  := build/$(MODEL)$(if $(TD),-td-$(subst +,p,$(subst $(comma),-,$(TD))))
DRIVER := $(if $(filter starforge%,$(MODEL)),test_jaco_solver.cc,test_jaco_solver_generic.cc)
GEN    := $(BUILD)/microphysics_func_jac.cc $(BUILD)/microphysics_outputs.cc $(BUILD)/jaco_eos.cc $(BUILD)/jaco_util.cc
SOLVER := $(BUILD)/jaco_solver.cc $(BUILD)/jaco_solver.h
CXXFLAGS ?= -O2 -g -Wall
CPPFLAGS += -I$(BUILD) $(if $(HDF5_BASE),-I$(HDF5_BASE)/include)
LDLIBS   += $(if $(HDF5_BASE),-L$(HDF5_BASE)/lib -Xlinker -rpath -Xlinker $(HDF5_BASE)/lib) -lhdf5

COMPILE  = $(CXX) $(CPPFLAGS) $(CXXFLAGS)
GEN_OBJ := $(GEN:.cc=.o)
REF_SRC := $(BUILD)/ref/jaco_solver.cc
REF_RENAME := -Djaco_solve=jaco_ref_solve -Djaco_solve_chemistry=jaco_ref_solve_chemistry \
	-Djaco_solver_default_settings=jaco_ref_solver_default_settings -Djaco_print_state=jaco_ref_print_state

.PHONY: test compare codegen clean FORCE
test: $(BUILD)/test_jaco_solver
	./$< $(BUILD)
compare: $(BUILD)/test_jaco_compare
	./$< $(BUILD) compare

# Always rerun: the codegen only rewrites files whose content changed, so a stale build/ cannot hide
# a model change.
codegen:
	$(PYTHON) -m jaco.codegen.gizmo.gizmo $(MODEL) --language c --ext .cc --output-dir $(BUILD) $(if $(TD),--time-dependent $(TD))
$(GEN) $(BUILD)/microphysics_func_jac.h: codegen ;

$(SOLVER): $(BUILD)/%: ../../cooling/% | codegen
	cp $< $@

# rewritten only when its content changes
$(REF_SRC): FORCE | codegen
	@mkdir -p $(dir $@)
	@git -C ../.. show $(REF):cooling/jaco_solver.cc > $@.new && { cmp -s $@.new $@ && rm $@.new || mv $@.new $@; }

# rewritten only when the flags change, so that changing OPTFLAGS rebuilds every object
$(BUILD)/flags: FORCE | codegen
	@echo '$(COMPILE) $(OPTFLAGS)' | cmp -s - $@ || echo '$(COMPILE) $(OPTFLAGS)' > $@

DEPS = $(BUILD)/flags $(BUILD)/microphysics_func_jac.h $(BUILD)/jaco_solver.h
$(BUILD)/microphysics_func_jac.o $(BUILD)/microphysics_outputs.o $(BUILD)/jaco_eos.o: $(BUILD)/%.o: $(BUILD)/%.cc $(DEPS)
	$(COMPILE) $(OPTFLAGS) -fno-associative-math -c $< -o $@
$(BUILD)/jaco_util.o $(BUILD)/jaco_solver.o: $(BUILD)/%.o: $(BUILD)/%.cc $(DEPS)
	$(COMPILE) $(OPTFLAGS) -c $< -o $@
$(BUILD)/jaco_solver_ref.o: $(REF_SRC) $(DEPS)
	$(COMPILE) $(OPTFLAGS) $(REF_RENAME) -c $< -o $@
$(BUILD)/main.o: $(DRIVER) $(DEPS)
	$(COMPILE) -c $< -o $@
$(BUILD)/main_compare.o: test_jaco_solver.cc $(DEPS)
	$(COMPILE) -DJACO_COMPARE -c $< -o $@

$(BUILD)/test_jaco_solver: $(BUILD)/main.o $(BUILD)/jaco_solver.o $(GEN_OBJ)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $^ -o $@ $(LDLIBS)
$(BUILD)/test_jaco_compare: $(BUILD)/main_compare.o $(BUILD)/jaco_solver.o $(BUILD)/jaco_solver_ref.o $(GEN_OBJ)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -rf build
