# Standalone test of cooling/jaco_solver.cc outside GIZMO: regenerate the jaco code for MODEL into
# build/, compile the solver against it, replay known failure cases, sweep a (n, T0, dt, pdv, u0)
# grid, and check every answer independently. Exits non-zero on any failure.
#
#   make -C test/jaco_solver               # needs the jaco package importable by $(PYTHON)
#
# The solver sources are copied into build/ so that their #include "microphysics_func_jac.h" finds
# the freshly generated header, never a stale one left in cooling/ by a GIZMO build of another model.
# HDF5 is found via HDF5_BASE (set by `module load hdf5` on FI systems) or the default search paths.

PYTHON ?= python3
MODEL  ?= starforge
BUILD  := build
GEN    := $(BUILD)/microphysics_func_jac.cc $(BUILD)/jaco_eos.cc $(BUILD)/jaco_util.cc
SOLVER := $(BUILD)/jaco_solver.cc $(BUILD)/jaco_solver.h
CXXFLAGS ?= -O2 -g -Wall
CPPFLAGS += -I$(BUILD) $(if $(HDF5_BASE),-I$(HDF5_BASE)/include)
LDLIBS   += $(if $(HDF5_BASE),-L$(HDF5_BASE)/lib -Xlinker -rpath -Xlinker $(HDF5_BASE)/lib) -lhdf5

.PHONY: test codegen clean
test: $(BUILD)/test_jaco_solver
	./$(BUILD)/test_jaco_solver $(BUILD)

# Always rerun: the codegen only rewrites files whose content changed, so a stale build/ cannot hide
# a model change.
codegen:
	$(PYTHON) -m jaco.codegen.gizmo.gizmo $(MODEL) --language c --ext .cc --output-dir $(BUILD)
$(GEN): codegen ;

$(BUILD)/jaco_solver.%: ../../cooling/jaco_solver.% | codegen
	cp $< $@

$(BUILD)/test_jaco_solver: test_jaco_solver.cc $(SOLVER) $(GEN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) test_jaco_solver.cc $(BUILD)/jaco_solver.cc $(GEN) -o $@ $(LDLIBS)

clean:
	rm -rf $(BUILD)
