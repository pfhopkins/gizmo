# Standalone checks of the jaco-generated sources outside GIZMO: regenerate the code for MODEL into
# build/, compile the EOS test against it (linking the RHS/Jacobian, the outputs and the table loader too, so
# they must build), and run it.
#
#   make -C test/jaco_CIE                    # needs the jaco package importable by $(PYTHON)
#
# HDF5 is found via HDF5_BASE (set by `module load hdf5` on FI systems) or the default search paths.

PYTHON ?= python3
MODEL  ?= starforge
BUILD  := build
GEN    := $(BUILD)/microphysics_func_jac.cc $(BUILD)/microphysics_outputs.cc $(BUILD)/jaco_eos.cc $(BUILD)/jaco_util.cc
CXXFLAGS ?= -O2 -g -Wall
CPPFLAGS += -I$(BUILD) $(if $(HDF5_BASE),-I$(HDF5_BASE)/include)
LDLIBS   += $(if $(HDF5_BASE),-L$(HDF5_BASE)/lib -Xlinker -rpath -Xlinker $(HDF5_BASE)/lib) -lhdf5

.PHONY: test codegen clean
test: $(BUILD)/test_jaco_eos
	./$(BUILD)/test_jaco_eos

# Always rerun: the codegen only rewrites files whose content changed, so a stale build/ cannot hide
# a model change.
codegen:
	$(PYTHON) -m jaco.codegen.gizmo.gizmo $(MODEL) --language c --ext .cc --output-dir $(BUILD)
$(GEN): codegen ;

$(BUILD)/test_jaco_eos: test_jaco_eos.cc jaco_test_stubs.cc $(GEN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -rf $(BUILD)
