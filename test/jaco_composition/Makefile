# Standalone check of cooling/jaco_composition.h; exits non-zero on failure.
#   make -C test/jaco_composition
CXXFLAGS ?= -O2 -g -Wall

.PHONY: test clean
test: test_jaco_composition
	./test_jaco_composition

test_jaco_composition: test_jaco_composition.cc ../../cooling/jaco_composition.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f test_jaco_composition
