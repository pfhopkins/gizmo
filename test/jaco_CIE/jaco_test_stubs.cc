/* Minimal stubs for standalone jaco unit tests outside of GIZMO */
#include <stdio.h>
#include <stdlib.h>

/* Global state stubs */
struct { double MinEgySpec; double MinGasTemp; int ComovingIntegrationOn; double Time; } All = {0, 1.0, 0, 0};
int ThisTask = 0;

/* endrun stub */
extern "C" void endrun(int code) {
    fprintf(stderr, "endrun(%d) called in unit test\n", code);
    exit(code);
}
