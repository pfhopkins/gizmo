#ifndef RESOLVEDISM_SELFCHECK_H
#define RESOLVEDISM_SELFCHECK_H
/* Runtime invariant checks for the test harness; compiled only with STELLA_SELFCHECK.
 * Log format (parsed by stella/tests/harness): "STELLA_SELFCHECK <tag> RAN|FAIL ..." */
#include <stdio.h>

#ifdef STELLA_SELFCHECK
#define STELLA_CHECK(tag, ok, fmt, ...) do { \
    if(!(ok)) { printf("STELLA_SELFCHECK %s FAIL Task=%d " fmt "\n", tag, ThisTask, ##__VA_ARGS__); fflush(stdout); } \
} while(0)
/* per call site: prints the first and every 1000th call, n = cumulative count */
#define STELLA_CHECK_RAN(tag, n) do { \
    static long stella_ran_calls_ = 0, stella_ran_n_ = 0; \
    if((n) > 0) { stella_ran_n_ += (long)(n); \
        if(++stella_ran_calls_ == 1 || stella_ran_calls_ % 1000 == 0) { printf("STELLA_SELFCHECK %s RAN Task=%d n=%ld\n", tag, ThisTask, stella_ran_n_); fflush(stdout); } } \
} while(0)
#else
#define STELLA_CHECK(tag, ok, fmt, ...) do {} while(0)
#define STELLA_CHECK_RAN(tag, n) do {} while(0)
#endif

#endif
