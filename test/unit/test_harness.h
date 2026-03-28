#pragma once
// Minimal C++ unit test harness. No external dependencies.
//
// Usage:
//   #include "test_harness.h"
//   TEST_CASE("descriptive name") {
//       CHECK(1 + 1 == 2);
//       CHECK_CLOSE(3.14, 3.14159, 0.01);
//   }
//   TEST_MAIN()  // expands to main(), runs all registered tests
//
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <functional>

struct TestCase {
    const char* name;
    std::function<void()> func;
};

inline std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> cases;
    return cases;
}

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> func) {
        test_registry().push_back({name, std::move(func)});
    }
};

inline int& check_fail_count() { static int n = 0; return n; }
inline int& check_pass_count() { static int n = 0; return n; }

#define TEST_PASTE_(a, b) a##b
#define TEST_PASTE(a, b) TEST_PASTE_(a, b)
#define TEST_CASE_IMPL(id, name) \
    static void TEST_PASTE(test_func_, id)(); \
    static TestRegistrar TEST_PASTE(test_reg_, id)(name, TEST_PASTE(test_func_, id)); \
    static void TEST_PASTE(test_func_, id)()
#define TEST_CASE(name) TEST_CASE_IMPL(__COUNTER__, name)

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        check_fail_count()++; \
    } else { \
        check_pass_count()++; \
    } \
} while(0)

#define CHECK_CLOSE(a, b, tol) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (std::fabs(_a - _b) > _t) { \
        std::fprintf(stderr, "  FAIL: %s:%d: %s ~= %s (%.15g != %.15g, tol=%.3g)\n", \
                     __FILE__, __LINE__, #a, #b, _a, _b, _t); \
        check_fail_count()++; \
    } else { \
        check_pass_count()++; \
    } \
} while(0)

#define TEST_MAIN() \
int main() { \
    int total_failures = 0; \
    for (auto& tc : test_registry()) { \
        check_fail_count() = 0; \
        check_pass_count() = 0; \
        tc.func(); \
        int f = check_fail_count(), p = check_pass_count(); \
        if (f > 0) { \
            std::fprintf(stderr, "FAIL: %s (%d/%d checks passed)\n", tc.name, p, p+f); \
            total_failures += f; \
        } else { \
            std::fprintf(stdout, "  ok: %s (%d checks)\n", tc.name, p); \
        } \
    } \
    if (total_failures > 0) { \
        std::fprintf(stderr, "\n%d check(s) FAILED\n", total_failures); \
        return 1; \
    } \
    std::fprintf(stdout, "\nAll %zu test(s) passed.\n", test_registry().size()); \
    return 0; \
}
