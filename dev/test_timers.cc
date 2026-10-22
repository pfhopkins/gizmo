/* Standalone behavioral checks; see Makefile.timers and docs/timers.md. */
#include "../src/system/timer.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

using namespace gizmo::timing;

namespace {
std::atomic<int> pushes{0}, pops{0};
void push(const char *) {++pushes;}
void pop() {++pops;}

void work()
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
    while(std::chrono::steady_clock::now() < end) {std::atomic_signal_fence(std::memory_order_seq_cst);}
}

TimerRecord get(const char *path, std::size_t thread = 0)
{
    for(const auto &record : TimerSnapshot()) {
        if(record.path == path && record.thread == thread) {return record;}
    }
    std::fprintf(stderr, "Missing timer: %s on thread %zu\n", path, thread);
    std::abort();
}

void early_return()
{
    GIZMO_TIMER_SCOPE("Return");
    return;
}

void recurse(int n)
{
    GIZMO_TIMER_SCOPE("Recurse");
    if(n) {recurse(n - 1);}
}
} // namespace

int main(int argc, char **argv)
{
    RegionHooks hooks;
    hooks.push = push; hooks.pop = pop;
    hooks.main_thread_only = argc > 1 && !std::strcmp(argv[1], "--main-hooks");
    const bool disabled = argc > 1 && !std::strcmp(argv[1], "--disabled");
    TimerInitialize(!disabled, hooks);
    if(disabled) {
        assert(!TimerEnabled());
        assert(TimerStart(nullptr) == 0);
        assert(TimerStop("never-started"));
        early_return();
        std::thread worker([] {GIZMO_TIMER_SCOPE("Worker");}); worker.join();
        assert(TimerSnapshot().empty());
        assert(pushes == 0 && pops == 0 && TimerErrorCount() == 0);
        TimerShutdown();
        std::puts("PASS: disabled timers have no records, hooks or errors");
        return 0;
    }

    // Parent = self + direct children, including repeated calls and live spans.
    TimerStart("Tree");
    work();
    for(int i = 0; i < 3; ++i) {
        TimerStart("Tree::TreeSolve");
        work();
        {GIZMO_TIMER_SCOPE("small_function"); work();}
        assert(TimerStop("TreeSolve"));
    }
    const auto live = get("Tree");
    const auto child = get("Tree::TreeSolve");
    assert(live.active == 1 && live.completed == 0 && live.calls == 1);
    assert(live.inclusive >= child.inclusive);
    assert(std::abs(live.inclusive - live.exclusive - child.inclusive) < 1e-8);
    const auto next_live = get("Tree");
    assert(next_live.calls == 1 && next_live.active == 1);
    assert(next_live.inclusive >= live.inclusive); // snapshots do not reset totals
    assert(TimerStop("Tree"));
    const auto closed = get("Tree");
    assert(closed.completed == 1 && closed.active == 0);
    assert(std::abs(closed.inclusive - closed.exclusive - child.inclusive) < 1e-8);
    assert(child.calls == 3 && child.completed == 3);
    assert(get("Tree::TreeSolve::small_function").calls == 3);

    // Live nested children must be subtracted once when taking a snapshot.
    TimerStart("Live");
    TimerStart("Child");
    work();
    const auto snapshot = TimerSnapshot();
    TimerRecord parent_live, child_live;
    for(const auto &record : snapshot) {
        if(record.path == "Live") {parent_live = record;}
        if(record.path == "Live::Child") {child_live = record;}
    }
    assert(std::abs(parent_live.inclusive - parent_live.exclusive - child_live.inclusive) < 1e-8);
    assert(TimerStop()); assert(TimerStop());

    // A shared call site acquires the current parent on every entry.
    for(const char *parent : {"First", "Second"}) {
        GIZMO_TIMER_SCOPE(parent);
        early_return();
    }
    assert(get("First::Return").completed == 1);
    assert(get("Second::Return").completed == 1);
    try {GIZMO_TIMER_SCOPE("Exception"); throw std::runtime_error("test");}
    catch(const std::runtime_error &) {}
    assert(get("Exception").completed == 1);
    recurse(2);
    assert(get("Recurse::Recurse::Recurse").completed == 1);

    // Invalid stops/qualified starts preserve the active stack.
    auto errors = TimerErrorCount();
    TimerStart("Checked");
    assert(!TimerStop("Wrong"));
    assert(TimerStart("Other::Child") == 0);
    assert(TimerStart("Checked::Missing::Child") == 0);
    assert(TimerStart("bad\tname") == 0);
    assert(TimerStart("bad:name") == 0);
    assert(TimerStop("Checked"));
    assert(!TimerStop());
    assert(TimerErrorCount() == errors + 6);

    // Worker stacks are independent; statistics survive thread exit. Snapshot
    // concurrently with updates to exercise the synchronization contract.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    const int pushes_before_workers = pushes.load();
    std::vector<std::thread> workers;
    for(int t = 0; t < 4; ++t) {
        workers.emplace_back([&] {
            ++ready;
            while(!go.load()) {std::this_thread::yield();}
            for(int i = 0; i < 100; ++i) {
                GIZMO_TIMER_SCOPE("Worker");
                GIZMO_TIMER_SCOPE("Child");
            }
        });
    }
    while(ready != 4) {std::this_thread::yield();}
    go = true;
    for(int i = 0; i < 100; ++i) {
        for(const auto &record : TimerSnapshot()) {
            assert(record.completed + record.active == record.calls);
            assert(record.exclusive >= 0 && record.exclusive <= record.inclusive);
        }
    }
    for(auto &worker : workers) {worker.join();}
    if(hooks.main_thread_only) {assert(pushes == pushes_before_workers);}
    int worker_roots = 0;
    for(const auto &record : TimerSnapshot()) {
        if(record.path == "Worker") {
            assert(record.thread != 0 && record.completed == 100 && record.active == 0);
            ++worker_roots;
        }
    }
    assert(worker_roots == 4);

    // A scope cannot accidentally pop a matching numeric token on another thread.
    auto *foreign_scope = new TimerScope("Owner");
    errors = TimerErrorCount();
    std::thread wrong_owner([&] {GIZMO_TIMER_SCOPE("DifferentThread"); delete foreign_scope;});
    wrong_owner.join();
    assert(TimerErrorCount() == errors + 1);
    assert(TimerStop("Owner"));

    // Controlled shutdown closes live regions, balances hooks, and makes a
    // later automatic destructor harmless after its runtime has been finalized.
    {
        GIZMO_TIMER_SCOPE("Shutdown");
        TimerStart("UnwoundAtShutdown");
        TimerShutdown();
        assert(!TimerEnabled());
        assert(get("Shutdown").active == 0);
    }
    assert(pushes == pops);
    std::puts("PASS: nesting, live snapshots, context, recursion, unwind, validation, threads, hooks and shutdown");
}
