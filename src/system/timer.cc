#include "timer.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>

namespace gizmo {
namespace timing {
namespace {
using Clock = std::chrono::steady_clock;

struct Node {
    std::string name, path;
    std::map<std::string, std::size_t, std::less<>> children;
    std::uint64_t calls = 0, completed = 0;
    double inclusive = 0, exclusive = 0;
};

struct Frame {
    std::size_t node;
    std::uint64_t token;
    Clock::time_point start;
    double children = 0;
};

struct ThreadState {
    std::mutex mutex;
    std::size_t index = 0;
    std::deque<Node> nodes{1};  // anonymous root; never reported
    std::vector<Frame> stack;
    std::uint64_t next_token = 1;
};

struct Registry {
    std::atomic<bool> enabled{false};
    std::atomic<std::uint64_t> errors{0};
    std::mutex mutex;
    std::vector<std::shared_ptr<ThreadState>> threads;
    RegionHooks hooks;
    bool initialized = false;
};

Registry &registry()
{
    static Registry r;
    return r;
}

thread_local std::shared_ptr<ThreadState> local_state;

void error(const char *message, const char *name = "") noexcept
{
    const auto count = registry().errors.fetch_add(1, std::memory_order_relaxed);
    if(count < 8) {
        std::fprintf(stderr, "GIZMO timer error: %s [%s]%s\n", message,
                     name ? name : "(null)", count == 7 ? " (further messages suppressed)" : "");
    }
}

ThreadState &thread_state()
{
    if(!local_state) {
        auto state = std::make_shared<ThreadState>();
        state->stack.reserve(32);
        auto &r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        state->index = r.threads.size();
        r.threads.push_back(state); // retain statistics after a worker exits
        local_state = std::move(state);
    }
    return *local_state;
}

double seconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double>(end - begin).count();
}

/* Caller holds the owning ThreadState's mutex. */
void close_top(ThreadState &state) noexcept
{
    const auto &frame = state.stack.back();
    const double elapsed = seconds(frame.start, Clock::now());
    auto &node = state.nodes[frame.node];
    node.inclusive += elapsed;
    const double self = elapsed - frame.children;
    node.exclusive += self > 0 ? self : 0;
    ++node.completed;
    state.stack.pop_back();
    if(!state.stack.empty()) {state.stack.back().children += elapsed;}
    const auto &hooks = registry().hooks;
    if(hooks.pop && (!hooks.main_thread_only || state.index == 0)) {hooks.pop();}
}

bool stop_token(std::uint64_t token, const void *owner) noexcept
{
    if(token == 0 || !TimerEnabled()) {return true;}
    if(local_state.get() != owner) {error("scope destroyed on another thread"); return false;}
    auto &state = *local_state;
    std::lock_guard<std::mutex> guard(state.mutex);
    if(state.stack.empty() || state.stack.back().token != token) {
        error("scope does not match the active timer");
        return false;
    }
    close_top(state);
    return true;
}
} // namespace

void TimerInitialize(bool enabled, RegionHooks hooks)
{
    auto &r = registry();
    if(r.initialized) {error("TimerInitialize must be called only once"); return;}
    r.initialized = true;
    if((hooks.push == nullptr) != (hooks.pop == nullptr)) {
        error("region hooks must supply both push and pop");
        hooks = RegionHooks();
    }
    r.hooks = hooks;
    if(enabled) {thread_state();} // reserve thread 0 for the initializing thread
    r.enabled.store(enabled, std::memory_order_release);
}

bool TimerEnabled() noexcept
{
    return registry().enabled.load(std::memory_order_acquire);
}

std::uint64_t TimerStart(const char *name)
{
    if(!TimerEnabled()) {return 0;}
    if(!name || !name[0] || std::strpbrk(name, "\t\r\n")) {
        error("timer names must be nonempty and contain no tabs or newlines", name);
        return 0;
    }
    auto &state = thread_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    const std::size_t parent = state.stack.empty() ? 0 : state.stack.back().node;
    const char *leaf = name;
    if(std::strstr(name, "::")) {
        const auto &prefix = state.nodes[parent].path;
        if(prefix.empty() || std::strncmp(name, prefix.c_str(), prefix.size()) != 0 ||
           std::strncmp(name + prefix.size(), "::", 2) != 0) {
            error("qualified name does not match the active parent", name);
            return 0;
        }
        leaf = name + prefix.size() + 2;
        if(!leaf[0] || std::strstr(leaf, "::")) {
            error("qualified name must name an immediate child", name);
            return 0;
        }
    }
    if(std::strchr(leaf, ':')) {
        error("colons are reserved for qualified path separators", name);
        return 0;
    }
    auto &children = state.nodes[parent].children;
    auto found = children.find(leaf); // transparent lookup: no temporary string
    std::size_t id;
    if(found == children.end()) {
        id = state.nodes.size();
        Node node;
        node.name = leaf;
        node.path = parent ? state.nodes[parent].path + "::" + leaf : leaf;
        state.nodes.push_back(std::move(node));
        children.emplace(leaf, id);
    } else {id = found->second;}
    // Registration and the external push are outside this region's own clock.
    // Their overhead remains in the enclosing host region, if any.
    if(state.stack.size() == state.stack.capacity()) {
        state.stack.reserve(state.stack.capacity() * 2);
    }
    const auto &hooks = registry().hooks;
    if(hooks.push && (!hooks.main_thread_only || state.index == 0)) {hooks.push(state.nodes[id].path.c_str());}
    const auto token = state.next_token++;
    state.stack.push_back(Frame{id, token, Clock::now(), 0});
    ++state.nodes[id].calls;
    return token;
}

bool TimerStop(const char *name) noexcept
{
    if(!TimerEnabled()) {return true;}
    if(!local_state) {error("stop without an active timer", name); return false;}
    auto &state = *local_state;
    std::lock_guard<std::mutex> guard(state.mutex);
    if(state.stack.empty()) {error("stop without an active timer", name); return false;}
    const auto &node = state.nodes[state.stack.back().node];
    if(name && node.name != name && node.path != name) {
        error("stop does not match the active timer; stack left unchanged", name);
        return false;
    }
    close_top(state);
    return true;
}

TimerScope::TimerScope(const char *name) : token_(TimerStart(name)), owner_(local_state.get()) {}
TimerScope::~TimerScope() noexcept {stop_token(token_, owner_);}

std::vector<TimerRecord> TimerSnapshot()
{
    std::vector<std::shared_ptr<ThreadState>> threads;
    {
        std::lock_guard<std::mutex> guard(registry().mutex);
        threads = registry().threads;
    }
    std::vector<TimerRecord> result;
    for(const auto &state : threads) {
        std::lock_guard<std::mutex> guard(state->mutex);
        const auto now = Clock::now();
        const auto base = result.size();
        for(std::size_t id = 1; id < state->nodes.size(); ++id) {
            const auto &node = state->nodes[id];
            TimerRecord record;
            record.thread = state->index;
            record.path = node.path;
            record.calls = node.calls;
            record.completed = node.completed;
            record.inclusive = node.inclusive;
            record.exclusive = node.exclusive;
            result.push_back(std::move(record));
        }
        // Include live calls without closing them or changing accumulated totals.
        for(std::size_t i = 0; i < state->stack.size(); ++i) {
            const auto &frame = state->stack[i];
            auto &record = result[base + frame.node - 1];
            const double elapsed = seconds(frame.start, now);
            const double live_child = i + 1 < state->stack.size() ?
                seconds(state->stack[i + 1].start, now) : 0;
            const double self = elapsed - frame.children - live_child;
            record.inclusive += elapsed;
            record.exclusive += self > 0 ? self : 0;
            ++record.active;
        }
    }
    return result;
}

std::uint64_t TimerErrorCount() noexcept
{
    return registry().errors.load(std::memory_order_relaxed);
}

void TimerShutdown() noexcept
{
    if(!TimerEnabled()) {return;}
    if(!local_state || local_state->index != 0) {
        error("TimerShutdown must run on the initializing thread");
        return;
    }
    if(local_state) {
        std::lock_guard<std::mutex> guard(local_state->mutex);
        while(!local_state->stack.empty()) {close_top(*local_state);}
    }
    registry().enabled.store(false, std::memory_order_release);
}

} // namespace timing
} // namespace gizmo
