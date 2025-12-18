#include "telemetry.h"

#include <atomic>
#include <sstream>
#include <thread>

namespace BSPOT::telemetry {

namespace {
thread_local Recorder* g_current = nullptr;
std::atomic<Recorder*> g_shared{nullptr};
}

Recorder::Recorder() {
    snap.cpu_threads = std::max(1u, std::thread::hardware_concurrency());
}

void Recorder::start() {
    wall_start = Time::now();
    cpu_start = wall_start;
}

void Recorder::stop() {
    const auto now = Time::now();
    std::lock_guard<std::mutex> lk(mtx);
    snap.wall_seconds = TimeBetween(wall_start, now);
    snap.cpu_seconds = TimeBetween(cpu_start, now);
}

void Recorder::addCPUProjection(std::size_t points, double ms) {
    std::lock_guard<std::mutex> lk(mtx);
    snap.points_projected += points;
    snap.projection_calls += 1;
    snap.cpu_ms += ms;
}

void Recorder::addGPUProjection(std::size_t points, std::size_t h2d, std::size_t d2h, double kernel_ms) {
    std::lock_guard<std::mutex> lk(mtx);
    snap.gpu_enabled = true;
    snap.points_projected += points;
    snap.projection_calls += 1;
    snap.h2d_bytes += h2d;
    snap.d2h_bytes += d2h;
    snap.gpu_ms += kernel_ms;
}

void Recorder::setGPUInfo(const std::string& name, std::size_t total, std::size_t free) {
    std::lock_guard<std::mutex> lk(mtx);
    snap.gpu_enabled = true;
    snap.gpu_name = name;
    snap.gpu_mem_total = total;
    snap.gpu_mem_free = free;
}

Snapshot Recorder::snapshot() const {
    std::lock_guard<std::mutex> lk(mtx);
    return snap;
}

std::string Recorder::to_json() const {
    const auto s = snapshot();
    std::ostringstream oss;
    oss << "{";
    oss << "\"wall_seconds\":" << s.wall_seconds << ",";
    oss << "\"cpu_seconds\":" << s.cpu_seconds << ",";
    oss << "\"cpu_threads\":" << s.cpu_threads << ",";
    oss << "\"gpu_enabled\":" << (s.gpu_enabled ? "true" : "false") << ",";
    oss << "\"gpu_name\":\"" << s.gpu_name << "\",";
    oss << "\"gpu_mem_total\":" << s.gpu_mem_total << ",";
    oss << "\"gpu_mem_free\":" << s.gpu_mem_free << ",";
    oss << "\"points_projected\":" << s.points_projected << ",";
    oss << "\"projection_calls\":" << s.projection_calls << ",";
    oss << "\"h2d_bytes\":" << s.h2d_bytes << ",";
    oss << "\"d2h_bytes\":" << s.d2h_bytes << ",";
    oss << "\"gpu_ms\":" << s.gpu_ms << ",";
    oss << "\"cpu_ms\":" << s.cpu_ms;
    oss << "}";
    return oss.str();
}

ScopedRecorder::ScopedRecorder(Recorder* r) {
    g_current = r;
    g_shared.store(r, std::memory_order_release);
}

ScopedRecorder::~ScopedRecorder() {
    g_current = nullptr;
    g_shared.store(nullptr, std::memory_order_release);
}

Recorder* current() {
    if (g_current) return g_current;
    return g_shared.load(std::memory_order_acquire);
}

} // namespace BSPOT::telemetry
