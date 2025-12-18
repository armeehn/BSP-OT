#pragma once

#include "BSPOT.h"
#include <mutex>
#include <string>

namespace BSPOT::telemetry {

struct Snapshot {
    double wall_seconds = 0.0;
    double cpu_seconds = 0.0;
    int cpu_threads = 0;

    bool gpu_enabled = false;
    std::string gpu_name;
    std::size_t gpu_mem_total = 0;
    std::size_t gpu_mem_free = 0;

    std::size_t points_projected = 0;
    std::size_t projection_calls = 0;
    std::size_t h2d_bytes = 0;
    std::size_t d2h_bytes = 0;
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
};

class Recorder {
public:
    Recorder();
    ~Recorder() = default;

    void start();
    void stop();

    void addCPUProjection(std::size_t points, double ms);
    void addGPUProjection(std::size_t points, std::size_t h2d, std::size_t d2h, double kernel_ms);

    void setGPUInfo(const std::string& name, std::size_t total, std::size_t free);

    Snapshot snapshot() const;
    std::string to_json() const;

private:
    TimeStamp wall_start{};
    TimeStamp cpu_start{};
    mutable std::mutex mtx;
    Snapshot snap;
};

class ScopedRecorder {
public:
    explicit ScopedRecorder(Recorder* r);
    ~ScopedRecorder();
};

Recorder* current();

} // namespace BSPOT::telemetry
