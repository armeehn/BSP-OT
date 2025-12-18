#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace BSPOT::monitor {

struct Sample {
    double timestamp_s = 0.0;
    double cpu_percent = 0.0;

    std::uint64_t mem_total_kb = 0;
    std::uint64_t mem_available_kb = 0;
    std::uint64_t mem_used_kb = 0;

    bool gpu_available = false;
    std::string gpu_name;
    std::uint64_t gpu_mem_total_b = 0;
    std::uint64_t gpu_mem_used_b = 0;
    std::uint64_t gpu_mem_free_b = 0;
    unsigned gpu_util_percent = 0;
    unsigned gpu_mem_util_percent = 0;
};

class ResourceMonitor {
public:
    explicit ResourceMonitor(int gpu_index = 0);
    ~ResourceMonitor();

    void start(double interval_seconds = 0.5);
    void stop();
    bool running() const;

    Sample latest() const;

private:
    void loop(double interval_seconds);

    struct CpuTimes {
        std::uint64_t idle = 0;
        std::uint64_t total = 0;
    };

    static CpuTimes readCpuTimes();
    static bool readMemInfo(std::uint64_t& total_kb, std::uint64_t& available_kb);

    struct Nvml;
    Nvml* nvml = nullptr;

    std::atomic<bool> is_running{false};
    std::thread worker;

    mutable std::mutex mtx;
    Sample last;
};

} // namespace BSPOT::monitor
