#include "resource_monitor.h"

#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sstream>

namespace BSPOT::monitor {

namespace {
double nowSeconds() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    const auto dt = std::chrono::duration<double>(clock::now() - t0);
    return dt.count();
}
} // namespace

struct ResourceMonitor::Nvml {
    using nvmlReturn_t = int;
    struct nvmlDevice_st;
    using nvmlDevice_t = nvmlDevice_st*;

    struct nvmlUtilization_t {
        unsigned int gpu;
        unsigned int memory;
    };

    struct nvmlMemory_t {
        unsigned long long total;
        unsigned long long free;
        unsigned long long used;
    };

    void* lib = nullptr;
    bool ok = false;
    int device_index = 0;
    nvmlDevice_t device = nullptr;
    std::string name;

    nvmlReturn_t (*nvmlInit_v2)() = nullptr;
    nvmlReturn_t (*nvmlShutdown)() = nullptr;
    nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*) = nullptr;

    explicit Nvml(int gpu_index) : device_index(gpu_index) {
        lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (!lib) return;

        nvmlInit_v2 = reinterpret_cast<nvmlReturn_t (*)()>(dlsym(lib, "nvmlInit_v2"));
        nvmlShutdown = reinterpret_cast<nvmlReturn_t (*)()>(dlsym(lib, "nvmlShutdown"));
        nvmlDeviceGetHandleByIndex_v2 =
            reinterpret_cast<nvmlReturn_t (*)(unsigned int, nvmlDevice_t*)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
        nvmlDeviceGetName =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int)>(dlsym(lib, "nvmlDeviceGetName"));
        nvmlDeviceGetUtilizationRates =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*)>(dlsym(lib, "nvmlDeviceGetUtilizationRates"));
        nvmlDeviceGetMemoryInfo =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*)>(dlsym(lib, "nvmlDeviceGetMemoryInfo"));

        if (!nvmlInit_v2 || !nvmlShutdown || !nvmlDeviceGetHandleByIndex_v2 || !nvmlDeviceGetName ||
            !nvmlDeviceGetUtilizationRates || !nvmlDeviceGetMemoryInfo) {
            return;
        }

        if (nvmlInit_v2() != 0) return;
        if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_index), &device) != 0) return;

        char buf[256];
        std::memset(buf, 0, sizeof(buf));
        if (nvmlDeviceGetName(device, buf, sizeof(buf)) == 0) {
            name = buf;
        }
        ok = true;
    }

    ~Nvml() {
        if (ok && nvmlShutdown) {
            nvmlShutdown();
        }
        if (lib) {
            dlclose(lib);
            lib = nullptr;
        }
    }

    bool sample(Sample& out) const {
        if (!ok || !device) return false;

        nvmlUtilization_t util{};
        nvmlMemory_t mem{};

        if (nvmlDeviceGetUtilizationRates(device, &util) != 0) return false;
        if (nvmlDeviceGetMemoryInfo(device, &mem) != 0) return false;

        out.gpu_available = true;
        out.gpu_name = name;
        out.gpu_util_percent = util.gpu;
        out.gpu_mem_util_percent = util.memory;
        out.gpu_mem_total_b = static_cast<std::uint64_t>(mem.total);
        out.gpu_mem_free_b = static_cast<std::uint64_t>(mem.free);
        out.gpu_mem_used_b = static_cast<std::uint64_t>(mem.used);
        return true;
    }
};

ResourceMonitor::ResourceMonitor(int gpu_index) {
    nvml = new Nvml(gpu_index);
}

ResourceMonitor::~ResourceMonitor() {
    stop();
    delete nvml;
    nvml = nullptr;
}

void ResourceMonitor::start(double interval_seconds) {
    if (is_running.exchange(true)) return;
    worker = std::thread([this, interval_seconds] { loop(interval_seconds); });
}

void ResourceMonitor::stop() {
    if (!is_running.exchange(false)) return;
    if (worker.joinable()) worker.join();
}

bool ResourceMonitor::running() const {
    return is_running.load();
}

Sample ResourceMonitor::latest() const {
    std::lock_guard<std::mutex> lk(mtx);
    return last;
}

ResourceMonitor::CpuTimes ResourceMonitor::readCpuTimes() {
    std::ifstream in("/proc/stat");
    std::string cpu;
    std::uint64_t user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
    in >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    CpuTimes t;
    t.idle = idle + iowait;
    t.total = user + nice + system + idle + iowait + irq + softirq + steal;
    return t;
}

bool ResourceMonitor::readMemInfo(std::uint64_t& total_kb, std::uint64_t& available_kb) {
    std::ifstream in("/proc/meminfo");
    if (!in) return false;
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    total_kb = 0;
    available_kb = 0;
    while (in >> key >> value >> unit) {
        if (key == "MemTotal:") total_kb = value;
        if (key == "MemAvailable:") available_kb = value;
        if (total_kb && available_kb) return true;
    }
    return total_kb > 0;
}

void ResourceMonitor::loop(double interval_seconds) {
    CpuTimes prev = readCpuTimes();

    while (is_running.load()) {
        std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
        CpuTimes cur = readCpuTimes();

        const auto idle_delta = static_cast<double>(cur.idle - prev.idle);
        const auto total_delta = static_cast<double>(cur.total - prev.total);
        prev = cur;

        Sample s;
        s.timestamp_s = nowSeconds();
        if (total_delta > 0.0) {
            s.cpu_percent = 100.0 * (1.0 - idle_delta / total_delta);
        }

        std::uint64_t total_kb = 0, avail_kb = 0;
        if (readMemInfo(total_kb, avail_kb)) {
            s.mem_total_kb = total_kb;
            s.mem_available_kb = avail_kb;
            s.mem_used_kb = (total_kb > avail_kb) ? (total_kb - avail_kb) : 0;
        }

        if (nvml) {
            nvml->sample(s);
        }

        {
            std::lock_guard<std::mutex> lk(mtx);
            last = std::move(s);
        }
    }
}

} // namespace BSPOT::monitor
