#include "cuda_backend.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace BSPOT::cuda_backend {

__global__ void dotKernel(const scalar* points, int dim, int cols, const int* ids, int count, const scalar* dir,
                          scalar* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const int col = ids[idx];
    if (col >= cols) return;
    const scalar* base = points + static_cast<std::size_t>(col) * static_cast<std::size_t>(dim);
    scalar acc = 0;
    for (int k = 0; k < dim; ++k) {
        acc += base[k] * dir[k];
    }
    out[idx] = acc;
}

namespace {

struct DeviceMatrix {
    const scalar* host = nullptr;
    int dim = 0;
    int cols = 0;
    scalar* dev = nullptr;
    std::size_t bytes = 0;
};

struct Workspace {
    int* d_ids = nullptr;
    std::size_t ids_capacity = 0;
    scalar* d_dir = nullptr;
    std::size_t dir_capacity = 0;
    scalar* d_out = nullptr;
    std::size_t out_capacity = 0;
};

std::mutex g_mtx;
std::vector<DeviceMatrix> g_mats;
Workspace g_ws;

void freeWorkspace() {
    if (g_ws.d_ids) cudaFree(g_ws.d_ids);
    if (g_ws.d_dir) cudaFree(g_ws.d_dir);
    if (g_ws.d_out) cudaFree(g_ws.d_out);
    g_ws = {};
}

void freeMatrices() {
    for (auto& m : g_mats) {
        if (m.dev) cudaFree(m.dev);
        m = {};
    }
    g_mats.clear();
}

struct Cleanup {
    ~Cleanup() {
        std::lock_guard<std::mutex> lk(g_mtx);
        freeWorkspace();
        freeMatrices();
    }
};

Cleanup g_cleanup;

bool disabledByEnv() {
    const char* v = std::getenv("BSPOT_DISABLE_CUDA");
    return v && v[0] && v[0] != '0';
}

bool hasDevice() {
    static int cached = -1;
    if (cached != -1) return cached == 1;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    cached = (err == cudaSuccess && count > 0) ? 1 : 0;
    return cached == 1;
}

DeviceMatrix* getOrUploadMatrix(const scalar* points, int dim, int cols, bool& did_upload) {
    did_upload = false;
    for (auto& m : g_mats) {
        if (m.host == points && m.dim == dim && m.cols == cols && m.dev) {
            return &m;
        }
    }

    DeviceMatrix m;
    m.host = points;
    m.dim = dim;
    m.cols = cols;
    m.bytes = static_cast<std::size_t>(dim) * static_cast<std::size_t>(cols) * sizeof(scalar);

    if (cudaMalloc(&m.dev, m.bytes) != cudaSuccess) {
        return nullptr;
    }
    if (cudaMemcpy(m.dev, points, m.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(m.dev);
        return nullptr;
    }
    did_upload = true;

    constexpr std::size_t kMaxMatrices = 4;
    if (g_mats.size() >= kMaxMatrices) {
        auto& victim = g_mats.front();
        if (victim.dev) cudaFree(victim.dev);
        g_mats.erase(g_mats.begin());
    }
    g_mats.push_back(m);
    return &g_mats.back();
}

bool ensureWorkspace(int dim, int count) {
    if (g_ws.dir_capacity < static_cast<std::size_t>(dim)) {
        if (g_ws.d_dir) cudaFree(g_ws.d_dir);
        if (cudaMalloc(&g_ws.d_dir, static_cast<std::size_t>(dim) * sizeof(scalar)) != cudaSuccess) return false;
        g_ws.dir_capacity = static_cast<std::size_t>(dim);
    }
    if (g_ws.ids_capacity < static_cast<std::size_t>(count)) {
        if (g_ws.d_ids) cudaFree(g_ws.d_ids);
        if (cudaMalloc(&g_ws.d_ids, static_cast<std::size_t>(count) * sizeof(int)) != cudaSuccess) return false;
        g_ws.ids_capacity = static_cast<std::size_t>(count);
    }
    if (g_ws.out_capacity < static_cast<std::size_t>(count)) {
        if (g_ws.d_out) cudaFree(g_ws.d_out);
        if (cudaMalloc(&g_ws.d_out, static_cast<std::size_t>(count) * sizeof(scalar)) != cudaSuccess) return false;
        g_ws.out_capacity = static_cast<std::size_t>(count);
    }
    return true;
}

} // namespace

bool enabled() {
    if (disabledByEnv()) return false;
    return hasDevice();
}

bool projectDots(const scalar* points, int dim, int cols, const int* ids, int count, const scalar* dir, scalar* out,
                 telemetry::Recorder* rec) {
    if (!enabled()) return false;
    if (!points || !ids || !dir || !out || count <= 0 || dim <= 0) return false;

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    const std::size_t ids_bytes = static_cast<std::size_t>(count) * sizeof(int);
    const std::size_t dir_bytes = static_cast<std::size_t>(dim) * sizeof(scalar);
    const std::size_t out_bytes = static_cast<std::size_t>(count) * sizeof(scalar);

    std::lock_guard<std::mutex> lk(g_mtx);

    bool uploaded_points = false;
    DeviceMatrix* mat = getOrUploadMatrix(points, dim, cols, uploaded_points);
    if (!mat) return false;
    if (!ensureWorkspace(dim, count)) return false;

    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    cudaError_t err = cudaMemcpy(g_ws.d_ids, ids, ids_bytes, cudaMemcpyHostToDevice);
    err = err == cudaSuccess ? cudaMemcpy(g_ws.d_dir, dir, dir_bytes, cudaMemcpyHostToDevice) : err;
    if (err != cudaSuccess) {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
        return false;
    }

    cudaEventRecord(start_evt);
    dotKernel<<<blocks, threads>>>(mat->dev, dim, cols, g_ws.d_ids, count, g_ws.d_dir, g_ws.d_out);
    cudaEventRecord(stop_evt);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
        return false;
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_evt, stop_evt);

    if (cudaMemcpy(out, g_ws.d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
        return false;
    }

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);

    if (rec) {
        cudaDeviceProp prop{};
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
                rec->setGPUInfo(prop.name, total_b, free_b);
            } else {
                rec->setGPUInfo(prop.name, 0, 0);
            }
        }
        std::size_t h2d = ids_bytes + dir_bytes + (uploaded_points ? mat->bytes : 0);
        rec->addGPUProjection(static_cast<std::size_t>(count), h2d, out_bytes, ms);
    }
    return true;
}

} // namespace BSPOT::cuda_backend
