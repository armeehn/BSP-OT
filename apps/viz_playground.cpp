#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"

#include "../src/BSPOTWrapper.h"
#include "../src/PointCloudIO.h"
#include "../src/cloudutils.h"
#include "../src/cuda_backend.h"
#include "../src/plot.h"
#include "../src/resource_monitor.h"
#include "../src/sampling.h"
#include "../src/telemetry.h"
#include "../common/CLI11.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace BSPOT;

constexpr int static_dim = -1;
using Pts = Points<static_dim>;

int NA = 1000;
int NB = 1000;
int nb_plans = 16;
int dim = 3;
int nb_couplings = 1;
bool force_size = false;
bool viz = false;
bool dashboard = false;
bool auto_compute = true;
std::string mode = "bijection";
std::string mu_src;
std::string nu_src;
std::string metrics_out;
double metrics_interval = 0.5;
int gpu_index = 0;

Pts A, B;
BijectiveMatching T;
Coupling pi;
scalar last_cost = 0;
double last_compute_s = 0.0;
bool has_result = false;
bool mode_coupling = false;
float lerp_t = 0.0f;

polyscope::PointCloud* pcA = nullptr;
polyscope::PointCloud* pcB = nullptr;
polyscope::PointCloud* pcLerp = nullptr;
polyscope::CurveNetwork* plan_net = nullptr;

std::unique_ptr<monitor::ResourceMonitor> g_monitor;
std::unique_ptr<telemetry::Recorder> g_recorder;

static std::string escapeJson(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        if (c == '"') {
            out += "\\\"";
        } else if (c == '\\') {
            out += "\\\\";
        } else {
            out += c;
        }
    }
    return out;
}

static void writeMetrics(const std::string& path, const telemetry::Recorder* rec, const monitor::Sample* sample) {
    std::ofstream out(path);
    if (!out) {
        spdlog::error("failed to write metrics to {}", path);
        return;
    }
    out << "{";
    out << "\"telemetry\":";
    if (rec) {
        out << rec->to_json();
    } else {
        out << "null";
    }
    out << ",";
    out << "\"resources\":";
    if (sample) {
        out << "{";
        out << "\"timestamp_s\":" << sample->timestamp_s << ",";
        out << "\"cpu_percent\":" << sample->cpu_percent << ",";
        out << "\"mem_total_kb\":" << sample->mem_total_kb << ",";
        out << "\"mem_available_kb\":" << sample->mem_available_kb << ",";
        out << "\"mem_used_kb\":" << sample->mem_used_kb << ",";
        out << "\"gpu_available\":" << (sample->gpu_available ? "true" : "false") << ",";
        out << "\"gpu_name\":\"" << escapeJson(sample->gpu_name) << "\",";
        out << "\"gpu_mem_total_b\":" << sample->gpu_mem_total_b << ",";
        out << "\"gpu_mem_used_b\":" << sample->gpu_mem_used_b << ",";
        out << "\"gpu_mem_free_b\":" << sample->gpu_mem_free_b << ",";
        out << "\"gpu_util_percent\":" << sample->gpu_util_percent << ",";
        out << "\"gpu_mem_util_percent\":" << sample->gpu_mem_util_percent;
        out << "}";
    } else {
        out << "null";
    }
    out << "}";
}

static Pts forceToCols(const Pts& X, int target) {
    if (X.cols() == target) return X;
    const int rows = X.rows();
    const int cols = X.cols();
    Pts rslt(rows, target);
    if (cols <= 0) return rslt;

    static thread_local std::mt19937 gen(std::random_device{}());
    if (cols < target) {
        std::uniform_int_distribution<int> dist(0, cols - 1);
        for (int i = 0; i < target; ++i) {
            rslt.col(i) = X.col(dist(gen));
        }
    } else {
        std::vector<int> idx(cols);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), gen);
        for (int i = 0; i < target; ++i) {
            rslt.col(i) = X.col(idx[i]);
        }
    }
    return rslt;
}

static void updatePointCloud(polyscope::PointCloud* pc, const Pts& X) {
    if (!pc) return;
    if (X.rows() == 2) {
        pc->updatePointPositions2D(X.transpose());
    } else if (X.rows() == 3) {
        pc->updatePointPositions(X.transpose());
    }
}

static void clearResult() {
    has_result = false;
    if (plan_net) {
        polyscope::removeStructure(plan_net, false);
        plan_net = nullptr;
    }
    if (pcLerp) {
        polyscope::removeStructure(pcLerp, false);
        pcLerp = nullptr;
    }
}

static void refreshDisplay() {
    if (!viz) return;
    if (A.rows() != 2 && A.rows() != 3) {
        spdlog::warn("visualization supports only 2D or 3D, current dim={}", A.rows());
        return;
    }
    if (!pcA)
        pcA = display<static_dim>("A", A);
    else
        updatePointCloud(pcA, A);

    if (!pcB)
        pcB = display<static_dim>("B", B);
    else
        updatePointCloud(pcB, B);

    if (plan_net) {
        polyscope::removeStructure(plan_net, false);
        plan_net = nullptr;
    }

    if (has_result) {
        if (mode_coupling) {
            plan_net = plotCoupling("bspot", A, B, pi);
        } else {
            plan_net = plotMatching("bspot", A, B, T);
        }
    }
}

static void sampleInputs() {
    if (!mu_src.empty()) {
        A = ReadPointCloud<static_dim>(mu_src);
        dim = A.rows();
    } else {
        A = sampleUnitBall<static_dim>(NA, dim);
    }

    if (!nu_src.empty()) {
        B = ReadPointCloud<static_dim>(nu_src);
        if (dim == 0) dim = B.rows();
    } else {
        B = sampleUnitBall<static_dim>(NB, dim);
    }
    if (A.rows() != B.rows()) {
        spdlog::error("dimension mismatch: A is {}, B is {}", A.rows(), B.rows());
        return;
    }

    if (force_size) {
        if (NA > 0) A = forceToCols(A, NA);
        if (NB > 0) B = forceToCols(B, NB);
    }

    if (!mode_coupling && A.cols() != B.cols()) {
        const int target = std::min<int>(A.cols(), B.cols());
        spdlog::warn("bijection requires equal sizes, resizing to {}", target);
        A = forceToCols(A, target);
        B = forceToCols(B, target);
    }

    NA = A.cols();
    NB = B.cols();
    Normalize<static_dim>(A, Vector<static_dim>::Zero(A.rows()));
    Normalize<static_dim>(B, Vector<static_dim>::Zero(B.rows()));

    clearResult();
    refreshDisplay();
}

static scalar evalBijection(const Pts& X, const Pts& Y, const ints& plan) {
    return std::sqrt((X - Y(Eigen::all, plan)).squaredNorm());
}

static scalar evalCoupling(const Pts& X, const Pts& Y, const Coupling& coupling) {
    scalar C = 0;
    for (int k = 0; k < coupling.outerSize(); ++k) {
        for (Coupling::InnerIterator it(coupling, k); it; ++it) {
            C += (X.col(it.row()) - Y.col(it.col())).squaredNorm() * it.value();
        }
    }
    return C;
}

static void computeBijectionCore() {
    if (A.cols() != B.cols()) {
        spdlog::error("bijection expects equal sizes, got {} and {}", A.cols(), B.cols());
        return;
    }
    if (nb_plans <= 0) {
        spdlog::error("nb_plans must be positive");
        return;
    }
    std::vector<BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching<static_dim> BSP(A, B);
    auto cost = [&](size_t i, size_t j) { return (A.col(i) - B.col(j)).squaredNorm(); };
#pragma omp parallel for
    for (int i = 0; i < nb_plans; ++i) {
        plans[i] = BSP.computeGaussianMatching();
    }
    T = MergePlans(plans, cost, BijectiveMatching(), (A.cols() < 500000));
    last_cost = evalBijection(A, B, T);
    has_result = true;
}

static void computeCouplingCore() {
    if (nb_couplings <= 0) {
        spdlog::error("nb_couplings must be positive");
        return;
    }
    Atoms mu = UniformMass(A.cols());
    Atoms nu = UniformMass(B.cols());
    scalar cost_sum = 0;
    for (int i = 0; i < nb_couplings; ++i) {
        GeneralBSPMatching<static_dim> BSP(A, mu, B, nu);
        Coupling cur = BSP.computeCoupling();
        cost_sum += evalCoupling(A, B, cur);
        if (i + 1 == nb_couplings) {
            pi = std::move(cur);
        }
    }
    last_cost = cost_sum / static_cast<scalar>(nb_couplings);
    has_result = true;
}

static void computePlan() {
    has_result = false;
    last_cost = 0;
    g_recorder = std::make_unique<telemetry::Recorder>();
    g_recorder->start();
    auto start = Time::now();
    {
        telemetry::ScopedRecorder scoped(g_recorder.get());
        if (mode_coupling) {
            computeCouplingCore();
        } else {
            computeBijectionCore();
        }
    }
    g_recorder->stop();
    last_compute_s = TimeFrom(start);

    refreshDisplay();
    if (!mode_coupling && has_result && viz) {
        const scalar t = static_cast<scalar>(lerp_t);
        Pts L = A * (scalar(1) - t) + t * B(Eigen::all, T);
        if (!pcLerp)
            pcLerp = display<static_dim>("lerp", L);
        else
            updatePointCloud(pcLerp, L);
    }

    if (!metrics_out.empty()) {
        monitor::Sample sample;
        const monitor::Sample* sample_ptr = nullptr;
        if (g_monitor) {
            sample = g_monitor->latest();
            sample_ptr = &sample;
        }
        writeMetrics(metrics_out, g_recorder.get(), sample_ptr);
    }
}

static void updateLerp() {
    if (!viz || mode_coupling || !has_result) return;
    const scalar t = static_cast<scalar>(lerp_t);
    Pts L = A * (scalar(1) - t) + t * B(Eigen::all, T);
    if (!pcLerp)
        pcLerp = display<static_dim>("lerp", L);
    else
        updatePointCloud(pcLerp, L);
}

static void dashboardUI() {
    if (!dashboard) return;

    ImGui::Separator();
    ImGui::Text("Telemetry");
    if (g_recorder) {
        auto snap = g_recorder->snapshot();
        ImGui::Text("projections: %zu", snap.points_projected);
        ImGui::Text("projection calls: %zu", snap.projection_calls);
        ImGui::Text("gpu ms: %.2f", snap.gpu_ms);
        ImGui::Text("cpu ms: %.2f", snap.cpu_ms);
        ImGui::Text("h2d bytes: %zu", snap.h2d_bytes);
        ImGui::Text("d2h bytes: %zu", snap.d2h_bytes);
        if (snap.gpu_enabled) {
            ImGui::Text("gpu: %s", snap.gpu_name.c_str());
            ImGui::Text("gpu mem free/total: %.2f / %.2f MiB",
                        static_cast<double>(snap.gpu_mem_free) / (1024.0 * 1024.0),
                        static_cast<double>(snap.gpu_mem_total) / (1024.0 * 1024.0));
        }
    } else {
        ImGui::Text("no telemetry yet");
    }

    ImGui::Separator();
    ImGui::Text("Resources");
    if (g_monitor) {
        auto sample = g_monitor->latest();
        ImGui::Text("cpu usage: %.1f%%", sample.cpu_percent);
        ImGui::Text("ram used: %.2f / %.2f GiB",
                    static_cast<double>(sample.mem_used_kb) / (1024.0 * 1024.0),
                    static_cast<double>(sample.mem_total_kb) / (1024.0 * 1024.0));
        if (sample.gpu_available) {
            ImGui::Text("gpu: %s", sample.gpu_name.c_str());
            ImGui::Text("gpu util: %u%%", sample.gpu_util_percent);
            ImGui::Text("gpu mem util: %u%%", sample.gpu_mem_util_percent);
            ImGui::Text("gpu mem used: %.2f / %.2f MiB",
                        static_cast<double>(sample.gpu_mem_used_b) / (1024.0 * 1024.0),
                        static_cast<double>(sample.gpu_mem_total_b) / (1024.0 * 1024.0));
        } else {
            ImGui::Text("gpu: not available");
        }
    } else {
        ImGui::Text("resource monitor disabled");
    }
}

static void userCallback() {
    ImGui::Begin("BSP-OT Playground");
    const char* modes[] = {"bijection", "coupling"};
    int mode_idx = mode_coupling ? 1 : 0;
    if (ImGui::Combo("mode", &mode_idx, modes, 2)) {
        mode_coupling = (mode_idx == 1);
        clearResult();
    }
    ImGui::InputInt("A size", &NA);
    ImGui::InputInt("B size", &NB);
    ImGui::InputInt("nb_plans", &nb_plans);
    if (mode_coupling) {
        ImGui::InputInt("nb_couplings", &nb_couplings);
    }
    if (ImGui::Checkbox("dashboard", &dashboard)) {
        if (dashboard && !g_monitor) {
            g_monitor = std::make_unique<monitor::ResourceMonitor>(gpu_index);
            g_monitor->start(metrics_interval);
        } else if (!dashboard && g_monitor && metrics_out.empty()) {
            g_monitor->stop();
            g_monitor.reset();
        }
    }
    ImGui::Text("cuda available: %s", cuda_backend::enabled() ? "yes" : "no");

    if (ImGui::Button("Resample")) {
        sampleInputs();
    }
    ImGui::SameLine();
    if (ImGui::Button("Compute")) {
        computePlan();
    }

    if (!mode_coupling) {
        if (ImGui::SliderFloat("lerp", &lerp_t, 0.0f, 1.0f)) {
            updateLerp();
        }
    }

    ImGui::Separator();
    ImGui::Text("last compute: %.3f s", last_compute_s);
    ImGui::Text("last cost: %.6f", static_cast<double>(last_cost));

    if (!metrics_out.empty()) {
        if (ImGui::Button("Write metrics")) {
            monitor::Sample sample;
            const monitor::Sample* sample_ptr = nullptr;
            if (g_monitor) {
                sample = g_monitor->latest();
                sample_ptr = &sample;
            }
            writeMetrics(metrics_out, g_recorder.get(), sample_ptr);
        }
    }

    dashboardUI();
    ImGui::End();
}

int main(int argc, char** argv) {
    srand(time(nullptr));
    CLI::App app("BSP-OT visualization playground");
    app.add_option("--mode", mode, "bijection|coupling")->check(CLI::IsMember({"bijection", "coupling"}));
    app.add_option("--na", NA, "number of points in A");
    app.add_option("--nb", NB, "number of points in B");
    app.add_option("--nb_plans", nb_plans, "number of plans (bijection mode)");
    app.add_option("--nb_couplings", nb_couplings, "number of couplings (coupling mode)");
    app.add_option("--dim", dim, "dimension for generated point clouds");
    app.add_option("--mu_file", mu_src, "source point cloud file");
    app.add_option("--nu_file", nu_src, "target point cloud file");
    app.add_flag("--force_size", force_size, "force point cloud sizes after load");
    app.add_flag("--viz", viz, "enable polyscope UI");
    app.add_flag("--dashboard", dashboard, "show resource dashboard");
    bool no_compute = false;
    app.add_flag("--no_compute", no_compute, "skip initial compute");
    app.add_option("--metrics_out", metrics_out, "write metrics JSON after compute");
    app.add_option("--metrics_interval", metrics_interval, "resource sampling interval seconds");
    app.add_option("--gpu_index", gpu_index, "GPU index for NVML sampling");
    CLI11_PARSE(app, argc, argv);

    mode_coupling = (mode == "coupling");
    auto_compute = !no_compute;

    if (dashboard || !metrics_out.empty()) {
        g_monitor = std::make_unique<monitor::ResourceMonitor>(gpu_index);
        g_monitor->start(metrics_interval);
    }

    if (viz) {
        polyscope::init();
    }

    sampleInputs();

    if (auto_compute) {
        computePlan();
    }

    if (viz) {
        refreshDisplay();
        polyscope::state::userCallback = userCallback;
        polyscope::show();
    } else {
        if (!metrics_out.empty()) {
            monitor::Sample sample;
            const monitor::Sample* sample_ptr = nullptr;
            if (g_monitor) {
                sample = g_monitor->latest();
                sample_ptr = &sample;
            }
            writeMetrics(metrics_out, g_recorder.get(), sample_ptr);
        }
    }

    if (g_monitor) {
        g_monitor->stop();
    }
    return 0;
}
