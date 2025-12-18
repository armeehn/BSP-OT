#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark BSP-OT CPU vs GPU using viz_playground metrics.")
    p.add_argument("--build-dir", default="build-cuda-full", help="CMake build directory")
    p.add_argument("--exe", default="viz_playground", help="Executable name")
    p.add_argument("--mode", default="bijection", choices=["bijection", "coupling"], help="Mode for viz_playground")
    p.add_argument("--nb-plans", type=int, default=2, help="Number of plans (bijection only)")
    p.add_argument("--nb-couplings", type=int, default=1, help="Number of couplings (coupling mode)")
    p.add_argument("--dim", type=int, default=10, help="Dimension for generated point clouds")
    p.add_argument("--min", dest="min_n", type=int, default=4096, help="Minimum N for linear sweep")
    p.add_argument("--max", dest="max_n", type=int, default=16384, help="Maximum N for linear sweep")
    p.add_argument("--step", dest="step_n", type=int, default=4096, help="Step size for linear sweep")
    p.add_argument("--pow-min", dest="pow_min", type=int, default=12, help="Minimum r for N=2^r sweep")
    p.add_argument("--pow-max", dest="pow_max", type=int, default=14, help="Maximum r for N=2^r sweep")
    p.add_argument("--out-dir", default="benchmarks/out", help="Output directory")
    p.add_argument("--repeats", type=int, default=1, help="Repeats per N")
    p.add_argument("--threads", type=int, default=0, help="Set OMP_NUM_THREADS if > 0")
    return p.parse_args()


def linear_sweep(min_n, max_n, step_n):
    if step_n <= 0:
        return []
    return list(range(min_n, max_n + 1, step_n))


def pow2_sweep(pow_min, pow_max):
    return [2 ** r for r in range(pow_min, pow_max + 1)]


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mu = sum(values) / len(values)
    if len(values) < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return mu, var ** 0.5


def run_case(exe_path, n, mode, nb_plans, nb_couplings, metrics_path, env, dim):
    cmd = [
        str(exe_path),
        "--mode", mode,
        "--na", str(n),
        "--nb", str(n),
        "--nb_plans", str(nb_plans),
        "--nb_couplings", str(nb_couplings),
        "--dim", str(dim),
        "--metrics_out", str(metrics_path),
    ]
    start = time.time()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.time() - start

    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    telemetry = data.get("telemetry") or {}
    wall = float(telemetry.get("wall_seconds") or 0.0)
    cpu_ms = float(telemetry.get("cpu_ms") or 0.0)
    gpu_ms = float(telemetry.get("gpu_ms") or 0.0)
    if wall <= 0.0:
        wall = cpu_ms / 1000.0 if cpu_ms > 0.0 else elapsed

    return {
        "N": n,
        "wall_s": wall,
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "projection_calls": int(telemetry.get("projection_calls") or 0),
        "points_projected": int(telemetry.get("points_projected") or 0),
        "gpu_enabled": bool(telemetry.get("gpu_enabled")),
    }


def aggregate_runs(samples):
    wall_vals = [s["wall_s"] for s in samples]
    cpu_vals = [s["cpu_ms"] for s in samples]
    gpu_vals = [s["gpu_ms"] for s in samples]
    calls_vals = [s["projection_calls"] for s in samples]
    pts_vals = [s["points_projected"] for s in samples]

    wall_mean, wall_std = mean_std(wall_vals)
    cpu_mean, cpu_std = mean_std(cpu_vals)
    gpu_mean, gpu_std = mean_std(gpu_vals)
    calls_mean, calls_std = mean_std(calls_vals)
    pts_mean, pts_std = mean_std(pts_vals)

    return {
        "wall_mean_s": wall_mean,
        "wall_std_s": wall_std,
        "cpu_mean_ms": cpu_mean,
        "cpu_std_ms": cpu_std,
        "gpu_mean_ms": gpu_mean,
        "gpu_std_ms": gpu_std,
        "projection_calls_mean": calls_mean,
        "projection_calls_std": calls_std,
        "points_projected_mean": pts_mean,
        "points_projected_std": pts_std,
        "gpu_enabled": any(s["gpu_enabled"] for s in samples),
    }


def run_sweep(label, ns, exe_path, mode, nb_plans, nb_couplings, out_dir, threads, dim, repeats):
    metrics_dir = out_dir / "metrics" / label
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env_gpu = os.environ.copy()
    env_cpu = os.environ.copy()
    env_cpu["BSPOT_DISABLE_CUDA"] = "1"
    if threads and threads > 0:
        env_gpu["OMP_NUM_THREADS"] = str(threads)
        env_cpu["OMP_NUM_THREADS"] = str(threads)

    rows = []
    for n in ns:
        gpu_samples = []
        cpu_samples = []
        for rep in range(repeats):
            gpu_metrics = metrics_dir / f"gpu_n{n}_r{rep}.json"
            cpu_metrics = metrics_dir / f"cpu_n{n}_r{rep}.json"
            print(f"[{label}] N={n} GPU (rep {rep + 1}/{repeats})...")
            gpu_samples.append(run_case(exe_path, n, mode, nb_plans, nb_couplings, gpu_metrics, env_gpu, dim))
            print(f"[{label}] N={n} CPU (rep {rep + 1}/{repeats})...")
            cpu_samples.append(run_case(exe_path, n, mode, nb_plans, nb_couplings, cpu_metrics, env_cpu, dim))

        gpu = aggregate_runs(gpu_samples)
        cpu = aggregate_runs(cpu_samples)
        speedup = cpu["wall_mean_s"] / gpu["wall_mean_s"] if gpu["wall_mean_s"] > 0 else 0.0
        rows.append({
            "N": n,
            "cpu_wall_s": cpu["wall_mean_s"],
            "cpu_wall_std_s": cpu["wall_std_s"],
            "gpu_wall_s": gpu["wall_mean_s"],
            "gpu_wall_std_s": gpu["wall_std_s"],
            "cpu_ms_mean": cpu["cpu_mean_ms"],
            "cpu_ms_std": cpu["cpu_std_ms"],
            "gpu_ms_mean": gpu["gpu_mean_ms"],
            "gpu_ms_std": gpu["gpu_std_ms"],
            "cpu_projection_calls_mean": cpu["projection_calls_mean"],
            "gpu_projection_calls_mean": gpu["projection_calls_mean"],
            "cpu_points_projected_mean": cpu["points_projected_mean"],
            "gpu_points_projected_mean": gpu["points_projected_mean"],
            "gpu_enabled": gpu["gpu_enabled"],
            "speedup": speedup,
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def linspace(a, b, n):
    if n <= 1:
        return [a]
    step = (b - a) / float(n - 1)
    return [a + i * step for i in range(n)]


def nice_label(val, is_int=False):
    if is_int:
        return str(int(round(val)))
    if abs(val) < 1e-3:
        return f"{val:.3e}"
    return f"{val:.3g}"


def write_svg(path, title, series, x_label="N points", y_label="wall time (s)"):
    width, height = 900, 500
    margin_l, margin_r, margin_t, margin_b = 80, 20, 50, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xs = [x for _, _, data in series for x, _ in data]
    ys = [y for _, _, data in series for _, y in data]
    if not xs or not ys:
        return

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_max <= 0:
        y_max = 1.0

    x_pad = (x_max - x_min) * 0.05
    y_pad = y_max * 0.1
    x_min -= x_pad
    x_max += x_pad
    y_max += y_pad

    def map_x(x):
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def map_y(y):
        return margin_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    svg.append("<rect width='100%' height='100%' fill='white' />")
    svg.append(f"<text x='{width/2:.1f}' y='28' font-size='16' text-anchor='middle' fill='#111'>{title}</text>")

    # Axes
    x0, y0 = margin_l, margin_t + plot_h
    x1, y1 = margin_l + plot_w, margin_t
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#333' stroke-width='1' />")
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333' stroke-width='1' />")

    # Ticks
    for xt in linspace(x_min, x_max, 5):
        px = map_x(xt)
        svg.append(f"<line x1='{px}' y1='{y0}' x2='{px}' y2='{y0 + 6}' stroke='#333' />")
        svg.append(f"<text x='{px}' y='{y0 + 22}' font-size='11' text-anchor='middle' fill='#333'>{nice_label(xt, True)}</text>")
    for yt in linspace(y_min, y_max, 5):
        py = map_y(yt)
        svg.append(f"<line x1='{x0 - 6}' y1='{py}' x2='{x0}' y2='{py}' stroke='#333' />")
        svg.append(f"<text x='{x0 - 10}' y='{py + 4}' font-size='11' text-anchor='end' fill='#333'>{nice_label(yt)}</text>")

    svg.append(f"<text x='{width/2:.1f}' y='{height - 15}' font-size='12' text-anchor='middle' fill='#111'>{x_label}</text>")
    svg.append(f"<text x='18' y='{height/2:.1f}' font-size='12' text-anchor='middle' fill='#111' transform='rotate(-90 18 {height/2:.1f})'>{y_label}</text>")

    # Series
    legend_x = margin_l + plot_w - 180
    legend_y = margin_t + 10
    for i, (label, color, data) in enumerate(series):
        points = " ".join(f"{map_x(x):.1f},{map_y(y):.1f}" for x, y in data)
        svg.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}' />")
        for x, y in data:
            svg.append(f"<circle cx='{map_x(x):.1f}' cy='{map_y(y):.1f}' r='3' fill='{color}' />")
        ly = legend_y + i * 18
        svg.append(f"<rect x='{legend_x}' y='{ly - 8}' width='12' height='12' fill='{color}' />")
        svg.append(f"<text x='{legend_x + 18}' y='{ly + 2}' font-size='12' fill='#111'>{label}</text>")

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_summary(path, linear_rows, pow2_rows, linear_svg, pow2_svg):
    def table(rows):
        lines = ["| N | CPU wall (s) | GPU wall (s) | speedup |", "|---:|---:|---:|---:|"]
        for r in rows:
            cpu_std = r.get("cpu_wall_std_s", 0.0)
            gpu_std = r.get("gpu_wall_std_s", 0.0)
            lines.append(
                f"| {r['N']} | {r['cpu_wall_s']:.4f} +/- {cpu_std:.4f} | {r['gpu_wall_s']:.4f} +/- {gpu_std:.4f} | {r['speedup']:.2f}x |"
            )
        return "\n".join(lines)

    text = [
        "# BSP-OT CPU vs GPU Benchmarks",
        "",
        "## Linear sweep",
        "",
        f"![linear plot]({linear_svg.name})",
        "",
        table(linear_rows),
        "",
        "## Power-of-two sweep (N = 2^r)",
        "",
        f"![pow2 plot]({pow2_svg.name})",
        "",
        table(pow2_rows),
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe_path = Path(args.build_dir) / args.exe
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    linear_ns = linear_sweep(args.min_n, args.max_n, args.step_n)
    pow2_ns = pow2_sweep(args.pow_min, args.pow_max)

    if not linear_ns:
        raise ValueError("linear sweep produced no sizes")
    if not pow2_ns:
        raise ValueError("pow2 sweep produced no sizes")

    linear_rows = run_sweep(
        "linear",
        linear_ns,
        exe_path,
        args.mode,
        args.nb_plans,
        args.nb_couplings,
        out_dir,
        args.threads,
        args.dim,
        args.repeats,
    )
    pow2_rows = run_sweep(
        "pow2",
        pow2_ns,
        exe_path,
        args.mode,
        args.nb_plans,
        args.nb_couplings,
        out_dir,
        args.threads,
        args.dim,
        args.repeats,
    )

    linear_csv = out_dir / "linear.csv"
    pow2_csv = out_dir / "pow2.csv"
    write_csv(linear_csv, linear_rows)
    write_csv(pow2_csv, pow2_rows)

    linear_svg = out_dir / "linear.svg"
    pow2_svg = out_dir / "pow2.svg"

    linear_series = [
        ("CPU", "#d1495b", [(r["N"], r["cpu_wall_s"]) for r in linear_rows]),
        ("GPU", "#00798c", [(r["N"], r["gpu_wall_s"]) for r in linear_rows]),
    ]
    pow2_series = [
        ("CPU", "#d1495b", [(r["N"], r["cpu_wall_s"]) for r in pow2_rows]),
        ("GPU", "#00798c", [(r["N"], r["gpu_wall_s"]) for r in pow2_rows]),
    ]

    write_svg(linear_svg, "CPU vs GPU: linear sweep", linear_series)
    write_svg(pow2_svg, "CPU vs GPU: N = 2^r", pow2_series)

    summary_md = out_dir / "summary.md"
    write_summary(summary_md, linear_rows, pow2_rows, linear_svg, pow2_svg)

    print(f"Wrote {linear_csv}")
    print(f"Wrote {pow2_csv}")
    print(f"Wrote {linear_svg}")
    print(f"Wrote {pow2_svg}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
