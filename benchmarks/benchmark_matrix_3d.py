#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from math import sqrt
from pathlib import Path


def parse_list(arg, cast=int):
    if not arg:
        return []
    items = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mu = sum(values) / len(values)
    if len(values) < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return mu, sqrt(var)


def run_case(exe_path, mode, n, plans, couplings, metrics_path, env, dim):
    cmd = [
        str(exe_path),
        "--mode", mode,
        "--na", str(n),
        "--nb", str(n),
        "--nb_plans", str(plans),
        "--nb_couplings", str(couplings),
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


def run_matrix(exe_path, modes, sizes, plans_list, couplings_list, repeats, out_dir, threads, dim):
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env_gpu = os.environ.copy()
    env_cpu = os.environ.copy()
    env_cpu["BSPOT_DISABLE_CUDA"] = "1"
    if threads and threads > 0:
        env_gpu["OMP_NUM_THREADS"] = str(threads)
        env_cpu["OMP_NUM_THREADS"] = str(threads)

    rows = []
    matrix = {}

    for mode in modes:
        matrix.setdefault(mode, {})
        for couplings in couplings_list:
            c_key = str(couplings)
            matrix[mode].setdefault(c_key, {})
            for n in sizes:
                n_key = str(n)
                matrix[mode][c_key].setdefault(n_key, {})
                for plans in plans_list:
                    cell_key = str(plans)
                    matrix[mode][c_key][n_key].setdefault(cell_key, {})
                    gpu_samples = []
                    cpu_samples = []

                    for rep in range(repeats):
                        tag = f"{mode}_n{n}_p{plans}_c{couplings}_r{rep}"
                        gpu_metrics = metrics_dir / f"gpu_{tag}.json"
                        cpu_metrics = metrics_dir / f"cpu_{tag}.json"

                        print(f"[{mode}] N={n} plans={plans} couplings={couplings} GPU (rep {rep + 1}/{repeats})...")
                        gpu_samples.append(run_case(exe_path, mode, n, plans, couplings, gpu_metrics, env_gpu, dim))

                        print(f"[{mode}] N={n} plans={plans} couplings={couplings} CPU (rep {rep + 1}/{repeats})...")
                        cpu_samples.append(run_case(exe_path, mode, n, plans, couplings, cpu_metrics, env_cpu, dim))

                    gpu_agg = aggregate_runs(gpu_samples)
                    cpu_agg = aggregate_runs(cpu_samples)

                    speedup = cpu_agg["wall_mean_s"] / gpu_agg["wall_mean_s"] if gpu_agg["wall_mean_s"] > 0 else 0.0
                    matrix[mode][c_key][n_key][cell_key] = {
                        "cpu": cpu_agg,
                        "gpu": gpu_agg,
                        "speedup": speedup,
                    }

                    rows.append({
                        "mode": mode,
                        "N": n,
                        "nb_plans": plans,
                        "nb_couplings": couplings,
                        "cpu_wall_mean_s": cpu_agg["wall_mean_s"],
                        "cpu_wall_std_s": cpu_agg["wall_std_s"],
                        "gpu_wall_mean_s": gpu_agg["wall_mean_s"],
                        "gpu_wall_std_s": gpu_agg["wall_std_s"],
                        "cpu_ms_mean": cpu_agg["cpu_mean_ms"],
                        "gpu_ms_mean": gpu_agg["gpu_mean_ms"],
                        "cpu_projection_calls_mean": cpu_agg["projection_calls_mean"],
                        "gpu_projection_calls_mean": gpu_agg["projection_calls_mean"],
                        "cpu_points_projected_mean": cpu_agg["points_projected_mean"],
                        "gpu_points_projected_mean": gpu_agg["points_projected_mean"],
                        "gpu_enabled": gpu_agg["gpu_enabled"],
                        "speedup": speedup,
                    })
    return rows, matrix


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def lerp(a, b, t):
    return a + (b - a) * t


def color_for_speedup(value, vmin, vmax):
    if vmax <= vmin:
        return "#cccccc"
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # Red (slow) -> Gray (neutral) -> Green (fast)
    if t < 0.5:
        tt = t / 0.5
        r = int(lerp(209, 180, tt))
        g = int(lerp(73, 180, tt))
        b = int(lerp(91, 180, tt))
    else:
        tt = (t - 0.5) / 0.5
        r = int(lerp(180, 41, tt))
        g = int(lerp(180, 179, tt))
        b = int(lerp(180, 89, tt))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap_svg(path, title, sizes, plans, values):
    cell_w = 90
    cell_h = 40
    margin_l = 90
    margin_t = 60
    width = margin_l + cell_w * len(plans) + 20
    height = margin_t + cell_h * len(sizes) + 40

    vals = [values.get((n, p), 0.0) for n in sizes for p in plans]
    vmin = min(vals) if vals else 0.0
    vmax = max(vals) if vals else 1.0

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    svg.append("<rect width='100%' height='100%' fill='white' />")
    svg.append(f"<text x='{width/2:.1f}' y='28' font-size='16' text-anchor='middle' fill='#111'>{title}</text>")

    # Column labels (plans)
    for j, p in enumerate(plans):
        x = margin_l + j * cell_w + cell_w / 2
        svg.append(f"<text x='{x:.1f}' y='{margin_t - 14}' font-size='12' text-anchor='middle' fill='#111'>p={p}</text>")

    # Row labels (sizes)
    for i, n in enumerate(sizes):
        y = margin_t + i * cell_h + cell_h / 2 + 4
        svg.append(f"<text x='{margin_l - 10}' y='{y:.1f}' font-size='12' text-anchor='end' fill='#111'>N={n}</text>")

    for i, n in enumerate(sizes):
        for j, p in enumerate(plans):
            val = values.get((n, p), 0.0)
            color = color_for_speedup(val, vmin, vmax)
            x = margin_l + j * cell_w
            y = margin_t + i * cell_h
            svg.append(f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' fill='{color}' stroke='#333' />")
            svg.append(f"<text x='{x + cell_w/2:.1f}' y='{y + cell_h/2 + 4:.1f}' font-size='11' text-anchor='middle' fill='#111'>{val:.2f}x</text>")

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_line_plot_svg(path, title, series, x_label="N points", y_label="wall time (s)"):
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

    x0, y0 = margin_l, margin_t + plot_h
    x1, y1 = margin_l + plot_w, margin_t
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#333' stroke-width='1' />")
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333' stroke-width='1' />")

    for xt in [x_min, (x_min + x_max) / 2, x_max]:
        px = map_x(xt)
        svg.append(f"<line x1='{px}' y1='{y0}' x2='{px}' y2='{y0 + 6}' stroke='#333' />")
        svg.append(f"<text x='{px}' y='{y0 + 22}' font-size='11' text-anchor='middle' fill='#333'>{int(round(xt))}</text>")
    for yt in [y_min, (y_min + y_max) / 2, y_max]:
        py = map_y(yt)
        svg.append(f"<line x1='{x0 - 6}' y1='{py}' x2='{x0}' y2='{py}' stroke='#333' />")
        svg.append(f"<text x='{x0 - 10}' y='{py + 4}' font-size='11' text-anchor='end' fill='#333'>{yt:.3g}</text>")

    svg.append(f"<text x='{width/2:.1f}' y='{height - 15}' font-size='12' text-anchor='middle' fill='#111'>{x_label}</text>")
    svg.append(f"<text x='18' y='{height/2:.1f}' font-size='12' text-anchor='middle' fill='#111' transform='rotate(-90 18 {height/2:.1f})'>{y_label}</text>")

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


def table_from_matrix(sizes, plans, values, fmt="{:.3f}"):
    header = "| N | " + " | ".join(f"p={p}" for p in plans) + " |"
    sep = "|" + "---|" * (len(plans) + 1)
    lines = [header, sep]
    for n in sizes:
        row = [f"{n}"]
        for p in plans:
            val = values.get((n, p), 0.0)
            row.append(fmt.format(val))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(out_dir, rows, matrix, modes, sizes, plans_list, couplings_list):
    report = []
    report.append("# BSP-OT 3D Matrix Benchmark Report")
    report.append("")
    report.append("Axes:")
    report.append(f"- modes: {', '.join(modes)}")
    report.append(f"- sizes (N): {', '.join(str(n) for n in sizes)}")
    report.append(f"- nb_plans: {', '.join(str(p) for p in plans_list)}")
    report.append(f"- nb_couplings: {', '.join(str(c) for c in couplings_list)}")
    report.append("")

    pow2_sizes = [n for n in sizes if is_pow2(n)]
    if pow2_sizes:
        report.append("Power-of-two sizes detected:")
        report.append(f"- N = 2^r for r in [{', '.join(str(n) for n in pow2_sizes)}]")
        report.append("")

    for mode in modes:
        report.append(f"## Mode: {mode}")
        report.append("")
        for couplings in couplings_list:
            c_key = str(couplings)
            report.append(f"### nb_couplings = {couplings}")
            report.append("")
            speed_values = {}
            cpu_values = {}
            gpu_values = {}
            for n in sizes:
                for p in plans_list:
                    cell = matrix.get(mode, {}).get(c_key, {}).get(str(n), {}).get(str(p), {})
                    speed_values[(n, p)] = float(cell.get("speedup") or 0.0)
                    cpu = cell.get("cpu") or {}
                    gpu = cell.get("gpu") or {}
                    cpu_values[(n, p)] = float(cpu.get("wall_mean_s") or 0.0)
                    gpu_values[(n, p)] = float(gpu.get("wall_mean_s") or 0.0)

            report.append("#### Speedup matrix (CPU/GPU wall time)")
            report.append("")
            report.append(table_from_matrix(sizes, plans_list, speed_values, fmt="{:.2f}x"))
            report.append("")

            report.append("#### CPU wall time (s)")
            report.append("")
            report.append(table_from_matrix(sizes, plans_list, cpu_values, fmt="{:.4f}"))
            report.append("")

            report.append("#### GPU wall time (s)")
            report.append("")
            report.append(table_from_matrix(sizes, plans_list, gpu_values, fmt="{:.4f}"))
            report.append("")

            heatmap_path = out_dir / f"speedup_heatmap_{mode}_c{couplings}.svg"
            write_heatmap_svg(heatmap_path, f"Speedup heatmap ({mode}, couplings={couplings})", sizes, plans_list, speed_values)
            report.append(f"![speedup heatmap]({heatmap_path.name})")
            report.append("")

            cpu_mean_series = []
            gpu_mean_series = []
            for n in sizes:
                cpu_vals = [cpu_values.get((n, p), 0.0) for p in plans_list]
                gpu_vals = [gpu_values.get((n, p), 0.0) for p in plans_list]
                cpu_mean = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0
                gpu_mean = sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0
                cpu_mean_series.append((n, cpu_mean))
                gpu_mean_series.append((n, gpu_mean))

            line_path = out_dir / f"mean_wall_{mode}_c{couplings}.svg"
            write_line_plot_svg(
                line_path,
                f"Mean wall time vs N ({mode}, couplings={couplings})",
                [("CPU mean", "#d1495b", cpu_mean_series), ("GPU mean", "#00798c", gpu_mean_series)],
            )
            report.append(f"![mean wall time]({line_path.name})")
            report.append("")

            if pow2_sizes:
                report.append("#### Power-of-two subset (N = 2^r)")
                report.append("")
                report.append(table_from_matrix(pow2_sizes, plans_list, speed_values, fmt="{:.2f}x"))
                report.append("")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run a 3D CPU/GPU benchmark matrix for BSP-OT.")
    parser.add_argument("--build-dir", default="build-cuda-full", help="CMake build directory")
    parser.add_argument("--exe", default="viz_playground", help="Executable name")
    parser.add_argument("--modes", default="bijection,coupling", help="Comma-separated modes")
    parser.add_argument("--sizes", default="4096,8192,16384,32768", help="Comma-separated sizes")
    parser.add_argument("--plans", default="1,2,4", help="Comma-separated nb_plans values")
    parser.add_argument("--couplings", default="1,2,4", help="Comma-separated nb_couplings values")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per cell")
    parser.add_argument("--dim", type=int, default=10, help="Dimension for generated point clouds")
    parser.add_argument("--out-dir", default="benchmarks/out_matrix", help="Output directory")
    parser.add_argument("--threads", type=int, default=0, help="Set OMP_NUM_THREADS if > 0")
    args = parser.parse_args()

    sizes = parse_list(args.sizes, int)
    plans_list = parse_list(args.plans, int)
    couplings_list = parse_list(args.couplings, int)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if not sizes or not plans_list or not couplings_list or not modes:
        raise ValueError("sizes, plans, couplings, and modes must be non-empty")

    exe_path = Path(args.build_dir) / args.exe
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, matrix = run_matrix(
        exe_path=exe_path,
        modes=modes,
        sizes=sizes,
        plans_list=plans_list,
        couplings_list=couplings_list,
        repeats=args.repeats,
        out_dir=out_dir,
        threads=args.threads,
        dim=args.dim,
    )

    write_csv(out_dir / "results.csv", rows)
    (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    build_report(out_dir, rows, matrix, modes, sizes, plans_list, couplings_list)

    print(f"Wrote {out_dir / 'results.csv'}")
    print(f"Wrote {out_dir / 'matrix.json'}")
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
