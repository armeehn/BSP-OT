# BSP-OT Synthetic CPU/GPU Benchmarks

This folder contains a small benchmark harness that runs `viz_playground` in headless mode and records
CPU vs GPU timing for increasing point counts. It also writes a dedicated **power-of-two** section
where `N = 2^r` for a positive integer `r`.

## How to run

From the repo root:

```bash
python3 benchmarks/benchmark_cpu_gpu.py \
  --build-dir build-cuda-full \
  --mode bijection \
  --nb-plans 2 \
  --dim 10 \
  --repeats 3
```

The script generates:

- `benchmarks/out/linear.csv`
- `benchmarks/out/pow2.csv`
- `benchmarks/out/linear.svg` (CPU vs GPU plot)
- `benchmarks/out/pow2.svg` (CPU vs GPU plot for N = 2^r)
- `benchmarks/out/summary.md` (tables + embedded plots)

## Notes

- The CUDA path starts engaging once per-slice projections are large (currently ~4096 points).
- You can adjust sizes via `--min`, `--max`, `--step`, and `--pow-min`/`--pow-max`.
- If you want to cap CPU parallelism, use `--threads` to set `OMP_NUM_THREADS`.

## 3D matrix benchmark

Run a 3D grid across (mode, N, nb_plans), capturing CPU and GPU for every cell:

```bash
python3 benchmarks/benchmark_matrix_3d.py
```

Outputs land in `benchmarks/out_matrix/`:

- `results.csv` and `matrix.json`
- `report.md` with tables + plots

You can extend the matrix with couplings per iteration:

```bash
python3 benchmarks/benchmark_matrix_3d.py --couplings 1,2,4
```
