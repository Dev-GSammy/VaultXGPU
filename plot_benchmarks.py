#!/usr/bin/env python3
"""
Plot benchmark results comparing CPU (naive), SYCL CPU, SYCL GPU, and CUDA GPU
implementations across k-values 27-32.

Generates 6 plots:
  1. K31 CPU thread scaling (time vs threads)
  2. Best-time comparison across implementations for k27-k31
  3. CUDA GPU stage breakdown (T1, Sort/T2, Write) for k27-k31
  4a. Speedup factor: CPU-best vs GPUs at k31
  4b. CPU thread scaling efficiency (actual vs ideal)
  4c. Throughput log-scale: k vs time, all implementations

Usage: python3 plot_benchmarks.py [--outdir ./out]
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# K31 CPU thread scaling: threads -> minutes
CPU_THREAD_DATA = {
    1: 44.80, 2: 24.65, 4: 13.88, 8: 8.42, 16: 4.48, 32: 2.57,
    64: 1.61, 96: 1.33, 128: 1.21, 192: 1.01, 256: 0.98, 384: 0.87,
}

# SYCL GPU runs: k -> list of (T1, Sort/T2, Write, Total) in seconds
SYCL_GPU = {
    27: [(0.602, 0.565, 1.706, 3.536),
         (0.164, 0.292, 1.698, 2.818),
         (0.164, 0.292, 1.702, 2.820)],
    28: [(0.341, 0.507, 3.426, 5.521),
         (0.334, 0.506, 3.428, 5.514),
         (0.334, 0.506, 3.425, 5.509)],
    29: [(0.675, 1.040, 6.852, 11.080)] * 3,
    30: [(1.364, 2.153, 13.704, 22.384)] * 3,
    31: [(2.755, 4.457, 27.408, 45.412)] * 3,
}

# CUDA GPU runs: k -> list of (T1, Sort/T2, Write, Total) in seconds
CUDA_GPU = {
    27: [(0.218, 0.301, 2.279, 3.307),
         (0.222, 0.299, 2.254, 3.260),
         (0.221, 0.299, 2.222, 3.196)],
    28: [(0.453, 0.438, 4.151, 5.503),
         (0.453, 0.438, 4.260, 5.627),
         (0.453, 0.438, 4.350, 5.696)],
    29: [(0.923, 1.085, 8.606, 11.079),
         (0.923, 1.085, 8.424, 10.891),
         (0.923, 1.084, 8.226, 10.700)],
    30: [(1.868, 4.149, 16.849, 23.348),
         (1.867, 4.150, 15.945, 22.440),
         (1.868, 4.148, 16.340, 22.861)],
    31: [(3.767, 17.913, 33.002, 55.175),
         (3.766, 17.913, 31.627, 53.799)],
}

# SYCL CPU runs: k -> list of (T1, Sort/T2, Write, Total) in seconds
SYCL_CPU = {
    27: [(1.369, 4.063, 2.741, 8.855),
         (1.402, 3.971, 3.033, 9.088),
         (1.379, 3.977, 3.013, 9.085)],
    28: [(1.808, 6.248, 5.979, 15.087),
         (2.286, 6.156, 5.708, 15.219),
         (1.882, 6.159, 5.808, 14.912)],
    29: [(4.240, 10.660, 10.594, 27.286),
         (3.002, 10.450, 10.791, 26.144),
         (2.944, 10.428, 10.673, 25.904)],
    30: [(5.215, 19.693, 20.526, 48.992),
         (5.530, 19.983, 20.326, 49.724),
         (4.362, 19.797, 21.052, 48.956)],
    31: [(8.908, 40.230, 43.262, 100.042),
         (8.733, 39.720, 44.056, 99.166),
         (8.946, 39.852, 44.154, 99.600)],
    32: [(21.316, 91.922, 69.170, 197.442),
         (21.370, 92.742, 69.932, 198.055),
         (20.700, 91.025, 69.818, 193.731)],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_total_seconds(runs_dict):
    """For each k, return the minimum Total time across runs (in seconds)."""
    return {k: min(r[3] for r in runs) for k, runs in runs_dict.items()}

def best_stages_seconds(runs_dict):
    """For each k, return the (T1, Sort/T2, Write) from the run with min Total."""
    out = {}
    for k, runs in runs_dict.items():
        best_run = min(runs, key=lambda r: r[3])
        out[k] = best_run[:3]
    return out

# Naive CPU best time at k31 (in seconds) — best thread count from scaling data
CPU_BEST_K31_SEC = min(CPU_THREAD_DATA.values()) * 60.0  # minutes -> seconds

# Style
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "cpu_naive": "#d62728",   # red
    "sycl_cpu":  "#ff7f0e",   # orange
    "sycl_gpu":  "#2ca02c",   # green
    "cuda_gpu":  "#1f77b4",   # blue
}

# ---------------------------------------------------------------------------
# Plot 1: K31 CPU thread scaling
# ---------------------------------------------------------------------------

def plot_cpu_thread_scaling(outdir):
    threads = sorted(CPU_THREAD_DATA.keys())
    times = [CPU_THREAD_DATA[t] for t in threads]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(threads, times, "o-", color=COLORS["cpu_naive"],
            linewidth=2, markersize=7, label="Naive CPU (k=31)")

    # Annotate a few key points
    for t, tm in zip(threads, times):
        if t in (1, 32, 128, 384):
            ax.annotate(f"{tm:.2f} min",
                        xy=(t, tm), xytext=(8, 8),
                        textcoords="offset points", fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels(threads)
    ax.set_xlabel("Number of threads (log scale)")
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Naive CPU: K31 runtime vs thread count")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "1_cpu_thread_scaling.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plot 2: Best-time comparison across implementations (k27-k31)
# ---------------------------------------------------------------------------

def plot_best_time_comparison(outdir):
    ks = [27, 28, 29, 30, 31]
    sycl_cpu_best = best_total_seconds(SYCL_CPU)
    sycl_gpu_best = best_total_seconds(SYCL_GPU)
    cuda_gpu_best = best_total_seconds(CUDA_GPU)

    series = {
        "SYCL CPU":  [sycl_cpu_best[k] for k in ks],
        "SYCL GPU":  [sycl_gpu_best[k] for k in ks],
        "CUDA GPU":  [cuda_gpu_best[k] for k in ks],
    }
    color_map = {
        "SYCL CPU": COLORS["sycl_cpu"],
        "SYCL GPU": COLORS["sycl_gpu"],
        "CUDA GPU": COLORS["cuda_gpu"],
    }

    x = np.arange(len(ks))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color_map[name])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}s",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Best total time (seconds)")
    ax.set_title("Best-run time comparison: SYCL CPU vs SYCL GPU vs CUDA GPU")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "2_best_time_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plot 3: CUDA GPU stage breakdown (stacked) for k27-k31
# ---------------------------------------------------------------------------

def plot_cuda_stage_breakdown(outdir):
    ks = [27, 28, 29, 30, 31]
    stages = best_stages_seconds(CUDA_GPU)
    t1   = np.array([stages[k][0] for k in ks])
    t2   = np.array([stages[k][1] for k in ks])
    wr   = np.array([stages[k][2] for k in ks])
    tot  = t1 + t2 + wr

    # Proportions for the secondary view
    t1_p = t1 / tot * 100
    t2_p = t2 / tot * 100
    wr_p = wr / tot * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: stacked absolute time
    x = np.arange(len(ks))
    ax1.bar(x, t1, label="Table 1",      color="#8da0cb")
    ax1.bar(x, t2, bottom=t1, label="Sort / Table 2", color="#fc8d62")
    ax1.bar(x, wr, bottom=t1 + t2, label="Write",      color="#66c2a5")
    ax1.set_xticks(x); ax1.set_xticklabels([f"k={k}" for k in ks])
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("CUDA GPU: stage time per k-value (stacked)")
    ax1.legend()

    # Right: proportional view (100% stacked)
    ax2.bar(x, t1_p, label="Table 1",      color="#8da0cb")
    ax2.bar(x, t2_p, bottom=t1_p, label="Sort / Table 2", color="#fc8d62")
    ax2.bar(x, wr_p, bottom=t1_p + t2_p, label="Write",   color="#66c2a5")
    for i in range(len(ks)):
        ax2.text(i, t1_p[i] / 2, f"{t1_p[i]:.0f}%",
                 ha="center", va="center", fontsize=9)
        ax2.text(i, t1_p[i] + t2_p[i] / 2, f"{t2_p[i]:.0f}%",
                 ha="center", va="center", fontsize=9)
        ax2.text(i, t1_p[i] + t2_p[i] + wr_p[i] / 2, f"{wr_p[i]:.0f}%",
                 ha="center", va="center", fontsize=9, color="white")
    ax2.set_xticks(x); ax2.set_xticklabels([f"k={k}" for k in ks])
    ax2.set_ylabel("Share of total time (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("CUDA GPU: stage time proportions per k-value")
    ax2.legend()

    fig.tight_layout()
    path = os.path.join(outdir, "3_cuda_stage_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plot 4a: Speedup factor at k31
# ---------------------------------------------------------------------------

def plot_speedup_at_k31(outdir):
    cpu_naive_sec = CPU_BEST_K31_SEC                   # best thread count
    sycl_cpu_sec  = min(r[3] for r in SYCL_CPU[31])
    sycl_gpu_sec  = min(r[3] for r in SYCL_GPU[31])
    cuda_gpu_sec  = min(r[3] for r in CUDA_GPU[31])

    # Speedup is relative to the slowest baseline: naive CPU single thread (k31)
    cpu_naive_1thread_sec = CPU_THREAD_DATA[1] * 60.0

    impls = ["Naive CPU\n(1 thread)", "Naive CPU\n(best, 384 thr)",
             "SYCL CPU", "SYCL GPU", "CUDA GPU"]
    times = [cpu_naive_1thread_sec, cpu_naive_sec,
             sycl_cpu_sec, sycl_gpu_sec, cuda_gpu_sec]
    speedups = [cpu_naive_1thread_sec / t for t in times]
    colors = [COLORS["cpu_naive"], "#a83232",
              COLORS["sycl_cpu"], COLORS["sycl_gpu"], COLORS["cuda_gpu"]]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(impls, speedups, color=colors)
    for b, s, t in zip(bars, speedups, times):
        ax.annotate(f"{s:.1f}×\n({t:.1f}s)",
                    xy=(b.get_x() + b.get_width() / 2, s),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Speedup factor vs naive CPU (1 thread)")
    ax.set_title("k=31 speedup vs naive single-threaded CPU baseline")
    ax.set_ylim(0, max(speedups) * 1.15)
    fig.tight_layout()
    path = os.path.join(outdir, "4a_speedup_k31.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plot 4b: CPU thread scaling efficiency (actual vs ideal)
# ---------------------------------------------------------------------------

def plot_cpu_scaling_efficiency(outdir):
    threads = sorted(CPU_THREAD_DATA.keys())
    times   = np.array([CPU_THREAD_DATA[t] for t in threads])
    t1      = CPU_THREAD_DATA[1]
    actual_speedup = t1 / times
    ideal_speedup  = np.array(threads, dtype=float)
    efficiency = actual_speedup / ideal_speedup * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Speedup vs threads
    ax1.plot(threads, ideal_speedup,  "--", color="gray",
             label="Ideal (linear)", linewidth=1.5)
    ax1.plot(threads, actual_speedup, "o-", color=COLORS["cpu_naive"],
             label="Actual", linewidth=2, markersize=6)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log", base=2)
    ax1.set_xticks(threads); ax1.set_xticklabels(threads, rotation=45)
    ax1.set_xlabel("Threads"); ax1.set_ylabel("Speedup vs 1 thread")
    ax1.set_title("Naive CPU k=31: actual vs ideal speedup")
    ax1.legend()

    # Efficiency percentage
    ax2.plot(threads, efficiency, "s-", color=COLORS["cpu_naive"],
             linewidth=2, markersize=6)
    ax2.axhline(100, linestyle="--", color="gray", linewidth=1)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(threads); ax2.set_xticklabels(threads, rotation=45)
    ax2.set_xlabel("Threads"); ax2.set_ylabel("Parallel efficiency (%)")
    ax2.set_title("Naive CPU k=31: parallel efficiency")
    ax2.set_ylim(0, 110)

    fig.tight_layout()
    path = os.path.join(outdir, "4b_cpu_scaling_efficiency.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plot 4c: Throughput / scaling with k (log scale)
# ---------------------------------------------------------------------------

def plot_throughput_log(outdir):
    ks_all = [27, 28, 29, 30, 31, 32]

    sycl_cpu_best = best_total_seconds(SYCL_CPU)
    sycl_gpu_best = best_total_seconds(SYCL_GPU)
    cuda_gpu_best = best_total_seconds(CUDA_GPU)

    fig, ax = plt.subplots(figsize=(10, 6))

    # SYCL CPU: k27-32
    sx = ks_all
    sy = [sycl_cpu_best[k] for k in sx]
    ax.plot(sx, sy, "o-", color=COLORS["sycl_cpu"],
            label="SYCL CPU", linewidth=2, markersize=7)

    # SYCL GPU: k27-31
    gx = [27, 28, 29, 30, 31]
    gy = [sycl_gpu_best[k] for k in gx]
    ax.plot(gx, gy, "s-", color=COLORS["sycl_gpu"],
            label="SYCL GPU", linewidth=2, markersize=7)

    # CUDA GPU: k27-31
    cx = [27, 28, 29, 30, 31]
    cy = [cuda_gpu_best[k] for k in cx]
    ax.plot(cx, cy, "^-", color=COLORS["cuda_gpu"],
            label="CUDA GPU", linewidth=2, markersize=7)

    # Single naive-CPU reference point at k31 (best thread count)
    ax.plot([31], [CPU_BEST_K31_SEC], "*", color=COLORS["cpu_naive"],
            markersize=18, label=f"Naive CPU k=31 (best, {CPU_BEST_K31_SEC:.1f}s)")

    ax.set_yscale("log")
    ax.set_xticks(ks_all)
    ax.set_xlabel("k value")
    ax.set_ylabel("Best total time (seconds, log scale)")
    ax.set_title("Time vs k-value across implementations (log scale)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "4c_throughput_logscale.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plot 5: GPU scaling-with-problem-size ratio (CUDA vs SYCL GPU)
# ---------------------------------------------------------------------------

def plot_gpu_scaling_ratio(outdir):
    """For each k -> k+1 step, plot the runtime ratio T(k+1)/T(k).
    Ideal is 2.0x (problem size doubles, time doubles -> linear in N).
    < 2.0x = sublinear, > 2.0x = superlinear."""

    ks = [27, 28, 29, 30, 31]
    sycl_gpu_best = best_total_seconds(SYCL_GPU)
    cuda_gpu_best = best_total_seconds(CUDA_GPU)

    step_labels = [f"k{ks[i-1]}\u2192k{ks[i]}" for i in range(1, len(ks))]
    sycl_ratios = [sycl_gpu_best[ks[i]] / sycl_gpu_best[ks[i-1]]
                   for i in range(1, len(ks))]
    cuda_ratios = [cuda_gpu_best[ks[i]] / cuda_gpu_best[ks[i-1]]
                   for i in range(1, len(ks))]

    x = np.arange(len(step_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shaded band: sublinear region (below ideal 2.0)
    ax.axhspan(0, 2.0, color="#2ca02c", alpha=0.06, zorder=0)
    # Shaded band: superlinear region (above ideal)
    ax.axhspan(2.0, 3.0, color="#d62728", alpha=0.06, zorder=0)
    # Ideal line
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.5,
               label="Ideal linear (2.0\u00d7)", zorder=1)

    bars1 = ax.bar(x - width/2, sycl_ratios, width,
                   label="SYCL GPU", color=COLORS["sycl_gpu"], zorder=2)
    bars2 = ax.bar(x + width/2, cuda_ratios, width,
                   label="CUDA GPU", color=COLORS["cuda_gpu"], zorder=2)

    for b, v in zip(bars1, sycl_ratios):
        ax.annotate(f"{v:.2f}\u00d7",
                    xy=(b.get_x() + b.get_width()/2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    for b, v in zip(bars2, cuda_ratios):
        ax.annotate(f"{v:.2f}\u00d7",
                    xy=(b.get_x() + b.get_width()/2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    # Region labels (placed in clear corner spaces)
    ax.text(0.02, 0.05, "sublinear region\n(better than ideal)",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="#2ca02c",
            fontweight="bold", alpha=0.9,
            bbox=dict(facecolor="white", edgecolor="#2ca02c",
                      alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(0.98, 0.95, "superlinear region\n(worse than ideal)",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#d62728",
            fontweight="bold", alpha=0.9,
            bbox=dict(facecolor="white", edgecolor="#d62728",
                      alpha=0.85, boxstyle="round,pad=0.3"))

    ax.set_xticks(x)
    ax.set_xticklabels(step_labels)
    ax.set_xlabel("k-step (problem size doubles)")
    ax.set_ylabel("Runtime ratio T(k+1) / T(k)")
    ax.set_title("GPU scaling with problem size: SYCL GPU vs CUDA GPU")
    ax.set_ylim(0, max(max(sycl_ratios), max(cuda_ratios)) * 1.15)
    ax.legend(loc="upper left")

    fig.tight_layout()
    path = os.path.join(outdir, "5_gpu_scaling_ratio.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./out", help="output directory")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"writing plots to {os.path.abspath(args.outdir)}")
    plot_cpu_thread_scaling(args.outdir)
    plot_best_time_comparison(args.outdir)
    plot_cuda_stage_breakdown(args.outdir)
    plot_speedup_at_k31(args.outdir)
    plot_cpu_scaling_efficiency(args.outdir)
    plot_throughput_log(args.outdir)
    plot_gpu_scaling_ratio(args.outdir)
    print("done.")

if __name__ == "__main__":
    main()
