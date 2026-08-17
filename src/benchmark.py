"""
FPS / latency benchmarking harness.

Per research-plan-review.md Week 4: benchmark on your dev GPU AND on the Jetson
Orin Nano (the actual deployment target on the Sherpa RP) -- the Jetson numbers
are what a reviewer will look for first in a "mobile robots" edge-deployment
paper. This module is hardware-agnostic; run the same script on both machines
and diff the CSVs.
"""

from __future__ import annotations

import csv
import platform
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    label: str          # e.g. "depth-anything-v2-small_indoor_haze_sev3"
    device: str          # "cpu" / "cuda" / "jetson-orin-nano" (set manually per machine)
    hostname: str
    n_warmup: int
    n_runs: int
    mean_latency_s: float
    std_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    fps: float


def benchmark_callable(fn, *args, label: str = "run", device: str = "cpu",
                        n_warmup: int = 3, n_runs: int = 20, **kwargs) -> BenchmarkResult:
    """
    Time `fn(*args, **kwargs)` repeatedly. Use this around a single depth-model
    forward pass (e.g. `model.predict(image)`) -- warmup runs are excluded from
    the stats since first-call latency includes lazy CUDA/graph-compile overhead
    that isn't representative of steady-state operation.
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies)
    mean_lat = float(latencies.mean())

    return BenchmarkResult(
        label=label,
        device=device,
        hostname=platform.node(),
        n_warmup=n_warmup,
        n_runs=n_runs,
        mean_latency_s=mean_lat,
        std_latency_s=float(latencies.std()),
        p50_latency_s=float(np.percentile(latencies, 50)),
        p95_latency_s=float(np.percentile(latencies, 95)),
        fps=float(1.0 / mean_lat) if mean_lat > 0 else float("inf"),
    )


def write_csv(results: list[BenchmarkResult], path: str) -> None:
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


if __name__ == "__main__":
    # Self-test with a dummy workload standing in for a real model forward pass.
    def fake_forward(size: int = 512):
        x = np.random.rand(size, size).astype(np.float32)
        return (x @ x).sum()

    result = benchmark_callable(fake_forward, size=256, label="fake_forward_256",
                                 device="cpu-sandbox-selftest", n_warmup=2, n_runs=10)
    print(result)
    assert result.fps > 0
    assert result.n_runs == 10
    write_csv([result], "data/results/benchmark_selftest.csv")
    print("benchmark.py self-test: OK (wrote data/results/benchmark_selftest.csv)")
