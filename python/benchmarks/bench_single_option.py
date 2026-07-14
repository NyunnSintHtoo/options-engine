"""
Benchmark single-option BSM pricing: establish baseline latency.

Measures:
- p50, p99 latency (milliseconds)
- Throughput (options/sec)
- Scaling: scalar vs small array vs large array
"""

import time
import numpy as np
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.greeks.analytic import delta, gamma, vega, theta, rho


def benchmark_scalar(func, n_runs=10000):
    """Benchmark function on scalar inputs."""
    S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

    # Warmup
    for _ in range(100):
        func(S, K, T, r, vol)

    # Measure
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(S, K, T, r, vol)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    return {
        "p50": np.percentile(times, 50),
        "p99": np.percentile(times, 99),
        "mean": times.mean(),
        "std": times.std(),
        "throughput": n_runs / times.sum(),  # ops/sec
    }


def benchmark_array(func, array_size=100, n_runs=1000):
    """Benchmark function on array inputs."""
    S = np.ones(array_size) * 100.0
    K = np.ones(array_size) * 100.0
    T = np.ones(array_size) * 1.0
    r = np.ones(array_size) * 0.05
    vol = np.ones(array_size) * 0.2

    # Warmup
    for _ in range(10):
        func(S, K, T, r, vol)

    # Measure
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(S, K, T, r, vol)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return {
        "p50": np.percentile(times, 50),
        "p99": np.percentile(times, 99),
        "mean": times.mean(),
        "std": times.std(),
        "throughput": (array_size * n_runs) / times.sum(),  # ops/sec
    }


def main():
    print("=" * 80)
    print("BENCHMARK: Single-Option BSM Pricing")
    print("=" * 80)

    print("\n### Scalar Pricing ###\n")

    print("BSM Call (scalar):")
    result = benchmark_scalar(bsm_call, n_runs=10000)
    print(f"  p50: {result['p50']:.4f} ms")
    print(f"  p99: {result['p99']:.4f} ms")
    print(f"  Throughput: {result['throughput']:.0f} ops/sec")

    print("\nBSM Put (scalar):")
    result = benchmark_scalar(bsm_put, n_runs=10000)
    print(f"  p50: {result['p50']:.4f} ms")
    print(f"  p99: {result['p99']:.4f} ms")
    print(f"  Throughput: {result['throughput']:.0f} ops/sec")

    print("\nGreeks (scalar) - Delta:")
    result = benchmark_scalar(lambda S, K, T, r, vol: delta(S, K, T, r, vol, "call"), n_runs=10000)
    print(f"  p50: {result['p50']:.4f} ms")
    print(f"  p99: {result['p99']:.4f} ms")
    print(f"  Throughput: {result['throughput']:.0f} ops/sec")

    print("\nGreeks (scalar) - Gamma:")
    result = benchmark_scalar(gamma, n_runs=10000)
    print(f"  p50: {result['p50']:.4f} ms")
    print(f"  p99: {result['p99']:.4f} ms")
    print(f"  Throughput: {result['throughput']:.0f} ops/sec")

    print("\nGreeks (scalar) - Vega:")
    result = benchmark_scalar(vega, n_runs=10000)
    print(f"  p50: {result['p50']:.4f} ms")
    print(f"  p99: {result['p99']:.4f} ms")
    print(f"  Throughput: {result['throughput']:.0f} ops/sec")

    print("\n### Vectorized Pricing ###\n")

    for array_size in [10, 100, 1000]:
        print(f"BSM Call (array size={array_size}):")
        result = benchmark_array(bsm_call, array_size=array_size, n_runs=1000)
        per_option = result["mean"] / array_size
        print(f"  p50 total: {result['p50']:.4f} ms, per option: {per_option:.4f} ms")
        print(f"  p99 total: {result['p99']:.4f} ms, per option: {result['p99'] / array_size:.4f} ms")
        print(f"  Throughput: {result['throughput']:.0f} ops/sec")


if __name__ == "__main__":
    main()
