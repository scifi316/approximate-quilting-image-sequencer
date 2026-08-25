"""Benchmark candidate Faiss index types for this pipeline's actual usage
pattern: build once per dataset change, then many batched top_k searches per
target frame (the access pattern of stitch.matchFeaturesBatch).

Usage:
    python benchmarks/faiss_index_benchmark.py --dataset synthetic
    python benchmarks/faiss_index_benchmark.py --dataset real

--dataset real reconstructs descriptors from the already-built
individual_descriptors_faiss_index.bin at the repo root (run
src/database/build_database.py first if it doesn't exist yet).
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

DESCRIPTOR_DIM = 128
TOP_K = 7  # matches stitch.matchFeaturesBatch's default

# Faiss's IVF/PQ .train() re-scans the entire training set on every k-means
# iteration (25 by default) -- training on all 683K real descriptors was
# taking many minutes for no benefit, since a representative subsample
# trains equally good centroids. Cap it and add() the full dataset after.
TRAIN_SAMPLE_CAP = 200_000


def make_synthetic_descriptors(num_vectors=20000, dim=DESCRIPTOR_DIM, seed=0):
    """Clustered synthetic descriptors -- SIFT descriptors from real footage
    aren't uniform noise, they cluster around recurring visual patterns, and
    approximate indexes only pay off when there's real structure to exploit."""
    rng = np.random.default_rng(seed)
    num_clusters = 50
    centers = rng.uniform(0, 200, size=(num_clusters, dim)).astype("float32")
    assignments = rng.integers(0, num_clusters, size=num_vectors)
    noise = rng.normal(0, 10, size=(num_vectors, dim)).astype("float32")
    vectors = centers[assignments] + noise
    return np.clip(vectors, 0, None).astype("float32")


def load_real_descriptors(database_dir):
    index_path = Path(database_dir) / "individual_descriptors_faiss_index.bin"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- build the database first "
            f"(python src/database/build_database.py)"
        )
    flat_index = faiss.read_index(str(index_path))
    return flat_index.reconstruct_n(0, flat_index.ntotal)


def make_query_batch(descriptors, num_queries, seed=1):
    """Sample query vectors mimicking one target frame's worth of chunk
    descriptors handed to matchFeaturesBatch in a single search() call."""
    rng = np.random.default_rng(seed)
    size = min(num_queries, len(descriptors))
    idx = rng.choice(len(descriptors), size=size, replace=False)
    return descriptors[idx].copy()


def build_index(index_type, descriptors, use_gpu=False):
    """Construct and populate a Faiss index of the given type. Returns
    (index, build_time_seconds, gpu_resources). gpu_resources must be kept
    alive by the caller for as long as the index is used."""
    dim = descriptors.shape[1]
    n = len(descriptors)

    if index_type == "flat":
        index = faiss.IndexFlatL2(dim)
    elif index_type == "ivfflat":
        nlist = max(1, min(4096, int(4 * (n ** 0.5))))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)
    elif index_type == "ivfpq":
        nlist = max(1, min(4096, int(4 * (n ** 0.5))))
        m = 16  # sub-quantizers; DESCRIPTOR_DIM (128) must be divisible by m
        index = faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, nlist, m, 8)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    gpu_res = None
    if use_gpu:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    start = time.perf_counter()
    if not index.is_trained:
        if n > TRAIN_SAMPLE_CAP:
            rng = np.random.default_rng(2)
            sample = descriptors[rng.choice(n, size=TRAIN_SAMPLE_CAP, replace=False)]
        else:
            sample = descriptors
        index.train(sample)
    index.add(descriptors)
    build_time = time.perf_counter() - start

    if hasattr(index, "nprobe"):
        index.nprobe = 16

    return index, build_time, gpu_res


def index_size_bytes(index, use_gpu):
    cpu_index = faiss.index_gpu_to_cpu(index) if use_gpu else index
    fd, tmp_path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        faiss.write_index(cpu_index, tmp_path)
        return os.path.getsize(tmp_path)
    finally:
        os.unlink(tmp_path)


def recall_at_k(candidate_indices, baseline_indices):
    """Fraction of the exact baseline's top-k neighbors also returned by the
    candidate index, averaged across queries."""
    hits, total = 0, 0
    for cand_row, base_row in zip(candidate_indices, baseline_indices):
        base_set = {int(i) for i in base_row if i >= 0}
        if not base_set:
            continue
        cand_set = {int(i) for i in cand_row if i >= 0}
        hits += len(cand_set & base_set)
        total += len(base_set)
    return hits / total if total else float("nan")


def benchmark_index_type(index_type, descriptors, queries, baseline_indices, use_gpu=False, repeats=5):
    index, build_time, _gpu_res = build_index(index_type, descriptors, use_gpu=use_gpu)

    # Warm up + measure batched query latency, mirroring matchFeaturesBatch's
    # single search() call across an entire target frame's chunk descriptors.
    latencies = []
    result_indices = None
    for _ in range(repeats):
        start = time.perf_counter()
        _, result_indices = index.search(queries, TOP_K)
        latencies.append(time.perf_counter() - start)

    return {
        "index_type": index_type + ("+gpu" if use_gpu else ""),
        "build_time_s": build_time,
        "size_bytes": index_size_bytes(index, use_gpu),
        "avg_query_latency_s": sum(latencies) / len(latencies),
        "recall_at_k": recall_at_k(result_indices, baseline_indices),
    }


def run_benchmark(descriptors, num_queries, candidates, repeats=3):
    queries = make_query_batch(descriptors, num_queries)
    print(f"Descriptors: {len(descriptors)}, dim: {descriptors.shape[1]}, queries: {len(queries)}\n", flush=True)

    print("Building exact IndexFlatL2 baseline for recall comparison...", flush=True)
    baseline_start = time.perf_counter()
    baseline_index, _, _ = build_index("flat", descriptors)
    _, baseline_indices = baseline_index.search(queries, TOP_K)
    print(f"  done in {time.perf_counter() - baseline_start:.1f}s", flush=True)

    results = []
    for index_type, use_gpu in candidates:
        label = index_type + ("+gpu" if use_gpu else "")
        print(f"Benchmarking {label}...", flush=True)
        start = time.perf_counter()
        results.append(
            benchmark_index_type(index_type, descriptors, queries, baseline_indices,
                                  use_gpu=use_gpu, repeats=repeats)
        )
        print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)
    return results


def print_table(results):
    header = f"{'index_type':<14}{'build_s':>10}{'size_MB':>10}{'query_ms':>10}{'recall@7':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['index_type']:<14}{r['build_time_s']:>10.2f}{r['size_bytes'] / 1e6:>10.1f}"
              f"{r['avg_query_latency_s'] * 1000:>10.2f}{r['recall_at_k']:>10.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--database-dir", default=str(ROOT_DIR),
                         help="Directory containing individual_descriptors_faiss_index.bin (--dataset real)")
    parser.add_argument("--num-queries", type=int, default=2000,
                         help="Query batch size, approximating one target frame's chunk descriptors")
    parser.add_argument("--synthetic-size", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3, help="Query batches per candidate, for latency averaging")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="Faiss OpenMP thread count")
    parser.add_argument("--json", default=None, help="Optional path to dump results as JSON")
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)
    print(f"Faiss threads: {faiss.omp_get_max_threads()}")

    if args.dataset == "synthetic":
        descriptors = make_synthetic_descriptors(num_vectors=args.synthetic_size)
    else:
        descriptors = load_real_descriptors(args.database_dir)

    gpu_available = faiss.get_num_gpus() > 0
    print(f"GPU available: {gpu_available}")

    candidates = [
        ("flat", False),
        ("ivfflat", False),
        ("hnsw", False),
        ("ivfpq", False),
    ]
    if gpu_available:
        candidates += [("flat", True), ("ivfflat", True)]

    results = run_benchmark(descriptors, args.num_queries, candidates, repeats=args.repeats)
    print_table(results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
