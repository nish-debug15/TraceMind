"""
Build the HDBSCAN cluster index from the postmortem dataset.

Usage:
    python -m clustering.build_index

Loads data/postmortems.json, generates embeddings for all records,
runs HDBSCAN clustering, computes cluster centroids, and saves
the index to clustering/cluster_index.pkl.
"""
import json
import pickle
import time
import numpy as np
from pathlib import Path
from collections import Counter

INDEX_PATH = Path(__file__).parent / "cluster_index.pkl"
DATA_PATH = Path(__file__).parent.parent / "data" / "postmortems.json"


def build(data_path: str = None, min_cluster_size: int = 5, min_samples: int = 3):
    """Build the cluster index and save to disk.

    Args:
        data_path: Path to postmortems.json. Defaults to data/postmortems.json.
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        min_samples: HDBSCAN min_samples parameter.
    """
    data_path = Path(data_path) if data_path else DATA_PATH

    print("=" * 60)
    print("CLUSTER-01: Building cluster index")
    print("=" * 60)

    # ---- Step 1: Load data ----
    print(f"\n[1/4] Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records")

    # ---- Step 2: Generate embeddings ----
    print(f"\n[2/4] Generating embeddings (model: all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Combine raw_log_excerpt + root_cause for richer embedding
    texts = []
    for r in records:
        log = r.get("raw_log_excerpt", "")[:1000]
        rc = r.get("root_cause", "")[:1000]
        combined = f"{log} [SEP] {rc}"
        texts.append(combined)

    t0 = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64
    )
    elapsed = time.time() - t0
    print(f"  Embedded {len(texts)} records in {elapsed:.1f}s")
    print(f"  Embedding shape: {embeddings.shape}")

    # ---- Step 3: Run HDBSCAN ----
    print(f"\n[3/4] Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # L2 on normalized vectors ≈ cosine
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )

    t0 = time.time()
    labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - t0
    print(f"  Clustering completed in {elapsed:.1f}s")

    # ---- Compute statistics ----
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    noise_pct = n_noise / len(labels) * 100

    print(f"\n  --- Cluster Statistics ---")
    print(f"  Total records:    {len(labels)}")
    print(f"  Clusters found:   {n_clusters}")
    print(f"  Noise points:     {n_noise} ({noise_pct:.1f}%)")

    # Cluster size distribution
    label_counts = Counter(labels)
    if n_clusters > 0:
        cluster_sizes = [v for k, v in label_counts.items() if k != -1]
        print(f"  Largest cluster:  {max(cluster_sizes)} records")
        print(f"  Smallest cluster: {min(cluster_sizes)} records")
        print(f"  Median cluster:   {int(np.median(cluster_sizes))} records")

        # Show top 5 clusters with sample incident IDs
        print(f"\n  --- Top 5 Largest Clusters ---")
        sorted_clusters = sorted(
            [(k, v) for k, v in label_counts.items() if k != -1],
            key=lambda x: -x[1]
        )
        for cluster_id, size in sorted_clusters[:5]:
            members = [records[i]["incident_id"] for i, l in enumerate(labels) if l == cluster_id]
            sample = members[:3]
            print(f"  Cluster {cluster_id}: {size} records — e.g., {sample}")

    # ---- Step 4: Compute centroids and save ----
    print(f"\n[4/4] Computing cluster centroids and saving index...")

    centroids = []
    cluster_ids = []
    for cid in sorted(unique_labels - {-1}):
        mask = labels == cid
        centroid = embeddings[mask].mean(axis=0)
        # Re-normalize the centroid
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroids.append(centroid)
        cluster_ids.append(int(cid))

    centroids = np.array(centroids, dtype=np.float32) if centroids else np.empty((0, 384), dtype=np.float32)

    index = {
        "centroids": centroids,
        "cluster_ids": cluster_ids,
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(noise_pct, 1),
        "params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "model": "all-MiniLM-L6-v2",
            "n_records": len(records),
        },
        "incident_ids": [r["incident_id"] for r in records],
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    print(f"  Saved index to {INDEX_PATH} ({size_mb:.1f} MB)")
    print(f"\n{'='*60}")
    print(f"Done. {n_clusters} clusters, {noise_pct:.1f}% noise.")
    print(f"{'='*60}")

    return index


if __name__ == "__main__":
    build()
