import json
import time
import numpy as np
from pathlib import Path
import hdbscan

def run():
    DATA_PATH = Path("n:/gitt/TraceMind/data/postmortems.json")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Embeddings
    print("Generating embeddings...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    for r in records:
        log = r.get("raw_log_excerpt", "")[:1000]
        rc = r.get("root_cause", "")[:1000]
        texts.append(f"{log} [SEP] {rc}")
    
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64)

    combos = [(5, 3), (3, 2), (8, 5)]
    results = []

    for min_c, min_s in combos:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_c,
            min_samples=min_s,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))
        noise_pct = n_noise / len(labels) * 100
        results.append((min_c, min_s, n_clusters, n_noise, noise_pct))
        print(f"min_cluster_size={min_c}, min_samples={min_s} -> clusters: {n_clusters}, noise: {n_noise} ({noise_pct:.1f}%)")

if __name__ == "__main__":
    run()
