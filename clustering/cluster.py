"""
TraceMind Clustering Module — BLOCK-CLUSTER-01

Generates sentence-transformer embeddings and assigns incoming log text
to pre-computed HDBSCAN clusters of historical postmortem data.

Public API (imported by backend):
    embed(text: str) -> list[float]
    assign_cluster(text: str) -> dict
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_model = None
_cluster_index: Optional[dict] = None

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path(__file__).parent / "cluster_index.pkl"
DATA_PATH = Path(__file__).parent.parent / "data" / "postmortems.json"


def _get_model():
    """Lazy-load the sentence-transformers model (cached after first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    """Generate a 384-dimensional embedding for the input text.

    Args:
        text: Raw log text or incident description.

    Returns:
        List of 384 floats — the dense embedding vector.
    """
    model = _get_model()
    # Truncate very long inputs to avoid OOM (model max is 256 tokens)
    text = text[:2000]
    vec = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return vec.tolist()


def _load_cluster_index() -> dict:
    """Load the pre-computed cluster index from disk.
    Auto-builds if the pickle file is missing.
    """
    global _cluster_index
    if _cluster_index is not None:
        return _cluster_index

    if not INDEX_PATH.exists():
        print(f"[clustering] cluster_index.pkl not found at {INDEX_PATH}. Building now...")
        from clustering.build_index import build
        build()

    with open(INDEX_PATH, "rb") as f:
        _cluster_index = pickle.load(f)
    return _cluster_index


def assign_cluster(text: str) -> dict:
    """Assign input text to the nearest existing cluster.

    Embeds the input, computes cosine similarity against all cluster
    centroids, and returns the best match.

    Args:
        text: Raw log text to classify.

    Returns:
        dict with keys:
            cluster_id (int): ID of the nearest cluster (-1 = noise/novel)
            is_noise (bool): True if the input doesn't match any cluster well
            confidence (float): Cosine similarity to the nearest centroid (0-1)
    """
    index = _load_cluster_index()
    centroids = index["centroids"]        # shape: (n_clusters, 384)
    cluster_ids = index["cluster_ids"]    # list of cluster IDs (matching centroid rows)

    # Embed the input
    vec = np.array(embed(text), dtype=np.float32)

    if len(centroids) == 0:
        return {"cluster_id": -1, "is_noise": True, "confidence": 0.0}

    # Cosine similarity (embeddings are already L2-normalized)
    similarities = centroids @ vec
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    # Threshold: if similarity is below 0.3, treat as noise/novel
    NOISE_THRESHOLD = 0.3
    if best_score < NOISE_THRESHOLD:
        return {
            "cluster_id": -1,
            "is_noise": True,
            "confidence": round(best_score, 4)
        }

    return {
        "cluster_id": int(cluster_ids[best_idx]),
        "is_noise": False,
        "confidence": round(best_score, 4)
    }
