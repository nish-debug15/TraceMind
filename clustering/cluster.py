def embed(text: str) -> list[float]:
    """Generate embedding for input text. MOCK: returns dummy 384-dim vector."""
    return [0.01] * 384  # MOCK — Pragun replaces with real sentence-transformers embedding

def assign_cluster(text: str) -> dict:
    """Assign input text to nearest cluster. MOCK: returns hardcoded cluster."""
    return {
        "cluster_id": 0,
        "is_noise": False,
        "confidence": 0.87
    }  # MOCK — Pragun replaces with real HDBSCAN assignment
