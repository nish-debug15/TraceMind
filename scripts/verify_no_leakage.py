"""
scripts/verify_no_leakage.py
----------------------------
Verifies retrieval against the train index (rag/chroma_db_train):
1. Loads data/splits/test.json (all 93 high-quality incidents).
2. For each test incident, queries rag/chroma_db_train using its raw_log_excerpt (top_k=3).
3. Verifies that NONE of the retrieved incident_ids equal the query incident_id (zero self-retrieval).
4. Verifies that NONE of the retrieved incident_ids belong to the same company as the query incident (zero same-company retrieval).
5. Computes the top-1 retrieval similarity score for each test incident.
6. Prints a formatted summary:
   - Total queries: N
   - Self-retrieval: 0
   - Same-company retrieval: 0
   - Similarity distribution: min, max, mean, median
   - % of test incidents with top-1 similarity < 0.40 (retrieval floor check)
7. Exits with code 0 if all checks pass, non-zero if any leakage detected.
"""

import sys
import json
from pathlib import Path
import numpy as np
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.json"
CHROMA_TRAIN_DIR = PROJECT_ROOT / "rag" / "chroma_db_train"
COLLECTION_NAME = "postmortems_train"


def verify_no_leakage(top_k: int = 3):
    print("=" * 70)
    print("VERIFYING RETRIEVAL AGAINST TRAIN INDEX (NO LEAKAGE CHECKS)")
    print("=" * 70)

    if not TEST_FILE.exists():
        print(f"Error: {TEST_FILE} not found. Run scripts/make_split.py first.")
        sys.exit(1)

    if not CHROMA_TRAIN_DIR.exists():
        print(f"Error: {CHROMA_TRAIN_DIR} not found. Ingest train split first.")
        sys.exit(1)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_records = json.load(f)

    train_company_map = {r["incident_id"]: r.get("company", "") for r in train_records}

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_TRAIN_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    from clustering.cluster import _get_model
    model = _get_model()

    test_logs = [r.get("raw_log_excerpt", "")[:1000] for r in test_records]
    print(f"Embedding {len(test_logs)} test queries...")
    query_embeddings = model.encode(test_logs, batch_size=64, show_progress_bar=False, normalize_embeddings=True).tolist()

    print(f"Querying ChromaDB '{COLLECTION_NAME}' (top_k={top_k})...\n")
    query_results = collection.query(query_embeddings=query_embeddings, n_results=top_k)

    self_retrieval_count = 0
    same_company_retrieval_count = 0
    top1_similarities = []
    leakage_failures = []

    for i, test_r in enumerate(test_records):
        query_id = test_r["incident_id"]
        query_company = test_r["company"]

        retrieved_ids = query_results["ids"][i]
        retrieved_distances = query_results["distances"][i]
        retrieved_metas = query_results["metadatas"][i]

        # Top-1 cosine similarity: 1.0 - (distance / 2.0)
        top1_dist = retrieved_distances[0]
        top1_sim = 1.0 - (top1_dist / 2.0)
        top1_similarities.append(top1_sim)

        # Check all top-k retrieved results for leakage
        for rank, (ret_id, ret_dist, ret_meta) in enumerate(zip(retrieved_ids, retrieved_distances, retrieved_metas), start=1):
            ret_company = ret_meta.get("company") or train_company_map.get(ret_id, "unknown")

            if ret_id == query_id:
                self_retrieval_count += 1
                leakage_failures.append({
                    "type": "SELF_RETRIEVAL",
                    "query_id": query_id,
                    "retrieved_id": ret_id,
                    "rank": rank,
                })

            if ret_company.lower() == query_company.lower():
                same_company_retrieval_count += 1
                leakage_failures.append({
                    "type": "SAME_COMPANY_LEAKAGE",
                    "query_id": query_id,
                    "query_company": query_company,
                    "retrieved_id": ret_id,
                    "retrieved_company": ret_company,
                    "rank": rank,
                })

    min_sim = float(np.min(top1_similarities))
    max_sim = float(np.max(top1_similarities))
    mean_sim = float(np.mean(top1_similarities))
    median_sim = float(np.median(top1_similarities))
    below_floor = [s for s in top1_similarities if s < 0.40]
    below_floor_pct = (len(below_floor) / len(top1_similarities)) * 100

    print("-" * 50)
    print("LEAKAGE VERIFICATION SUMMARY:")
    print("-" * 50)
    print(f"Total queries:                 {len(test_records)}")
    print(f"Self-retrieval:                {self_retrieval_count}")
    print(f"Same-company retrieval:        {same_company_retrieval_count}")
    print(f"Similarity distribution (top-1):")
    print(f"  - Min:                       {min_sim:.4f}")
    print(f"  - Max:                       {max_sim:.4f}")
    print(f"  - Mean:                      {mean_sim:.4f}")
    print(f"  - Median:                    {median_sim:.4f}")
    print(f"Top-1 similarity < 0.40 floor: {len(below_floor)}/{len(top1_similarities)} ({below_floor_pct:.2f}%)")
    print("-" * 50)

    if leakage_failures:
        print("\n[FAILED] Data leakage detected:")
        for fail in leakage_failures:
            print(f"  {fail}")
        sys.exit(1)

    print("\n[PASSED] Zero self-retrieval and zero same-company leakage across all test queries.")
    sys.exit(0)


if __name__ == "__main__":
    verify_no_leakage(top_k=3)
