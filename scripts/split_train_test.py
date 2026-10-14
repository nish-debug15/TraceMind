"""
Create a stratified train/test split (80/20) of data/postmortems.json based on quality.
Ensures zero data leakage between ChromaDB retrieval index and evaluation test set.
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SRC_FILE = DATA_DIR / "postmortems.json"
TRAIN_FILE = DATA_DIR / "train.json"
TEST_FILE = DATA_DIR / "test.json"

def split_dataset(test_ratio: float = 0.20, seed: int = 42):
    with open(SRC_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Group by quality
    by_quality = {}
    for r in records:
        q = r.get("quality", "high")
        by_quality.setdefault(q, []).append(r)

    rng = random.Random(seed)
    train_records = []
    test_records = []

    print(f"Total records in source: {len(records)}")
    for q, group in by_quality.items():
        rng.shuffle(group)
        n_test = int(round(len(group) * test_ratio))
        test_subset = group[:n_test]
        train_subset = group[n_test:]
        print(f"Quality '{q}': total={len(group)} -> train={len(train_subset)}, test={len(test_subset)}")
        train_records.extend(train_subset)
        test_records.extend(test_subset)

    # Shuffle both splits with deterministic seed
    rng.shuffle(train_records)
    rng.shuffle(test_records)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    print(f"\nSaved {len(train_records)} records to {TRAIN_FILE}")
    print(f"Saved {len(test_records)} records to {TEST_FILE}")

    # Double check intersection of incident_id
    train_ids = {r["incident_id"] for r in train_records}
    test_ids = {r["incident_id"] for r in test_records}
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Critical leakage! Overlap detected: {overlap}"
    print("Leakage verification passed: 0 incident IDs shared between train and test.")

if __name__ == "__main__":
    split_dataset()
