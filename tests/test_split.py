"""
tests/test_split.py
-------------------
Pytest suite verifying that the train/test split is strictly leakage-free,
company-grouped, high-quality, within sizing bounds, and deterministic.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.make_split import (
    TRAIN_SPLIT_FILE,
    TEST_SPLIT_FILE,
    REPORT_FILE,
    DATA_PATH,
    generate_split,
)
from clustering.cluster import _get_model


def _load_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_split_files_exist():
    """train.json, test.json, split_report.json all exist and are non-empty."""
    for path in (TRAIN_SPLIT_FILE, TEST_SPLIT_FILE, REPORT_FILE):
        assert path.exists(), f"File {path} does not exist"
        assert path.stat().st_size > 0, f"File {path} is empty"

    train_data = _load_json(TRAIN_SPLIT_FILE)
    test_data = _load_json(TEST_SPLIT_FILE)
    report = _load_json(REPORT_FILE)

    assert len(train_data) > 0, "train.json has 0 records"
    assert len(test_data) > 0, "test.json has 0 records"
    assert isinstance(report, dict) and len(report) > 0, "split_report.json is invalid"


def test_no_incident_id_overlap():
    """set(train_ids) & set(test_ids) == set()"""
    train_data = _load_json(TRAIN_SPLIT_FILE)
    test_data = _load_json(TEST_SPLIT_FILE)

    train_ids = {r["incident_id"] for r in train_data}
    test_ids = {r["incident_id"] for r in test_data}

    overlap = train_ids & test_ids
    assert overlap == set(), f"Found {len(overlap)} overlapping incident_ids: {overlap}"


def test_no_company_overlap():
    """set(train_companies) & set(test_companies) == set()"""
    train_data = _load_json(TRAIN_SPLIT_FILE)
    test_data = _load_json(TEST_SPLIT_FILE)

    train_companies = {r["company"] for r in train_data}
    test_companies = {r["company"] for r in test_data}

    overlap = train_companies & test_companies
    assert overlap == set(), f"Found {len(overlap)} overlapping companies: {overlap}"


def test_test_set_all_high_quality():
    """All test records must have quality == 'high'."""
    test_data = _load_json(TEST_SPLIT_FILE)
    non_high = [r for r in test_data if r.get("quality") != "high"]
    assert len(non_high) == 0, f"Found {len(non_high)} records in test set with non-high quality: {non_high}"


def test_split_sizes():
    """
    - test is within 15-20% of high-quality corpus (~82-109 incidents).
    - train has expected count.
    - no single company > 25% of test set.
    """
    corpus = _load_json(DATA_PATH)
    total_high = sum(1 for r in corpus if r.get("quality") == "high")
    target_min = int(round(0.15 * total_high))
    target_max = int(round(0.20 * total_high))

    test_data = _load_json(TEST_SPLIT_FILE)
    train_data = _load_json(TRAIN_SPLIT_FILE)
    report = _load_json(REPORT_FILE)

    # Test size check
    test_count = len(test_data)
    assert target_min <= test_count <= target_max, (
        f"Test set size {test_count} outside target range [{target_min}, {target_max}]"
    )

    # Expected train count: 828 - test_records - excluded_test_low == len(train_data)
    assert len(train_data) == report["train_records"], (
        f"Train records count mismatch: {len(train_data)} vs report {report['train_records']}"
    )

    # Max single company share in test set capped at <= 25%
    company_counts = {}
    for r in test_data:
        company_counts[r["company"]] = company_counts.get(r["company"], 0) + 1

    max_comp_count = max(company_counts.values())
    max_comp_share = (max_comp_count / test_count) * 100
    assert max_comp_share <= 25.0, (
        f"Company exceeds 25% of test set: {max_comp_count}/{test_count} ({max_comp_share:.2f}%)"
    )


def test_no_near_duplicate_cross_split():
    """
    Compute pairwise cosine similarity of root_cause between test and train records.
    Assert max similarity < 0.95.
    Fail with top 10 most similar pairs if violated.
    """
    train_data = _load_json(TRAIN_SPLIT_FILE)
    test_data = _load_json(TEST_SPLIT_FILE)

    model = _get_model()
    train_rcs = [r.get("root_cause", "")[:500] for r in train_data]
    test_rcs = [r.get("root_cause", "")[:500] for r in test_data]

    train_embs = model.encode(train_rcs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    test_embs = model.encode(test_rcs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    sim_matrix = np.dot(test_embs, train_embs.T)
    max_sim = float(np.max(sim_matrix))

    if max_sim >= 0.95:
        # Extract top 10 most similar pairs
        flat_indices = np.argsort(sim_matrix.ravel())[::-1][:10]
        top_10 = []
        for idx in flat_indices:
            t_idx, tr_idx = np.unravel_index(idx, sim_matrix.shape)
            top_10.append({
                "sim": float(sim_matrix[t_idx, tr_idx]),
                "test_id": test_data[t_idx]["incident_id"],
                "test_company": test_data[t_idx]["company"],
                "train_id": train_data[tr_idx]["incident_id"],
                "train_company": train_data[tr_idx]["company"],
                "test_rc": test_data[t_idx]["root_cause"][:100],
                "train_rc": train_data[tr_idx]["root_cause"][:100]
            })
        pytest.fail(f"Max cross-split similarity {max_sim:.4f} >= 0.95. Top 10 pairs:\n{json.dumps(top_10, indent=2)}")

    assert max_sim < 0.95, f"Max similarity {max_sim} >= 0.95"


def test_split_is_deterministic():
    """Re-running make_split with seed 42 produces byte-identical files."""
    train_bytes_before = TRAIN_SPLIT_FILE.read_bytes()
    test_bytes_before = TEST_SPLIT_FILE.read_bytes()
    report_bytes_before = REPORT_FILE.read_bytes()

    generate_split(seed=42)

    assert TRAIN_SPLIT_FILE.read_bytes() == train_bytes_before, "train.json changed after re-running with seed 42"
    assert TEST_SPLIT_FILE.read_bytes() == test_bytes_before, "test.json changed after re-running with seed 42"
    assert REPORT_FILE.read_bytes() == report_bytes_before, "split_report.json changed after re-running with seed 42"
