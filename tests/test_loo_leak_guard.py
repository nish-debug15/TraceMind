"""
Unit tests for Leave-One-Out (LOO) sampler and retrieval leak guard (Block 5).
Verifies:
1. sample.json contains exactly 93 items across eligible companies.
2. retrieval.json contains 93 queries under conditions (a), (b), (c).
3. Leak Guard: Query incident ID and source URL NEVER appear in retrieved hits for (a), (b), or (c).
4. Near-duplicate filter removed >= 1 hit in condition (c) vs (a) across the set.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "eval" / "loo" / "sample.json"
RETRIEVAL_FILE = PROJECT_ROOT / "eval" / "loo" / "retrieval.json"


@pytest.fixture(scope="module")
def sample_data():
    assert SAMPLE_FILE.exists(), f"Missing {SAMPLE_FILE}"
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def retrieval_data():
    assert RETRIEVAL_FILE.exists(), f"Missing {RETRIEVAL_FILE}"
    with open(RETRIEVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_loo_sample_size_and_caps(sample_data):
    sample = sample_data.get("sample", [])
    assert len(sample) == 93, f"Expected 93 sample incidents, got {len(sample)}"

    comp_dist = sample_data.get("metadata", {}).get("company_distribution", {})
    assert len(comp_dist) >= 10, f"Expected >= 10 companies, got {len(comp_dist)}"
    for comp, count in comp_dist.items():
        assert count <= 12, f"Company {comp} exceeds max cap of 12 (count: {count})"


def test_loo_retrieval_item_count(retrieval_data):
    retrieval = retrieval_data.get("retrieval", [])
    assert len(retrieval) == 93, f"Expected 93 retrieval records, got {len(retrieval)}"


def test_loo_leak_guard(retrieval_data):
    """Assert query incident's own ID and source_url NEVER appear in retrieved hits for (a), (b), or (c)."""
    retrieval = retrieval_data.get("retrieval", [])
    for item in retrieval:
        qid = item["query_incident_id"]
        q_url = (item.get("query_source_url") or "").strip()

        for cond_key in [
            "condition_a_same_company_filtered",
            "condition_b_cross_company_filtered",
            "condition_c_same_company_unfiltered"
        ]:
            hits = item[cond_key]["hits"]
            for h in hits:
                cid = h["incident_id"]
                curl = (h.get("source_url") or "").strip()
                assert cid != qid, f"LEAK DETECTED: Query ID {qid} retrieved in {cond_key}"
                if q_url and curl:
                    assert curl != q_url, f"LEAK DETECTED: Query URL {q_url} retrieved in {cond_key} (hit {cid})"


def test_near_duplicate_filter_active(retrieval_data):
    """Near-duplicate filter must have removed >= 1 hit in condition (c) vs (a) across the set."""
    stats = retrieval_data.get("metadata", {}).get("summary_stats", {})
    removed_a = stats.get("condition_a_total_near_duplicates_filtered", 0)
    assert removed_a >= 1, f"Expected >= 1 near duplicates removed in (a), got {removed_a}"

    div_top1 = stats.get("queries_with_divergent_top1_between_a_and_c", 0)
    assert div_top1 >= 1, f"Expected >= 1 queries with divergent top-1 between (a) and (c), got {div_top1}"
