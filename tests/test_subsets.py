"""
tests/test_subsets.py
---------------------
Tests asserting the statistical and algorithmic independence of the
retrieval abstention mask (sim < 0.40) and the ground-truth boilerplate noise mask.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
RVZ_FILE = PROJECT_ROOT / "eval" / "results" / "rag_vs_zero_shot.json"
GT_FLAGS_FILE = PROJECT_ROOT / "eval" / "results" / "gt_noise_flags.json"
OVERLAP_FILE = PROJECT_ROOT / "eval" / "results" / "subset_overlap.json"


def test_clean_subset_computed_independently():
    """Assert that clean == all_ids minus gt_noise_flagged_ids computed independently of the abstention mask."""
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    all_ids = set(r["incident_id"] for r in test_records)

    with open(GT_FLAGS_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    flagged_ids = set(item["incident_id"] for item in gt_data["flagged_incidents"])

    expected_clean_ids = all_ids - flagged_ids

    # Load from overlap audit
    with open(OVERLAP_FILE, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    actual_clean_ids = set(audit_data["id_lists"]["clean"])
    assert actual_clean_ids == expected_clean_ids, "Clean set must strictly equal all_ids minus flagged_ids"
    assert len(flagged_ids) == 7
    assert len(actual_clean_ids) == 86


def test_abstained_and_flagged_are_not_identical():
    """Verify that abstained and gt_noise_flagged are not identical (stopping condition check)."""
    with open(OVERLAP_FILE, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    abstained = set(audit_data["id_lists"]["abstained"])
    flagged = set(audit_data["id_lists"]["gt_noise_flagged"])
    answered = set(audit_data["id_lists"]["answered"])
    clean = set(audit_data["id_lists"]["clean"])

    # Must NOT be identical
    assert abstained != flagged, "Abstained and flagged sets must not be identical"
    assert clean != answered, "Clean and answered sets must not be identical"

    # Confirm complete disjointness between abstained and flagged
    assert abstained.isdisjoint(flagged), "Abstained and flagged sets must be completely disjoint"
    assert len(abstained.intersection(flagged)) == 0


def test_partition_completeness():
    """Verify that both partitions form complete, disjoint partitions of the 93 test incidents."""
    with open(OVERLAP_FILE, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    all_count = audit_data["total_test_incidents"]
    assert all_count == 93

    abstained = set(audit_data["id_lists"]["abstained"])
    answered = set(audit_data["id_lists"]["answered"])
    assert len(abstained.union(answered)) == 93
    assert abstained.isdisjoint(answered)

    flagged = set(audit_data["id_lists"]["gt_noise_flagged"])
    clean = set(audit_data["id_lists"]["clean"])
    assert len(flagged.union(clean)) == 93
    assert flagged.isdisjoint(clean)
