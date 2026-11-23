"""
Tests for Block 5: LOO Retrieval Consistency
Asserts that eval/results/loo_consistency.json exists,
all 8 divergent queries are explained by near-duplicate filtering,
and the similarity and Jaccard distributions match expected ranges.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSISTENCY_FILE = PROJECT_ROOT / "eval" / "results" / "loo_consistency.json"


def test_loo_consistency_report_exists():
    assert CONSISTENCY_FILE.exists()
    with open(CONSISTENCY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    recon = data["reconciliation"]
    assert recon["condition_a_total_near_duplicates_filtered"] == 11
    assert recon["condition_b_total_near_duplicates_filtered"] == 0
    assert recon["queries_with_divergent_top3_between_a_and_c"] == 8
    assert recon["queries_with_divergent_top1_between_a_and_c"] == 7
    assert recon["unexplained_differences_count"] == 0
    assert "CONSISTENT" in recon["status"]


def test_loo_consistency_distributions():
    with open(CONSISTENCY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    dist_cos = data["distributions_condition_c_top1"]["cosine_similarity"]
    assert dist_cos["min"] == 0.4056
    assert dist_cos["median"] == 0.6869
    assert dist_cos["max"] == 0.9905
    assert dist_cos["count_ge_0_90"] == 6
    assert dist_cos["count_ge_0_95"] == 2

    dist_jac = data["distributions_condition_c_top1"]["token_5gram_jaccard"]
    assert dist_jac["min"] == 0.0
    assert dist_jac["max"] == 1.0
    assert dist_jac["count_ge_0_80"] == 7
