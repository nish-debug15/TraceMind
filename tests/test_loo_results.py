"""
Unit tests for Leave-One-Out (LOO) regime results (Block 6).
Verifies:
1. JSON file exists and contains valid metadata.
2. Top-level regime string is exactly 'leave-one-out on train companies; NOT comparable to company-disjoint test numbers'.
3. Relevance@1 distributions exist for conditions (a), (b), (c) and sum to 93.
4. Formats baseline and gate_v2 contain arms, deltas, and remediation.
5. Verdict is in allowed set {'RAG helps when same-company history exists', 'no detected effect'}.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "eval" / "results" / "loo_regime_results.json"

ALLOWED_VERDICTS = {"RAG helps when same-company history exists", "no detected effect"}
EXPECTED_REGIME = "leave-one-out on train companies; NOT comparable to company-disjoint test numbers"


@pytest.fixture(scope="module")
def loo_results():
    assert RESULTS_FILE.exists(), f"Missing {RESULTS_FILE}"
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_regime_string(loo_results):
    assert loo_results.get("regime") == EXPECTED_REGIME, (
        f"Expected regime string '{EXPECTED_REGIME}', got '{loo_results.get('regime')}'"
    )


def test_metadata_schema(loo_results):
    meta = loo_results.get("metadata", {})
    assert meta.get("num_incidents") == 93
    assert meta.get("num_companies") >= 10
    assert meta.get("family_size_m") == 5
    assert meta.get("bootstrap_resamples") == 10000


def test_relevance_at_1_distributions(loo_results):
    rel = loo_results.get("relevance_at_1", {})
    assert "condition_a" in rel
    assert "condition_b" in rel
    assert "condition_c" in rel

    for cond_name, cdist in rel.items():
        assert cdist.get("num_judged") == 93
        total_counts = (
            cdist["score_0_unrelated"]["count"]
            + cdist["score_1_partially_relevant"]["count"]
            + cdist["score_2_highly_relevant"]["count"]
        )
        assert total_counts == 93, f"{cond_name}: score counts sum to {total_counts}, expected 93"
        assert 0.0 <= cdist["mean_relevance_score"] <= 2.0


def test_formats_and_deltas(loo_results):
    formats = loo_results.get("formats", {})
    assert "baseline" in formats
    assert "gate_v2" in formats

    expected_arms = {"zero_shot", "rag_condition_a_same_company", "rag_condition_b_cross_company"}
    expected_deltas = {"rag_a_minus_zeroshot", "rag_b_minus_zeroshot", "rag_a_minus_rag_b_same_company_premium"}
    expected_metrics = {"rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "cosine", "length_matched_rougeL"}

    for fmt_name, fmt_data in formats.items():
        arms = fmt_data.get("arms", {})
        assert set(arms.keys()) == expected_arms, f"Unexpected arms in {fmt_name}: {set(arms.keys())}"
        for arm_name, arm_metrics in arms.items():
            for m in expected_metrics:
                assert m in arm_metrics, f"Missing metric {m} in {fmt_name} -> {arm_name}"

        deltas = fmt_data.get("deltas", {})
        assert set(deltas.keys()) == expected_deltas, f"Unexpected deltas in {fmt_name}: {set(deltas.keys())}"
        for dname, dmetrics in deltas.items():
            for m in expected_metrics:
                assert m in dmetrics, f"Missing metric {m} in {fmt_name} -> {dname}"
                assert "adjusted_ci" in dmetrics[m]
                assert "status" in dmetrics[m]


def test_verdict(loo_results):
    verdict = loo_results.get("verdict")
    assert verdict in ALLOWED_VERDICTS, f"Invalid verdict: {verdict}"
