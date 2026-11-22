"""
Unit tests for E1 random-precedent control analysis (Block 4).
Verifies:
1. JSON exists and has valid metadata.
2. Contains both formats ('baseline', 'gate_v2') and 3 arms each.
3. Contains deltas for real_minus_random, random_minus_zeroshot, real_minus_zeroshot.
4. Contains word counts and length-matched ROUGE-L.
5. Verdict belongs to allowed set {'style_priming', 'precedent_content', 'inconclusive'}.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTROL_FILE = PROJECT_ROOT / "eval" / "results" / "random_precedent_control.json"

ALLOWED_VERDICTS = {"style_priming", "precedent_content", "inconclusive"}


@pytest.fixture(scope="module")
def control_data():
    assert CONTROL_FILE.exists(), f"Missing {CONTROL_FILE}"
    with open(CONTROL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_metadata_schema(control_data):
    meta = control_data.get("metadata", {})
    assert meta.get("num_incidents") == 93
    assert meta.get("num_companies") == 40
    assert meta.get("bootstrap_resamples") == 10000
    assert meta.get("seed") == 42
    assert meta.get("family_size_m") == 5


def test_formats_and_arms(control_data):
    formats = control_data.get("formats", {})
    assert "baseline" in formats
    assert "gate_v2" in formats

    expected_arms = {"zero_shot", "random_precedent", "real_rag"}
    expected_metrics = {"rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "cosine", "length_matched_rougeL"}

    for fmt_name, fmt_data in formats.items():
        arms = fmt_data.get("arms", {})
        assert set(arms.keys()) == expected_arms, f"Unexpected arms in {fmt_name}: {set(arms.keys())}"
        for arm_name, arm_metrics in arms.items():
            for m in expected_metrics:
                assert m in arm_metrics, f"Missing metric {m} in {fmt_name} -> {arm_name}"
                assert "mean" in arm_metrics[m]
                assert "unadjusted_ci_95" in arm_metrics[m]
                assert "adjusted_ci" in arm_metrics[m]
            assert "word_count" in arm_metrics
            assert "mean" in arm_metrics["word_count"]
            assert "median" in arm_metrics["word_count"]


def test_deltas_schema(control_data):
    formats = control_data.get("formats", {})
    expected_deltas = {"real_minus_random", "random_minus_zeroshot", "real_minus_zeroshot"}
    expected_metrics = {"rouge1_f1", "rouge2_f1", "rougeL_f1", "bert_score_f1", "cosine", "length_matched_rougeL"}

    for fmt_name, fmt_data in formats.items():
        deltas = fmt_data.get("deltas", {})
        assert set(deltas.keys()) == expected_deltas, f"Unexpected deltas in {fmt_name}: {set(deltas.keys())}"
        for delta_name, delta_metrics in deltas.items():
            for m in expected_metrics:
                assert m in delta_metrics, f"Missing metric {m} in {fmt_name} -> {delta_name}"
                assert "mean" in delta_metrics[m]
                assert "adjusted_ci" in delta_metrics[m]
                assert "status" in delta_metrics[m]
                assert delta_metrics[m]["status"] in {"excludes zero", "unadjusted only", "crosses zero"}


def test_remediation_both_parsed(control_data):
    formats = control_data.get("formats", {})
    for fmt_name, fmt_data in formats.items():
        rem = fmt_data.get("remediation_both_parsed", {})
        assert "real_minus_random" in rem
        assert "random_minus_zeroshot" in rem
        assert "real_minus_zeroshot" in rem


def test_verdicts(control_data):
    top_verdict = control_data.get("verdict")
    assert top_verdict in ALLOWED_VERDICTS, f"Invalid top verdict: {top_verdict}"

    formats = control_data.get("formats", {})
    for fmt_name, fmt_data in formats.items():
        fmt_verdict = fmt_data.get("verdict")
        assert fmt_verdict in ALLOWED_VERDICTS, f"Invalid verdict for {fmt_name}: {fmt_verdict}"
