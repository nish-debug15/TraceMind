"""
Tests for Block 2: Scorer Regression Pin
Pins the canonical scorer module outputs against the previously committed numbers
for the 4 control_2x2 cells within 1e-4 tolerance.
"""

import json
from pathlib import Path
import pytest
from eval.scorer import score_cell

try:
    import bert_score
except ImportError:
    bert_score = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"
COMMITTED_2X2_FILE = PROJECT_ROOT / "eval" / "results" / "control_2x2.json"


@pytest.fixture(scope="module")
def control_2x2_data():
    if bert_score is None:
        pytest.skip("bert-score not installed; skipping scorer regression tests.")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)
    gt_map = {r["incident_id"]: r["root_cause"] for r in test_records}
    valid_ids = [r["incident_id"] for r in test_records]
    ground_truth = [gt_map[i] for i in valid_ids]

    with open(CACHE_DIR / "zeroshot_generations.json", "r", encoding="utf-8") as f:
        cell_b_zs = json.load(f)
    with open(CACHE_DIR / "rag_generations.json", "r", encoding="utf-8") as f:
        cell_b_rag = json.load(f)
    with open(CACHE_DIR / "ablation_zero_shot_gate_format.json", "r", encoding="utf-8") as f:
        cell_g_zs = json.load(f)
    with open(CACHE_DIR / "ablation_relevance_gate_v2.json", "r", encoding="utf-8") as f:
        cell_g_rag = json.load(f)

    with open(COMMITTED_2X2_FILE, "r", encoding="utf-8") as f:
        committed = json.load(f)["cells"]

    cells_raw = {
        "baseline_zeroshot": [cell_b_zs[i]["root_cause"] for i in valid_ids],
        "baseline_rag": [cell_b_rag[i]["root_cause"] for i in valid_ids],
        "gate_v2_zeroshot": [cell_g_zs[i]["root_cause"] for i in valid_ids],
        "gate_v2_rag": [cell_g_rag[i]["root_cause"] for i in valid_ids],
    }
    return ground_truth, cells_raw, committed


@pytest.mark.parametrize("cell_name", [
    "baseline_zeroshot",
    "baseline_rag",
    "gate_v2_zeroshot",
    "gate_v2_rag",
])
def test_control_2x2_regression_pins(control_2x2_data, cell_name):
    ground_truth, cells_raw, committed = control_2x2_data
    preds = cells_raw[cell_name]
    expected = committed[cell_name]

    result = score_cell(preds, ground_truth)
    summary = result["summary"]

    assert abs(summary["rougeL_f1"] - expected["rougeL_f1"]["mean"]) <= 1e-4, (
        f"{cell_name} ROUGE-L mismatch: {summary['rougeL_f1']} vs {expected['rougeL_f1']['mean']}"
    )
    assert abs(summary["bert_score_f1"] - expected["bert_score_f1"]["mean"]) <= 1e-4, (
        f"{cell_name} BERTScore mismatch: {summary['bert_score_f1']} vs {expected['bert_score_f1']['mean']}"
    )
    assert abs(summary["cosine"] - expected["cosine"]["mean"]) <= 1e-4, (
        f"{cell_name} Cosine mismatch: {summary['cosine']} vs {expected['cosine']['mean']}"
    )
    assert abs(summary["word_count"]["mean"] - expected["word_count"]["mean"]) <= 1e-4, (
        f"{cell_name} WC Mean mismatch: {summary['word_count']['mean']} vs {expected['word_count']['mean']}"
    )
    assert abs(summary["word_count"]["median"] - expected["word_count"]["median"]) <= 1e-4, (
        f"{cell_name} WC Median mismatch: {summary['word_count']['median']} vs {expected['word_count']['median']}"
    )
