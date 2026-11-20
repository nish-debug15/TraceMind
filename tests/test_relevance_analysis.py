import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_analysis.json"


def test_relevance_analysis_keys_and_schema():
    assert ANALYSIS_FILE.exists(), f"Analysis file {ANALYSIS_FILE} does not exist"
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level keys
    assert "exploratory" in data
    assert data["exploratory"] is True, "Expected exploratory: true"

    assert "metadata" in data
    assert "relevance_rates" in data
    assert "similarity_prediction" in data
    assert "stratified_gain_by_relevance" in data
    assert "guardrail_check" in data

    # Check relevance rates
    rr = data["relevance_rates"]
    for k in ["relevance_at_1_score_2", "relevance_at_1_score_ge_1", "best_of_3_score_2", "best_of_3_score_ge_1"]:
        assert k in rr
        assert 0.0 <= rr[k]["fraction"] <= 1.0
        assert "company_clustered_ci_95" in rr[k]

    # Check similarity prediction & AUROC
    sp = data["similarity_prediction"]
    assert "spearman_rho" in sp
    assert "auroc_relevance_ge_1" in sp
    auroc = sp["auroc_relevance_ge_1"]
    assert 0.0 <= auroc <= 1.0, f"AUROC {auroc} not in [0, 1]"

    # Check stratified gains
    sg = data["stratified_gain_by_relevance"]
    assert "score_0" in sg
    assert "score_1" in sg
    assert "score_2" in sg
    assert sg["score_2"]["low_n"] is True
    assert sg["score_0"]["low_n"] is False

    # Check guardrail check
    gc = data["guardrail_check"]
    assert "abstained_n7" in gc
    assert "answered_n86" in gc
    assert gc["abstained_n7"]["n"] == 7
    assert gc["answered_n86"]["n"] == 86
