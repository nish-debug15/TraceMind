"""
Tests for Block 6: LOO Stratified Deltas
Asserts that eval/results/loo_stratified.json exists,
contains strata for scores 0, 1, 2 with accurate n and n_companies,
and verifies low_n status.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATIFIED_FILE = PROJECT_ROOT / "eval" / "results" / "loo_stratified.json"


def test_loo_stratified_exists_and_valid():
    assert STRATIFIED_FILE.exists()
    with open(STRATIFIED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    strata = data["strata"]
    assert "score_0" in strata
    assert "score_1" in strata
    assert "score_2" in strata

    # Score 0
    s0 = strata["score_0"]
    assert s0["n"] == 31
    assert s0["n_companies"] == 14
    assert s0["low_n"] is False
    assert abs(s0["deltas_rag_a_minus_zeroshot"]["rougeL_f1"]["mean"] - (-0.0146)) < 1e-4

    # Score 1
    s1 = strata["score_1"]
    assert s1["n"] == 48
    assert s1["n_companies"] == 16
    assert s1["low_n"] is False
    assert abs(s1["deltas_rag_a_minus_zeroshot"]["rougeL_f1"]["mean"] - (-0.0068)) < 1e-4

    # Score 2
    s2 = strata["score_2"]
    assert s2["n"] == 14
    assert s2["n_companies"] == 9
    assert s2["low_n"] is False
    assert abs(s2["deltas_rag_a_minus_zeroshot"]["rougeL_f1"]["mean"] - (-0.0379)) < 1e-4
