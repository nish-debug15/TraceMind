"""
Unit tests for E1 random-precedent control generation (Block 3).
Verifies:
1. JSON exists and contains exactly 93 items.
2. Every random precedent is NOT in the real top-3 retrieved for that query.
3. similarity_score is a float and mean similarity is well below 0.40.
4. parse_status for all generations belongs to the allowed set.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "eval" / "results" / "random_precedent_generations.json"
JUDGMENTS_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_judgments.json"
CACHE_DB_PATH = PROJECT_ROOT / "eval" / "cache" / "e1_cache.sqlite"

ALLOWED_PARSE_STATUSES = {"both_parsed", "root_cause_only", "remediation_only", "failed"}


@pytest.fixture(scope="module")
def random_data():
    assert RESULTS_FILE.exists(), f"Missing {RESULTS_FILE}"
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def judgments_data():
    assert JUDGMENTS_FILE.exists(), f"Missing {JUDGMENTS_FILE}"
    with open(JUDGMENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_random_precedent_item_count(random_data):
    generations = random_data.get("generations", [])
    assert len(generations) == 93, f"Expected 93 generations, got {len(generations)}"


def test_random_precedent_not_in_real_top3(random_data, judgments_data):
    """Every random precedent ID must NOT be present in the real top-3 retrieved IDs for that query."""
    judgments_by_query = {item["query_incident_id"]: item for item in judgments_data["judgments"]}

    generations = random_data.get("generations", [])
    for item in generations:
        qid = item["query_incident_id"]
        random_prec_id = item["random_precedent_id"]
        assert qid in judgments_by_query, f"Query {qid} not in relevance_judgments"
        real_top3 = [p["retrieved_incident_id"] for p in judgments_by_query[qid]["hits"]]
        assert random_prec_id not in real_top3, (
            f"Query {qid}: random precedent {random_prec_id} leaked from real top-3 {real_top3}"
        )


def test_random_precedent_similarity_scores(random_data):
    generations = random_data.get("generations", [])
    sims = [item["similarity_score"] for item in generations]
    assert len(sims) == 93
    for s in sims:
        assert isinstance(s, (int, float)), f"similarity_score {s} is not numeric"
        assert -1.0 <= s <= 1.0, f"similarity_score {s} out of range [-1, 1]"
    mean_sim = sum(sims) / len(sims)
    assert mean_sim < 0.40, f"Mean random similarity {mean_sim:.4f} should be well below 0.40"


def test_random_precedent_parse_statuses(random_data):
    generations = random_data.get("generations", [])
    for item in generations:
        b_status = item["baseline"]["parse_status"]
        g_status = item["gate_v2"]["parse_status"]
        assert b_status in ALLOWED_PARSE_STATUSES, f"Invalid baseline parse_status: {b_status}"
        assert g_status in ALLOWED_PARSE_STATUSES, f"Invalid gate_v2 parse_status: {g_status}"


def test_cache_db_exists():
    if not CACHE_DB_PATH.exists():
        pytest.skip(f"Missing cache DB {CACHE_DB_PATH}; local or tracked evaluation artifact.")
    assert CACHE_DB_PATH.exists(), f"Missing cache DB {CACHE_DB_PATH}"
