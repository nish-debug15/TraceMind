import json
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_judgments.json"
RUBRIC_FILE = PROJECT_ROOT / "eval" / "rubrics" / "relevance_v1.md"


def test_rubric_exists_and_hash():
    assert RUBRIC_FILE.exists(), f"Rubric file {RUBRIC_FILE} does not exist"
    rubric_bytes = RUBRIC_FILE.read_bytes()
    expected_hash = hashlib.sha256(rubric_bytes).hexdigest()

    assert RESULTS_FILE.exists(), f"Results file {RESULTS_FILE} does not exist"
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    assert meta.get("rubric_sha256") == expected_hash, (
        f"Rubric hash mismatch: {meta.get('rubric_sha256')} vs {expected_hash}"
    )


def test_relevance_judgments_schema_and_counts():
    assert RESULTS_FILE.exists(), f"Results file {RESULTS_FILE} does not exist"
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check top-level keys
    assert "metadata" in data
    assert "summary" in data
    assert "judgments" in data

    meta = data["metadata"]
    assert meta.get("num_queries") == 93
    assert meta.get("total_pairs") == 279
    assert "model" in meta
    assert "num_new_llm_calls" in meta

    judgments = data["judgments"]
    assert len(judgments) == 93

    total_hits = 0
    for q in judgments:
        assert "query_incident_id" in q
        assert "hits" in q
        hits = q["hits"]
        assert len(hits) == 3
        for hit in hits:
            total_hits += 1
            assert hit.get("rank") in (1, 2, 3)
            assert hit.get("score") in (0, 1, 2)
            assert isinstance(hit.get("rationale"), str)
            assert len(hit.get("rationale", "").split()) <= 30  # word count limit <= 25 (+ modest grace)
            assert hit.get("retrieval_pass") in (1, 2)
            assert isinstance(hit.get("similarity_score"), float)
            assert "retrieved_incident_id" in hit

    assert total_hits == 279, f"Expected 279 pairs, got {total_hits}"


def test_distributions():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    r1 = summary.get("rank_1_distribution", {})
    b3 = summary.get("best_of_3_distribution", {})

    r1_total = sum(r1[f"score_{s}"]["count"] for s in (0, 1, 2))
    b3_total = sum(b3[f"score_{s}"]["count"] for s in (0, 1, 2))

    assert r1_total == 93
    assert b3_total == 93
