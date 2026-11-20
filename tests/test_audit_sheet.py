import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHEET_FILE = PROJECT_ROOT / "eval" / "audit" / "relevance_audit_sheet.csv"
KEY_FILE = PROJECT_ROOT / "eval" / "audit" / "relevance_audit_key.json"


def test_audit_sheet_and_key_structure():
    assert SHEET_FILE.exists(), f"Audit sheet {SHEET_FILE} does not exist"
    assert KEY_FILE.exists(), f"Audit key {KEY_FILE} does not exist"

    # 1. Check CSV sheet
    with open(SHEET_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Asserts 30 rows
    assert len(rows) == 30, f"Expected 30 rows, got {len(rows)}"

    # Asserts blinded: no judge score or rationale columns in sheet
    for r in rows:
        assert "judge_score" not in r
        assert "score" not in r
        assert "judge_rationale" not in r
        assert "rationale" not in r
        # human_label and human_notes columns exist and are blank
        assert "human_label" in r
        assert "human_notes" in r
        assert r["human_label"] == ""

    # 2. Check Key JSON
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    assert len(key_data) == 30

    # Key hashes / IDs match sheet pair_ids
    sheet_pair_ids = {r["pair_id"] for r in rows}
    key_pair_ids = set(key_data.keys())
    assert key_pair_ids == sheet_pair_ids, "Key pair_ids do not match sheet pair_ids"

    # 10/10/10 stratum sizes in key
    judge_scores = [v["judge_score"] for v in key_data.values()]
    assert judge_scores.count(0) == 10, f"Expected 10 score-0 in key, got {judge_scores.count(0)}"
    assert judge_scores.count(1) == 10, f"Expected 10 score-1 in key, got {judge_scores.count(1)}"
    assert judge_scores.count(2) == 10, f"Expected 10 score-2 in key, got {judge_scores.count(2)}"
