"""
TraceMind Manual Relevance Audit Sheet Generator (Block 5)
-----------------------------------------------------------
Samples 30 (query, precedent) pairs from the 279 judged pairs (seed 42),
stratified into 10 score-0, 10 score-1, and 10 score-2 pairs (using rank-1
and rank-2/3 pairs proportionately).

Writes:
- eval/audit/relevance_audit_sheet.csv (blinded: no judge scores/rationales)
- eval/audit/relevance_audit_key.json (unblinded scoring key)
"""

import sys
import re
import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JUDGMENTS_FILE = PROJECT_ROOT / "eval" / "results" / "relevance_judgments.json"
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
TRAIN_FILE = PROJECT_ROOT / "data" / "train.json"

AUDIT_DIR = PROJECT_ROOT / "eval" / "audit"
SHEET_FILE = AUDIT_DIR / "relevance_audit_sheet.csv"
KEY_FILE = AUDIT_DIR / "relevance_audit_key.json"

BASE_STOPLIST = {
    "github", "sentry", "cloudflare", "twilio", "discord", "shopify", "reddit",
    "atlassian", "datadog", "digitalocean", "circleci", "netlify", "openai",
    "dropbox", "squarespace", "launchdarkly", "google", "aws", "anthropic",
    "slack", "jira", "confluence", "bitbucket", "acuity", "copilot", "chatgpt",
    "travis", "travisci"
}

def load_stoplist() -> re.Pattern:
    stoplist = set(BASE_STOPLIST)
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            for r in json.load(f):
                c = r.get("company", "").strip().lower()
                if c:
                    stoplist.add(c)
    if TEST_FILE.exists():
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            for r in json.load(f):
                c = r.get("company", "").strip().lower()
                if c:
                    stoplist.add(c)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(c) for c in sorted(stoplist, key=len, reverse=True)) + r")\b",
        re.IGNORECASE
    )
    return pattern

COMPANY_PATTERN = load_stoplist()

def mask_vendor(text: str) -> str:
    if not text:
        return ""
    masked = COMPANY_PATTERN.sub("[VENDOR]", text)
    masked = re.sub(r"(\[VENDOR\]\s*)+", "[VENDOR] ", masked)
    return masked.strip()

def make_audit_sheet():
    if not JUDGMENTS_FILE.exists():
        raise FileNotFoundError(f"Judgments file {JUDGMENTS_FILE} not found.")

    with open(JUDGMENTS_FILE, "r", encoding="utf-8") as f:
        judgements_data = json.load(f)

    # Load full incident text for queries and precedents
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = {r["incident_id"]: r for r in json.load(f)}

    train_records = {}
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            train_records = {r["incident_id"]: r for r in json.load(f)}

    # Collect all 279 pairs
    all_pairs = []
    for q in judgements_data["judgments"]:
        q_id = q["query_incident_id"]
        q_raw = test_records.get(q_id, {})
        q_log = q_raw.get("raw_log_excerpt", "")
        q_rc = q_raw.get("root_cause", "")
        q_masked = f"Log:\n{mask_vendor(q_log)}\n\nRoot Cause:\n{mask_vendor(q_rc)}"

        for h in q["hits"]:
            p_id = h["retrieved_incident_id"]
            p_raw = train_records.get(p_id, {})
            p_log = p_raw.get("raw_log_excerpt", "")
            p_rc = p_raw.get("root_cause", "")
            p_masked = f"Log:\n{mask_vendor(p_log)}\n\nRoot Cause:\n{mask_vendor(p_rc)}"

            all_pairs.append({
                "query_id": q_id,
                "precedent_id": p_id,
                "rank": h["rank"],
                "similarity_score": h["similarity_score"],
                "query_text_masked": q_masked,
                "precedent_text_masked": p_masked,
                "judge_score": h["score"],
                "judge_rationale": h["rationale"]
            })

    # Group by judge score
    by_score = {0: [], 1: [], 2: []}
    for p in all_pairs:
        by_score[p["judge_score"]].append(p)

    rng = random.Random(42)

    def sample_proportionate_by_rank(items, n_target, rng_inst):
        rank1 = [it for it in items if it["rank"] == 1]
        rank_other = [it for it in items if it["rank"] in (2, 3)]
        
        # Calculate proportionate allocation
        prop_r1 = len(rank1) / len(items)
        n_r1 = int(round(n_target * prop_r1))
        n_other = n_target - n_r1

        # Sample rank 1
        if len(rank1) >= n_r1:
            sampled_r1 = rng_inst.sample(rank1, n_r1)
        else:
            sampled_r1 = [rng_inst.choice(rank1) for _ in range(n_r1)] if rank1 else []

        # Sample rank other
        if len(rank_other) >= n_other:
            sampled_other = rng_inst.sample(rank_other, n_other)
        else:
            sampled_other = [rng_inst.choice(rank_other) for _ in range(n_other)] if rank_other else []

        return sampled_r1 + sampled_other

    # Sample exactly 10 per stratum
    sampled_0 = sample_proportionate_by_rank(by_score[0], 10, rng)
    sampled_1 = sample_proportionate_by_rank(by_score[1], 10, rng)
    sampled_2 = sample_proportionate_by_rank(by_score[2], 10, rng)

    assert len(sampled_0) == 10
    assert len(sampled_1) == 10
    assert len(sampled_2) == 10

    # Combine and shuffle so strata are blinded and randomized
    combined = sampled_0 + sampled_1 + sampled_2
    rng.shuffle(combined)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    sheet_rows = []
    key_dict = {}

    for idx, item in enumerate(combined, start=1):
        pair_id = f"pair_{idx:02d}"
        
        sheet_rows.append({
            "pair_id": pair_id,
            "query_id": item["query_id"],
            "precedent_id": item["precedent_id"],
            "rank": item["rank"],
            "similarity_score": item["similarity_score"],
            "query_text_masked": item["query_text_masked"],
            "precedent_text_masked": item["precedent_text_masked"],
            "human_label": "",
            "human_notes": ""
        })

        key_dict[pair_id] = {
            "query_id": item["query_id"],
            "precedent_id": item["precedent_id"],
            "rank": item["rank"],
            "similarity_score": item["similarity_score"],
            "judge_score": item["judge_score"],
            "judge_rationale": item["judge_rationale"]
        }

    # Write CSV sheet
    fieldnames = [
        "pair_id", "query_id", "precedent_id", "rank", "similarity_score",
        "query_text_masked", "precedent_text_masked", "human_label", "human_notes"
    ]
    with open(SHEET_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sheet_rows)

    # Write Key JSON
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        json.dump(key_dict, f, indent=2)

    print(f"Created audit sheet: {SHEET_FILE} (30 rows, blinded)")
    print(f"Created audit key:   {KEY_FILE} (10 score-0, 10 score-1, 10 score-2)")

if __name__ == "__main__":
    make_audit_sheet()
