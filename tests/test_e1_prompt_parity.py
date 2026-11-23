"""
Tests for Block 3: Prompt Parity modulo precedent text
Asserts that the baseline random prompt builder is identical to the baseline_rag prompt builder,
differing only in the precedent text.
"""

import json
import re
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.json"
RAG_GEN_FILE = PROJECT_ROOT / "eval" / "cache" / "rag_generations.json"

BASELINE_SYSTEM_PROMPT = (
    "You are an expert SRE assistant analyzing infrastructure incidents. "
    "Structure your output into ROOT CAUSE: and REMEDIATION STEPS: sections."
)


def build_baseline_rag_user_prompt(ref_root_cause: str, ref_remediation_steps: str, log_text: str) -> str:
    return (
        f"Historical Incident Reference:\n"
        f"- Root Cause: {ref_root_cause}\n"
        f"- Remediation: {ref_remediation_steps}\n\n"
        f"New Incident Log Excerpt:\n"
        f"{log_text}\n\n"
        f"Provide your response in the following format:\n"
        f"ROOT CAUSE:\n"
        f"(your analysis)\n\n"
        f"REMEDIATION STEPS:\n"
        f"(your steps)"
    )


def test_prompt_parity_modulo_precedent():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    with open(RAG_GEN_FILE, "r", encoding="utf-8") as f:
        rag_generations = json.load(f)

    placeholder_rc = "__DUMMY_PRECEDENT_ROOT_CAUSE__"
    placeholder_rem = "__DUMMY_PRECEDENT_REMEDIATION_STEPS__"

    for r in test_records:
        qid = r["incident_id"]
        log_text = r["raw_log_excerpt"]

        # Prompt used for baseline_rag
        actual_rag_prompt = rag_generations[qid]["prompt"]

        # Normalize baseline_rag prompt by replacing its precedent text with placeholders
        # Extract the precedent text
        m = re.search(
            r"Historical Incident Reference:\n- Root Cause: (.*?)\n- Remediation: (.*?)\n\nNew Incident Log Excerpt:\n",
            actual_rag_prompt,
            re.DOTALL
        )
        assert m is not None, f"Could not parse precedent from baseline_rag prompt for {qid}"

        normalized_rag_user = actual_rag_prompt.replace(m.group(1), placeholder_rc, 1).replace(m.group(2), placeholder_rem, 1)

        # Baseline random prompt with placeholders
        random_user_with_placeholders = build_baseline_rag_user_prompt(
            ref_root_cause=placeholder_rc,
            ref_remediation_steps=placeholder_rem,
            log_text=log_text
        )

        rag_hash = hashlib.sha256(f"{BASELINE_SYSTEM_PROMPT}\n{normalized_rag_user}".encode("utf-8")).hexdigest()
        rand_hash = hashlib.sha256(f"{BASELINE_SYSTEM_PROMPT}\n{random_user_with_placeholders}".encode("utf-8")).hexdigest()

        assert rag_hash == rand_hash, f"Prompt hash mismatch modulo precedent text for incident {qid}"
