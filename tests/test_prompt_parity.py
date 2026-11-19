"""
tests/test_prompt_parity.py
---------------------------
Guardrail test asserting byte-level parity between the prompt evaluated
in eval/prompt_ablation.py and the prompt constructed at runtime by the backend.
Also verifies deterministic SHA-256 integrity against eval/results/prompt_hashes.json.
"""

import json
import hashlib
from pathlib import Path
import pytest

from rag.generate import PROMPT_REGISTRY as RUNTIME_REGISTRY
from rag.prompts import PROMPT_REGISTRY as EVAL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HASHES_FILE = PROJECT_ROOT / "eval" / "results" / "prompt_hashes.json"

FIXTURE_INCIDENT = {
    "raw_log_excerpt": "ERROR 2026-10-15 08:23:41 db-pool: connection pool exhausted, 50/50 connections in use",
    "root_cause": "Database connection pool exhausted due to unclosed cursors in payments-service",
    "remediation_steps": "1. Deploy hotfix closing cursors in db.go\n2. Increase max pool size to 100",
    "similarity": 0.5820
}


def test_relevance_gate_v2_prompt_parity():
    """Verify that evaluated prompt matches runtime prompt byte-for-byte."""
    # 1. Runtime path prompt construction
    runtime_def = RUNTIME_REGISTRY["relevance_gate_v2"]
    runtime_sys = runtime_def["system"]
    runtime_user = runtime_def["user_template"].format(
        ref_root_cause=FIXTURE_INCIDENT["root_cause"],
        ref_remediation_steps=FIXTURE_INCIDENT["remediation_steps"],
        log_text=FIXTURE_INCIDENT["raw_log_excerpt"],
        similarity=FIXTURE_INCIDENT["similarity"]
    )
    runtime_full = f"{runtime_sys}\n\n{runtime_user}"

    # 2. Evaluation path prompt construction
    eval_def = EVAL_REGISTRY["relevance_gate_v2"]
    eval_sys = eval_def["system"]
    eval_user = eval_def["user_template"].format(
        ref_root_cause=FIXTURE_INCIDENT["root_cause"],
        ref_remediation_steps=FIXTURE_INCIDENT["remediation_steps"],
        log_text=FIXTURE_INCIDENT["raw_log_excerpt"],
        similarity=FIXTURE_INCIDENT["similarity"]
    )
    eval_full = f"{eval_sys}\n\n{eval_user}"

    # Assert byte identity
    assert runtime_sys == eval_sys, f"System prompt diff: {runtime_sys!r} != {eval_sys!r}"
    assert runtime_user == eval_user, f"User prompt diff: {runtime_user!r} != {eval_user!r}"
    assert runtime_full == eval_full, f"Full prompt diff: {runtime_full!r} != {eval_full!r}"


def test_prompt_template_sha256_integrity():
    """Assert that prompt template SHA-256 matches frozen hash in prompt_hashes.json."""
    assert HASHES_FILE.exists(), f"Missing {HASHES_FILE}. Run scripts/generate_prompt_hashes.py first."

    with open(HASHES_FILE, "r", encoding="utf-8") as f:
        stored_hashes = json.load(f)

    for variant_name in ["baseline", "relevance_gate", "relevance_gate_v2", "zero_shot_gate_format"]:
        assert variant_name in RUNTIME_REGISTRY, f"Missing {variant_name} in RUNTIME_REGISTRY"
        assert variant_name in stored_hashes, f"Missing {variant_name} in prompt_hashes.json"

        v_def = RUNTIME_REGISTRY[variant_name]
        combined = f"{v_def['system']}\n\n{v_def['user_template']}"
        actual_sha = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        expected_sha = stored_hashes[variant_name]["combined_template_sha256"]

        assert actual_sha == expected_sha, (
            f"Prompt template hash mismatch for '{variant_name}'!\n"
            f"Expected: {expected_sha}\n"
            f"Actual:   {actual_sha}"
        )
