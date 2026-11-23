"""
Tests for Block 1: E1 Baseline Random-Arm Audit
Asserts that the audit correctly detects the prompt builder disparity,
word count deviation, and records the arm as INVALID in eval/DEVIATIONS.md.
"""

import json
import re
from pathlib import Path
import pytest
from scripts.audit_e1_baseline import audit_e1_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_e1_baseline_audit_detects_invalidation():
    results = audit_e1_baseline()
    assert results["status"] == "INVALID"
    assert results["prompt_hashes_differ"] is True
    # Word count should be close to gate_v2 (~40.7), and far from real_rag (~196.0)
    assert abs(results["mean_rc_words_random_baseline"] - 54.68) < 1.0
    assert abs(results["mean_rc_words_gate_v2"] - 40.67) < 1.0
    assert results["mean_rc_words_random_baseline"] < 100.0
    assert results["mean_rc_words_real_rag"] > 180.0
    assert results["remediation_parse_rate"] == 1.0


def test_deviations_md_marks_arm_invalid():
    deviations_path = PROJECT_ROOT / "eval" / "DEVIATIONS.md"
    assert deviations_path.exists()
    content = deviations_path.read_text(encoding="utf-8")
    assert "Deviation 2" in content
    assert "INVALID" in content
    assert "eval/run_random_precedent_gen.py" in content
