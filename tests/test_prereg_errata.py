"""
Tests for Block 4: PREREG Errata and Directional Verdicts
Asserts that PREREG.md is unchanged, PREREG_ERRATA.md exists,
and random_precedent_control_directional.json contains proper directional verdicts.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_prereg_md_untouched():
    prereg_path = PROJECT_ROOT / "eval" / "PREREG.md"
    assert prereg_path.exists()
    content = prereg_path.read_text(encoding="utf-8")
    assert "Pre-Registered Decision Rules: Experiment 1 (E1) & Experiment 2 (E2)" in content


def test_prereg_errata_exists_and_defines_sign():
    errata_path = PROJECT_ROOT / "eval" / "PREREG_ERRATA.md"
    assert errata_path.exists()
    content = errata_path.read_text(encoding="utf-8")
    assert "precedent_harmful_or_artifact" in content
    assert "precedent_content" in content
    assert "real_minus_random > 0" in content or "real_minus_random" in content


def test_directional_verdicts_json():
    dir_path = PROJECT_ROOT / "eval" / "results" / "random_precedent_control_directional.json"
    assert dir_path.exists()
    with open(dir_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Original verdict fields must be preserved
    assert data["formats"]["gate_v2"]["verdict"] == "inconclusive"
    assert data["formats"]["baseline"]["verdict"] == "precedent_content"

    # Directional fields must be added
    assert data["formats"]["gate_v2"]["verdict_directional"] == "inconclusive"
    assert data["formats"]["baseline"]["verdict_directional"] == "precedent_harmful_or_artifact"
    assert data["verdict_directional"] == "inconclusive"
