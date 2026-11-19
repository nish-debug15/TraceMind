"""
scripts/generate_prompt_hashes.py
---------------------------------
Generates deterministic SHA-256 hashes of all prompt templates in PROMPT_REGISTRY.
Used by tests/test_prompt_parity.py to guarantee evaluated prompts match runtime prompts.
"""

import json
import hashlib
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.generate import PROMPT_REGISTRY
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "prompt_hashes.json"


def generate_hashes():
    hashes = {}
    for name, p_def in PROMPT_REGISTRY.items():
        sys_txt = p_def["system"]
        user_txt = p_def["user_template"]
        combined = f"{sys_txt}\n\n{user_txt}"
        hashes[name] = {
            "system_sha256": hashlib.sha256(sys_txt.encode("utf-8")).hexdigest(),
            "user_template_sha256": hashlib.sha256(user_txt.encode("utf-8")).hexdigest(),
            "combined_template_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print(f"[Artifact] Saved prompt hashes to {OUTPUT_FILE}")
    for name, h in hashes.items():
        print(f"  {name:<22} -> combined: {h['combined_template_sha256']}")

    return hashes


if __name__ == "__main__":
    generate_hashes()
