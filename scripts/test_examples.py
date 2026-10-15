"""
scripts/test_examples.py
------------------------
Runs the 4 example incidents through /analyze and reports:
- Latency
- is_mock
- model
- root_cause (first 200 chars)
- number of remediation steps
- Verifies no empty fields
"""

import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from app import SAMPLE_LOGS

client = TestClient(app)

print("=" * 70)
print("RUNNING 4 EXAMPLE INCIDENTS THROUGH /analyze")
print("=" * 70)

for i, log in enumerate(SAMPLE_LOGS, start=1):
    t0 = time.time()
    resp = client.post("/analyze", json={"log_text": log})
    t1 = time.time()

    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
    data = resp.json()

    trace = data.get("trace", {})
    root_cause = data.get("root_cause", "")
    remediation = data.get("remediation_steps", "")
    is_mock = data.get("is_mock")
    model = trace.get("model")

    steps = [s for s in remediation.split("\n") if s.strip()]

    print(f"\n--- Example {i} ---")
    print(f"Latency:           {t1 - t0:.3f}s")
    print(f"is_mock:           {is_mock}")
    print(f"model:             {model}")
    print(f"root_cause:        {root_cause[:200]}")
    print(f"remediation_steps: {len(steps)} items (preview: {remediation[:80]})")

    assert root_cause, f"Example {i}: root_cause is empty"
    assert remediation, f"Example {i}: remediation_steps is empty"
    assert model, f"Example {i}: trace.model is empty"

print("\n" + "=" * 70)
print("ALL 4 EXAMPLES VERIFIED (NO EMPTY FIELDS)")
print("=" * 70)
