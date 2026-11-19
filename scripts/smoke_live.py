#!/usr/bin/env python3
"""
Smoke test suite for TraceMind Backend.
Can be executed against a live deployment or locally via FastAPI TestClient.

Usage:
    python scripts/smoke_live.py
    BACKEND_URL=https://nishhhx0-tracemind-backend.hf.space python scripts/smoke_live.py
"""

import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "test.json"


def run_smoke_tests(base_url: str = None, client = None) -> bool:
    """
    Run smoke tests against either:
      1. A TestClient instance (client)
      2. An HTTP backend URL (base_url or BACKEND_URL env var)
    Returns True if all assertions pass, False otherwise.
    """
    if client is None:
        target_url = (base_url or os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")
        import requests
        http_session = requests.Session()
        print(f"[*] Running smoke tests against live target: {target_url}")
    else:
        target_url = ""
        http_session = client
        print("[*] Running smoke tests against local FastAPI TestClient")

    # Load 2 real test set logs
    if not DATA_SPLITS_PATH.exists():
        print(f"[FAIL] Missing test split file: {DATA_SPLITS_PATH}")
        return False

    with open(DATA_SPLITS_PATH, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    if len(test_records) < 2:
        print("[FAIL] data/splits/test.json contains fewer than 2 incidents.")
        return False

    all_passed = True

    # 1. Test GET /health
    t0 = time.time()
    try:
        if client:
            resp = client.get("/health")
        else:
            resp = http_session.get(f"{target_url}/health", timeout=15)
        
        lat_ms = (time.time() - t0) * 1000
        if resp.status_code != 200:
            print(f"[FAIL] GET /health returned HTTP {resp.status_code} ({lat_ms:.1f}ms)")
            return False
        
        health_data = resp.json()
        assert health_data.get("status") == "ok", f"Expected {{'status': 'ok'}}, got: {health_data}"
        print(f"[PASS] GET /health -> 200 OK ({lat_ms:.1f}ms)")
    except Exception as e:
        lat_ms = (time.time() - t0) * 1000
        print(f"[FAIL] GET /health failed ({lat_ms:.1f}ms): {e}")
        return False

    # 2. Test POST /analyze with 2 real test set incidents
    for idx in [0, 1]:
        record = test_records[idx]
        incident_id = record.get("incident_id", f"idx-{idx}")
        company = record.get("company", "unknown")
        raw_log = record.get("raw_log_excerpt", "")
        
        t0 = time.time()
        try:
            payload = {"log_text": raw_log}
            if client:
                resp = client.post("/analyze", json=payload)
            else:
                resp = http_session.post(f"{target_url}/analyze", json=payload, timeout=60)
            
            lat_ms = (time.time() - t0) * 1000
            if resp.status_code != 200:
                print(f"[FAIL] POST /analyze [{incident_id} / {company}] returned HTTP {resp.status_code} ({lat_ms:.1f}ms)")
                all_passed = False
                continue

            data = resp.json()

            # Schema assertions
            # root_cause non-empty
            rc = data.get("root_cause")
            assert rc and isinstance(rc, str) and len(rc.strip()) > 0, "root_cause is empty or missing"

            # remediation_steps non-empty
            rem = data.get("remediation_steps")
            assert rem and isinstance(rem, str) and len(rem.strip()) > 0, "remediation_steps is empty or missing"

            # retrieval_pass in (1, 2)
            retrieval_pass = data.get("retrieval_pass")
            assert retrieval_pass in (1, 2), f"retrieval_pass expected 1 or 2, got: {retrieval_pass}"

            # trace object schema
            trace = data.get("trace")
            assert isinstance(trace, dict), "trace field is missing or not a dictionary"
            assert "cluster_id" in trace and isinstance(trace["cluster_id"], int), "trace.cluster_id missing or not int"
            assert "is_noise" in trace and isinstance(trace["is_noise"], bool), "trace.is_noise missing or not bool"
            assert isinstance(trace.get("similarity_score"), (int, float)), "trace.similarity_score missing or not float"
            assert trace.get("prompt_used") and isinstance(trace["prompt_used"], str), "trace.prompt_used missing or empty"
            assert trace.get("model") and isinstance(trace["model"], str), "trace.model missing or empty"

            # retrieved_incident inside trace
            retrieved = trace.get("retrieved_incident")
            assert isinstance(retrieved, dict), "trace.retrieved_incident is missing or not a dictionary"
            for required_key in ["incident_id", "company", "root_cause", "source_url"]:
                assert required_key in retrieved, f"trace.retrieved_incident missing key '{required_key}'"

            cluster_label = f"Cluster #{trace['cluster_id']}" if not trace["is_noise"] else "Noise (-1)"
            print(
                f"[PASS] POST /analyze [{incident_id} / {company}] -> 200 OK "
                f"({lat_ms:.1f}ms | pass: {retrieval_pass} | {cluster_label} | sim: {trace['similarity_score']:.4f})"
            )
        except Exception as e:
            lat_ms = (time.time() - t0) * 1000
            print(f"[FAIL] POST /analyze [{incident_id} / {company}] failed ({lat_ms:.1f}ms): {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
