"""
Smoke test suite executed locally via FastAPI TestClient without external network.
"""
import os
import unittest.mock as mock
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from scripts.smoke_live import run_smoke_tests


def test_smoke_local():
    from backend.main import ip_limiter
    ip_limiter.requests.clear()
    client = TestClient(app)
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        success = run_smoke_tests(client=client)
        assert success is True, "Local smoke tests failed"


@pytest.mark.live
def test_smoke_live_hf_space():
    """Smoke test executed against live deployed Hugging Face Space."""
    target_url = os.environ.get("BACKEND_URL", "https://nishhhx0-tracemind-backend.hf.space")
    success = run_smoke_tests(base_url=target_url)
    assert success is True, "Live Space smoke tests failed"
