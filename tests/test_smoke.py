"""
Smoke test suite executed locally via FastAPI TestClient without external network.
"""
from fastapi.testclient import TestClient
from backend.main import app
from scripts.smoke_live import run_smoke_tests


def test_smoke_local():
    client = TestClient(app)
    success = run_smoke_tests(client=client)
    assert success is True, "Local smoke tests failed"
