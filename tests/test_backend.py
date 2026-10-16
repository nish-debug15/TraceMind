import os
import unittest.mock as mock
from fastapi.testclient import TestClient
from backend.main import app
from rag.gen_cache import get_cached_generation, set_cached_generation

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_valid():
    response = client.post("/analyze", json={"log_text": "Connection timeout"})
    assert response.status_code == 200
    data = response.json()
    assert "incident_id" in data
    assert "root_cause" in data
    assert "remediation_steps" in data
    assert "trace" in data

def test_analyze_empty():
    response = client.post("/analyze", json={"log_text": ""})
    assert response.status_code == 422

def test_analyze_mock_when_groq_key_unset():
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}):
        response = client.post("/analyze", json={"log_text": "Connection timeout pool exhausted"})
        assert response.status_code == 200
        data = response.json()
        assert data["is_mock"] is True
        assert "[MOCK]" in data["root_cause"]

def test_analyze_loud_502_on_live_failure():
    """Live call failure after retries must return HTTP 502 with message and provider_status."""
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_invalid_test_key"}):
        # Mock requests.post to simulate 401 unauthorized / invalid key from Groq
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"error": {"message": "Invalid API Key", "type": "invalid_request_error"}}'

        with mock.patch("requests.post", return_value=mock_resp):
            response = client.post("/analyze", json={"log_text": "Connection timeout pool exhausted"})
            assert response.status_code == 502
            data = response.json()
            assert "message" in data
            assert data["provider_status"] == 401
            assert "Invalid API Key" in data["message"]

def test_gen_cache_roundtrip(tmp_path):
    test_model = "llama-3.3-70b-versatile"
    test_variant = "test_unit"
    test_query = "Error: Out of memory killed process 1234"
    
    set_cached_generation(
        model=test_model,
        prompt_variant=test_variant,
        query_text_or_id=test_query,
        root_cause="Memory allocation exceeded system limit",
        remediation_steps="Increase pod memory limits to 8Gi",
        raw_output="ROOT CAUSE:\nMemory allocation exceeded system limit\n\nREMEDIATION STEPS:\nIncrease pod memory limits to 8Gi"
    )

    cached = get_cached_generation(
        model=test_model,
        prompt_variant=test_variant,
        query_text_or_id=test_query
    )

    assert cached is not None
    assert cached["root_cause"] == "Memory allocation exceeded system limit"
    assert cached["remediation_steps"] == "Increase pod memory limits to 8Gi"
    assert cached["cache_hit"] is True
