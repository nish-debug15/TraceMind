from fastapi.testclient import TestClient
from backend.main import app

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
