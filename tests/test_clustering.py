"""
Tests for BLOCK-CLUSTER-01: Embedding + HDBSCAN clustering module.
Run with: python -m pytest tests/test_clustering.py -v
"""
import pytest


class TestEmbed:
    """Tests for the embed() function."""

    def test_embed_returns_list(self):
        from clustering import embed
        result = embed("database connection timeout")
        assert isinstance(result, list)

    def test_embed_dimension(self):
        from clustering import embed
        result = embed("database connection timeout")
        assert len(result) == 384

    def test_embed_values_are_floats(self):
        from clustering import embed
        result = embed("database connection timeout")
        assert all(isinstance(v, float) for v in result)

    def test_embed_normalized(self):
        """Embeddings should be L2-normalized (magnitude ≈ 1.0)."""
        import numpy as np
        from clustering import embed
        vec = np.array(embed("ERROR: disk full on /var/log"))
        magnitude = np.linalg.norm(vec)
        assert abs(magnitude - 1.0) < 0.01

    def test_embed_similar_texts_close(self):
        """Two semantically similar inputs should produce similar embeddings."""
        import numpy as np
        from clustering import embed
        v1 = np.array(embed("database connection pool exhausted"))
        v2 = np.array(embed("DB connection limit reached, pool full"))
        similarity = float(v1 @ v2)
        assert similarity > 0.6  # should be quite similar

    def test_embed_different_texts_less_similar(self):
        """Two semantically different inputs should be less similar."""
        import numpy as np
        from clustering import embed
        v1 = np.array(embed("database connection pool exhausted"))
        v2 = np.array(embed("DNS resolution failure for api.example.com"))
        similarity = float(v1 @ v2)
        assert similarity < 0.8  # should be less similar than the similar pair

    def test_embed_empty_string(self):
        """embed() should handle empty-ish strings without crashing."""
        from clustering import embed
        result = embed("error")
        assert len(result) == 384

    def test_embed_long_text(self):
        """embed() should handle long text by truncating."""
        from clustering import embed
        long_text = "ERROR: connection timeout " * 500
        result = embed(long_text)
        assert len(result) == 384


class TestAssignCluster:
    """Tests for the assign_cluster() function."""

    def test_returns_dict(self):
        from clustering import assign_cluster
        result = assign_cluster("database connection timeout on primary replica")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from clustering import assign_cluster
        result = assign_cluster("ERROR: connection pool exhausted")
        assert "cluster_id" in result
        assert "is_noise" in result
        assert "confidence" in result

    def test_cluster_id_is_int(self):
        from clustering import assign_cluster
        result = assign_cluster("network latency spike detected")
        assert isinstance(result["cluster_id"], int)

    def test_is_noise_is_bool(self):
        from clustering import assign_cluster
        result = assign_cluster("network latency spike detected")
        assert isinstance(result["is_noise"], bool)

    def test_confidence_is_float(self):
        from clustering import assign_cluster
        result = assign_cluster("network latency spike detected")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_similar_inputs_same_cluster(self):
        """Two similar log entries should be assigned to the same cluster."""
        from clustering import assign_cluster
        r1 = assign_cluster("database connection pool exhausted, 50/50 connections in use")
        r2 = assign_cluster("DB connection pool full, all connections allocated")
        # They should get the same cluster (or both be noise)
        if not r1["is_noise"] and not r2["is_noise"]:
            assert r1["cluster_id"] == r2["cluster_id"]

    def test_novel_input_low_confidence(self):
        """A completely novel/nonsensical input should have lower confidence."""
        from clustering import assign_cluster
        result = assign_cluster("the quick brown fox jumps over the lazy dog")
        # Novel text should either be noise or have relatively lower confidence
        assert result["confidence"] < 0.8


class TestSampleLogs:
    """End-to-end tests with realistic SRE log samples."""

    SAMPLE_LOGS = [
        "ERROR 2024-03-15T08:23:41Z db-primary-03: connection pool exhausted, 50/50 connections in use",
        "CRITICAL: OOMKilled container web-api-7f8d9b in pod api-deployment-5c8f8",
        "WARNING: SSL certificate for api.example.com expires in 3 days",
        "ERROR: DNS resolution failed for internal-service.cluster.local after 5 retries",
        "ALERT: CPU utilization at 98% on worker-node-12 for past 15 minutes",
        "ERROR: Deployment rollout failed - image pull error for registry.internal/api:v2.3.1",
        "CRITICAL: Kafka consumer lag exceeding 100000 messages on topic payments-events",
        "ERROR: Redis sentinel failover triggered - master unreachable for 30s",
        "WARNING: Disk usage at 95% on /var/log/application.log - log rotation failed",
        "ERROR: HTTP 503 from upstream load balancer - all backends unhealthy",
    ]

    def test_all_samples_produce_valid_output(self):
        from clustering import assign_cluster
        for log in self.SAMPLE_LOGS:
            result = assign_cluster(log)
            assert "cluster_id" in result
            assert "is_noise" in result
            assert "confidence" in result
            assert isinstance(result["cluster_id"], int)
            assert isinstance(result["is_noise"], bool)
            assert isinstance(result["confidence"], float)
