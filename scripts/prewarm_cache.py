"""
scripts/prewarm_cache.py
-----------------------
Pre-warms the SQLite generation cache (rag/cache/gen_cache.sqlite3)
for all preset example chips in the TraceMind frontend.
This guarantees that demo interactions never fail due to Groq rate limits,
never suffer cold-start latency, and run instantly offline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.generate import generate_rca, LLM_MODEL
from rag.gen_cache import get_cached_generation

PRESETS = [
    {
        "name": "DB pool exhausted",
        "log": """[2026-10-15 08:23:41] [ERROR] [db-proxy-pool] Connection timeout acquiring connection from pool.
FATAL: remaining connection slots are reserved for non-replication superuser connections.
PoolExhaustedError: database connection pool exhausted, 50/50 connections in use
[2026-10-15 08:23:42] [WARN] [health-check] Liveness probe failed for pod db-proxy-6b8f7d9""",
    },
    {
        "name": "Kubelet OOMKilled",
        "log": """CRITICAL 2026-10-14T14:12:05Z kubelet worker-node-04: OOMKilled process 18421 (payments-api) total-vm:8421048kB, anon-rss:4194304kB. Memory limit 4Gi exceeded.""",
    },
    {
        "name": "Redis sentinel failover",
        "log": """ERROR: Redis sentinel failover triggered - master 10.0.4.12:6379 unreachable for 30s. Failover elected replica 10.0.4.13:6379 as new master. Write latency spiked to 2400ms.""",
    },
    {
        "name": "SSL verify failed",
        "log": """WARNING: SSL certificate verification failed for api.stripe.com. OpenSSL Error: certificate has expired on 2026-10-14 23:59:59 GMT. TLS handshake terminated.""",
    },
]

def prewarm():
    print(f"Pre-warming generation cache for {len(PRESETS)} frontend examples using model '{LLM_MODEL}'...")
    for p in PRESETS:
        name = p["name"]
        log = p["log"]
        print(f"\nProcessing '{name}'...")
        res = generate_rca(log, use_cache=True)
        if res.get("is_mock"):
            print(f"  [WARN] Output was mock (check GROQ_API_KEY).")
        else:
            cached_status = "CACHE HIT" if res.get("cached") else "LIVE GENERATION CACHED"
            print(f"  [{cached_status}] Similarity: {res['similarity_score']}")
            first_line = res["root_cause"].splitlines()[0] if res.get("root_cause") else ""
            print(f"  Root Cause Preview: {first_line}")
    print("\nPre-warming complete. All preset examples are cached.")

if __name__ == "__main__":
    prewarm()
