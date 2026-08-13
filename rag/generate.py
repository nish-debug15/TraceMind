def generate_rca(log_text: str) -> dict:
    """Full RAG pipeline. MOCK: returns hardcoded example response."""
    return {
        "root_cause": "Database connection pool exhaustion caused by a misconfigured max_connections parameter after a routine deployment. The connection limit was set to 50 instead of 500, causing cascading timeouts across dependent services.",
        "remediation_steps": "1. Increase max_connections to 500 in the database configuration.\n2. Restart the database service to apply the new connection limit.\n3. Monitor connection pool utilization for the next 24 hours.\n4. Add connection pool alerting threshold at 80% capacity.",
        "retrieved_incident": {
            "incident_id": "INC-2024-0142",
            "raw_log_excerpt": "ERROR 2024-03-15T08:23:41Z db-primary-03: connection pool exhausted, 50/50 connections in use, 127 requests queued",
            "root_cause": "Connection pool limit misconfiguration during v2.3.1 deployment",
            "remediation_steps": "Rolled back connection limit to previous value of 500, implemented connection pool monitoring",
            "source_url": "https://github.com/danluu/post-mortems"
        },
        "similarity_score": 0.89,
        "prompt_used": "[MOCK PROMPT] You are an SRE assistant analyzing infrastructure incidents...\n\nHistorical Incident:\n- Root Cause: Connection pool limit misconfiguration during v2.3.1 deployment\n- Remediation: Rolled back connection limit to previous value of 500\n\nNew Log Input:\n{log_text}\n\nProvide a root cause analysis and remediation steps."
    }  # MOCK — Unique replaces with real ChromaDB retrieval + LLM generation
