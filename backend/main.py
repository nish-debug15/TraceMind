from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import json
import logging
import time
import threading
from pathlib import Path
from typing import Optional, List
from collections import defaultdict

from backend.config import ALLOWED_ORIGINS, LLM_MODEL, IP_RATE_LIMIT_RPM
from backend.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    PipelineTrace,
    RetrievedIncident,
    ClusterProfile,
)

# Import stubs with mock data
from clustering import assign_cluster
from rag import generate_rca
from rag.generate import LLMGenerationError

app = FastAPI(title="TraceMind Backend")

# Lock CORS strictly to allowed origins (no wildcard .*.vercel.app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROFILES_PATH = Path(__file__).parent.parent / "clustering" / "cluster_profiles.json"
_CLUSTER_PROFILES: dict[int, dict] = {}


def get_cluster_profile(cluster_id: int) -> Optional[dict]:
    """Retrieve cluster profile by ID from clustering/cluster_profiles.json."""
    global _CLUSTER_PROFILES
    if not _CLUSTER_PROFILES and PROFILES_PATH.exists():
        try:
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
                _CLUSTER_PROFILES = {int(k): v for k, v in raw.items()}
        except Exception as e:
            logging.error(f"Failed to load cluster profiles: {e}")
    return _CLUSTER_PROFILES.get(cluster_id)


class IPRateLimiter:
    """Thread-safe sliding window rate limiter per client IP."""
    def __init__(self, max_rpm: int = 15):
        self.max_rpm = max_rpm
        self.window = 60.0
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        if self.max_rpm <= 0:
            return True
        with self.lock:
            now = time.time()
            # Retain only timestamps within the rolling window
            self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window]
            if len(self.requests[client_ip]) >= self.max_rpm:
                return False
            self.requests[client_ip].append(now)
            return True


ip_limiter = IPRateLimiter(max_rpm=IP_RATE_LIMIT_RPM)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        content = dict(exc.detail)
        if "detail" not in content and "message" in content:
            content["detail"] = content["message"]
        return JSONResponse(status_code=exc.status_code, content=content)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "message": str(exc.detail)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred processing the request"}
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/clusters", response_model=List[ClusterProfile])
def list_clusters():
    """List all failure-grounded cluster profiles."""
    get_cluster_profile(-1)
    return sorted(list(_CLUSTER_PROFILES.values()), key=lambda c: c.get("id", 0))


@app.get("/clusters/{cluster_id}", response_model=ClusterProfile)
def get_cluster(cluster_id: int):
    """Get failure-grounded metadata, playbook fixes, and examples for a specific cluster."""
    profile = get_cluster_profile(cluster_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    return profile


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_log(request: AnalyzeRequest, req: Request):
    # Enforce per-IP rate limiting
    client_ip = req.client.host if req.client else "unknown"
    forwarded = req.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    if not ip_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Rate limit exceeded. Maximum {IP_RATE_LIMIT_RPM} requests per minute per IP.",
                "provider_status": 429
            }
        )

    try:
        # Run clustering
        cluster_info = assign_cluster(request.log_text)
        cluster_id = cluster_info["cluster_id"]
        profile_data = get_cluster_profile(cluster_id)
        cluster_profile_obj = ClusterProfile(**profile_data) if profile_data else None
        
        # Run RAG retrieval and generation with caching enabled in production
        use_cache = os.getenv("ENABLE_GEN_CACHE", "true").lower() == "true"
        if "PYTEST_CURRENT_TEST" in os.environ:
            use_cache = False
        rag_info = generate_rca(request.log_text, use_cache=use_cache)
        
        # Assemble PipelineTrace
        retrieval_pass = rag_info.get("retrieval_pass", 1 if rag_info.get("similarity_score", 0) >= 0.40 else 2)
        trace = PipelineTrace(
            cluster_id=cluster_id,
            is_noise=cluster_info["is_noise"],
            retrieved_incident=RetrievedIncident(**rag_info["retrieved_incident"]),
            similarity_score=rag_info["similarity_score"],
            retrieval_pass=retrieval_pass,
            prompt_used=rag_info["prompt_used"],
            model=rag_info.get("model", LLM_MODEL),
            cluster_profile=cluster_profile_obj,
        )
        
        # Assemble Final Response
        return AnalyzeResponse(
            raw_log_excerpt=request.log_text,
            root_cause=rag_info["root_cause"],
            remediation_steps=rag_info["remediation_steps"],
            source_url=rag_info["retrieved_incident"]["source_url"],
            retrieval_pass=retrieval_pass,
            trace=trace,
            cluster_profile=cluster_profile_obj,
            is_mock=rag_info.get("is_mock", False)
        )
    except LLMGenerationError as e:
        logging.error(f"LLM Generation Error: {e.message}")
        raise HTTPException(
            status_code=502,
            detail={
                "message": e.message,
                "provider_status": e.provider_status or 502
            }
        )
    except Exception as e:
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=504, detail={"message": "LLM Request Timeout", "provider_status": 504})
        raise
