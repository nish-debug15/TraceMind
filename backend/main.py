from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
import time
import threading
from collections import defaultdict

from backend.config import ALLOWED_ORIGINS, LLM_MODEL, IP_RATE_LIMIT_RPM
from backend.models import AnalyzeRequest, AnalyzeResponse, PipelineTrace, RetrievedIncident

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
        
        # Run RAG retrieval and generation with caching enabled in production
        use_cache = os.getenv("ENABLE_GEN_CACHE", "true").lower() == "true"
        if "PYTEST_CURRENT_TEST" in os.environ:
            use_cache = False
        rag_info = generate_rca(request.log_text, use_cache=use_cache)
        
        # Assemble PipelineTrace
        retrieval_pass = rag_info.get("retrieval_pass", 1 if rag_info.get("similarity_score", 0) >= 0.40 else 2)
        trace = PipelineTrace(
            cluster_id=cluster_info["cluster_id"],
            is_noise=cluster_info["is_noise"],
            retrieved_incident=RetrievedIncident(**rag_info["retrieved_incident"]),
            similarity_score=rag_info["similarity_score"],
            retrieval_pass=retrieval_pass,
            prompt_used=rag_info["prompt_used"],
            model=rag_info.get("model", LLM_MODEL),
        )
        
        # Assemble Final Response
        return AnalyzeResponse(
            raw_log_excerpt=request.log_text,
            root_cause=rag_info["root_cause"],
            remediation_steps=rag_info["remediation_steps"],
            source_url=rag_info["retrieved_incident"]["source_url"],
            retrieval_pass=retrieval_pass,
            trace=trace,
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
