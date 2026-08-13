from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from backend.config import ALLOWED_ORIGINS
from backend.models import AnalyzeRequest, AnalyzeResponse, PipelineTrace, RetrievedIncident

# Import stubs with mock data
from clustering import assign_cluster
from rag import generate_rca

app = FastAPI(title="TraceMind Backend")

# Setup CORS with wildcard support for vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def analyze_log(request: AnalyzeRequest):
    try:
        # MOCK MODE: call stub clustering function
        cluster_info = assign_cluster(request.log_text)
        
        # MOCK MODE: call stub RAG function
        rag_info = generate_rca(request.log_text)
        
        # Assemble PipelineTrace
        trace = PipelineTrace(
            cluster_id=cluster_info["cluster_id"],
            is_noise=cluster_info["is_noise"],
            retrieved_incident=RetrievedIncident(**rag_info["retrieved_incident"]),
            similarity_score=rag_info["similarity_score"],
            prompt_used=rag_info["prompt_used"]
        )
        
        # Assemble Final Response
        return AnalyzeResponse(
            raw_log_excerpt=request.log_text,
            root_cause=rag_info["root_cause"],
            remediation_steps=rag_info["remediation_steps"],
            source_url=rag_info["retrieved_incident"]["source_url"],
            trace=trace
        )
    except Exception as e:
        # For simplicity, assuming timeout might raise TimeoutError in the future
        if isinstance(e, TimeoutError):
            raise HTTPException(status_code=504, detail="LLM Request Timeout")
        raise
