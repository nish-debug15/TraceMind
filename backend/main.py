from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from backend.config import ALLOWED_ORIGINS, LLM_MODEL
from backend.models import AnalyzeRequest, AnalyzeResponse, PipelineTrace, RetrievedIncident

# Import stubs with mock data
from clustering import assign_cluster
from rag import generate_rca
from rag.generate import LLMGenerationError

app = FastAPI(title="TraceMind Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def analyze_log(request: AnalyzeRequest):
    try:
        # Run clustering
        cluster_info = assign_cluster(request.log_text)
        
        # Run RAG retrieval and generation
        rag_info = generate_rca(request.log_text)
        
        # Assemble PipelineTrace
        trace = PipelineTrace(
            cluster_id=cluster_info["cluster_id"],
            is_noise=cluster_info["is_noise"],
            retrieved_incident=RetrievedIncident(**rag_info["retrieved_incident"]),
            similarity_score=rag_info["similarity_score"],
            prompt_used=rag_info["prompt_used"],
            model=rag_info.get("model", LLM_MODEL),
        )
        
        # Assemble Final Response
        return AnalyzeResponse(
            raw_log_excerpt=request.log_text,
            root_cause=rag_info["root_cause"],
            remediation_steps=rag_info["remediation_steps"],
            source_url=rag_info["retrieved_incident"]["source_url"],
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
