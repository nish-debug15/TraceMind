from pydantic import BaseModel, Field
from typing import Optional
import uuid

class AnalyzeRequest(BaseModel):
    log_text: str = Field(..., min_length=1, description="Raw log text to analyze")

class RetrievedIncident(BaseModel):
    incident_id: str
    raw_log_excerpt: str
    root_cause: str
    remediation_steps: str
    source_url: str

class PipelineTrace(BaseModel):
    cluster_id: int
    is_noise: bool
    retrieved_incident: RetrievedIncident
    similarity_score: float
    prompt_used: str

class AnalyzeResponse(BaseModel):
    incident_id: str = Field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8].upper()}")
    raw_log_excerpt: str
    root_cause: str
    remediation_steps: str
    source_url: str
    trace: PipelineTrace
