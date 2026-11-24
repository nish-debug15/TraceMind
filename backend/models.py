from pydantic import BaseModel, Field
from typing import Optional, List
import uuid

class AnalyzeRequest(BaseModel):
    log_text: str = Field(..., min_length=1, description="Raw log text to analyze")

class RetrievedIncident(BaseModel):
    incident_id: str
    raw_log_excerpt: str
    root_cause: str
    remediation_steps: str
    source_url: str
    company: Optional[str] = None

class ClusterExample(BaseModel):
    company: str
    root_cause: str
    source_url: str

class ClusterSource(BaseModel):
    top_company: Optional[str] = None
    top_share: Optional[float] = None
    company_count: Optional[int] = None

class ClusterProfile(BaseModel):
    id: int
    name: str
    description: str
    size: int = 0
    confidence: Optional[str] = "ok"
    source: Optional[ClusterSource] = None
    keywords: List[str] = Field(default_factory=list)
    typical_fixes: List[str] = Field(default_factory=list)
    examples: List[ClusterExample] = Field(default_factory=list)

class PipelineTrace(BaseModel):
    cluster_id: int
    is_noise: bool
    retrieved_incident: RetrievedIncident
    similarity_score: float
    retrieval_pass: Optional[int] = 1
    prompt_used: str
    model: Optional[str] = None
    cluster_profile: Optional[ClusterProfile] = None

class AnalyzeResponse(BaseModel):
    incident_id: str = Field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8].upper()}")
    raw_log_excerpt: str
    root_cause: str
    remediation_steps: str
    source_url: str
    retrieval_pass: Optional[int] = 1
    trace: PipelineTrace
    cluster_profile: Optional[ClusterProfile] = None
    is_mock: bool = False
