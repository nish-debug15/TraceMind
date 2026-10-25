import os

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    os.getenv("FRONTEND_URL", "").strip(),
    os.getenv("VERCEL_URL", "").strip(),
]

ALLOWED_ORIGINS = [
    (f"https://{o}" if not o.startswith("http") else o)
    for o in ALLOWED_ORIGINS
    if o
]

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_RPM = int(os.getenv("LLM_MAX_RPM", "20"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
