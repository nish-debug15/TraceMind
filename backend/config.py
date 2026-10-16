import os

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    os.getenv("FRONTEND_URL", ""),
]

ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]
ALLOWED_ORIGINS.append("https://*.vercel.app")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_RPM = int(os.getenv("LLM_MAX_RPM", "20"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
