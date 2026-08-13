import os

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    os.getenv("FRONTEND_URL", ""),
]

ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]
ALLOWED_ORIGINS.append("https://*.vercel.app")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
