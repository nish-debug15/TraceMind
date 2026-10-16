"""
scripts/probe_groq.py
---------------------
Probes Groq API chat completions with:
- llama-3.3-70b-versatile
- llama-3.1-8b-instant
Prints status, latency, response preview, and all x-ratelimit-* headers.
"""

import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODELS_TO_PROBE = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def probe_model(model_name: str, api_key: str):
    print("=" * 70)
    print(f"PROBING GROQ MODEL: {model_name}")
    print("=" * 70)

    if not api_key:
        print("ERROR: GROQ_API_KEY is not set in environment or .env file.")
        return False, None, {}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an SRE assistant."},
            {"role": "user", "content": "Respond with: TraceMind Groq online."},
        ],
        "max_tokens": 50,
        "temperature": 0.2,
    }

    t0 = time.time()
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30.0)
        latency = time.time() - t0
    except Exception as e:
        latency = time.time() - t0
        print(f"Network error probing {model_name} (latency: {latency:.3f}s): {e}")
        return False, latency, {}

    print(f"HTTP Status: {resp.status_code}")
    print(f"Latency:     {latency:.3f}s")

    # Extract x-ratelimit headers
    ratelimit_headers = {k: v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit-")}
    print("\nRate Limit Headers (x-ratelimit-*):")
    if ratelimit_headers:
        for k, v in sorted(ratelimit_headers.items()):
            print(f"  {k}: {v}")
    else:
        print("  (None returned)")

    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"\nResponse Body Preview:\n  {content.strip()}")
        print(f"\n[PASSED] Model {model_name} is LIVE and working!")
        return True, latency, ratelimit_headers
    else:
        print(f"\n[FAILED] Model {model_name} returned error: {resp.text}")
        return False, latency, ratelimit_headers


def main():
    if not GROQ_API_KEY:
        print("No GROQ_API_KEY found in .env. Please set GROQ_API_KEY=gsk_... first.")
        sys.exit(1)

    results = {}
    for model in MODELS_TO_PROBE:
        success, latency, headers = probe_model(model, GROQ_API_KEY)
        results[model] = {"success": success, "latency": latency, "headers": headers}
        print("\n")

    print("=" * 70)
    print("PROBE SUMMARY:")
    for m, r in results.items():
        status_str = "WORKING" if r["success"] else "FAILED"
        lat_str = f"{r['latency']:.3f}s" if r["latency"] is not None else "N/A"
        print(f"  - {m}: {status_str} (latency: {lat_str})")
    print("=" * 70)


if __name__ == "__main__":
    main()
