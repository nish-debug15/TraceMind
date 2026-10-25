import os
import re
import json
import time
import logging
import threading
import requests
import chromadb
from pathlib import Path
from clustering.cluster import embed

RAG_DIR = Path(__file__).parent
DATA_PATH = RAG_DIR.parent / "data" / "train.json"
CHROMA_DB_PATH = RAG_DIR / "chroma_db"
COLLECTION_NAME = "postmortems"

# Initialize Chroma client
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

def ingest(
    force: bool = False,
    source: str | Path | None = None,
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
):
    """Ingest postmortems from source JSON into ChromaDB."""
    source_path = Path(source) if source else DATA_PATH
    persist_dir = Path(persist_directory) if persist_directory else CHROMA_DB_PATH
    col_name = collection_name if collection_name else COLLECTION_NAME

    print(f"Ingesting data from {source_path} into ChromaDB at {persist_dir} (collection: {col_name})...")
    
    target_client = client if (persist_dir.resolve() == CHROMA_DB_PATH.resolve()) else chromadb.PersistentClient(path=str(persist_dir))

    with open(source_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    expected_count = len(records)

    try:
        collection = target_client.get_collection(name=col_name)
        if force:
            print(f"Force re-ingestion requested. Deleting existing collection '{col_name}'...")
            target_client.delete_collection(name=col_name)
            collection = target_client.create_collection(name=col_name)
        elif collection.count() == expected_count:
            print(f"Collection '{col_name}' already contains exactly {expected_count} records, skipping ingestion.")
            return collection
        else:
            print(f"Collection '{col_name}' count is {collection.count()} (expected {expected_count} records). Rebuilding...")
            target_client.delete_collection(name=col_name)
            collection = target_client.create_collection(name=col_name)
    except Exception:
        collection = target_client.create_collection(name=col_name)

    docs = []
    metadatas = []
    ids = []
    
    print(f"Preparing {len(records)} records for Chroma...")
    for i, r in enumerate(records):
        log = r.get("raw_log_excerpt", "")[:1000]
        rc = r.get("root_cause", "")[:1000]
        text = f"{log} [SEP] {rc}"
        
        docs.append(text)
        meta = {
            "incident_id": r.get("incident_id", f"inc-{i}"),
            "raw_log_excerpt": log,
            "root_cause": rc,
            "remediation_steps": r.get("remediation_steps", ""),
            "source_url": r.get("source_url", ""),
            "quality": r.get("quality", "high"),
        }
        if "company" in r:
            meta["company"] = r["company"]
        metadatas.append(meta)
        ids.append(r.get("incident_id", f"inc-{i}"))

    print(f"Embedding {len(docs)} records for Chroma...")
    from clustering.cluster import _get_model
    model = _get_model()
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=False, normalize_embeddings=True).tolist()
    
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    print("Ingestion complete.")
    return collection

def retrieve(
    log_text: str,
    k: int = 1,
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
    return_all: bool = False,
):
    """Retrieve the most similar historical incident from ChromaDB, preferring quality:high."""
    if persist_directory or collection_name:
        p_dir = Path(persist_directory) if persist_directory else CHROMA_DB_PATH
        c_name = collection_name if collection_name else COLLECTION_NAME
        target_client = client if (p_dir.resolve() == CHROMA_DB_PATH.resolve()) else chromadb.PersistentClient(path=str(p_dir))
        collection = target_client.get_collection(name=c_name)
    else:
        collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = embed(log_text)
    
    # Try high quality first
    results_high = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"quality": "high"}
    )

    if return_all:
        matches = []
        if results_high.get('metadatas') and results_high['metadatas'][0]:
            for meta, dist in zip(results_high['metadatas'][0], results_high['distances'][0]):
                sim = 1.0 - (dist / 2.0)
                matches.append((meta, sim))
        if len(matches) < k:
            results_all = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            if results_all.get('metadatas') and results_all['metadatas'][0]:
                seen_ids = {m[0]['incident_id'] for m in matches}
                for meta, dist in zip(results_all['metadatas'][0], results_all['distances'][0]):
                    if meta['incident_id'] not in seen_ids:
                        sim = 1.0 - (dist / 2.0)
                        matches.append((meta, sim))
                    if len(matches) >= k:
                        break
        return matches
    
    if results_high['metadatas'] and results_high['metadatas'][0]:
        distance = results_high['distances'][0][0]
        similarity = 1.0 - (distance / 2.0)
        # Similarity floor for high quality
        if similarity >= 0.40:
            return results_high['metadatas'][0][0], similarity
            
    # Fallback to any quality if no high quality match found or similarity < 0.40
    results_all = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    if not results_all['metadatas'] or not results_all['metadatas'][0]:
        return None, 0.0
        
    distance = results_all['distances'][0][0]
    similarity = 1.0 - (distance / 2.0)
    return results_all['metadatas'][0][0], similarity

LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "20"))
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMGenerationError(Exception):
    """Raised when live LLM generation or parsing fails."""
    def __init__(self, message: str, provider_status: int = 502):
        super().__init__(message)
        self.message = message
        self.provider_status = provider_status


class RateLimiter:
    """Thread-safe client-side rate limiter respecting LLM_MAX_RPM."""
    def __init__(self, max_rpm: int = 20):
        self.max_rpm = max_rpm
        self.window = 60.0
        self.timestamps = []
        self.lock = threading.Lock()

    def acquire(self):
        if self.max_rpm <= 0:
            return
        with self.lock:
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < self.window]
            if len(self.timestamps) >= self.max_rpm:
                sleep_sec = self.window - (now - self.timestamps[0]) + 0.1
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < self.window]
            self.timestamps.append(now)


_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM)


def generate_rca(
    log_text: str,
    use_cache: bool = False,
    prompt_variant: str = "default",
    model_override: str | None = None
) -> dict:
    """Full RAG pipeline. Retrieves context and generates RCA using Groq API."""
    try:
        match_meta, similarity = retrieve(log_text, k=1)
    except Exception:
        # In case it hasn't been ingested yet
        ingest()
        match_meta, similarity = retrieve(log_text, k=1)
        
    if not match_meta:
        return {"error": "No historical context found."}
        
    model_name = model_override or os.environ.get("LLM_MODEL", LLM_MODEL)

    system_prompt = (
        "You are an expert Site Reliability Engineer (SRE) assistant analyzing infrastructure incidents.\n"
        "Analyze the incident log excerpt and correlate it with the provided historical incident.\n"
        "You must structure your response into exactly two sections labeled as follows:\n\n"
        "ROOT CAUSE:\n"
        "<concise, precise analysis of the technical root cause>\n\n"
        "REMEDIATION STEPS:\n"
        "<numbered actionable mitigation and remediation steps>\n\n"
        "Do not output internal chain-of-thought, reasoning scratchpads, or conversational preambles."
    )

    user_prompt = f"""Historical Incident Reference:
- Root Cause: {match_meta['root_cause']}
- Remediation: {match_meta['remediation_steps']}

New Incident Log Excerpt:
{log_text}"""

    prompt_used = f"{system_prompt}\n\n{user_prompt}"

    if use_cache:
        try:
            from rag.gen_cache import get_cached_generation, set_cached_generation
            cached = get_cached_generation(model=model_name, prompt_variant=prompt_variant, query_text_or_id=log_text)
            if cached:
                return {
                    "root_cause": cached["root_cause"],
                    "remediation_steps": cached["remediation_steps"],
                    "retrieved_incident": match_meta,
                    "similarity_score": round(similarity, 4),
                    "prompt_used": prompt_used,
                    "is_mock": False,
                    "model": model_name,
                    "cached": True,
                }
        except Exception as e:
            logging.warning(f"Failed to check gen_cache: {e}")

    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    is_mock = not bool(groq_api_key)

    if is_mock:
        # Fallback to mock for testing without API key
        root_cause = "[MOCK] Analyzed based on historical data."
        remediation_steps = "[MOCK] Restart service and monitor."
        return {
            "root_cause": root_cause,
            "remediation_steps": remediation_steps,
            "retrieved_incident": match_meta,
            "similarity_score": round(similarity, 4),
            "prompt_used": prompt_used,
            "is_mock": True,
            "model": model_name,
        }

    # Live call to Groq
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    last_error_status = 502
    last_error_msg = ""
    raw_content = ""

    # Up to 4 attempts (1 initial + up to 3 retries on 429, 1 retry on 5xx/timeout)
    for attempt in range(4):
        _rate_limiter.acquire()
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
                timeout=60.0
            )

            if resp.status_code == 200:
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"] or ""
                break
            elif resp.status_code == 429:
                last_error_status = 429
                last_error_msg = resp.text
                if attempt < 3:
                    retry_after = resp.headers.get("Retry-After")
                    reset_req = resp.headers.get("x-ratelimit-reset-requests")
                    sleep_sec = 2.0 * (attempt + 1)
                    if retry_after:
                        try:
                            sleep_sec = float(retry_after)
                        except ValueError:
                            pass
                    elif reset_req:
                        try:
                            if "s" in reset_req:
                                sleep_sec = float(reset_req.replace("s", "").strip())
                            elif "m" in reset_req:
                                parts = reset_req.split("m")
                                sleep_sec = float(parts[0]) * 60 + float(parts[1].replace("s", ""))
                        except Exception:
                            pass
                    time.sleep(max(1.0, min(sleep_sec, 60.0)))
                    continue
                else:
                    raise LLMGenerationError(
                        f"Groq 429 Rate Limit Exceeded after 3 retries: {last_error_msg}",
                        provider_status=429
                    )
            elif resp.status_code >= 500:
                last_error_status = resp.status_code
                last_error_msg = resp.text
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                else:
                    raise LLMGenerationError(
                        f"Groq server error ({resp.status_code}) after retry: {last_error_msg}",
                        provider_status=resp.status_code
                    )
            else:
                raise LLMGenerationError(
                    f"Groq API error ({resp.status_code}): {resp.text}",
                    provider_status=resp.status_code
                )
        except requests.exceptions.Timeout as e:
            last_error_status = 504
            last_error_msg = str(e)
            if attempt == 0:
                time.sleep(2.0)
                continue
            else:
                raise LLMGenerationError(f"Groq request timeout after 60s: {e}", provider_status=504)
        except requests.exceptions.RequestException as e:
            last_error_status = 502
            last_error_msg = str(e)
            if attempt == 0:
                time.sleep(2.0)
                continue
            else:
                raise LLMGenerationError(f"Groq network error: {e}", provider_status=502)

    if not raw_content:
        raise LLMGenerationError(
            f"Empty response from Groq API: {last_error_msg}",
            provider_status=last_error_status
        )

    # Clean any leaked reasoning text (e.g. <think>...</think>)
    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    if not cleaned_text:
        cleaned_text = raw_content.strip()

    # Parse ROOT CAUSE and REMEDIATION STEPS defensively
    norm_text = cleaned_text
    for marker in ["### ROOT CAUSE", "## ROOT CAUSE", "ROOT CAUSE:", "Root Cause:"]:
        if marker in norm_text:
            norm_text = norm_text.replace(marker, "ROOT CAUSE:")
            break

    for marker in ["### REMEDIATION STEPS", "## REMEDIATION STEPS", "REMEDIATION STEPS:", "Remediation Steps:", "REMEDIATION:"]:
        if marker in norm_text:
            norm_text = norm_text.replace(marker, "REMEDIATION STEPS:")
            break

    if "ROOT CAUSE:" not in norm_text:
        raise LLMGenerationError(f"Groq output missing required 'ROOT CAUSE' section: {cleaned_text[:200]}", provider_status=502)
    if "REMEDIATION STEPS:" not in norm_text:
        raise LLMGenerationError(f"Groq output missing required 'REMEDIATION STEPS' section: {cleaned_text[:200]}", provider_status=502)

    parts = norm_text.split("REMEDIATION STEPS:")
    rc_part = parts[0]
    if "ROOT CAUSE:" in rc_part:
        rc_part = rc_part.split("ROOT CAUSE:")[-1]
    root_cause = rc_part.strip()
    remediation_steps = parts[1].strip()

    if not root_cause:
        raise LLMGenerationError("Parsed 'ROOT CAUSE' section is blank.", provider_status=502)
    if not remediation_steps:
        raise LLMGenerationError("Parsed 'REMEDIATION STEPS' section is blank.", provider_status=502)

    if use_cache:
        try:
            from rag.gen_cache import set_cached_generation
            set_cached_generation(
                model=model_name,
                prompt_variant=prompt_variant,
                query_text_or_id=log_text,
                root_cause=root_cause,
                remediation_steps=remediation_steps,
                raw_output=raw_content
            )
        except Exception as e:
            logging.warning(f"Failed to write gen_cache: {e}")

    return {
        "root_cause": root_cause,
        "remediation_steps": remediation_steps,
        "retrieved_incident": match_meta,
        "similarity_score": round(similarity, 4),
        "prompt_used": prompt_used,
        "is_mock": False,
        "model": model_name,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TraceMind RAG Ingestion CLI")
    parser.add_argument("--ingest-train", action="store_true", help="Ingest train split into rag/chroma_db_train")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    parser.add_argument("--source", type=str, default=None, help="Source JSON file")
    parser.add_argument("--persist", type=str, default=None, help="Persist directory")
    parser.add_argument("--collection", type=str, default=None, help="Collection name")
    cli_args = parser.parse_args()

    if cli_args.ingest_train:
        train_path = RAG_DIR.parent / "data" / "splits" / "train.json"
        persist_dir = RAG_DIR / "chroma_db_train"
        ingest(
            force=cli_args.force,
            source=train_path,
            persist_directory=persist_dir,
            collection_name="postmortems_train"
        )
    else:
        ingest(
            force=cli_args.force,
            source=cli_args.source,
            persist_directory=cli_args.persist,
            collection_name=cli_args.collection
        )
