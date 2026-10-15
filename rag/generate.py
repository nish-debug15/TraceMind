import os
import json
import chromadb
from pathlib import Path
from clustering.cluster import embed
from huggingface_hub import InferenceClient

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

LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

def generate_rca(log_text: str) -> dict:
    """Full RAG pipeline. Retrieves context and generates RCA using HF Inference API."""
    try:
        match_meta, similarity = retrieve(log_text, k=1)
    except Exception:
        # In case it hasn't been ingested yet
        ingest()
        match_meta, similarity = retrieve(log_text, k=1)
        
    if not match_meta:
        return {"error": "No historical context found."}
        
    system_prompt = (
        "You are an expert Site Reliability Engineer (SRE) assistant analyzing infrastructure incidents.\n"
        "Reasoning: low\n"
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
    
    hf_token = os.environ.get("HF_TOKEN")
    is_mock = not bool(hf_token)
    model_name = LLM_MODEL

    if is_mock:
        # Fallback to mock for testing without API key
        root_cause = "[MOCK] Analyzed based on historical data."
        remediation_steps = "[MOCK] Restart service and monitor."
    else:
        import time
        import re
        import logging

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        last_error = None
        raw_content = ""
        hf_client = InferenceClient(token=hf_token, timeout=60.0)

        chat_kwargs = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.2,
        }

        # 1 retry with backoff on 429/5xx
        for attempt in range(2):
            try:
                try:
                    response = hf_client.chat_completion(**chat_kwargs, extra_body={"reasoning_effort": "low"})
                except TypeError:
                    response = hf_client.chat_completion(**chat_kwargs)
                
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                
                # Check reasoning/reasoning_content field defensively if content is empty
                if not content.strip():
                    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                    if reasoning:
                        logging.warning("Main message content was empty; using reasoning content fallback.")
                        content = reasoning
                    else:
                        raise ValueError("Received empty content and empty reasoning from LLM response.")

                raw_content = content
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                is_retryable = any(code in err_str for code in ["429", "500", "502", "503", "504", "Rate limit", "timed out", "timeout"])
                if attempt == 0 and is_retryable:
                    time.sleep(2.0)
                    continue
                else:
                    break

        if not raw_content and last_error is not None:
            # Live call failed — return explicit error, is_mock must NEVER be true
            return {
                "root_cause": f"(Error calling LLM) {str(last_error)}",
                "remediation_steps": "N/A",
                "retrieved_incident": match_meta,
                "similarity_score": round(similarity, 4),
                "prompt_used": prompt_used,
                "is_mock": False,
                "model": model_name,
            }

        try:
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
                raise ValueError(f"LLM output missing required 'ROOT CAUSE' section: {cleaned_text[:200]}")
            if "REMEDIATION STEPS:" not in norm_text:
                raise ValueError(f"LLM output missing required 'REMEDIATION STEPS' section: {cleaned_text[:200]}")

            parts = norm_text.split("REMEDIATION STEPS:")
            rc_part = parts[0]
            if "ROOT CAUSE:" in rc_part:
                rc_part = rc_part.split("ROOT CAUSE:")[-1]
            root_cause = rc_part.strip()
            remediation_steps = parts[1].strip()

            if not root_cause:
                raise ValueError("Parsed 'ROOT CAUSE' section is blank.")
            if not remediation_steps:
                raise ValueError("Parsed 'REMEDIATION STEPS' section is blank.")
        except Exception as e:
            return {
                "root_cause": f"(Error parsing LLM response) {str(e)}",
                "remediation_steps": "N/A",
                "retrieved_incident": match_meta,
                "similarity_score": round(similarity, 4),
                "prompt_used": prompt_used,
                "is_mock": False,
                "model": model_name,
            }

    return {
        "root_cause": root_cause,
        "remediation_steps": remediation_steps,
        "retrieved_incident": match_meta,
        "similarity_score": round(similarity, 4),
        "prompt_used": prompt_used,
        "is_mock": is_mock,
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
