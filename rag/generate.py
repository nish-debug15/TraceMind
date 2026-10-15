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
        
    prompt = f"""You are an SRE assistant analyzing infrastructure incidents. Use the following historical incident to understand the context and format, then analyze the New Log Input and provide a precise Root Cause Analysis and actionable Remediation Steps.

Historical Incident:
- Root Cause: {match_meta['root_cause']}
- Remediation: {match_meta['remediation_steps']}

New Log Input:
{log_text}

Provide your response in the following format:
Root Cause: (your analysis)
Remediation Steps: (your steps)
"""
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Fallback to mock for testing without API key
        generated_text = "Root Cause: [MOCK] Analyzed based on historical data.\nRemediation Steps: [MOCK] Restart service and monitor."
    else:
        try:
            hf_client = InferenceClient(token=hf_token)
            response = hf_client.text_generation(
                prompt, 
                model="mistralai/Mistral-7B-Instruct-v0.3",
                max_new_tokens=256,
                temperature=0.2
            )
            generated_text = response
        except Exception as e:
            generated_text = f"Root Cause: (Error calling LLM) {str(e)}\nRemediation Steps: N/A"

    # Simple parsing logic
    root_cause = generated_text
    remediation_steps = ""
    if "Remediation Steps:" in generated_text:
        parts = generated_text.split("Remediation Steps:")
        root_cause = parts[0].replace("Root Cause:", "").strip()
        remediation_steps = parts[1].strip()

    return {
        "root_cause": root_cause,
        "remediation_steps": remediation_steps,
        "retrieved_incident": match_meta,
        "similarity_score": round(similarity, 4),
        "prompt_used": prompt,
        "is_mock": not bool(hf_token)
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
