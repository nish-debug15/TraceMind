import os
import json
import chromadb
from pathlib import Path
from clustering.cluster import embed
from huggingface_hub import InferenceClient

RAG_DIR = Path(__file__).parent
DATA_PATH = RAG_DIR.parent / "data" / "postmortems.json"
CHROMA_DB_PATH = RAG_DIR / "chroma_db"
COLLECTION_NAME = "postmortems"

# Initialize Chroma client
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

def ingest():
    """Ingest postmortems into ChromaDB."""
    print("Ingesting data into ChromaDB...")
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Collection already exists, skipping ingestion.")
        return
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    docs = []
    metadatas = []
    ids = []
    embeddings = []
    
    print(f"Embedding {len(records)} records for Chroma...")
    for i, r in enumerate(records):
        log = r.get("raw_log_excerpt", "")[:1000]
        rc = r.get("root_cause", "")[:1000]
        text = f"{log} [SEP] {rc}"
        
        docs.append(text)
        metadatas.append({
            "incident_id": r.get("incident_id", f"inc-{i}"),
            "raw_log_excerpt": log,
            "root_cause": rc,
            "remediation_steps": r.get("remediation_steps", ""),
            "source_url": r.get("source_url", "")
        })
        ids.append(r.get("incident_id", f"inc-{i}"))
        embeddings.append(embed(text))
    
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    print("Ingestion complete.")

def retrieve(log_text: str, k: int = 1):
    """Retrieve the most similar historical incident from ChromaDB."""
    collection = client.get_collection(name=COLLECTION_NAME)
    query_embedding = embed(log_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    if not results['metadatas'] or not results['metadatas'][0]:
        return None, 0.0
        
    match_meta = results['metadatas'][0][0]
    distance = results['distances'][0][0]
    # L2 distance on normalized vectors: distance = 2 * (1 - cosine_sim) -> cosine_sim = 1 - (distance / 2)
    similarity = 1.0 - (distance / 2.0)
    
    return match_meta, similarity

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
        generated_text = "Root Cause: (Mocked because HF_TOKEN is not set) Analyzed based on historical data.\nRemediation Steps: Restart service and monitor."
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
        "prompt_used": prompt
    }

if __name__ == "__main__":
    ingest()
