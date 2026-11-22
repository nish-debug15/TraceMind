"""
eval/loo/retrieval.py
---------------------
Runs retrieval on the 93 LOO sample incidents from rag/chroma_db_train under 3 conditions:
  (a) same-company allowed (self excluded, identical source_url excluded, near-duplicate filter applied)
  (b) cross-company only (same company excluded, self excluded, near-duplicate filter applied)
  (c) same-company allowed (self excluded, identical source_url excluded, NO near-duplicate filter)

Pool size: n_results=100 for both quality:high and all-corpus passes.
Near-duplicate definition (eval/PREREG.md):
- Embedding cosine similarity >= 0.95
OR
- 5-gram token-shingle Jaccard similarity >= 0.80 (on log excerpt or root cause)

HARD ASSERTS:
- Every query has >= 3 valid hits in EACH of conditions (a), (b), (c) after filtering.
- Leak guard: Query incident's own ID and source_url NEVER appear in retrieved hits.

Outputs:
- eval/loo/retrieval_pool_diff.json (diff report between 30-pool and 100-pool)
- eval/loo/retrieval.json (100-pool retrieval results)
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import chromadb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustering.cluster import embed

SAMPLE_FILE = PROJECT_ROOT / "eval" / "loo" / "sample.json"
CHROMA_TRAIN_DIR = PROJECT_ROOT / "rag" / "chroma_db_train"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "loo" / "retrieval.json"
DIFF_FILE = PROJECT_ROOT / "eval" / "loo" / "retrieval_pool_diff.json"


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def get_5gram_shingles(text: str) -> Set[Tuple[str, ...]]:
    tokens = tokenize_words(text)
    if len(tokens) < 5:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


def compute_jaccard(s1: Set[Tuple[str, ...]], s2: Set[Tuple[str, ...]]) -> float:
    if not s1 or not s2:
        return 0.0
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def is_near_duplicate(query: Dict[str, Any], cand_meta: Dict[str, Any], sim: float) -> Tuple[bool, str, float]:
    if sim >= 0.95:
        return True, "cosine_sim_ge_0.95", round(sim, 4)

    q_log_shingles = get_5gram_shingles(query.get("raw_log_excerpt", ""))
    c_log_shingles = get_5gram_shingles(cand_meta.get("raw_log_excerpt", ""))
    j_log = compute_jaccard(q_log_shingles, c_log_shingles)

    q_rc_shingles = get_5gram_shingles(query.get("root_cause", ""))
    c_rc_shingles = get_5gram_shingles(cand_meta.get("root_cause", ""))
    j_rc = compute_jaccard(q_rc_shingles, c_rc_shingles)

    max_j = max(j_log, j_rc)
    if max_j >= 0.80:
        reason = "log_jaccard_ge_0.80" if j_log >= j_rc else "root_cause_jaccard_ge_0.80"
        return True, reason, round(max_j, 4)

    return False, "none", round(max_j, 4)


def run_loo_retrieval() -> Dict[str, Any]:
    print("=" * 60)
    print("TraceMind E2: Leave-One-Out (LOO) Retrieval (Pool Size = 100)")
    print("=" * 60)

    # 1. Load existing 30-pool retrieval if available for diff comparison
    old_retrieval_map = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_retrieval_map = {item["query_incident_id"]: item for item in old_data.get("retrieval", [])}
            print(f"Loaded existing 30-pool retrieval ({len(old_retrieval_map)} queries) for diff comparison.")
        except Exception as e:
            print(f"Notice: Could not load previous retrieval for diff: {e}")

    assert SAMPLE_FILE.exists(), f"Missing LOO sample file: {SAMPLE_FILE}"
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    sample_incidents = sample_data["sample"]
    num_queries = len(sample_incidents)
    print(f"Loaded {num_queries} LOO sample queries across {sample_data['metadata']['num_sampled_companies']} companies.")

    assert CHROMA_TRAIN_DIR.exists(), f"Missing train Chroma DB: {CHROMA_TRAIN_DIR}"
    client = chromadb.PersistentClient(path=str(CHROMA_TRAIN_DIR))
    collection = client.get_collection("postmortems_train")
    print(f"Connected to ChromaDB 'postmortems_train' with {collection.count()} indexed postmortems.")

    print("\nPre-embedding query log excerpts...")
    query_logs = [inc.get("raw_log_excerpt", "") for inc in sample_incidents]
    query_embeddings = embed(query_logs)

    retrieval_results = []
    total_near_dups_removed_a = 0
    total_near_dups_removed_b = 0
    queries_with_divergent_top1_ac = 0
    queries_with_divergent_top3_ac = 0

    diff_records = []
    diff_count_a = 0
    diff_count_b = 0
    diff_count_c = 0
    recovered_count_b = 0

    for idx, query in enumerate(sample_incidents):
        qid = query["incident_id"]
        q_comp = query.get("company", "Unknown")
        q_url = (query.get("source_url") or "").strip()
        q_emb = query_embeddings[idx]

        # Query pool from Chroma with n_results=100 uniformly for high-quality and all-corpus passes
        res_high = collection.query(
            query_embeddings=[q_emb],
            n_results=100,
            where={"quality": "high"}
        )
        res_all = collection.query(
            query_embeddings=[q_emb],
            n_results=100
        )

        # Merge pool maintaining order (pass 1 high quality first, then pass 2)
        candidate_pool: List[Dict[str, Any]] = []
        seen_cand_ids: Set[str] = set()

        if res_high.get("metadatas") and res_high["metadatas"][0]:
            for meta, dist in zip(res_high["metadatas"][0], res_high["distances"][0]):
                cid = meta["incident_id"]
                if cid not in seen_cand_ids:
                    seen_cand_ids.add(cid)
                    sim = 1.0 - (dist / 2.0)
                    candidate_pool.append({"metadata": meta, "similarity": sim, "pass": 1})

        if res_all.get("metadatas") and res_all["metadatas"][0]:
            for meta, dist in zip(res_all["metadatas"][0], res_all["distances"][0]):
                cid = meta["incident_id"]
                if cid not in seen_cand_ids:
                    seen_cand_ids.add(cid)
                    sim = 1.0 - (dist / 2.0)
                    candidate_pool.append({"metadata": meta, "similarity": sim, "pass": 2})

        # Condition (a): same-company allowed, self excluded, identical source_url excluded, near-duplicate filtered
        hits_a: List[Dict[str, Any]] = []
        near_dups_a: List[Dict[str, Any]] = []
        for cand in candidate_pool:
            meta = cand["metadata"]
            cid = meta["incident_id"]
            curl = (meta.get("source_url") or "").strip()
            sim = cand["similarity"]

            # Leak guard check
            if cid == qid or (q_url and curl and curl == q_url):
                continue

            near_dup, reason, val = is_near_duplicate(query, meta, sim)
            if near_dup:
                near_dups_a.append({
                    "incident_id": cid,
                    "company": meta.get("company", "Unknown"),
                    "similarity": round(sim, 4),
                    "reason": reason,
                    "metric_value": val
                })
                continue

            hits_a.append({
                "rank": len(hits_a) + 1,
                "incident_id": cid,
                "company": meta.get("company", "Unknown"),
                "similarity_score": round(sim, 4),
                "is_same_company": bool(meta.get("company") == q_comp),
                "retrieval_pass": cand["pass"],
                "root_cause": meta.get("root_cause", ""),
                "remediation_steps": meta.get("remediation_steps", ""),
                "raw_log_excerpt": meta.get("raw_log_excerpt", ""),
                "source_url": curl,
            })
            if len(hits_a) >= 3:
                break

        # Condition (b): cross-company only, self excluded, identical source_url excluded, near-duplicate filtered
        hits_b: List[Dict[str, Any]] = []
        near_dups_b: List[Dict[str, Any]] = []
        for cand in candidate_pool:
            meta = cand["metadata"]
            cid = meta["incident_id"]
            curl = (meta.get("source_url") or "").strip()
            c_comp = meta.get("company", "Unknown")
            sim = cand["similarity"]

            # Leak guard & cross-company check
            if cid == qid or (q_url and curl and curl == q_url) or (c_comp == q_comp):
                continue

            near_dup, reason, val = is_near_duplicate(query, meta, sim)
            if near_dup:
                near_dups_b.append({
                    "incident_id": cid,
                    "company": c_comp,
                    "similarity": round(sim, 4),
                    "reason": reason,
                    "metric_value": val
                })
                continue

            hits_b.append({
                "rank": len(hits_b) + 1,
                "incident_id": cid,
                "company": c_comp,
                "similarity_score": round(sim, 4),
                "is_same_company": False,
                "retrieval_pass": cand["pass"],
                "root_cause": meta.get("root_cause", ""),
                "remediation_steps": meta.get("remediation_steps", ""),
                "raw_log_excerpt": meta.get("raw_log_excerpt", ""),
                "source_url": curl,
            })
            if len(hits_b) >= 3:
                break

        # Condition (c): same-company allowed, self excluded, identical source_url excluded, NO near-duplicate filter
        hits_c: List[Dict[str, Any]] = []
        for cand in candidate_pool:
            meta = cand["metadata"]
            cid = meta["incident_id"]
            curl = (meta.get("source_url") or "").strip()
            sim = cand["similarity"]

            # Leak guard check
            if cid == qid or (q_url and curl and curl == q_url):
                continue

            hits_c.append({
                "rank": len(hits_c) + 1,
                "incident_id": cid,
                "company": meta.get("company", "Unknown"),
                "similarity_score": round(sim, 4),
                "is_same_company": bool(meta.get("company") == q_comp),
                "retrieval_pass": cand["pass"],
                "root_cause": meta.get("root_cause", ""),
                "remediation_steps": meta.get("remediation_steps", ""),
                "raw_log_excerpt": meta.get("raw_log_excerpt", ""),
                "source_url": curl,
            })
            if len(hits_c) >= 3:
                break

        # --- HARD ASSERTS (Rule 2) ---
        assert len(hits_a) >= 3, f"HARD ASSERT FAILED: Query {qid} ({q_comp}) has {len(hits_a)} hits in Condition (a) (expected >= 3)"
        assert len(hits_b) >= 3, f"HARD ASSERT FAILED: Query {qid} ({q_comp}) has {len(hits_b)} hits in Condition (b) (expected >= 3)"
        assert len(hits_c) >= 3, f"HARD ASSERT FAILED: Query {qid} ({q_comp}) has {len(hits_c)} hits in Condition (c) (expected >= 3)"

        # --- LEAK GUARD HARD ASSERTS ---
        for cond_name, hits in [("condition_a", hits_a), ("condition_b", hits_b), ("condition_c", hits_c)]:
            for h in hits:
                if h["incident_id"] == qid:
                    raise RuntimeError(f"CRITICAL LEAK GUARD FAILURE: Query ID {qid} appeared in {cond_name} hits!")
                if q_url and h["source_url"] and h["source_url"] == q_url:
                    raise RuntimeError(f"CRITICAL LEAK GUARD FAILURE: Query URL {q_url} appeared in {cond_name} hit {h['incident_id']}!")

        total_near_dups_removed_a += len(near_dups_a)
        total_near_dups_removed_b += len(near_dups_b)

        top1_a = hits_a[0]["incident_id"]
        top1_c = hits_c[0]["incident_id"]
        if top1_a != top1_c:
            queries_with_divergent_top1_ac += 1

        top3_a_ids = [h["incident_id"] for h in hits_a]
        top3_b_ids = [h["incident_id"] for h in hits_b]
        top3_c_ids = [h["incident_id"] for h in hits_c]
        if top3_a_ids != top3_c_ids:
            queries_with_divergent_top3_ac += 1

        # Diff against 30-pool
        old_item = old_retrieval_map.get(qid)
        if old_item:
            old_a = [h["incident_id"] for h in old_item.get("condition_a_same_company_filtered", {}).get("hits", [])]
            old_b = [h["incident_id"] for h in old_item.get("condition_b_cross_company_filtered", {}).get("hits", [])]
            old_c = [h["incident_id"] for h in old_item.get("condition_c_same_company_unfiltered", {}).get("hits", [])]

            chg_a = bool(old_a != top3_a_ids)
            chg_b = bool(old_b != top3_b_ids)
            chg_c = bool(old_c != top3_c_ids)
            rec_b = bool(len(old_b) == 0 and len(top3_b_ids) >= 3)

            if chg_a:
                diff_count_a += 1
            if chg_b:
                diff_count_b += 1
            if chg_c:
                diff_count_c += 1
            if rec_b:
                recovered_count_b += 1

            diff_records.append({
                "query_incident_id": qid,
                "company": q_comp,
                "condition_a_changed": chg_a,
                "condition_b_changed": chg_b,
                "condition_c_changed": chg_c,
                "condition_b_recovered_from_zero": rec_b,
                "old_top3": {"condition_a": old_a, "condition_b": old_b, "condition_c": old_c},
                "new_top3": {"condition_a": top3_a_ids, "condition_b": top3_b_ids, "condition_c": top3_c_ids},
            })

        retrieval_results.append({
            "query_incident_id": qid,
            "query_company": q_comp,
            "query_source_url": q_url,
            "condition_a_same_company_filtered": {
                "hits": hits_a,
                "near_duplicates_removed": near_dups_a,
                "count_near_duplicates_removed": len(near_dups_a),
            },
            "condition_b_cross_company_filtered": {
                "hits": hits_b,
                "near_duplicates_removed": near_dups_b,
                "count_near_duplicates_removed": len(near_dups_b),
            },
            "condition_c_same_company_unfiltered": {
                "hits": hits_c,
            },
            "near_dup_changed_top1": bool(top1_a != top1_c),
            "near_dup_changed_top3": bool(top3_a_ids != top3_c_ids),
        })

    # Save diff report first
    if diff_records:
        diff_report = {
            "metadata": {
                "num_queries": num_queries,
                "old_pool_size": 30,
                "new_pool_size": 100,
                "queries_changed_condition_a_top3": diff_count_a,
                "queries_changed_condition_b_top3": diff_count_b,
                "queries_changed_condition_c_top3": diff_count_c,
                "condition_b_recovered_from_zero_hits": recovered_count_b,
            },
            "diff_summary": (
                f"Expanded retrieval candidate pool from 30 to 100. "
                f"Condition (a) top-3 changed in {diff_count_a}/{num_queries} queries; "
                f"Condition (b) top-3 changed in {diff_count_b}/{num_queries} queries (recovering {recovered_count_b} Twilio queries from 0 hits to 3 hits); "
                f"Condition (c) top-3 changed in {diff_count_c}/{num_queries} queries."
            ),
            "per_query_diff": diff_records,
        }
        DIFF_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DIFF_FILE, "w", encoding="utf-8") as f:
            json.dump(diff_report, f, indent=2)
        print(f"[Diff Report] Successfully written to {DIFF_FILE}")
        print(f"  Condition A changed: {diff_count_a}/{num_queries}")
        print(f"  Condition B changed: {diff_count_b}/{num_queries} (recovered {recovered_count_b} from 0 hits)")
        print(f"  Condition C changed: {diff_count_c}/{num_queries}")

    summary_stats = {
        "num_queries": num_queries,
        "leak_guard_status": "PASSED (zero self-id or source-url leaks across all 3 conditions)",
        "all_queries_have_ge_3_hits": True,
        "condition_a_total_near_duplicates_filtered": total_near_dups_removed_a,
        "condition_b_total_near_duplicates_filtered": total_near_dups_removed_b,
        "queries_with_divergent_top1_between_a_and_c": queries_with_divergent_top1_ac,
        "queries_with_divergent_top3_between_a_and_c": queries_with_divergent_top3_ac,
        "condition_a_same_company_in_top1_count": sum(1 for r in retrieval_results if r["condition_a_same_company_filtered"]["hits"][0]["is_same_company"]),
        "condition_c_same_company_in_top1_count": sum(1 for r in retrieval_results if r["condition_c_same_company_unfiltered"]["hits"][0]["is_same_company"]),
    }

    results_obj = {
        "metadata": {
            "source_corpus": "rag/chroma_db_train (collection: postmortems_train)",
            "sample_file": str(SAMPLE_FILE.relative_to(PROJECT_ROOT)),
            "num_queries": num_queries,
            "candidate_pool_size": 100,
            "near_dup_thresholds": {
                "cosine_sim_ge": 0.95,
                "5gram_shingle_jaccard_ge": 0.80,
            },
            "summary_stats": summary_stats,
        },
        "retrieval": retrieval_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_obj, f, indent=2)

    print(f"\n[LOO Retrieval] Successfully saved retrieval to {OUTPUT_FILE}")
    print(f"Summary stats:")
    for k, v in summary_stats.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return results_obj


if __name__ == "__main__":
    run_loo_retrieval()
