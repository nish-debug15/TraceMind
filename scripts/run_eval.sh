#!/usr/bin/env bash
# One-command reproducible evaluation of TraceMind RAG vs Zero-Shot baseline
# Evaluates on data/splits/test.json against train-only index rag/chroma_db_train

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "TraceMind: One-Command Reproducible Evaluation"
echo "Dataset: data/splits/test.json (93 held-out test incidents)"
echo "Index:   rag/chroma_db_train (662 postmortems_train records)"
echo "============================================================"

python eval/evaluate.py "$@"
