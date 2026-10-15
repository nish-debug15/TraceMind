import os
import gradio as gr
from backend.main import app as fastapi_app
from backend.models import AnalyzeRequest
from backend.main import analyze_log

# Auto-build indices on cold start if running on fresh cloud host
from clustering.cluster import INDEX_PATH
from clustering.build_index import build
from rag.generate import ingest

# 1. ZeroGPU runtime compatibility hook
try:
    import spaces
    @spaces.GPU
    def zero_gpu_anchor():
        """Satisfies Hugging Face ZeroGPU runtime requirement."""
        return True
except (ImportError, Exception):
    def zero_gpu_anchor():
        return True

# 2. Cold-start index verification
if not INDEX_PATH.exists():
    print("[Startup] Cluster index not found. Building from train.json...")
    build()

print("[Startup] Verifying ChromaDB collection...")
ingest()

# 3. Build interactive Gradio demo
with gr.Blocks(title="TraceMind RCA Backend") as demo:
    gr.Markdown("# 🧠 TraceMind — SRE Root Cause Analysis Service")
    gr.Markdown(
        "This Hugging Face Space hosts the **FastAPI backend** for TraceMind.\n\n"
        "- **API Status:** `GET /health`\n"
        "- **RCA Endpoint:** `POST /analyze`\n"
        "- **Vercel Frontend:** Connects to this endpoint via `NEXT_PUBLIC_API_URL`."
    )
    
    with gr.Row():
        log_input = gr.Textbox(
            label="Raw Log Excerpt",
            lines=5,
            placeholder="Paste log traces here...",
            value="[ERROR] [db-proxy-pool] Connection timeout acquiring connection from pool. database connection pool exhausted, 50/50 connections in use"
        )
    analyze_btn = gr.Button("Analyze Incident", variant="primary")
    
    with gr.Row():
        rc_output = gr.Textbox(label="Root Cause", lines=4)
        rem_output = gr.Textbox(label="Remediation Steps", lines=4)
        
    trace_output = gr.JSON(label="Pipeline Trace (Similarity, Retrieved Incident, Prompt)")
    
    def on_analyze(log_text):
        if not log_text.strip():
            return "Please provide log text.", "", {}
        try:
            req = AnalyzeRequest(log_text=log_text)
            res = analyze_log(req)
            return res.root_cause, res.remediation_steps, res.trace.model_dump()
        except Exception as e:
            return f"Error: {str(e)}", "", {}

    analyze_btn.click(on_analyze, inputs=[log_input], outputs=[rc_output, rem_output, trace_output])

# 4. Attach backend FastAPI routes (/health, /analyze) to Gradio's internal FastAPI app
demo.app.include_router(fastapi_app.router)

# 5. Launch using native Gradio launcher (compatible with HF Spaces & ZeroGPU)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
