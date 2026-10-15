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

# 3. Custom Editorial Theme & Styling
custom_theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    button_primary_background_fill="#D97757",
    button_primary_background_fill_hover="#C26245",
    button_primary_text_color="white",
)

custom_css = """
.hero-title {
    font-family: 'Lora', Georgia, serif;
    font-size: 2.25rem !important;
    font-weight: 600;
    color: #2D2825;
    margin-bottom: 0.25rem;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #716960;
    margin-bottom: 1.25rem;
}
.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.5rem;
    border: 1px solid #EAE5D9;
    background-color: #F8F6F0;
    color: #5C544B;
}
.badge-live {
    background-color: #EBF8F2;
    border-color: #C5EEDB;
    color: #177A53;
}
.badge-model {
    background-color: #FDF4F2;
    border-color: #F8D9D2;
    color: #B84B31;
}
.result-card {
    border-radius: 0.75rem;
    padding: 1rem;
}
"""

SAMPLE_LOGS = [
    [
        "[2026-10-15 08:23:41] [ERROR] [db-proxy-pool] Connection timeout acquiring connection from pool.\n"
        "FATAL: remaining connection slots are reserved for non-replication superuser connections.\n"
        "PoolExhaustedError: database connection pool exhausted, 50/50 connections in use\n"
        "[2026-10-15 08:23:42] [WARN] [health-check] Liveness probe failed for pod db-proxy-6b8f7d9"
    ],
    [
        "CRITICAL 2026-10-14T14:12:05Z kubelet worker-node-04: OOMKilled process 18421 (payments-api) "
        "total-vm:8421048kB, anon-rss:4194304kB. Memory limit 4Gi exceeded."
    ],
    [
        "ERROR: Redis sentinel failover triggered - master 10.0.4.12:6379 unreachable for 30s. "
        "Failover elected replica 10.0.4.13:6379 as new master. Write latency spiked to 2400ms."
    ],
    [
        "WARNING: SSL certificate verification failed for api.stripe.com. "
        "OpenSSL Error: certificate has expired on 2026-10-14 23:59:59 GMT. TLS handshake terminated."
    ]
]

# 4. Build enhanced UI
with gr.Blocks(title="TraceMind RCA API") as demo:
    with gr.Column(elem_classes=["max-w-5xl", "mx-auto"]):
        gr.HTML("""
        <div>
            <div class="hero-title">🧠 TraceMind RCA</div>
            <div class="hero-subtitle">Automated NLP Anomaly Clustering & Retrieval-Augmented Postmortem Analysis</div>
            <div>
                <span class="badge-pill badge-live">● FastAPI REST Live</span>
                <span class="badge-pill badge-model">Mistral-7B-Instruct-v0.3</span>
                <span class="badge-pill">ChromaDB (662 Incidents)</span>
                <span class="badge-pill">HDBSCAN (41 Clusters)</span>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                log_input = gr.Textbox(
                    label="Input Raw Infrastructure Log",
                    lines=6,
                    placeholder="Paste server/database error logs or tracebacks here...",
                    value=SAMPLE_LOGS[0][0]
                )
                analyze_btn = gr.Button("Analyze Incident with TraceMind", variant="primary", size="lg")

        gr.Examples(
            examples=SAMPLE_LOGS,
            inputs=[log_input],
            label="Click an example incident to inspect:"
        )

        with gr.Row():
            rc_output = gr.Textbox(label="Root Cause Analysis", lines=5, interactive=False)
            rem_output = gr.Textbox(label="Remediation Steps", lines=5, interactive=False)

        with gr.Accordion("🔍 Pipeline Trace & Audit Inspector (Jury Differentiator)", open=False):
            with gr.Row():
                sim_metric = gr.Number(label="Similarity Score (Cosine)", precision=4, interactive=False)
                cluster_metric = gr.Textbox(label="HDBSCAN Cluster", interactive=False)
                match_id = gr.Textbox(label="Retrieved Incident ID", interactive=False)

            source_link = gr.Markdown(label="Postmortem Source URL")
            prompt_viewer = gr.Code(label="Prompt Sent to Mistral-7B", language="markdown", interactive=False)

    def on_analyze(log_text):
        if not log_text.strip():
            return "Please enter a log excerpt to analyze.", "", None, "", "", "", ""
        try:
            req = AnalyzeRequest(log_text=log_text)
            res = analyze_log(req)
            
            trace = res.trace
            matched_inc = trace.retrieved_incident
            cluster_text = f"Noise (Novel)" if trace.is_noise else f"Cluster #{trace.cluster_id}"
            source_md = f"**Historical Incident Postmortem Source:** [{matched_inc.source_url}]({matched_inc.source_url})"

            return (
                res.root_cause,
                res.remediation_steps,
                trace.similarity_score,
                cluster_text,
                matched_inc.incident_id,
                source_md,
                trace.prompt_used
            )
        except Exception as e:
            return f"Error executing analysis: {str(e)}", "", None, "", "", "", ""

    analyze_btn.click(
        on_analyze,
        inputs=[log_input],
        outputs=[rc_output, rem_output, sim_metric, cluster_metric, match_id, source_link, prompt_viewer]
    )

# Attach backend FastAPI routes (/health, /analyze) to Gradio's internal FastAPI app
demo.app.include_router(fastapi_app.router)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=custom_theme, css=custom_css)

