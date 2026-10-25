import os
from pathlib import Path
import gradio as gr
from backend.main import app as fastapi_app
from backend.models import AnalyzeRequest
from backend.main import analyze_log

# Auto-build indices on cold start if running on fresh cloud host
from clustering.cluster import INDEX_PATH
from clustering.build_index import build
from rag.generate import ingest, LLM_MODEL

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

# 2. Hook into Gradio's App.create_app so all launches include FastAPI REST routes
_original_create_app = gr.routes.App.create_app

@staticmethod
def _custom_create_app(*args, **kwargs):
    app = _original_create_app(*args, **kwargs)
    app.include_router(fastapi_app.router)
    app.include_router(fastapi_app.router, prefix="/gradio_api")
    for exc, handler in fastapi_app.exception_handlers.items():
        app.add_exception_handler(exc, handler)
    return app

gr.routes.App.create_app = _custom_create_app

# 3. Cold-start index verification
if not INDEX_PATH.exists():
    print("[Startup] Cluster index not found. Building from train.json...")
    build()

print("[Startup] Verifying ChromaDB collection...")
ingest()

# 4. Load CSS & Build Editorial Base Theme
CSS_PATH = Path(__file__).parent / "styles.css"
custom_css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""

editorial_theme = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#FAF9F5",
    body_background_fill_dark="#FAF9F5",
    body_text_color="#1F1E1D",
    body_text_color_dark="#1F1E1D",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_width="1px",
    block_border_color="#EAE5D9",
    block_border_color_dark="#EAE5D9",
    block_radius="10px",
    block_shadow="0 1px 2px rgba(0, 0, 0, 0.04)",
    button_primary_background_fill="#D97757",
    button_primary_background_fill_hover="#C4623F",
    button_primary_text_color="#FFFFFF",
    input_background_fill="#F5F3EC",
    input_background_fill_dark="#F5F3EC",
    input_border_color="#EAE5D9",
    input_border_color_dark="#EAE5D9",
    input_border_color_focus="#D97757",
    input_radius="8px",
)

SAMPLE_LOGS = [
    (
        "[2026-10-15 08:23:41] [ERROR] [db-proxy-pool] Connection timeout acquiring connection from pool.\n"
        "FATAL: remaining connection slots are reserved for non-replication superuser connections.\n"
        "PoolExhaustedError: database connection pool exhausted, 50/50 connections in use\n"
        "[2026-10-15 08:23:42] [WARN] [health-check] Liveness probe failed for pod db-proxy-6b8f7d9"
    ),
    (
        "CRITICAL 2026-10-14T14:12:05Z kubelet worker-node-04: OOMKilled process 18421 (payments-api) "
        "total-vm:8421048kB, anon-rss:4194304kB. Memory limit 4Gi exceeded."
    ),
    (
        "ERROR: Redis sentinel failover triggered - master 10.0.4.12:6379 unreachable for 30s. "
        "Failover elected replica 10.0.4.13:6379 as new master. Write latency spiked to 2400ms."
    ),
    (
        "WARNING: SSL certificate verification failed for api.stripe.com. "
        "OpenSSL Error: certificate has expired on 2026-10-14 23:59:59 GMT. TLS handshake terminated."
    ),
]

def render_pills_html(is_mock: bool) -> str:
    mode_class = "pill-mock" if is_mock else "pill-live"
    dot_class = "dot-mock" if is_mock else "dot-live"
    mode_text = "MOCK" if is_mock else "Live"
    model_display = LLM_MODEL.split("/")[-1] if "/" in LLM_MODEL else LLM_MODEL
    return f"""
    <div class="pills-container">
        <span class="status-pill">Corpus (828 incidents)</span>
        <span class="status-pill">Clusters (48)</span>
        <span class="status-pill">Model ({model_display})</span>
        <span class="status-pill {mode_class}">
            <span class="pill-dot {dot_class}"></span> Mode ({mode_text})
        </span>
    </div>
    """

initial_is_mock = not bool(os.environ.get("GROQ_API_KEY"))

# 5. Build Editorial UI
head_html = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Lora:ital,wght@0,500;0,600;1,400&display=swap" rel="stylesheet">
"""

with gr.Blocks(title="TraceMind RCA API") as demo:
    with gr.Column(elem_classes=["main-wrapper"]):
        # Header
        gr.HTML(f"""
        <div class="hero-header">
            <h1 class="hero-title">🧠 TraceMind</h1>
            <div class="hero-subtitle">Automated NLP Anomaly Clustering & Retrieval-Augmented Postmortem Correlation</div>
        </div>
        """)
        
        status_pills_view = gr.HTML(value=render_pills_html(initial_is_mock))

        # Input Card
        with gr.Column(elem_classes=["editorial-card"]):
            gr.HTML('<span class="section-label">RAW INFRASTRUCTURE LOG</span>')
            log_input = gr.Textbox(
                lines=8,
                show_label=False,
                placeholder="Paste unstructured infrastructure logs (Kubernetes OOMKilled, Database pool exhaust, Network timeouts, Redis sentinel)...",
                elem_classes=["code-editor-input"]
            )

            with gr.Row():
                analyze_btn = gr.Button("Analyze Incident", variant="primary", elem_classes=["analyze-btn"])
                clear_btn = gr.ClearButton(components=[log_input], value="Clear", variant="secondary")

            gr.Examples(
                examples=SAMPLE_LOGS,
                inputs=log_input,
                label="Select Incident Benchmark"
            )

        # Output Card
        with gr.Column(elem_classes=["editorial-card"]):
            gr.HTML('<span class="section-label">ROOT CAUSE ANALYSIS</span>')
            mock_banner = gr.HTML(visible=initial_is_mock)
            rc_output = gr.Markdown(elem_classes=["rc-markdown"])

            gr.HTML('<span class="section-label" style="margin-top: 24px;">RECOMMENDED REMEDIATION</span>')
            rem_output = gr.Markdown(elem_classes=["rem-markdown"])

            # Trace Inspector
            with gr.Accordion("Show Pipeline Trace", open=True, elem_classes=["trace-accordion"]):
                trace_summary_view = gr.HTML()
                gr.HTML('<div style="height: 8px;"></div>')
                with gr.Accordion("View Exact Prompt Sent to LLM", open=False):
                    prompt_viewer = gr.Code(language="markdown", interactive=False, show_label=False)

    def on_analyze(log_text: str):
        if not log_text or not log_text.strip():
            return (
                "*Please input a valid log excerpt to run analysis.*",
                "*No remediation steps available.*",
                gr.update(visible=False),
                render_pills_html(initial_is_mock),
                """
                <div class="trace-metric-grid">
                    <div class="trace-metric-box">
                        <div class="trace-metric-label">Detected Anomaly & Cluster</div>
                        <div class="trace-metric-value">—</div>
                    </div>
                    <div class="trace-metric-box">
                        <div class="trace-metric-label">Matched Incident & Similarity</div>
                        <div class="trace-metric-value">—</div>
                    </div>
                </div>
                """,
                ""
            )
        try:
            req = AnalyzeRequest(log_text=log_text)
            res = analyze_log(req)
            
            trace = res.trace
            matched_inc = trace.retrieved_incident
            sim = trace.similarity_score
            sim_pct = max(0, min(100, int(sim * 100)))
            cluster_text = "Noise (Novel Anomaly)" if trace.is_noise else f"Cluster #{trace.cluster_id}"
            
            # Format remediation cleanly
            rem_text = res.remediation_steps.strip()
            
            # Mock handling
            is_mock = res.is_mock
            mock_html = (
                '<div class="mock-banner-box">⚠️ <strong>Mock response:</strong> LLM not configured</div>'
                if is_mock else ""
            )
            mock_update = gr.update(value=mock_html, visible=is_mock)
            pills_html = render_pills_html(is_mock)

            # Trace summary HTML with similarity progress bar and clickable source link
            trace_html = f"""
            <div class="trace-metric-grid">
                <div class="trace-metric-box">
                    <div class="trace-metric-label">Detected Anomaly & Cluster</div>
                    <div class="trace-metric-value">{cluster_text}</div>
                </div>
                <div class="trace-metric-box">
                    <div class="trace-metric-label">Matched Incident ID & Similarity</div>
                    <div class="trace-metric-value">{matched_inc.incident_id} · {sim:.4f}</div>
                    <div class="sim-bar-container">
                        <div class="sim-bar-fill" style="width: {sim_pct}%;"></div>
                    </div>
                </div>
            </div>
            <div class="trace-source-wrapper">
                <strong>Historical Postmortem Source:</strong> 
                <a href="{matched_inc.source_url}" target="_blank" rel="noopener noreferrer" class="trace-source-link">{matched_inc.source_url}</a>
            </div>
            """

            return (
                res.root_cause,
                rem_text,
                mock_update,
                pills_html,
                trace_html,
                trace.prompt_used
            )
        except Exception as e:
            if hasattr(e, "detail") and isinstance(e.detail, dict):
                error_msg = e.detail.get("message", str(e.detail))
            elif hasattr(e, "message"):
                error_msg = e.message
            else:
                error_msg = str(e)
            err_html = (
                f'<div class="error-banner-box" style="background-color: #FDF2F2; border: 1px solid #F5D8D3; '
                f'color: #B04A3A; padding: 12px 16px; border-radius: 8px; font-size: 13px; font-weight: 500; '
                f'margin-bottom: 16px; line-height: 1.5;">'
                f'❌ <strong>Analysis Failed (HTTP 502):</strong> {error_msg}</div>'
            )
            return (
                '<div class="result-empty" style="color: #B04A3A;">Analysis failed. Please check the error above.</div>',
                '<div class="result-empty">N/A</div>',
                gr.update(value=err_html, visible=True),
                render_pills_html(False),
                """
                <div class="trace-metric-grid">
                    <div class="trace-metric-box">
                        <div class="trace-metric-label">Detected Anomaly & Cluster</div>
                        <div class="trace-metric-value" style="color: #B04A3A;">Error</div>
                    </div>
                    <div class="trace-metric-box">
                        <div class="trace-metric-label">Matched Incident & Similarity</div>
                        <div class="trace-metric-value" style="color: #B04A3A;">Failed</div>
                    </div>
                </div>
                """,
                ""
            )

    analyze_btn.click(
        on_analyze,
        inputs=[log_input],
        outputs=[rc_output, rem_output, mock_banner, status_pills_view, trace_summary_view, prompt_viewer]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=editorial_theme,
        css=custom_css,
        head=head_html,
    )
