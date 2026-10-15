import os
from pathlib import Path
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

# 3. Load CSS & Build Editorial Base Theme
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
    return f"""
    <div class="pills-container">
        <span class="status-pill">Corpus (828 incidents)</span>
        <span class="status-pill">Clusters (48)</span>
        <span class="status-pill">Model (Mistral-7B-Instruct-v0.3)</span>
        <span class="status-pill {mode_class}">
            <span class="pill-dot {dot_class}"></span> Mode ({mode_text})
        </span>
    </div>
    """

initial_is_mock = not bool(os.environ.get("HF_TOKEN"))

# 4. Build Editorial UI
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
                placeholder="Paste infrastructure error logs or stack traces here...",
                value=SAMPLE_LOGS[0],
                elem_classes=["log-textarea"],
            )
            analyze_btn = gr.Button(
                "Analyze incident",
                variant="primary",
                elem_classes=["analyze-button"]
            )

        # Example Chips
        with gr.Column():
            gr.HTML('<span class="section-label">TRY AN EXAMPLE</span>')
            with gr.Row(elem_classes=["chips-container"]):
                chip_db = gr.Button("DB pool exhausted", elem_classes=["example-chip"])
                chip_oom = gr.Button("Kubelet OOMKilled", elem_classes=["example-chip"])
                chip_redis = gr.Button("Redis sentinel failover", elem_classes=["example-chip"])
                chip_ssl = gr.Button("SSL verify failed", elem_classes=["example-chip"])

        # Chip click interactions
        chip_db.click(fn=lambda: SAMPLE_LOGS[0], inputs=[], outputs=[log_input])
        chip_oom.click(fn=lambda: SAMPLE_LOGS[1], inputs=[], outputs=[log_input])
        chip_redis.click(fn=lambda: SAMPLE_LOGS[2], inputs=[], outputs=[log_input])
        chip_ssl.click(fn=lambda: SAMPLE_LOGS[3], inputs=[], outputs=[log_input])

        # Mock Banner (shown when is_mock is true)
        mock_banner = gr.HTML(visible=False)

        # Results Grid (2 Equal Cards, stacks on mobile)
        with gr.Row(elem_classes=["results-grid"]):
            with gr.Column(elem_classes=["result-card"]):
                gr.HTML('<h3 class="result-heading">Root cause</h3>')
                rc_output = gr.Markdown(
                    value='<div class="result-empty">Run an analysis to see results.</div>',
                    elem_classes=["result-body"]
                )

            with gr.Column(elem_classes=["result-card"]):
                gr.HTML('<h3 class="result-heading">Remediation steps</h3>')
                rem_output = gr.Markdown(
                    value='<div class="result-empty">Run an analysis to see results.</div>',
                    elem_classes=["result-body"]
                )

        # Pipeline Trace Accordion (Directly under results)
        with gr.Accordion("Pipeline trace (audit the reasoning)", open=False, elem_classes=["trace-card"]):
            trace_summary_view = gr.HTML(
                value="""
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
                <div class="trace-source-wrapper">Run an incident analysis to inspect retrieved postmortem context.</div>
                """
            )
            prompt_viewer = gr.Code(
                label="Prompt Payload Delivered to LLM",
                language="markdown",
                elem_classes=["prompt-code"],
                interactive=False
            )

        # Footer
        gr.HTML("""
        <div class="editorial-footer">
            TraceMind RCA · Auditable Infrastructure Postmortem Correlation · 
            <a href="https://github.com/nish-debug15/TraceMind" target="_blank" rel="noopener noreferrer">github.com/nish-debug15/TraceMind</a>
        </div>
        """)

    def on_analyze(log_text):
        if not log_text.strip():
            return (
                '<div class="result-empty">Please enter a log excerpt to analyze.</div>',
                '<div class="result-empty">Please enter a log excerpt to analyze.</div>',
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
            return (
                f"**Error executing analysis:** {str(e)}",
                "N/A",
                gr.update(visible=False),
                render_pills_html(initial_is_mock),
                "",
                ""
            )

    analyze_btn.click(
        on_analyze,
        inputs=[log_input],
        outputs=[rc_output, rem_output, mock_banner, status_pills_view, trace_summary_view, prompt_viewer]
    )

# Attach backend FastAPI routes (/health, /analyze) to Gradio's internal FastAPI app
demo.app.include_router(fastapi_app.router)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=editorial_theme, css=custom_css, head=head_html)

