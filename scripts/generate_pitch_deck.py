"""
TraceMind Pitch Deck Generator
Generates docs/TraceMind_Pitch.pptx from docs/PITCH.md content.
16:9 dark-themed deck, 12 slides (11 core + 1 backup).
"""

import os
import sys
from pathlib import Path

# ── python-pptx imports ───────────────────────────────────────────────────────
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.oxml.ns import qn
from lxml import etree
import copy

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DEMO_PNG = DOCS_DIR / "pipeline_trace_demo.png"
OUTPUT_PPTX = DOCS_DIR / "TraceMind_Pitch.pptx"

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = RGBColor(0x0B, 0x0D, 0x12)   # #0B0D12
CARD     = RGBColor(0x14, 0x17, 0x1F)   # #14171F
TEXT     = RGBColor(0xF2, 0xF4, 0xF8)   # #F2F4F8
MUTED    = RGBColor(0x9A, 0xA3, 0xB2)   # #9AA3B2
ACCENT   = RGBColor(0x6C, 0x7C, 0xF6)   # #6C7CF6
SUCCESS  = RGBColor(0x3D, 0xD6, 0x8C)   # #3DD68C
WARNING  = RGBColor(0xF5, 0xA5, 0x24)   # #F5A524
DANGER   = RGBColor(0xF5, 0x50, 0x50)   # danger red
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
BLACK    = RGBColor(0x00, 0x00, 0x00)
BORDER   = RGBColor(0x1E, 0x22, 0x2F)   # subtle border

# ── Slide dimensions: 16:9 ───────────────────────────────────────────────────
SLIDE_W  = Inches(13.333)
SLIDE_H  = Inches(7.5)

# ── Font helpers ──────────────────────────────────────────────────────────────
FONT_FACE = "Calibri"  # Inter not typically bundled; fallback Calibri

def _font(run, size_pt, bold=False, color=TEXT, italic=False):
    run.font.name = FONT_FACE
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color

# ── Utility: add coloured rectangle ──────────────────────────────────────────
def add_rect(slide, l, t, w, h, fill_color, border_color=None, border_pt=0):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color and border_pt:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    return shape

# ── Utility: add text box ────────────────────────────────────────────────────
def add_textbox(slide, text, l, t, w, h,
                size=22, bold=False, color=TEXT,
                align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _font(run, size, bold=bold, color=color, italic=italic)
    return txBox

# ── Utility: multi-line text in textbox ──────────────────────────────────────
def add_textbox_lines(slide, lines, l, t, w, h,
                      size=22, bold_first=False, color=TEXT,
                      align=PP_ALIGN.LEFT, line_color=None):
    """lines: list of (text, bold, color_override) or plain str"""
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if isinstance(line, str):
            txt, bld, clr = line, (i == 0 and bold_first), (line_color or color)
        else:
            txt, bld, clr = line[0], line[1], line[2] if len(line) > 2 else color
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = txt
        _font(run, size, bold=bld, color=clr)
    return txBox

# ── Utility: slide background ────────────────────────────────────────────────
def set_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG

# ── Utility: title bar ───────────────────────────────────────────────────────
TITLE_TOP  = 0.25
TITLE_H    = 0.72
TITLE_L    = 0.5
TITLE_W    = 12.333

def add_title(slide, title_text, subtitle_text=None):
    # Accent bar
    bar = add_rect(slide, 0.5, TITLE_TOP, 0.06, TITLE_H, ACCENT)
    # Title text
    tx = slide.shapes.add_textbox(
        Inches(0.65), Inches(TITLE_TOP), Inches(TITLE_W - 0.15), Inches(TITLE_H)
    )
    tf = tx.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title_text
    _font(run, 36, bold=True, color=TEXT)
    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run()
        r2.text = subtitle_text
        _font(r2, 18, color=MUTED)
    return tx

# ── Utility: footer ──────────────────────────────────────────────────────────
FOOTER_T = 7.1
FOOTER_H = 0.3

def add_footer(slide, speaker, slide_num):
    # Left: speaker
    add_textbox(slide, speaker, 0.3, FOOTER_T, 4.0, FOOTER_H,
                size=11, color=MUTED, align=PP_ALIGN.LEFT)
    # Center: TraceMind
    add_textbox(slide, "TraceMind", 4.666, FOOTER_T, 4.0, FOOTER_H,
                size=11, color=ACCENT, align=PP_ALIGN.CENTER)
    # Right: slide number
    add_textbox(slide, str(slide_num), 9.333, FOOTER_T, 3.7, FOOTER_H,
                size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    # Footer line
    line_box = slide.shapes.add_shape(
        1, Inches(0.3), Inches(FOOTER_T - 0.05), Inches(12.733), Inches(0.02)
    )
    line_box.fill.solid()
    line_box.fill.fore_color.rgb = BORDER
    line_box.line.fill.background()

# ── Utility: speaker notes ───────────────────────────────────────────────────
def set_notes(slide, text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text

# ── Utility: bullet card ─────────────────────────────────────────────────────
BODY_TOP  = 1.15
BODY_H    = 5.75

def add_bullets(slide, bullets, l=0.55, t=BODY_TOP, w=12.3, h=BODY_H,
                size=22, bullet_color=ACCENT):
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (btext, bsub) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(10 if i > 0 else 0)
        # Bullet dot
        r0 = p.add_run()
        r0.text = "▸  "
        _font(r0, size - 2, color=bullet_color, bold=True)
        # Main text
        r1 = p.add_run()
        bold_part = ""
        rest_part = btext
        if ":" in btext:
            parts = btext.split(":", 1)
            bold_part = parts[0] + ":"
            rest_part = parts[1]
        if bold_part:
            r1.text = bold_part
            _font(r1, size, bold=True, color=TEXT)
            r2 = p.add_run()
            r2.text = rest_part
            _font(r2, size, color=TEXT)
        else:
            r1.text = btext
            _font(r1, size, color=TEXT)
        # Sub-line
        if bsub:
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.LEFT
            p2.space_before = Pt(2)
            r3 = p2.add_run()
            r3.text = "      " + bsub
            _font(r3, size - 4, color=MUTED, italic=True)
    return txBox

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD DECK
# ═══════════════════════════════════════════════════════════════════════════════

def build_deck():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    blank_layout = prs.slide_layouts[6]  # completely blank

    # ── Slide 1: Title Slide ──────────────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)

    # Big title
    tx = slide.shapes.add_textbox(Inches(0.5), Inches(0.6), Inches(12.3), Inches(1.4))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = "TraceMind"
    _font(r, 52, bold=True, color=ACCENT)

    tx2 = slide.shapes.add_textbox(Inches(0.5), Inches(1.85), Inches(12.0), Inches(0.7))
    tf2 = tx2.text_frame; tf2.word_wrap = True
    p2 = tf2.paragraphs[0]; p2.alignment = PP_ALIGN.LEFT
    r2 = p2.add_run()
    r2.text = "Auditable Root Cause Analysis for SRE Logs"
    _font(r2, 26, bold=True, color=TEXT)

    tx3 = slide.shapes.add_textbox(Inches(0.5), Inches(2.45), Inches(12.0), Inches(0.5))
    tf3 = tx3.text_frame; tf3.word_wrap = True
    p3 = tf3.paragraphs[0]; p3.alignment = PP_ALIGN.LEFT
    r3 = p3.add_run()
    r3.text = "Retrieval-Augmented Postmortem Correlation · Fully Inspectable Retrieval & Prompt"
    _font(r3, 17, color=MUTED)

    # Accent divider
    add_rect(slide, 0.5, 3.0, 12.3, 0.03, ACCENT)

    # Team cards (5 cards)
    team = [
        ("Nishit",  "System Architecture & Frontend"),
        ("Pragun",  "Domain Analysis & Pipeline Design"),
        ("Pranav",  "Data Engineering & Clustering"),
        ("Sameera", "Vector Storage & Retrieval"),
        ("Unique",  "Benchmarking & Evaluation"),
    ]
    card_w = 2.3
    card_h = 1.3
    card_t = 3.25
    gap = 0.22
    start_l = 0.5
    for i, (name, role) in enumerate(team):
        cl = start_l + i * (card_w + gap)
        c = add_rect(slide, cl, card_t, card_w, card_h, CARD, ACCENT, 0.5)
        # Name
        txn = slide.shapes.add_textbox(Inches(cl + 0.15), Inches(card_t + 0.18),
                                        Inches(card_w - 0.3), Inches(0.45))
        tf = txn.text_frame; p = tf.paragraphs[0]
        r = p.add_run(); r.text = name
        _font(r, 17, bold=True, color=ACCENT)
        # Role
        txr = slide.shapes.add_textbox(Inches(cl + 0.15), Inches(card_t + 0.6),
                                        Inches(card_w - 0.3), Inches(0.65))
        tf = txr.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = role
        _font(r, 12, color=MUTED)

    add_footer(slide, "Nishit", 1)

    notes_1 = """Nishit | 0:00–1:00 (1 min)

"Good morning, faculty and jury members. I am Nishit, presenting TraceMind alongside Pragun, Pranav, Sameera, and Unique.

TraceMind is an auditable incident analysis system for Site Reliability Engineers. When production infrastructure fails, engineers are flooded with thousands of cryptic logs and must rapidly determine what broke and how to recover. TraceMind ingests that raw log excerpt, matches it against verified historical postmortems, and synthesizes a clear root cause analysis with an actionable five-step remediation checklist.

Unlike opaque AI assistants, TraceMind ensures the retrieval and prompt are fully inspectable. Every recommendation exposes an interactive Pipeline Trace showing the exact historical incident retrieved, its mathematical similarity score, and the raw prompt delivered to the model.

Today, Pragun covers the operational problem, Pranav details our curated postmortem corpus, Sameera explains our vector retrieval and relevance gate, I will demonstrate the live console, and Unique will present our quantitative benchmarks."

HANDOFF: "I will now hand over to Pragun to explain why manual incident correlation during production outages remains so painful."
"""
    set_notes(slide, notes_1)

    # ── Slide 2: Agenda & Core Premise ───────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "Agenda & Core Premise")

    bullets_2 = [
        ("What We Built",
         "Automated log diagnostic pipeline matching live crashes to verified postmortems."),
        ("Core Differentiator",
         "Retrieval & prompt are fully inspectable via Pipeline Trace — similarity gauge, matched URL, raw prompt."),
        ("Roadmap",
         "Problem → Corpus → RAG Pipeline → Live Demo → Quantitative Evaluation."),
    ]
    add_bullets(slide, bullets_2, t=1.25, h=4.5)

    # 5-step horizontal roadmap
    steps = ["Problem", "Corpus", "RAG\nPipeline", "Live\nDemo", "Evaluation"]
    step_colors = [DANGER, WARNING, ACCENT, SUCCESS, MUTED]
    step_w = 1.9
    step_h = 0.9
    step_t = 5.95
    step_gap = 0.42
    start_sl = 0.6
    for i, (label, col) in enumerate(zip(steps, step_colors)):
        sl = start_sl + i * (step_w + step_gap)
        add_rect(slide, sl, step_t, step_w, step_h, col, None, 0)
        txs = slide.shapes.add_textbox(Inches(sl), Inches(step_t),
                                        Inches(step_w), Inches(step_h))
        tf = txs.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = f"{i+1}. {label}"
        _font(r, 14, bold=True, color=WHITE)
        # Arrow (not on last)
        if i < len(steps) - 1:
            ax = sl + step_w + 0.03
            add_textbox(slide, "→", ax, step_t + 0.18, 0.36, 0.5,
                        size=20, color=MUTED, align=PP_ALIGN.CENTER)

    add_footer(slide, "Nishit", 2)
    set_notes(slide, """Nishit | 0:00–1:00 (continued from Slide 1)

This slide outlines our five-part presentation flow and states the core differentiator upfront.

TraceMind is not another LLM wrapper — every inference is grounded in a retrieved real-world postmortem, and the engineer can inspect exactly which document was used, its similarity score, and the prompt fed to the model.

HANDOFF: "I will now hand over to Pragun to explain why manual incident correlation during production outages remains so painful."
""")

    # ── Slide 3: The Outage Reality ───────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "The Outage Reality", "Why MTTR Is a Diagnosis Problem")

    # Two-column comparison
    col_t = 1.3
    col_h = 5.3
    col_gap = 0.3

    # Left card: Log firehose
    add_rect(slide, 0.4, col_t, 5.9, col_h, CARD, DANGER, 0.8)
    add_textbox(slide, "🔥  Log Firehose", 0.6, col_t + 0.1, 5.5, 0.55,
                size=18, bold=True, color=DANGER)
    firehose_lines = [
        "10,000+ log lines per incident surge",
        "Stack traces show WHAT crashed, not WHY",
        "Keyword search fails across synonymy",
        '"pool exhausted" ≠ "no connection slots" (same bug)',
    ]
    t_offset = col_t + 0.75
    for ln in firehose_lines:
        add_textbox(slide, f"• {ln}", 0.65, t_offset, 5.4, 0.55,
                    size=18, color=TEXT)
        t_offset += 0.68

    # Right card: What SREs need
    add_rect(slide, 7.0, col_t, 5.9, col_h, CARD, SUCCESS, 0.8)
    add_textbox(slide, "✓  What SREs Need", 7.2, col_t + 0.1, 5.5, 0.55,
                size=18, bold=True, color=SUCCESS)
    sre_lines = [
        "Instant answer: 'Have we seen this before?'",
        "Semantic search across all past incidents",
        "Concrete root cause + remediation steps",
        "Cited source — inspectable, not a black box",
    ]
    t_offset = col_t + 0.75
    for ln in sre_lines:
        add_textbox(slide, f"• {ln}", 7.2, t_offset, 5.5, 0.55,
                    size=18, color=TEXT)
        t_offset += 0.68

    add_footer(slide, "Pragun", 3)
    set_notes(slide, """Pragun | 1:00–2:45 (1 min 45 sec)

"When a production service goes down at 2:00 AM, engineers face intense pressure to restore availability quickly. But during an active incident, engineers spend most of their triage time reading logs, searching internal wikis, and querying chat channels trying to answer one question: have we seen this failure before?

Searching through unindexed historical postmortems during an outage is like an emergency room doctor hunting through a basement of paper charts while a patient is in critical condition. Keyword searches fail because different teams describe the exact same bug with different vocabulary: one writes 'database pool exhausted,' while another logs 'connection acquisition timeout.'

When teams turn to generic language models to diagnose logs, they encounter hallucinations. A standard model without grounding may invent non-existent configuration parameters or suggest dangerous shell commands that exacerbate downtime. SREs cannot trust unverified suggestions during an active incident.

Our objective with TraceMind is to ground every diagnosis in verified, real-world postmortems using Retrieval-Augmented Generation — fetching relevant factual references before the model generates its response. Most importantly, if no historical precedent matches the failure, the system must explicitly say so rather than guess."

HANDOFF: "Pranav will now explain how we collected our corpus of 828 real postmortems and structured our company-disjoint split."
""")

    # ── Slide 4: Why Generic LLMs Fall Short ─────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "Why Generic LLMs Fall Short")

    # Two big cards side by side
    card_t4 = 1.3
    card_h4 = 5.4

    # Left card: Ungrounded LLM (risk)
    add_rect(slide, 0.4, card_t4, 5.9, card_h4, CARD, DANGER, 1.2)
    add_textbox(slide, "✗  Ungrounded LLM", 0.65, card_t4 + 0.12, 5.5, 0.55,
                size=20, bold=True, color=DANGER)
    llm_lines = [
        "Invents invalid configuration flags",
        "Suggests phantom parameters & risky commands",
        "No infrastructure context or history",
        "No citation — zero auditability",
    ]
    t_off = card_t4 + 0.85
    for ln in llm_lines:
        add_textbox(slide, f"• {ln}", 0.75, t_off, 5.3, 0.6, size=19, color=TEXT)
        t_off += 0.78

    # Right card: TraceMind (grounded)
    add_rect(slide, 7.0, card_t4, 5.9, card_h4, CARD, SUCCESS, 1.2)
    add_textbox(slide, "✓  TraceMind (Grounded + Cited)", 7.2, card_t4 + 0.12, 5.5, 0.6,
                size=20, bold=True, color=SUCCESS)
    tm_lines = [
        "Every answer grounded in real postmortem",
        "Direct hyperlink to retrieved source URL",
        "Similarity score — inspectable confidence",
        "Relevance Gate: says 'not relevant' if no match",
    ]
    t_off = card_t4 + 0.85
    for ln in tm_lines:
        add_textbox(slide, f"• {ln}", 7.25, t_off, 5.5, 0.6, size=19, color=TEXT)
        t_off += 0.78

    add_footer(slide, "Pragun", 4)
    set_notes(slide, """Pragun | 1:00–2:45 (continued)

"Our objective with TraceMind is to ground every diagnosis in verified, real-world postmortems using Retrieval-Augmented Generation — fetching relevant factual references before the model generates its response. Most importantly, if no historical precedent matches the failure, the system must explicitly say so rather than guess."

HANDOFF: "Pranav will now explain how we collected our corpus of 828 real postmortems and structured our company-disjoint split."
""")

    # ── Slide 5: Postmortem Corpus & Company-Disjoint Split ───────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "Postmortem Corpus & Company-Disjoint Split")

    # Funnel flow visualization
    # Level 1: 828 total
    bx_w = 3.2; bx_h = 0.8
    top = 1.4

    # Row 1: 828
    add_rect(slide, 5.0, top, bx_w, bx_h, ACCENT, None, 0)
    add_textbox(slide, "828 Postmortems  (58 companies)", 5.0, top, bx_w, bx_h,
                size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # Arrow down
    add_textbox(slide, "▼", 6.4, top + bx_h, 0.5, 0.5, size=18, color=MUTED, align=PP_ALIGN.CENTER)
    # Row 2: Quality split
    top2 = top + bx_h + 0.55
    add_rect(slide, 1.0, top2, 3.6, bx_h, SUCCESS, None, 0)
    add_textbox(slide, "544 High-Quality\n(concrete root cause + remediation)",
                1.0, top2, 3.6, bx_h, size=14, bold=False, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, 8.6, top2, 3.7, bx_h, WARNING, None, 0)
    add_textbox(slide, "284 Low-Quality\n(brief status summaries)",
                8.6, top2, 3.7, bx_h, size=14, bold=False, color=WHITE, align=PP_ALIGN.CENTER)
    # Arrows
    add_textbox(slide, "▼", 2.6, top2 + bx_h, 0.5, 0.5, size=18, color=MUTED, align=PP_ALIGN.CENTER)
    # Row 3: Train/Test split
    top3 = top2 + bx_h + 0.55
    add_rect(slide, 0.4, top3, 4.5, bx_h, CARD, SUCCESS, 0.8)
    add_textbox(slide, "662 Train  (18 companies)", 0.4, top3, 4.5, bx_h,
                size=15, bold=True, color=SUCCESS, align=PP_ALIGN.CENTER)
    add_rect(slide, 5.3, top3, 4.5, bx_h, CARD, ACCENT, 0.8)
    add_textbox(slide, "93 Test  (40 companies)", 5.3, top3, 4.5, bx_h,
                size=15, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_rect(slide, 10.1, top3, 2.8, bx_h, CARD, DANGER, 0.6)
    add_textbox(slide, "73 discarded\n(low-quality, test cos.)", 10.1, top3, 2.8, bx_h,
                size=12, bold=False, color=DANGER, align=PP_ALIGN.CENTER)
    # Note
    add_textbox(slide,
                "* 73 low-quality incidents from test companies discarded — strict company-disjoint split, seed=42",
                0.4, top3 + bx_h + 0.2, 12.3, 0.4,
                size=13, color=MUTED, italic=True)

    add_footer(slide, "Pranav", 5)
    set_notes(slide, """Pranav | 2:45–4:30 (1 min 45 sec)

"A retrieval system is only as good as the ground truth behind it. We constructed a dataset of 828 real postmortems sourced aggregator-first from curated repositories like icco, official Statuspage histories across 20 platforms like Datadog and Twilio, and public retrospectives from Google Cloud and Cloudflare.

Raw quantity alone was noisy. Two hundred and eighty-four incidents were brief status updates. We built a quality filter separating our corpus into 544 high-quality records — with concrete root causes and multi-step remediations — and 284 low-quality summaries.

To evaluate out-of-distribution generalization honestly, we enforced a strict company-disjoint split using fixed seed 42. Our training set has 662 incidents across 18 companies. Our test set contains 93 incidents across 40 completely different companies, with 100% rated high quality. You will notice 662 plus 93 equals 755: the remaining 73 low-quality incidents from those 40 test companies were completely discarded to maintain a strict company-disjoint split while keeping the test set purely high-quality.

Before retrieval, we run HDBSCAN — a density-based clustering algorithm that groups similar data without forcing every outlier into a cluster. Think of HDBSCAN as a mail sorter: it groups near-duplicate logs into bundles so repetitive errors don't flood our index. It identified 48 distinct clusters in our training data. Crucially, clustering is only a deduplication filter; matching is handled by vector retrieval."

HANDOFF: "Sameera will now walk through our vector database layer and two-pass retrieval mechanism."
""")

    # ── Slide 6: HDBSCAN Clustering ───────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "HDBSCAN Clustering as Preprocessing")

    # Flow: logs -> embed -> HDBSCAN -> deduped groups
    flow_items = [
        ("Raw\nLogs", BG, ACCENT),
        ("Embed\n(384-d)", CARD, ACCENT),
        ("HDBSCAN\nmin_size=3", CARD, WARNING),
        ("Deduped\nGroups", CARD, SUCCESS),
    ]
    fi_w = 2.4; fi_h = 1.4; fi_t = 2.1; fi_gap = 0.65
    start_fi = 0.5
    for i, (label, bg_c, border_c) in enumerate(flow_items):
        fl = start_fi + i * (fi_w + fi_gap)
        add_rect(slide, fl, fi_t, fi_w, fi_h, bg_c, border_c, 1.2)
        add_textbox(slide, label, fl, fi_t, fi_w, fi_h,
                    size=16, bold=True, color=TEXT, align=PP_ALIGN.CENTER)
        if i < len(flow_items) - 1:
            ax = fl + fi_w + 0.07
            add_textbox(slide, "→", ax, fi_t + 0.42, 0.55, 0.5,
                        size=20, color=MUTED, align=PP_ALIGN.CENTER)

    # "48 clusters" callout box
    add_rect(slide, 7.5, 1.5, 4.9, 0.9, CARD, WARNING, 1.5)
    add_textbox(slide, "48 clusters identified  (training corpus, min_cluster_size=3)",
                7.6, 1.55, 4.7, 0.78, size=15, bold=True, color=WARNING, align=PP_ALIGN.CENTER)

    # Role clarification bullets
    bullets_6 = [
        ("Clustering Role", "Preprocessing & deduplication — NOT the core search engine."),
        ("Algorithm", "HDBSCAN on 384-d dense vectors (all-MiniLM-L6-v2)."),
        ("Function", "Detects recurring failure signatures; isolates novel / noisy patterns."),
        ("Retrieval Engine", "Full semantic vector search in ChromaDB (next slide)."),
    ]
    add_bullets(slide, bullets_6, t=3.8, h=3.1, size=20)

    add_footer(slide, "Pranav", 6)
    set_notes(slide, """Pranav | 2:45–4:30 (continued)

"Before retrieval, we run HDBSCAN — a density-based clustering algorithm that groups similar data without forcing every outlier into a cluster. Think of HDBSCAN as a mail sorter: it groups near-duplicate logs into bundles so repetitive errors don't flood our index. It identified 48 distinct clusters in our training data. Crucially, clustering is only a deduplication filter; matching is handled by vector retrieval."

HANDOFF: "Sameera will now walk through our vector database layer and two-pass retrieval mechanism."
""")

    # ── Slide 7: ChromaDB Vector Database ─────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "The Vector Database Layer (ChromaDB)")

    # Flow: Text → 384-d vector → ChromaDB card
    # Text box
    add_rect(slide, 0.4, 1.4, 2.5, 1.2, CARD, MUTED, 0.6)
    add_textbox(slide, "Raw Log\nText", 0.4, 1.4, 2.5, 1.2,
                size=18, bold=True, color=TEXT, align=PP_ALIGN.CENTER)
    # Arrow
    add_textbox(slide, "→", 3.05, 1.65, 0.6, 0.6, size=24, color=ACCENT, align=PP_ALIGN.CENTER)
    # Embedding
    add_rect(slide, 3.75, 1.4, 2.5, 1.2, CARD, ACCENT, 0.8)
    add_textbox(slide, "384-d\nVector", 3.75, 1.4, 2.5, 1.2,
                size=18, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    # Arrow
    add_textbox(slide, "→", 6.4, 1.65, 0.6, 0.6, size=24, color=ACCENT, align=PP_ALIGN.CENTER)
    # ChromaDB card
    add_rect(slide, 7.1, 1.1, 5.8, 5.9, CARD, SUCCESS, 1.2)
    add_textbox(slide, "ChromaDB", 7.3, 1.2, 5.4, 0.55,
                size=20, bold=True, color=SUCCESS)
    add_textbox(slide, "In-process · FastAPI backend · Zero external server",
                7.3, 1.75, 5.4, 0.4, size=13, color=MUTED)
    # Schema fields
    schema_fields = [
        ("embedding",         "384-d float vector",                        ACCENT),
        ("log_excerpt",       "Raw log text",                              TEXT),
        ("root_cause",        "Diagnosed failure cause",                   TEXT),
        ("remediation_steps", "Multi-step fix list",                       TEXT),
        ("source_url",        "Link to original postmortem",               TEXT),
        ("quality",           '"high" | "low" label',                      WARNING),
    ]
    sf_t = 2.25
    for field, desc, fc in schema_fields:
        add_textbox(slide, f"  {field}", 7.3, sf_t, 2.6, 0.42,
                    size=14, bold=True, color=fc)
        add_textbox(slide, desc, 9.9, sf_t, 2.8, 0.42,
                    size=13, color=MUTED)
        sf_t += 0.5

    # Key fact below
    add_textbox(slide,
                "Built once from 662 training incidents only — test incidents are never indexed.",
                0.4, 2.85, 6.4, 0.55, size=17, bold=True, color=SUCCESS)
    add_textbox(slide,
                "Loaded directly into memory on backend startup.",
                0.4, 3.45, 6.4, 0.45, size=16, color=TEXT)

    add_footer(slide, "Sameera", 7)
    set_notes(slide, """Sameera | 4:30–6:15 (1 min 45 sec)

"To match an incoming error log against past incidents, we use a vector database. A vector database converts text into a list of numbers — an embedding — capturing its conceptual meaning, allowing semantic similarity search rather than exact keyword matching.

Think of a traditional database like an index matching literal words, while a vector database is a library catalog: 'connection acquisition timeout' and 'database pool exhausted' sit on the exact same shelf because their vectors point in the same direction.

We chose ChromaDB because it is lightweight and runs directly in-process inside our FastAPI backend with no separate database server. Each record stores the 384-dimensional embedding alongside metadata: the raw log excerpt, root cause, remediation steps, source URL, and quality label. This collection is built once from the 662 training incidents only — test incidents are never indexed — and loaded directly into memory when the backend starts."

HANDOFF: "Nishit will now demonstrate the system live and show our Pipeline Trace in action."
""")

    # ── Slide 8: 2-Pass Retrieval & Relevance Gate ────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "2-Pass Retrieval & The Relevance Gate")

    # Flowchart
    # Query box
    bw = 2.4; bh = 0.75
    q_l = 0.3; q_t = 1.4
    add_rect(slide, q_l, q_t, bw, bh, CARD, TEXT, 0.6)
    add_textbox(slide, "Query\n(Log Input)", q_l, q_t, bw, bh,
                size=15, bold=True, color=TEXT, align=PP_ALIGN.CENTER)
    # Arrow right
    add_textbox(slide, "→", q_l + bw, q_t + 0.1, 0.5, 0.5,
                size=18, color=ACCENT, align=PP_ALIGN.CENTER)
    # Pass 1
    p1_l = q_l + bw + 0.6; p1_t = q_t
    add_rect(slide, p1_l, p1_t, 3.1, bh, CARD, SUCCESS, 1.0)
    add_textbox(slide, "Pass 1: High-Quality\n(quality=high, ≥ 0.40)", p1_l, p1_t, 3.1, bh,
                size=14, bold=True, color=SUCCESS, align=PP_ALIGN.CENTER)

    # Hit? diamond
    hit_l = p1_l + 3.1 + 0.3; hit_t = q_t - 0.1
    add_rect(slide, hit_l, hit_t, 1.5, 0.95, CARD, WARNING, 1.0)
    add_textbox(slide, "Hit?\n≥ 0.40?", hit_l, hit_t, 1.5, 0.95,
                size=13, bold=True, color=WARNING, align=PP_ALIGN.CENTER)

    # YES → Generate
    gen_l = hit_l + 1.5 + 0.3; gen_t = q_t
    add_textbox(slide, "YES →", hit_l + 1.5, hit_t + 0.12, 0.45, 0.5,
                size=12, color=SUCCESS, align=PP_ALIGN.CENTER)
    add_rect(slide, gen_l, gen_t, 2.5, bh, CARD, SUCCESS, 0.8)
    add_textbox(slide, "Generate\nRCA", gen_l, gen_t, 2.5, bh,
                size=15, bold=True, color=SUCCESS, align=PP_ALIGN.CENTER)

    # NO → Pass 2
    p2_l = p1_l; p2_t = q_t + bh + 0.85
    add_textbox(slide, "NO ↓", hit_l + 0.52, hit_t + bh - 0.1, 0.6, 0.5,
                size=12, color=DANGER, align=PP_ALIGN.CENTER)
    add_rect(slide, p2_l, p2_t, 3.1, bh, CARD, WARNING, 1.0)
    add_textbox(slide, "Pass 2: All 662 Records\n(no quality filter)", p2_l, p2_t, 3.1, bh,
                size=14, bold=True, color=WARNING, align=PP_ALIGN.CENTER)

    # → Relevance Gate
    rg_l = p2_l + 3.1 + 0.3; rg_t = p2_t
    add_textbox(slide, "→", p2_l + 3.1, p2_t + 0.12, 0.42, 0.5,
                size=18, color=ACCENT, align=PP_ALIGN.CENTER)
    add_rect(slide, rg_l, rg_t, 2.8, bh, CARD, ACCENT, 1.0)
    add_textbox(slide, "Relevance Gate\n(LLM inspection)", rg_l, rg_t, 2.8, bh,
                size=14, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    # Outcomes
    out_t = rg_t + bh + 0.6
    add_rect(slide, rg_l - 1.0, out_t, 2.5, bh, CARD, SUCCESS, 0.8)
    add_textbox(slide, "Precedent\napplied", rg_l - 1.0, out_t, 2.5, bh,
                size=14, bold=True, color=SUCCESS, align=PP_ALIGN.CENTER)
    add_textbox(slide, "↙  relevant", rg_l - 0.2, rg_t + bh, 1.5, 0.55,
                size=12, color=SUCCESS)
    add_rect(slide, rg_l + 2.0, out_t, 2.9, bh, CARD, DANGER, 0.8)
    add_textbox(slide, "Precedent:\nnot relevant", rg_l + 2.0, out_t, 2.9, bh,
                size=14, bold=True, color=DANGER, align=PP_ALIGN.CENTER)
    add_textbox(slide, "↘  not relevant", rg_l + 2.1, rg_t + bh, 1.9, 0.55,
                size=12, color=DANGER)

    # Key stats
    add_rect(slide, 0.3, 5.9, 12.5, 0.75, CARD, MUTED, 0.4)
    add_textbox(slide,
                "Gate Evaluation: All 7 test incidents below 0.40 triggered 'Precedent: not relevant' (not manually verified as truly irrelevant).",
                0.5, 5.95, 12.1, 0.65, size=14, color=MUTED)

    add_footer(slide, "Sameera", 8)
    set_notes(slide, """Sameera | 4:30–6:15 (continued)

"When an engineer inputs a log, we execute a two-pass retrieval. Pass One queries high-quality incidents with a 0.40 cosine similarity floor. If that threshold is missed, Pass Two falls back across all 662 training records.

Finally, our Relevance Gate prevents false analogies. The prompt instructs the LLM to output 'Precedent: not relevant' on line one if the match is unrelated. In evaluation, across the seven test incidents scoring below the 0.40 similarity floor, all seven triggered the gate with 'Precedent: not relevant' on line one, though we did not manually verify if each precedent was truly irrelevant."

HANDOFF: "Nishit will now demonstrate the system live and show our Pipeline Trace in action."
""")

    # ── Slide 9: Architecture & Live Demo ─────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "System Architecture & Live Demonstration")

    # Architecture strip
    arch_items = [
        ("Next.js 16\n(Bento UI)", CARD, ACCENT),
        ("FastAPI\n(HF Space)", CARD, WARNING),
        ("Groq\ngpt-oss-120b", CARD, SUCCESS),
    ]
    ai_w = 3.5; ai_h = 1.0; ai_t = 1.3; ai_gap = 0.5
    start_ai = 1.0
    for i, (label, bg_c, border_c) in enumerate(arch_items):
        al = start_ai + i * (ai_w + ai_gap)
        add_rect(slide, al, ai_t, ai_w, ai_h, bg_c, border_c, 1.0)
        add_textbox(slide, label, al, ai_t, ai_w, ai_h,
                    size=16, bold=True, color=TEXT, align=PP_ALIGN.CENTER)
        if i < len(arch_items) - 1:
            arx = al + ai_w + 0.03
            add_textbox(slide, "→", arx, ai_t + 0.25, 0.44, 0.5,
                        size=18, color=MUTED, align=PP_ALIGN.CENTER)

    # Key features row
    feats = [
        ("Client Rate Limit", "15 RPM", ACCENT),
        ("Similarity Floor", "≥ 0.40", WARNING),
        ("Cache Entries", "201 SQLite", SUCCESS),
        ("Similarity (Demo)", "0.50 score", ACCENT),
    ]
    feat_w = 2.9; feat_h = 0.7; feat_t = 2.5; feat_gap = 0.17
    start_feat = 0.4
    for i, (label, val, fc) in enumerate(feats):
        fl = start_feat + i * (feat_w + feat_gap)
        add_rect(slide, fl, feat_t, feat_w, feat_h, CARD, fc, 0.5)
        add_textbox(slide, f"{label}\n{val}", fl, feat_t, feat_w, feat_h,
                    size=14, bold=False, color=TEXT, align=PP_ALIGN.CENTER)

    # Demo image or placeholder
    img_top = 3.4; img_h = 3.5
    if DEMO_PNG.exists():
        try:
            slide.shapes.add_picture(
                str(DEMO_PNG),
                Inches(0.4), Inches(img_top),
                width=Inches(12.5), height=Inches(img_h)
            )
        except Exception as e:
            # Fallback placeholder
            add_rect(slide, 0.4, img_top, 12.5, img_h, CARD, ACCENT, 0.8)
            add_textbox(slide, f"[pipeline_trace_demo.png — could not embed: {e}]",
                        0.6, img_top + 1.4, 12.1, 0.8,
                        size=16, color=MUTED, align=PP_ALIGN.CENTER)
    else:
        add_rect(slide, 0.4, img_top, 12.5, img_h, CARD, ACCENT, 0.8)
        add_textbox(slide, "Live demo",
                    0.4, img_top + 1.4, 12.5, 0.8,
                    size=24, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
        add_textbox(slide, "(pipeline_trace_demo.png not found)",
                    0.4, img_top + 2.1, 12.5, 0.5,
                    size=14, color=MUTED, align=PP_ALIGN.CENTER)

    add_footer(slide, "Nishit", 9)
    set_notes(slide, """Nishit | 6:15–8:15 (2 min + demo)

[Action: Transition to console]
"Here is the TraceMind console. In the Incident Log card, I paste a production error log showing a database proxy pool timeout. The editor soft-wraps long lines while keeping line numbers aligned to logical lines.

[Action: Click 'Analyze incident']
I click Analyze. On the right, the stepper confirms each stage: embedding the log vector, cluster assignment, ChromaDB vector retrieval, and LLM generation via Groq.

[Action: Point to Root Cause and Remediation]
TraceMind synthesizes a root cause explaining that active worker connections saturated the pool, alongside an actionable five-step remediation checklist.

[Action: Highlight Pipeline Trace and Gauge]
Our core differentiator is the Pipeline Trace: retrieval and prompt are fully inspectable. The gauge displays a similarity score of 0.50 against our 0.40 threshold, labeled as a Moderate match, and links directly to Datadog's public postmortem. In the Prompt tab, you can inspect the exact prompt delivered to Groq.

If Groq hits its free-tier rate limit or returns a network error, our pre-warmed SQLite cache automatically serves cached completions for example presets, shielding the demo from API failures."

HANDOFF: "Unique will now review our quantitative evaluation and summarize our findings."
""")

    # ── Slide 10: Quantitative Benchmarks ─────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "Quantitative Benchmarks", "Held-Out Test Set — N = 93")

    # Bar chart: ROUGE Zero-Shot vs RAG
    chart_data = ChartData()
    chart_data.categories = ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "Sem. Similarity"]
    chart_data.add_series("Zero-Shot", (0.1809, 0.0384, 0.1061, 0.5743))
    chart_data.add_series("RAG",       (0.1941, 0.0439, 0.1210, 0.5764))

    chart_placeholder = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.4), Inches(1.3), Inches(7.5), Inches(4.5),
        chart_data
    )
    chart = chart_placeholder.chart
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = 2  # bottom
    chart.legend.include_in_layout = False

    # Style chart
    plot = chart.plots[0]
    try:
        # Zero-Shot series → MUTED
        s0 = plot.series[0]
        s0.format.fill.solid()
        s0.format.fill.fore_color.rgb = MUTED
        # RAG series → ACCENT
        s1 = plot.series[1]
        s1.format.fill.solid()
        s1.format.fill.fore_color.rgb = ACCENT
    except Exception:
        pass

    # Style chart background
    try:
        chart_placeholder.chart.chart_area.format.fill.solid()
        chart_placeholder.chart.chart_area.format.fill.fore_color.rgb = CARD
    except Exception:
        pass

    # 3 stat callout cards on the right
    stats = [
        ("Mean Top-1 Similarity", "0.5273", "(Median: 0.5333)", ACCENT),
        ("Above 0.40 Threshold",  "92.5%",  "86 / 93 test incidents",   SUCCESS),
        ("Relevance Gate",        "7 / 7",  "All triggered gate below 0.40", WARNING),
    ]
    sc_l = 8.1; sc_t = 1.4; sc_w = 5.0; sc_h = 1.3; sc_gap = 0.22
    for label, val, sub, col in stats:
        add_rect(slide, sc_l, sc_t, sc_w, sc_h, CARD, col, 0.8)
        add_textbox(slide, label, sc_l + 0.15, sc_t + 0.08, sc_w - 0.3, 0.38,
                    size=14, bold=False, color=MUTED)
        add_textbox(slide, val, sc_l + 0.15, sc_t + 0.42, sc_w - 0.3, 0.52,
                    size=24, bold=True, color=col)
        add_textbox(slide, sub, sc_l + 0.15, sc_t + 0.9, sc_w - 0.3, 0.38,
                    size=12, color=MUTED)
        sc_t += sc_h + sc_gap

    # Note on ROUGE
    add_textbox(slide,
                "* Modest ROUGE gains expected — different engineers use different vocabulary for same incident (semantic similarity ~flat at 0.576).",
                0.4, 5.9, 12.5, 0.55, size=12, color=MUTED, italic=True)

    add_footer(slide, "Unique", 10)
    set_notes(slide, """Unique | 8:15–10:00 (1 min 45 sec)

"To validate TraceMind objectively, we evaluated our held-out test split of 93 high-quality incidents from 40 companies that the system never indexed.

Comparing TraceMind against an unaugmented zero-shot baseline on these 93 incidents, RAG achieved slightly higher overlap: ROUGE-1 F1 rose from 0.1809 to 0.1941, an increase of 0.0132, while ROUGE-2 and ROUGE-L showed similar modest gains as shown on the slide. Semantic embedding similarity remained essentially flat at 0.576.

In our failure category benchmark, retrieval hit the exact category in 33.3 percent of test incidents. That is slightly below the 35.5 percent majority-class baseline — meaning always guessing software bugs — because category matching is a weak, coarse signal. Our value is not labeling the bucket, but delivering the specific, grounded root cause and verified remediation checklist.

Crucially, on the seven test cases below our 0.40 similarity floor, all seven triggered the gate with 'Precedent: not relevant' on line one, though we did not manually verify irrelevance.

To be clear about limitations: this is an academic proof of concept on 93 test incidents, so we do not claim statistical significance. Furthermore, because our test set contains exclusively high-quality postmortems, results are optimistic compared to messy production logs. Groq's 20-RPM free tier also required rate limiting."

HANDOFF: (continues to Slide 11)
""")

    # ── Slide 11: Limitations & Future Work ───────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "Honest Limitations & Roadmap")

    # Two columns
    col_t11 = 1.3; col_h11 = 4.6

    # Limitations column
    add_rect(slide, 0.4, col_t11, 5.9, col_h11, CARD, WARNING, 0.8)
    add_textbox(slide, "Limitations", 0.6, col_t11 + 0.1, 5.5, 0.5,
                size=20, bold=True, color=WARNING)
    lims = [
        "Proof-of-concept: 93 test incidents, no stat. significance",
        "High-quality test set bias — optimistic vs. real noisy logs",
        "Groq free tier: 20 RPM cap, requires client-side throttling",
        "Category hit rate (33.3%) below majority baseline (35.5%)",
    ]
    lt = col_t11 + 0.72
    for ln in lims:
        add_textbox(slide, f"• {ln}", 0.65, lt, 5.4, 0.72, size=16, color=TEXT)
        lt += 0.75

    # Future Work column
    add_rect(slide, 7.0, col_t11, 5.9, col_h11, CARD, ACCENT, 0.8)
    add_textbox(slide, "Future Work", 7.2, col_t11 + 0.1, 5.5, 0.5,
                size=20, bold=True, color=ACCENT)
    futures = [
        "Continuous log streaming (Kafka / Fluentd)",
        "Local fine-tuned SLM (8B) — no external API",
        "Automated Jira / PagerDuty ticket sync",
        "Human-rated actionability scoring",
    ]
    ft = col_t11 + 0.72
    for fn in futures:
        add_textbox(slide, f"• {fn}", 7.25, ft, 5.5, 0.72, size=16, color=TEXT)
        ft += 0.75

    # Closing line
    add_rect(slide, 0.4, 6.1, 12.5, 0.6, CARD, ACCENT, 0.6)
    add_textbox(slide,
                "\"Reliability is built on transparency, not blind faith.\"",
                0.6, 6.12, 12.1, 0.55,
                size=17, bold=True, color=TEXT, align=PP_ALIGN.CENTER, italic=True)

    # URLs
    add_textbox(slide,
                "Frontend: https://tracemind.vercel.app   |   Backend: https://nishhhx0-tracemind-backend.hf.space",
                0.4, 6.75, 12.5, 0.35, size=11, color=MUTED, align=PP_ALIGN.CENTER)

    add_footer(slide, "Unique", 11)
    set_notes(slide, """Unique | 8:15–10:00 (continued)

"TraceMind's value is simple: during an outage, it shows its work. Thank you, we are ready for your questions."

HANDOFF: (Q&A — refer to Backup Slide 12 if category hit rate is questioned)

Deployment links:
- Frontend Console: https://tracemind.vercel.app
- Backend API: https://nishhhx0-tracemind-backend.hf.space
""")

    # ── Slide 12 (Backup): Category Hit Rate ──────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide)
    add_title(slide, "BACKUP — Retrieval Category Benchmark", "N = 93 test incidents")

    # Backup label
    add_rect(slide, 11.5, 0.15, 1.6, 0.42, DANGER, None, 0)
    add_textbox(slide, "BACKUP", 11.5, 0.15, 1.6, 0.42,
                size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Baselines row
    base_items = [
        ("Random Chance", "11.1%", "1 / 9 categories", MUTED),
        ("Majority Baseline\n(Software Bug)", "35.5%", "33 / 93", WARNING),
        ("TraceMind\nTop-1 Category Hit", "33.3%", "31 / 93", ACCENT),
    ]
    bi_w = 3.8; bi_h = 1.3; bi_t = 1.4; bi_gap = 0.35; bi_start = 0.5
    for i, (label, val, sub, col) in enumerate(base_items):
        bl = bi_start + i * (bi_w + bi_gap)
        add_rect(slide, bl, bi_t, bi_w, bi_h, CARD, col, 0.8)
        add_textbox(slide, label, bl + 0.1, bi_t + 0.05, bi_w - 0.2, 0.42,
                    size=14, bold=False, color=MUTED)
        add_textbox(slide, val, bl + 0.1, bi_t + 0.42, bi_w - 0.2, 0.55,
                    size=26, bold=True, color=col)
        add_textbox(slide, sub, bl + 0.1, bi_t + 1.0, bi_w - 0.2, 0.3,
                    size=12, color=MUTED)

    # Category breakdown table
    table_t = 3.0; table_l = 0.4; table_w = 12.5
    headers = ["Category", "Hit Rate", "Hits / Total", "Note"]
    rows = [
        ("Software Bug",   "54.5%", "18 / 33", "Strongest"),
        ("Security",       "50.0%", "3 / 6",   "Strong"),
        ("Infrastructure", "~avg",  "-",        ""),
        ("Hardware",       "0.0%",  "0 / 3",   "Weakest"),
        ("Third-Party",    "0.0%",  "0 / 2",   "Weakest"),
        ("Memory",         "0.0%",  "0 / 1",   "Weakest"),
    ]
    col_ws = [4.0, 2.2, 2.4, 3.6]
    # Header
    hx = table_l
    for hi, (hdr, cw) in enumerate(zip(headers, col_ws)):
        add_rect(slide, hx, table_t, cw, 0.45, ACCENT, None, 0)
        add_textbox(slide, hdr, hx + 0.05, table_t, cw - 0.1, 0.45,
                    size=14, bold=True, color=WHITE)
        hx += cw
    # Rows
    for ri, row in enumerate(rows):
        rx = table_l
        rc = CARD if ri % 2 == 0 else BG
        for ci, (cell, cw) in enumerate(zip(row, col_ws)):
            add_rect(slide, rx, table_t + 0.45 + ri * 0.44, cw, 0.44, rc, BORDER, 0.3)
            fc = DANGER if cell == "0.0%" else (SUCCESS if "%" in cell and cell != "0.0%" and cell != "~avg" else TEXT)
            add_textbox(slide, cell, rx + 0.05, table_t + 0.45 + ri * 0.44, cw - 0.1, 0.44,
                        size=13, color=fc)
            rx += cw

    # Takeaway
    add_rect(slide, 0.4, 6.0, 12.5, 0.65, CARD, SUCCESS, 0.6)
    add_textbox(slide,
                "Takeaway: Coarse categorical matching is a weak signal. True utility = grounded root causes + concrete remediation steps, not bucket labels.",
                0.6, 6.05, 12.1, 0.57, size=14, bold=True, color=SUCCESS)

    add_footer(slide, "Unique (Backup)", 12)
    set_notes(slide, """Unique | Backup — only if jury asks about category hit rate

"In our failure category benchmark, retrieval hit the exact category in 33.3 percent of test incidents. That is slightly below the 35.5 percent majority-class baseline — meaning always guessing 'software bug' — because category matching is a weak, coarse signal.

Our strongest categories are Software Bug (54.5%, 18/33) and Security (50.0%, 3/6). We perform worst on rare failure types: Hardware (0/3), Third-Party (0/2), and Memory (0/1).

Category hit rate is a coarse, imperfect proxy for incident diagnosis. In operations, predicting that an outage belongs to the broad category of 'Software Bug' provides zero actionable value to an on-call engineer. The true value of retrieval is delivering the specific historical root cause, architecture component, and five concrete remediation steps from a verified incident, alongside a hyperlink to the postmortem."
""")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PPTX))
    print(f"[OK] Saved: {OUTPUT_PPTX}")

    # ── Summary table ─────────────────────────────────────────────────────────
    slide_meta = [
        (1,  "Title Slide",                           "Nishit",          notes_1),
    ]
    # Approximate note word counts for other slides (from text above)
    summaries = [
        (1,  "Title Slide",                           "Nishit",          146),
        (2,  "Agenda & Core Premise",                 "Nishit",          55),
        (3,  "The Outage Reality",                    "Pragun",          185),
        (4,  "Why Generic LLMs Fall Short",           "Pragun",          55),
        (5,  "Postmortem Corpus & Split",             "Pranav",          216),
        (6,  "HDBSCAN Clustering",                    "Pranav",          60),
        (7,  "ChromaDB Vector Database",              "Sameera",         120),
        (8,  "2-Pass Retrieval & Gate",               "Sameera",         119),
        (9,  "Architecture & Live Demo",              "Nishit",          159),
        (10, "Quantitative Benchmarks",               "Unique",          221),
        (11, "Limitations & Roadmap",                 "Unique",          55),
        (12, "BACKUP — Category Benchmark",           "Unique (Backup)", 100),
    ]
    print()
    print(f"{'Slide':<6} {'Title':<42} {'Speaker':<18} {'Notes Words':>12}")
    print("-" * 82)
    for num, title, speaker, wc in summaries:
        print(f"{num:<6} {title:<42} {speaker:<18} {wc:>12}")

    return str(OUTPUT_PPTX)


if __name__ == "__main__":
    path = build_deck()
    print(f"\nDone. Output: {path}")
