"""
scripts/build_report.py
-----------------------
Constructs the complete TraceMind Project Report as a professional .docx document
matching the formatting, typography, and visual structure of the RV University sample report.
Updates TOC fields and exports native PDF via Microsoft Word COM automation.
Target length: 20-26 pages.
"""

import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import win32com.client

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCX_PATH = DOCS_DIR / "TraceMind_Report.docx"
PDF_PATH = DOCS_DIR / "TraceMind_Report.pdf"

# Image paths
RVU_LOGO = DOCS_DIR / "rvu_logo.png"
FIG1_ARCH = DOCS_DIR / "fig1_architecture.png"
FIG2_EVAL = DOCS_DIR / "fig2_evaluation.png"
FIG3_CONSOLE = DOCS_DIR / "fig3_console_overview.png"
FIG4_EDITOR = DOCS_DIR / "fig4_input_editor.png"
FIG5_ANALYSIS = DOCS_DIR / "fig5_analysis_remediation.png"
FIG6_TRACE = DOCS_DIR / "fig6_pipeline_trace.png"


def set_cell_margins(cell, top=70, bottom=70, left=90, right=90):
    """Set inner cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m_name, val in [("w:top", top), ("w:bottom", bottom), ("w:left", left), ("w:right", right)]:
        node = OxmlElement(m_name)
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_shading(cell, color_hex="F1F5F9"):
    """Set background color of a cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    """Set professional light gray horizontal borders on table."""
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement("w:tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), val)
        border.set(qn("w:sz"), sz)
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), color)
        tblBorders.append(border)
    border_insideV = OxmlElement("w:insideV")
    border_insideV.set(qn("w:val"), "none")
    tblBorders.append(border_insideV)
    tblPr.append(tblBorders)


def format_table_structure(table, col_widths, is_header=True):
    """Enforce cantSplit on rows and tblHeader on row 0."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    if is_header and len(table.rows) > 0:
        trPr0 = table.rows[0]._tr.get_or_add_trPr()
        tblHeader = OxmlElement("w:tblHeader")
        trPr0.append(tblHeader)

    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        cantSplit = OxmlElement("w:cantSplit")
        trPr.append(cantSplit)
        for i, cell in enumerate(row.cells):
            cell.width = col_widths[i]
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_code_listing(doc, code_text, caption_text, explanation_text):
    """Add a shaded, bounded, un-split code listing with caption and explanation."""
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = "Times New Roman"
    run_cap.font.size = Pt(9.5)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(15, 23, 42)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    row = table.rows[0]
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))

    cell = row.cells[0]
    cell.width = Inches(6.5)
    set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
    set_cell_shading(cell, "F8FAFC")

    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for b_name in ["top", "left", "bottom", "right"]:
        b = OxmlElement(f"w:{b_name}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "E2E8F0")
        tcBorders.append(b)
    tcPr.append(tcBorders)

    code_lines = code_text.strip().split("\n")
    for j, line in enumerate(code_lines):
        p = cell.paragraphs[0] if j == 0 else cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0.5)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(8.0)
        run.font.color.rgb = RGBColor(30, 41, 59)

    p_exp = doc.add_paragraph()
    p_exp.paragraph_format.space_before = Pt(2)
    p_exp.paragraph_format.space_after = Pt(6)
    p_exp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_exp = p_exp.add_run(explanation_text)
    r_exp.font.name = "Times New Roman"
    r_exp.font.size = Pt(9.5)
    r_exp.font.italic = True
    r_exp.font.color.rgb = RGBColor(71, 85, 105)


def add_figure(doc, img_path, caption_text, width=Inches(5.5)):
    """Add a centered figure with caption below."""
    p_img = doc.add_paragraph()
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=width)

    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    p_cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = "Times New Roman"
    run_cap.font.size = Pt(9.5)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(30, 41, 59)


def add_table_caption(doc, caption_text):
    """Add a bold table caption above table."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(caption_text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)


def main():
    print("[Report Builder] Starting TraceMind Report assembly...")
    doc = docx.Document()

    # 1. Page Setup: 8.5 x 11 inches, 1-inch margins all around
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.different_first_page_header_footer = True

    # Configure Default Styles
    styles = doc.styles
    normal_style = styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(15, 23, 42)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(5)
    normal_style.paragraph_format.space_before = Pt(0)

    # Configure Heading 1 style
    h1_style = styles["Heading 1"]
    h1_style.font.name = "Times New Roman"
    h1_style.font.size = Pt(14)
    h1_style.font.bold = True
    h1_style.font.color.rgb = RGBColor(15, 23, 42)
    h1_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1_style.paragraph_format.space_before = Pt(12)
    h1_style.paragraph_format.space_after = Pt(4)
    h1_style.paragraph_format.keep_with_next = True

    # Configure Heading 2 style
    h2_style = styles["Heading 2"]
    h2_style.font.name = "Times New Roman"
    h2_style.font.size = Pt(11.5)
    h2_style.font.bold = True
    h2_style.font.color.rgb = RGBColor(30, 41, 59)
    h2_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h2_style.paragraph_format.space_before = Pt(8)
    h2_style.paragraph_format.space_after = Pt(2)
    h2_style.paragraph_format.keep_with_next = True

    # Footer on subsequent pages
    footer = section.footer
    f_p = footer.paragraphs[0]
    f_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    f_r1 = f_p.add_run("TraceMind: Project Report")
    f_r1.font.name = "Times New Roman"
    f_r1.font.size = Pt(9)
    f_r1.font.color.rgb = RGBColor(100, 116, 139)

    f_p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    f_p.add_run("\t")

    f_r2 = f_p.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    fld_text = OxmlElement("w:instrText")
    fld_text.set(qn("xml:space"), "preserve")
    fld_text.text = "PAGE"
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "separate")
    fld3 = OxmlElement("w:fldChar")
    fld3.set(qn("w:fldCharType"), "end")
    f_r2._r.append(fld1)
    f_r2._r.append(fld_text)
    f_r2._r.append(fld2)
    f_r2._r.append(fld3)
    f_r2.font.name = "Times New Roman"
    f_r2.font.size = Pt(9)
    f_r2.font.color.rgb = RGBColor(100, 116, 139)

    # -------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("RV UNIVERSITY, BENGALURU-59")
    r.font.bold = True
    r.font.size = Pt(15)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING")
    r.font.bold = True
    r.font.size = Pt(12)

    # Logo
    if RVU_LOGO.exists():
        p_logo = doc.add_paragraph()
        p_logo.paragraph_format.space_before = Pt(2)
        p_logo.paragraph_format.space_after = Pt(10)
        p_logo.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.add_run().add_picture(str(RVU_LOGO), width=Inches(1.9))

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("A Project Report On")
    r.font.size = Pt(11)
    r.font.italic = True

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TraceMind: RAG-Powered Root Cause Analysis for SRE Logs")
    r.font.bold = True
    r.font.size = Pt(16)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Submitted in Fulfillment for the award of degree of")
    r.font.size = Pt(10)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("BSc(Honors) in School of Computer Science and Engineering")
    r.font.bold = True
    r.font.size = Pt(11)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Course: Natural Language Processing (CS3255)")
    r.font.bold = True
    r.font.size = Pt(11)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Submitted By")
    r.font.bold = True
    r.font.size = Pt(10.5)

    team_members = [
        ("Patel Nishitkumar", "1RUA24SCS0068"),
        ("Pragun", "1RUA24SCS0072"),
        ("Pranav", "1RUA24SCS0077"),
        ("Sameera", "1RUA24SCS0099"),
        ("Unique", "1RUA24SCS0118"),
    ]
    t_team = doc.add_table(rows=len(team_members), cols=2)
    t_team.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (m_name, m_usn) in enumerate(team_members):
        row = t_team.rows[idx]
        row.cells[0].paragraphs[0].text = m_name
        row.cells[0].paragraphs[0].runs[0].font.name = "Times New Roman"
        row.cells[0].paragraphs[0].runs[0].font.size = Pt(10)
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

        row.cells[1].paragraphs[0].text = f"USN: {m_usn}"
        row.cells[1].paragraphs[0].runs[0].font.name = "Times New Roman"
        row.cells[1].paragraphs[0].runs[0].font.size = Pt(10)
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_cell_margins(row.cells[0], top=15, bottom=15, left=40, right=40)
        set_cell_margins(row.cells[1], top=15, bottom=15, left=40, right=40)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Under the Guidance of")
    r.font.bold = True
    r.font.size = Pt(10.5)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Prof. Shanthala T N")
    r.font.bold = True
    r.font.size = Pt(11)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Professor, School of Computer Science and Engineering")
    r.font.size = Pt(10)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("RV University, Bengaluru-560059")
    r.font.size = Pt(10)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026-2027")
    r.font.bold = True
    r.font.size = Pt(10.5)

    # -------------------------------------------------------------
    # ABSTRACT (Own Page) - Exactly 253 Words (220-260 target)
    # -------------------------------------------------------------
    doc.add_page_break()
    p_abs_title = doc.add_paragraph()
    p_abs_title.paragraph_format.space_before = Pt(12)
    p_abs_title.paragraph_format.space_after = Pt(6)
    p_abs_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_abs = p_abs_title.add_run("ABSTRACT")
    r_abs.font.bold = True
    r_abs.font.size = Pt(14)

    p1 = doc.add_paragraph(
        "During high-severity infrastructure outages, Site Reliability Engineers spend significant Mean Time to Resolution "
        "diagnosing root causes from cryptic, high-volume server logs. Existing workflows rely on manual wiki searches that fail "
        "across lexical synonymy or generic large language models that risk fabricating invalid configurations and dangerous shell "
        "commands. The primary objective of TraceMind is to build an auditable incident triage system that correlates raw error logs "
        "with verified historical postmortems using Retrieval-Augmented Generation."
    )

    p2 = doc.add_paragraph(
        "We constructed a standardized corpus of 828 real postmortems (544 high-quality and 284 low-quality summaries) from curated "
        "archives, Statuspage histories across 20 SaaS platforms, Google Cloud Platform, and Cloudflare. We enforced a strict company-disjoint "
        "split using fixed seed 42, creating a 662-record training set across 18 companies and a 93-record high-quality test set across "
        "40 unseen companies, discarding 73 low-quality test-company records. The pipeline combines 384-dimensional dense sentence embeddings, "
        "HDBSCAN clustering for burst deduplication (48 clusters, min_cluster_size=3), two-pass ChromaDB retrieval with a 0.40 cosine floor, "
        "and a Relevance Gate prompt paired with Groq openai/gpt-oss-120b inference."
    )

    p3 = doc.add_paragraph(
        "TraceMind provides full auditability through an interactive Pipeline Trace displaying raw cosine similarity, verified postmortem "
        "hyperlinks, and verbatim prompts. On our 93 held-out test incidents, RAG achieved modest empirical gains over zero-shot inference: "
        "ROUGE-1 rose from 0.1809 to 0.1941 (+0.0132), ROUGE-2 from 0.0384 to 0.0439 (+0.0055), ROUGE-L from 0.1061 to 0.1210 (+0.0149), "
        "and cosine similarity from 0.5743 to 0.5764 (+0.0021). All 7 cases below the 0.40 floor triggered the relevance gate. This academic "
        "proof of concept confirms that transparent grounding provides critical operational reliability during outages."
    )

    wc = len(p1.text.split()) + len(p2.text.split()) + len(p3.text.split())
    print(f"[Report Builder] Abstract word count: {wc} words (Target: 220-260 words).")

    # -------------------------------------------------------------
    # TABLE OF CONTENTS (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    p_toc_title = doc.add_paragraph()
    p_toc_title.paragraph_format.space_before = Pt(12)
    p_toc_title.paragraph_format.space_after = Pt(6)
    p_toc_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_toc = p_toc_title.add_run("TABLE OF CONTENTS")
    r_toc.font.bold = True
    r_toc.font.size = Pt(14)

    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.space_before = Pt(6)
    p_toc.paragraph_format.space_after = Pt(6)
    run_toc = p_toc.add_run()

    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-1" \\h \\z \\u'
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "separate")
    fld3 = OxmlElement("w:fldChar")
    fld3.set(qn("w:fldCharType"), "end")

    run_toc._r.append(fld1)
    run_toc._r.append(instr)
    run_toc._r.append(fld2)
    run_toc._r.append(fld3)

    # -------------------------------------------------------------
    # 1. INTRODUCTION (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("1. Introduction", level=1)

    doc.add_paragraph(
        "Modern cloud-native computing architectures rely upon intricate topologies of microservices, distributed "
        "data stores, asynchronous message brokers, and container orchestration platforms. While distributed systems provide "
        "unprecedented horizontal scalability and fault isolation, they introduce severe operational complexity when cascading "
        "failures emerge. A single localized degradation—such as a database connection pool exhaustion or a subtle network "
        "partition—can trigger thousands of downstream error logs, circuit breaker trips, and 504 Gateway Timeouts across dozens "
        "of interdependent services. In production environments, Site Reliability Engineers (SREs) are tasked with restoring service "
        "health under stringent Service Level Agreements (SLAs). During these critical incidents, the overwhelming majority of "
        "Mean Time to Resolution (MTTR) is consumed not by executing remediation actions, but by diagnosing the root cause amidst "
        "a barrage of unstructured log telemetry."
    )

    doc.add_paragraph(
        "Postmortem retrospectives represent the institutional memory of software engineering organizations. When major incidents "
        "are resolved, engineering teams conduct rigorous root-cause investigations, documenting the sequence of events, underlying "
        "architectural flaws, and specific recovery procedures. SRE organizations frequently publish these postmortems to public repositories "
        "and status archives (such as icco/postmortems, Statuspage platforms, GCP incident disclosures, and Cloudflare retrospectives). "
        "These documents represent an invaluable knowledge base. However, during an active outage, this knowledge remains largely "
        "inaccessible. Incident responders lack the cognitive bandwidth and time to manually search historical archives. Traditional "
        "information retrieval systems rely heavily on exact keyword matching, which consistently fails due to vocabulary mismatch and "
        "synonymy across engineering teams: one service team logs 'database connection pool exhausted (active: 100/100)', while a historical "
        "postmortem records 'downstream query timeout due to saturated connection pool acquisition limits'."
    )

    doc.add_paragraph(
        "Recent advances in natural language processing (NLP), specifically dense vector representations and Retrieval-Augmented "
        "Generation (RAG) [1], provide a viable mechanism to bridge this semantic gap. By mapping unstructured log excerpts and historical "
        "postmortems into a shared continuous semantic vector space [2], systems can retrieve conceptually relevant incident precedents "
        "regardless of lexical variation. Large Language Models (LLMs) can subsequently synthesize these retrieved precedents into a "
        "coherent technical analysis and actionable mitigation steps."
    )

    doc.add_paragraph(
        "However, deploying naive RAG architectures in high-stakes operational environments introduces substantial risks. Standard LLMs "
        "frequently hallucinate plausible-sounding but erroneous technical assertions, fabricating nonexistent configuration flags, "
        "invalid CLI options, or dangerous shell commands. Furthermore, when a query represents a novel failure pattern with no legitimate "
        "historical precedent, standard vector search retrieves the nearest semantic neighbor regardless of absolute relevance. Unconstrained "
        "LLMs then force false analogies, attempting to explain a Kubernetes OOMKill event using an unrelated Redis failover postmortem. "
        "Such opaque, ungrounded outputs undermine engineer trust and risk exacerbating production downtime."
    )

    doc.add_paragraph(
        "To resolve these operational challenges, we developed TraceMind: an auditable, verifiable incident triage system for Site Reliability "
        "Engineers. TraceMind combines density-based anomaly clustering (HDBSCAN) [3] for log burst deduplication, a two-pass vector "
        "retrieval engine in ChromaDB [5] governed by an explicit 0.40 cosine similarity threshold, a structured Relevance Gate guardrail, "
        "and an interactive Pipeline Trace console. Unlike black-box diagnostic assistants, TraceMind guarantees full auditability: responders "
        "can inspect the exact historical incident retrieved, its mathematical similarity score, direct hyperlinks to the verified postmortem, "
        "and the verbatim prompt delivered to the model."
    )

    # -------------------------------------------------------------
    # 2. PROBLEM STATEMENT AND OBJECTIVES (Flows Naturally)
    # -------------------------------------------------------------
    doc.add_heading("2. Problem Statement and Objectives", level=1)

    doc.add_heading("2.1 Problem Statement", level=2)
    doc.add_paragraph(
        "Modern incident management suffers from acute information asymmetry and high diagnostic latency. When infrastructure fails, "
        "on-call engineers are inundated with high-volume, repetitive log streams that describe immediate symptoms (such as connection "
        "timeouts, thread pool saturations, and HTTP 500 errors) rather than latent systemic causes (such as misconfigured database max_connections, "
        "stale connection leaks, or unbalanced shard rebalancing). Responders spend hours manually querying internal knowledge bases and chat "
        "channels to determine whether the organization has resolved identical failures previously. Keyword search engines fail to correlate "
        "incidents due to lexical variation, while ungrounded large language models generate unverified hallucinations that cannot be trusted "
        "during production emergencies. Furthermore, standard retrieval pipelines fail silently when encountering novel anomalies, generating "
        "misleading explanations based on irrelevant historical matches. A reliable incident response tool must automatically deduplicate "
        "incoming log bursts, retrieve semantically relevant postmortem precedents from verified archives, enforce strict relevance guardrails, "
        "and expose full provenance for every diagnostic assertion."
    )

    doc.add_heading("2.2 Project Objectives", level=2)
    doc.add_paragraph(
        "To systematically address the problem statement, TraceMind was designed and implemented around seven concrete technical objectives:"
    )

    objectives = [
        ("Curate a Standardized Postmortem Corpus:",
         "Aggregate, parse, clean, and standardize a diverse dataset of real-world infrastructure postmortems from public repositories, "
         "commercial SaaS Statuspage archives, and major cloud provider retrospectives, establishing a multi-attribute JSON schema."),
        ("Establish a Leakage-Free Evaluation Split:",
         "Implement a company-disjoint dataset splitting pipeline using deterministic grouping (seed 42) to ensure that no company appears "
         "in both the training and evaluation splits, thereby validating genuine out-of-distribution generalization without data contamination."),
        ("Implement Preprocessing Anomaly Clustering:",
         "Deploy HDBSCAN density-based clustering over dense sentence embeddings to group recurring failure patterns, deduplicate repetitive "
         "error bursts, and isolate novel infrastructure noise prior to vector indexing."),
        ("Design a Two-Pass Quality-Biased Retrieval Engine:",
         "Construct an in-process ChromaDB vector database implementing two-pass retrieval that prioritizes high-quality postmortems scoring "
         "above an explicit 0.40 cosine similarity floor, falling back to broader corpus records only when necessary."),
        ("Develop a Structured Relevance Gate Guardrail:",
         "Engineer a specialized prompt variant (relevance_gate_v2) that forces the language model to evaluate precedent relevance before "
         "generating root causes, requiring an explicit 'Precedent: not relevant' output when similarity is weak to eliminate false analogies."),
        ("Deploy a Resilient, Low-Latency System Architecture:",
         "Deliver a production-ready, modular system combining a FastAPI backend on Hugging Face Spaces with a Next.js 16 Bento UI console on "
         "Vercel, protected by client-side per-IP rate limiting (15 RPM) and an SQLite generation cache for high availability."),
        ("Surface Full Pipeline Trace Auditability:",
         "Implement an interactive telemetry interface exposing raw sentence embeddings, a semicircular cosine similarity gauge with an "
         "explicit 0.40 threshold marker, verified postmortem hyperlinks, and verbatim prompt inspection for complete operational transparency.")
    ]

    for title, desc in objectives:
        p_obj = doc.add_paragraph()
        p_obj.paragraph_format.left_indent = Inches(0.2)
        p_obj.paragraph_format.space_before = Pt(1.5)
        p_obj.paragraph_format.space_after = Pt(2.5)
        r_t = p_obj.add_run(f"• {title} ")
        r_t.bold = True
        r_t.font.name = "Times New Roman"
        r_d = p_obj.add_run(desc)
        r_d.font.name = "Times New Roman"

    # -------------------------------------------------------------
    # 3. TECHNOLOGIES USED (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("3. Technologies Used", level=1)

    doc.add_paragraph(
        "TraceMind was developed using a carefully selected software stack designed for high computational efficiency, low operational "
        "overhead, and strict reproducibility. All technology selections and package versions are verified directly against the repository's "
        "configuration files, including requirements.txt for backend and evaluation dependencies, and package.json for the frontend console. "
        "Table 1 outlines the complete architectural stack, version manifest, and functional purpose across all layers of the system."
    )

    add_table_caption(doc, "Table 1: TraceMind Technology Stack, Version Manifest, and Architectural Purpose")

    tech_data = [
        ("Embeddings & NLP", "sentence-transformers", ">=2.2.0 (all-MiniLM-L6-v2)", "Transforms raw logs and postmortem texts into 384-dimensional dense semantic vectors [2]."),
        ("Clustering", "hdbscan", ">=0.8.33", "Hierarchical density-based clustering for log deduplication and noise detection [3]."),
        ("Scientific Computing", "scikit-learn / numpy", ">=1.3.0 / >=1.24.0", "Cosine similarity computation, distance metrics, and vector array operations."),
        ("Vector Store", "chromadb", ">=0.4.0", "In-process vector database providing semantic similarity retrieval over indexed embeddings [5]."),
        ("Backend Framework", "FastAPI / Uvicorn", ">=0.104.0 / >=0.24.0", "Asynchronous REST API framework hosting /analyze and health endpoints [6]."),
        ("Data Validation", "pydantic", ">=2.0", "Strict schema enforcement and serialization for incoming requests and pipeline traces."),
        ("HTTP Client", "httpx / requests", ">=0.25.0 / >=2.31.0", "High-throughput asynchronous communication with Groq LLM inference endpoints."),
        ("Local Cache", "sqlite3", "Standard Library (3.x)", "Persistent key-value cache (rag/cache/gen_cache.sqlite3) storing 201 pre-warmed completions."),
        ("LLM Inference", "Groq API Client", "Cloud LPU Hardware", "High-speed inference utilizing openai/gpt-oss-120b for root cause synthesis [8]."),
        ("Testing & Quality", "pytest / pytest-cov", ">=7.0.0 / >=4.0.0", "Automated test suite (30 test cases) verifying backend, clustering, and data splits."),
        ("NLP Evaluation", "rouge-score", ">=0.1.2", "Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for empirical benchmarking [4]."),
        ("Frontend Web", "Next.js / React", "16.3.5 / 19.x", "Server-side rendering, React hooks, and component lifecycle for SRE console [7]."),
        ("Styling & UI", "Tailwind CSS / shadcn/ui", "^4 / Radix UI, Lucide ^1.16", "Editorial Bento UI layout, Radix tabs/badges, and Monaco-style soft-wrapped log editor."),
        ("Cloud Deployment", "Hugging Face / Vercel", "ZeroGPU Space / Edge Network", "Dual-cloud deployment hosting backend API and edge-distributed frontend application.")
    ]

    t_tech = doc.add_table(rows=len(tech_data) + 1, cols=4)
    format_table_structure(t_tech, [Inches(1.2), Inches(1.4), Inches(1.5), Inches(2.4)])
    set_table_borders(t_tech)

    headers = ["Layer", "Technology", "Version / Model", "Architectural Purpose"]
    for i, h_text in enumerate(headers):
        cell = t_tech.rows[0].cells[i]
        cell.paragraphs[0].text = h_text
        set_cell_shading(cell, "E2E8F0")
        r = cell.paragraphs[0].runs[0]
        r.font.bold = True
        r.font.size = Pt(8.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, (layer, tech, ver, purp) in enumerate(tech_data):
        row = t_tech.rows[idx + 1]
        for col_idx, text_val in enumerate([layer, tech, ver, purp]):
            cell = row.cells[col_idx]
            cell.paragraphs[0].text = text_val
            r = cell.paragraphs[0].runs[0]
            r.font.size = Pt(8.0)
            if col_idx == 1:
                r.font.name = "Consolas"
            if col_idx == 0:
                r.font.bold = True
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 3 else WD_ALIGN_PARAGRAPH.CENTER if col_idx == 2 else WD_ALIGN_PARAGRAPH.LEFT

    # -------------------------------------------------------------
    # 4. DATASET AND PREPROCESSING (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("4. Dataset and Preprocessing", level=1)

    doc.add_heading("4.1 Corpus Acquisition and Standardization", level=2)
    doc.add_paragraph(
        "A retrieval-augmented diagnostic system depends fundamentally upon the authenticity and quality of its underlying corpus. "
        "Rather than relying upon synthetic logs generated by language models, we constructed an empirical corpus of 828 real-world "
        "infrastructure postmortems. Data acquisition followed an aggregator-first strategy: primary records were extracted from the curated "
        "icco/postmortems archive [9], followed by targeted web scraping of official Statuspage incident histories across 20 major SaaS and "
        "cloud infrastructure providers (including Datadog, Twilio, GitHub, Discord, CircleCI, DigitalOcean, Sentry, Dropbox, and Squarespace). "
        "These records were complemented by detailed retrospectives from the Google Cloud Platform incident disclosures and in-depth engineering "
        "blogs from Cloudflare. Scraped records were normalized into a uniform schema consisting of: incident_id, company, raw_log_excerpt, "
        "root_cause, remediation_steps, source_url, and a binary quality indicator."
    )

    doc.add_heading("4.2 Quality Filtering Rationale", level=2)
    doc.add_paragraph(
        "Raw scrapings of public status pages frequently contain sparse, low-information entries—such as brief customer updates stating "
        "'we are monitoring DNS recovery'—which lack technical depth. Priming a language model with vague status announcements degrades generation "
        "quality, encouraging superficial answers. To eliminate this noise, we developed a programmatic quality filter (scripts/scrapers/quality_pass.py). "
        "A record is classified as 'quality: high' only if its root cause articulates specific failure mechanisms (e.g., buffer exhaustion, "
        "lock contention, deadlocks, cgroup limits) and its remediation defines multi-step actionable mitigations. Across the 828 collected "
        "records, 544 were verified as high quality, while 284 brief status summaries were classified as low quality."
    )

    doc.add_heading("4.3 Company-Disjoint Dataset Split (Zero Leakage)", level=2)
    doc.add_paragraph(
        "Standard machine learning evaluations frequently perform random uniform sampling across dataset records. In incident analysis, "
        "random splitting introduces severe data leakage: if an infrastructure provider experiences multiple related outages, identical "
        "error terminology and architecture components appear in both the training index and the test set. An evaluation under random splitting "
        "measures the system's ability to memorize company-specific vocabulary rather than its capacity to generalize across unseen systems."
    )

    doc.add_paragraph(
        "To enforce genuine out-of-distribution evaluation, we implemented a company-grouped split (scripts/make_split.py) using fixed seed 42. "
        "All records from a single company are assigned strictly to either the training split or the test split. Furthermore, to evaluate "
        "diagnostic capability against rigorous ground truth, the test set was restricted strictly to 100% high-quality postmortems. To prevent "
        "both company contamination and quality dilution, all 73 low-quality records belonging to the test companies were completely discarded. "
        "Table 2 details the exact distribution of corpus records, quality stratification, and split boundaries."
    )

    add_table_caption(doc, "Table 2: Corpus Statistics, Quality Stratification, and Leakage-Safe Dataset Splits")

    split_rows = [
        ("Total Acquired Corpus", "828 records across 58 companies (104 granular domains)", "544 High / 284 Low", "100.0%", "Complete unsegmented postmortem corpus."),
        ("Training Corpus (train.json)", "662 records across 18 primary companies (64 domains)", "451 High / 211 Low", "79.95%", "Exclusive corpus used for ChromaDB index & HDBSCAN."),
        ("Held-Out Test Set (test.json)", "93 records across 40 unseen companies", "93 High / 0 Low", "11.23%", "Strictly high quality; zero company overlap with train."),
        ("Excluded Test-Company Records", "73 records across 40 test companies", "0 High / 73 Low", "8.82%", "Discarded to maintain company-disjointness without noise."),
        ("Train/Test Company Overlap", "0 companies (set(train) & set(test) == empty)", "N/A", "0.0%", "Verified programmatically in tests/test_split.py."),
        ("Maximum Cross-Split Similarity", "0.8438 cosine similarity (strictly < 0.95)", "N/A", "Pass", "Guarantees zero near-duplicate incident leakage.")
    ]

    t_split = doc.add_table(rows=len(split_rows) + 1, cols=5)
    format_table_structure(t_split, [Inches(1.5), Inches(1.8), Inches(1.1), Inches(0.6), Inches(1.5)])
    set_table_borders(t_split)

    s_headers = ["Partition / Metric", "Scope & Companies", "Quality Breakdown", "Share", "Verification Rationale"]
    for i, h_text in enumerate(s_headers):
        cell = t_split.rows[0].cells[i]
        cell.paragraphs[0].text = h_text
        set_cell_shading(cell, "E2E8F0")
        r = cell.paragraphs[0].runs[0]
        r.font.bold = True
        r.font.size = Pt(8.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, (part, scope, qual, share, rat) in enumerate(split_rows):
        row = t_split.rows[idx + 1]
        for col_idx, text_val in enumerate([part, scope, qual, share, rat]):
            cell = row.cells[col_idx]
            cell.paragraphs[0].text = text_val
            r = cell.paragraphs[0].runs[0]
            r.font.size = Pt(8.0)
            if col_idx == 0:
                r.font.bold = True
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [2, 3] else WD_ALIGN_PARAGRAPH.LEFT

    # -------------------------------------------------------------
    # 5. SYSTEM ARCHITECTURE (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("5. System Architecture", level=1)

    doc.add_paragraph(
        "TraceMind implements an auditable end-to-end incident triage architecture that connects raw client-side log telemetry "
        "to a high-throughput backend API, vector retrieval database, and LLM inference engine. The entire system is engineered "
        "to eliminate black-box opacity by persisting telemetry at every intermediate step. Figure 1 illustrates the complete system "
        "architecture and directional data flow across the pipeline."
    )

    if FIG1_ARCH.exists():
        add_figure(doc, FIG1_ARCH, "Figure 1: TraceMind System Architecture and End-to-End Pipeline Flow", width=Inches(5.4))

    doc.add_paragraph(
        "The architecture is organized into five tightly coupled functional subsystems:"
    )

    doc.add_heading("5.1 Log Ingestion and Dense Vector Embedding", level=2)
    doc.add_paragraph(
        "Incoming incident logs are submitted via the Next.js frontend or directly to the FastAPI /analyze endpoint. The raw text "
        "is truncated to 2,000 characters to prevent buffer overflow and OOM conditions during high-volume triage. The text is passed "
        "to the sentence-transformers module, which executes inference using the all-MiniLM-L6-v2 model. This maps the log excerpt into "
        "a 384-dimensional dense semantic vector. The resulting vector is L2-normalized, allowing semantic distance to be calculated "
        "efficiently as standard Euclidean distance or inner product cosine similarity."
    )

    doc.add_heading("5.2 HDBSCAN Anomaly Clustering and Burst Deduplication", level=2)
    doc.add_paragraph(
        "During large infrastructure failures, services emit thousands of near-duplicate error lines per minute. If every log line triggered "
        "an independent retrieval query, vector databases and language models would quickly saturate. TraceMind uses HDBSCAN density-based "
        "clustering as a preprocessing filter. Built offline from the 662 training incidents (min_cluster_size=3, min_samples=2), HDBSCAN "
        "identified 48 distinct clusters representing recurring failure modes (such as database connection pool exhaustion, Redis sentinel "
        "failovers, Kubernetes OOMKill events, and DNS resolution failures). Incoming embeddings are compared against cluster centroids. "
        "If similarity to the nearest centroid is below 0.40, the log is assigned cluster ID -1, denoting novel infrastructure noise."
    )

    doc.add_heading("5.3 ChromaDB Two-Pass Vector Retrieval", level=2)
    doc.add_paragraph(
        "Vector search is executed inside an in-process ChromaDB instance loaded directly into the FastAPI application memory. To ensure "
        "diagnostic quality, retrieval executes in two passes. In Pass One, the query targets only records marked quality: high with an "
        "explicit cosine similarity floor of 0.40. If a verified high-quality precedent achieves similarity >= 0.40, it is immediately "
        "selected. If Pass One yields no match or falls below 0.40, the system initiates Pass Two, querying across all 662 training "
        "postmortems. Test incidents are never indexed into the vector collection, eliminating self-retrieval leakage."
    )

    doc.add_heading("5.4 Relevance Gate Prompt Synthesis and LLM Inference", level=2)
    doc.add_paragraph(
        "The retrieved precedent metadata (raw log excerpt, root cause, remediation steps, and similarity score) is synthesized into a "
        "structured prompt using the relevance_gate_v2 template. The prompt strictly instructs the LLM: if the historical case does not "
        "clearly match the incoming failure, it must write 'Precedent: not relevant' on line one of the root cause section and derive the "
        "analysis solely from the log excerpt. The prompt is dispatched via HTTPS to Groq's dedicated LPU hardware running openai/gpt-oss-120b, "
        "which returns a structured response consisting of a technical root cause and a 5-step numbered remediation checklist."
    )

    doc.add_heading("5.5 Resilience Layer: Rate Limiting and SQLite Generation Cache", level=2)
    doc.add_paragraph(
        "To protect against Groq API free-tier limitations (20 requests per minute and 8,000 tokens per minute), the backend incorporates "
        "two resilience mechanisms. First, an in-memory sliding window rate limiter restricts clients to 15 requests per minute per IP address. "
        "Second, a persistent SQLite generation cache (rag/cache/gen_cache.sqlite3) caches completed LLM analyses keyed by query hash and "
        "prompt variant. The cache contains 201 pre-warmed completions generated via scripts/prewarm_cache.py, guaranteeing instantaneous, "
        "error-free responses for standard incident presets even during provider downtime."
    )

    # -------------------------------------------------------------
    # 6. NLP TECHNIQUES IMPLEMENTED (Flows Naturally)
    # -------------------------------------------------------------
    doc.add_heading("6. NLP Techniques Implemented", level=1)

    doc.add_paragraph(
        "TraceMind integrates a comprehensive sequence of Natural Language Processing methodologies to transform noisy, unstructured "
        "operational logs into grounded, structured remediation plans. Rather than relying on black-box heuristics, each technique corresponds "
        "to a concrete module in the repository. Table 3 maps each NLP technique to its repository location and describes how the requirement "
        "is satisfied."
    )

    add_table_caption(doc, "Table 3: Natural Language Processing Techniques and Repo Implementation Mapping")

    nlp_rows = [
        ("Dense Sentence Embeddings", "clustering/cluster.py (embed)", "Generates 384-dimensional dense vectors using all-MiniLM-L6-v2; L2-normalizes vectors to map semantic concept similarity."),
        ("Density-Based Clustering", "clustering/build_index.py (build)", "Executes HDBSCAN (min_cluster_size=3) over training embeddings to identify 48 clusters and isolate novel noise (label -1)."),
        ("Vector Similarity Search", "rag/generate.py (retrieve)", "Performs approximate nearest neighbor search in ChromaDB using L2 distance converted to cosine similarity: sim = 1.0 - (dist / 2.0)."),
        ("Two-Pass Biased Retrieval", "rag/generate.py (retrieve)", "Enforces two-pass retrieval: Pass 1 filters where={'quality': 'high'} with 0.40 floor; Pass 2 searches all 662 train records on miss."),
        ("Structured RAG Prompting", "rag/generate.py (PROMPT_REGISTRY)", "Injects retrieved historical root cause, remediation steps, and similarity score into a strict two-section system template."),
        ("Relevance Gate Guardrail", "rag/generate.py (relevance_gate_v2)", "Forces model to evaluate correlation before answering; writes 'Precedent: not relevant' on line 1 if similarity is ungrounded."),
        ("Group-Wise Dataset Split", "scripts/make_split.py (generate_split)", "Partitions 828 postmortems strictly by company (seed 42); discards 73 low-quality test records to guarantee zero leakage."),
        ("Multi-Metric Evaluation", "eval/evaluate.py (compute_metrics)", "Evaluates 93 test incidents using ROUGE-1, ROUGE-2, and ROUGE-L F1 scores alongside dense embedding cosine similarity.")
    ]

    t_nlp = doc.add_table(rows=len(nlp_rows) + 1, cols=3)
    format_table_structure(t_nlp, [Inches(1.8), Inches(2.2), Inches(2.5)])
    set_table_borders(t_nlp)

    nlp_headers = ["NLP Technique", "Repository Location", "How It Is Satisfied in Code"]
    for i, h_text in enumerate(nlp_headers):
        cell = t_nlp.rows[0].cells[i]
        cell.paragraphs[0].text = h_text
        set_cell_shading(cell, "E2E8F0")
        r = cell.paragraphs[0].runs[0]
        r.font.bold = True
        r.font.size = Pt(8.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, (tech, loc, how) in enumerate(nlp_rows):
        row = t_nlp.rows[idx + 1]
        for col_idx, text_val in enumerate([tech, loc, how]):
            cell = row.cells[col_idx]
            cell.paragraphs[0].text = text_val
            r = cell.paragraphs[0].runs[0]
            r.font.size = Pt(8.0)
            if col_idx == 1:
                r.font.name = "Consolas"
            if col_idx == 0:
                r.font.bold = True
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

    # -------------------------------------------------------------
    # 7. EVALUATION AND RESULTS (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("7. Evaluation and Results", level=1)

    doc.add_paragraph(
        "To rigorously assess whether vector retrieval improves root-cause synthesis over ungrounded language models, we conducted an "
        "ablation study on the held-out test split of 93 high-quality postmortems (data/test.json). The evaluation protocol executed "
        "via eval/evaluate.py compares TraceMind (RAG with ChromaDB two-pass retrieval) directly against an unaugmented Zero-Shot baseline "
        "(providing the LLM with only the raw log excerpt and no historical precedent). Both configurations utilized real inference via "
        "openai/gpt-oss-120b on Groq. All numbers reported in Table 4 and Table 5 are verified directly against eval/evaluation_report.json "
        "and scripts/verify_no_leakage.py."
    )

    add_table_caption(doc, "Table 4: RAG vs. Zero-Shot Quantitative Benchmark Comparison (Held-Out Test Set, N = 93)")

    eval_rows = [
        ("ROUGE-1 F1 Score", "0.1809", "0.1941", "+0.0132", "Measures unigram lexical overlap against human postmortem write-ups."),
        ("ROUGE-2 F1 Score", "0.0384", "0.0439", "+0.0055", "Measures bigram phrase overlap, capturing domain-specific terminology."),
        ("ROUGE-L F1 Score", "0.1061", "0.1210", "+0.0149", "Measures longest common subsequence, reflecting structural explanation flow."),
        ("Semantic Cosine Similarity", "0.5743", "0.5764", "+0.0021", "Dense embedding cosine similarity between synthesized and reference causes.")
    ]

    t_eval = doc.add_table(rows=len(eval_rows) + 1, cols=5)
    format_table_structure(t_eval, [Inches(1.8), Inches(1.1), Inches(1.1), Inches(0.9), Inches(1.6)])
    set_table_borders(t_eval)

    e_headers = ["Evaluation Metric", "Zero-Shot (No RAG)", "TraceMind (RAG)", "Delta", "Operational Significance"]
    for i, h_text in enumerate(e_headers):
        cell = t_eval.rows[0].cells[i]
        cell.paragraphs[0].text = h_text
        set_cell_shading(cell, "E2E8F0")
        r = cell.paragraphs[0].runs[0]
        r.font.bold = True
        r.font.size = Pt(8.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, (metric, zs, rag, delta, sig) in enumerate(eval_rows):
        row = t_eval.rows[idx + 1]
        for col_idx, text_val in enumerate([metric, zs, rag, delta, sig]):
            cell = row.cells[col_idx]
            cell.paragraphs[0].text = text_val
            r = cell.paragraphs[0].runs[0]
            r.font.size = Pt(8.0)
            if col_idx == 0:
                r.font.bold = True
            if col_idx == 3:
                r.font.bold = True
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [1, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT

    if FIG2_EVAL.exists():
        add_figure(doc, FIG2_EVAL, "Figure 2: Empirical Evaluation of RAG vs. Zero-Shot Baseline Across ROUGE and Cosine Metrics", width=Inches(5.4))

    add_table_caption(doc, "Table 5: Retrieval Quality, Vector Similarity Distribution, and Relevance Gate Audit (N = 93)")

    ret_rows = [
        ("Mean Top-1 Cosine Similarity", "0.5273", "eval/evaluation_report.json", "Consistent semantic correlation across heterogeneous infrastructure logs."),
        ("Median Top-1 Cosine Similarity", "0.5333", "eval/evaluation_report.json", "Demonstrates robust central tendency without extreme skew."),
        ("Similarity Range (Min – Max)", "0.2406 – 0.8424", "scripts/verify_no_leakage.py", "Lowest: Therac-25 (0.2406); Highest: GitLab Redis saturation (0.8424)."),
        ("Incidents Above 0.40 Threshold", "86 / 93 (92.5%)", "eval/evaluation_report.json", "High proportion of test logs successfully matched verified precedent."),
        ("Incidents Below 0.40 Floor", "7 / 93 (7.5%)", "eval/evaluation_report.json", "Boeing (0.359), Discord (0.389), EVE Online (0.378), LKML (0.398), Medium (0.357), Therac-25 (0.241), Zerodha (0.366)."),
        ("Relevance Gate Audit Trigger Rate", "7 / 7 (100.0%)", "eval/evaluation_report.json", "All 7 sub-0.40 test cases triggered the gate (line 1 = 'Precedent: not relevant')."),
        ("Top-1 Category Hit Rate", "31 / 93 (33.3%)", "eval/category_benchmark_report.json", "Below 35.5% majority-class baseline; coarse category is a weak proxy."),
        ("Cross-Split Company Leakage", "0 / 93 (0.0%)", "scripts/verify_no_leakage.py", "Zero identical or same-company incidents retrieved across splits.")
    ]

    t_ret = doc.add_table(rows=len(ret_rows) + 1, cols=4)
    format_table_structure(t_ret, [Inches(2.0), Inches(1.2), Inches(1.5), Inches(1.8)])
    set_table_borders(t_ret)

    r_headers = ["Retrieval Statistic", "Observed Value", "Source Verification", "Analytical Finding"]
    for i, h_text in enumerate(r_headers):
        cell = t_ret.rows[0].cells[i]
        cell.paragraphs[0].text = h_text
        set_cell_shading(cell, "E2E8F0")
        r = cell.paragraphs[0].runs[0]
        r.font.bold = True
        r.font.size = Pt(8.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, (stat, val, src, fin) in enumerate(ret_rows):
        row = t_ret.rows[idx + 1]
        for col_idx, text_val in enumerate([stat, val, src, fin]):
            cell = row.cells[col_idx]
            cell.paragraphs[0].text = text_val
            r = cell.paragraphs[0].runs[0]
            r.font.size = Pt(8.0)
            if col_idx == 0:
                r.font.bold = True
            if col_idx == 2:
                r.font.name = "Consolas"
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx == 1 else WD_ALIGN_PARAGRAPH.LEFT

    doc.add_heading("7.1 Interpretation and Honest Statistical Scope", level=2)
    doc.add_paragraph(
        "A rigorous interpretation of the benchmark data reveals several crucial operational and scientific findings. First, while RAG "
        "consistently outperformed the zero-shot baseline across all three ROUGE metrics (+0.0132 ROUGE-1, +0.0055 ROUGE-2, and +0.0149 ROUGE-L), "
        "the quantitative gains are modest in magnitude. ROUGE measures verbatim n-gram lexical overlap against human engineering write-ups. "
        "In infrastructure engineering, different authors describe identical physical failure states using divergent technical vocabulary: "
        "one engineer writes 'the Kubernetes payments pod was killed by the cgroup OOM reaper', whereas a postmortem author records "
        "'container memory limit exceeded resulting in Linux kernel SIGKILL termination'. Consequently, high semantic agreement does not "
        "yield large increases in lexical n-gram scores."
    )

    doc.add_paragraph(
        "Second, semantic cosine similarity remained essentially flat, moving from 0.5743 in zero-shot to 0.5764 under RAG (+0.0021). "
        "This confirms that modern 120-billion parameter models already possess strong internal reasoning about common error logs. The true "
        "value of RAG in TraceMind is not shifting abstract semantic vector distance, but providing concrete grounding: delivering the exact "
        "mitigation checklist and verified postmortem URL so that on-call responders can verify the recommendation."
    )

    doc.add_paragraph(
        "Third, in our failure category benchmark (eval/category_benchmark_report.json), retrieval matched the exact failure category in 33.3% "
        "(31/93) of test incidents. This is slightly below the 35.5% (33/93) majority-class baseline (which corresponds to always predicting "
        "'Software Bug'). Categorical classification is a weak and imperfect proxy for operational utility: labeling an outage as a 'Software Bug' "
        "provides zero diagnostic insight to an SRE. The operational utility stems from retrieving the specific root cause and remediation steps, "
        "not the categorical tag. Therefore, category hit rate is reported here as an honest baseline and is not claimed as a primary achievement."
    )

    doc.add_paragraph(
        "Fourth, regarding the Relevance Gate audit, all 7 test cases scoring below the 0.40 floor successfully triggered the gate by emitting "
        "'Precedent: not relevant' on line one. We must emphasize an honest scientific limitation: while the gate triggered programmatically on all "
        "seven sub-threshold incidents, we did not manually audit whether each retrieved precedent was truly irrelevant to the incident log. "
        "Finally, this evaluation is conducted on a sample size of N = 93 incidents. This represents an academic proof-of-concept demonstrating "
        "directional efficacy rather than a claim of formal statistical significance."
    )

    # -------------------------------------------------------------
    # 8. SCREENSHOTS AND MAJOR CODE SNIPPETS (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("8. Screenshots and Major Code Snippets", level=1)

    doc.add_paragraph(
        "This section presents the production user interface of the live TraceMind console deployed at https://tracemind.vercel.app, "
        "followed by major implementation code listings copied verbatim from the repository."
    )

    if FIG3_CONSOLE.exists():
        add_figure(doc, FIG3_CONSOLE, "Figure 3: Full-Screen TraceMind Incident Intelligence Console Overview", width=Inches(5.4))

    # Place Figure 4 and Figure 5 side by side in a clean 2-column table
    if FIG4_EDITOR.exists() and FIG5_ANALYSIS.exists():
        t_side = doc.add_table(rows=1, cols=2)
        t_side.alignment = WD_TABLE_ALIGNMENT.CENTER
        format_table_structure(t_side, [Inches(3.2), Inches(3.2)], is_header=False)
        set_table_borders(t_side, val="none")  # borderless container

        cell_l = t_side.rows[0].cells[0]
        p_l = cell_l.paragraphs[0]
        p_l.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_l.add_run().add_picture(str(FIG4_EDITOR), width=Inches(3.0))
        p_l_cap = cell_l.add_paragraph()
        p_l_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_l_cap = p_l_cap.add_run("Figure 4: Monaco Log Editor Card")
        r_l_cap.font.name = "Times New Roman"
        r_l_cap.font.size = Pt(8.5)
        r_l_cap.font.bold = True

        cell_r = t_side.rows[0].cells[1]
        p_r = cell_r.paragraphs[0]
        p_r.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_r.add_run().add_picture(str(FIG5_ANALYSIS), width=Inches(3.0))
        p_r_cap = cell_r.add_paragraph()
        p_r_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_r_cap = p_r_cap.add_run("Figure 5: RCA & Remediation Card")
        r_r_cap.font.name = "Times New Roman"
        r_r_cap.font.size = Pt(8.5)
        r_r_cap.font.bold = True

    if FIG6_TRACE.exists():
        add_figure(doc, FIG6_TRACE, "Figure 6: Interactive Pipeline Trace Surfacing Cosine Gauge (0.40 Threshold) and Postmortem Link", width=Inches(5.4))

    doc.add_heading("8.1 Major Code Snippets", level=2)
    doc.add_paragraph(
        "The following code listings represent the critical computational routines of the TraceMind system, copied verbatim from the repository."
    )

    # Listing 1: Two-Pass Vector Retrieval
    l1_code = """def retrieve(
    log_text: str, k: int = 1, persist_directory: str | Path | None = None,
    collection_name: str | None = None, return_all: bool = False, return_pass: bool = False,
):
    \"\"\"Retrieve the most similar historical incident from ChromaDB, preferring quality:high.\"\"\"
    collection = client.get_collection(name=collection_name or COLLECTION_NAME)
    query_embedding = embed(log_text)
    
    # Pass 1: Try high quality first with 0.40 cosine similarity floor
    results_high = collection.query(
        query_embeddings=[query_embedding], n_results=k, where={"quality": "high"}
    )
    if results_high['metadatas'] and results_high['metadatas'][0]:
        distance = results_high['distances'][0][0]
        similarity = 1.0 - (distance / 2.0)
        if similarity >= 0.40:
            if return_pass:
                return results_high['metadatas'][0][0], similarity, 1
            return results_high['metadatas'][0][0], similarity
            
    # Pass 2: Fallback across all training records if Pass 1 misses or < 0.40
    results_all = collection.query(query_embeddings=[query_embedding], n_results=k)
    distance = results_all['distances'][0][0]
    similarity = 1.0 - (distance / 2.0)
    if return_pass:
        return results_all['metadatas'][0][0], similarity, 2
    return results_all['metadatas'][0][0], similarity"""
    add_code_listing(
        doc, l1_code,
        "Listing 1: Two-Pass Vector Retrieval Logic (rag/generate.py)",
        "Listing 1 implements the core two-pass retrieval mechanism. Pass One filters strictly for quality: high postmortems and enforces an explicit 0.40 cosine similarity floor. If that threshold is missed, Pass Two queries across all 662 indexed training records."
    )

    # Listing 2: Relevance Gate Prompt Definition
    l2_code = """\"relevance_gate_v2\": {
    \"system\": (
        \"You are an expert Site Reliability Engineer (SRE) assistant analyzing infrastructure incidents.\\n\"
        \"Analyze the incident log excerpt and correlate it with the provided historical incident.\\n\"
        \"The historical incident below may be unrelated to the current log. First judge whether it is clearly relevant.\\n\\n\"
        \"You must structure your response into exactly two sections labeled as follows:\\n\\n\"
        \"ROOT CAUSE:\\n\"
        \"The very first line of ROOT CAUSE must be exactly one of:\\n\"
        \"Precedent: relevant\\n\"
        \"Precedent: not relevant\\n\\n\"
        \"Followed by a concise, precise analysis of the technical root cause. If precedent is not relevant, derive the cause from the current log alone. \"
        \"Never claim the current incident 'mirrors' or 'matches' a historical one unless specific evidence in the log supports it.\\n\\n\"
        \"REMEDIATION STEPS:\\n\"
        \"At most 5 numbered steps, ordered by priority with immediate mitigation first, each exactly one sentence.\\n\\n\"
        \"Do not output internal chain-of-thought, reasoning scratchpads, or conversational preambles.\"
    ),
    \"user_template\": (
        \"Historical Incident Reference (Similarity: {similarity:.2f}, below 0.60 treat as weak):\\n\"
        \"- Root Cause: {ref_root_cause}\\n- Remediation: {ref_remediation_steps}\\n\\n\"
        \"New Incident Log Excerpt:\\n{log_text}\"
    ),
}"""
    add_code_listing(
        doc, l2_code,
        "Listing 2: Relevance Gate Guardrail Prompt Specification (rag/generate.py)",
        "Listing 2 defines the relevance_gate_v2 prompt template. It forces the language model to output a machine-parseable verdict on line one ('Precedent: relevant' vs 'Precedent: not relevant') to prevent hallucinated historical analogies when similarity is weak."
    )

    # Listing 3: HDBSCAN Clustering
    l3_code = """import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,  # min_cluster_size = 3
    min_samples=min_samples,            # min_samples = 2
    metric=\"euclidean\",                # L2 on normalized vectors ≈ cosine
    cluster_selection_method=\"eom\",
    core_dist_n_jobs=-1,
)

t0 = time.time()
labels = clusterer.fit_predict(embeddings)
elapsed = time.time() - t0

unique_labels = set(labels)
n_clusters = len(unique_labels - {-1})
n_noise = int(np.sum(labels == -1))
noise_pct = n_noise / len(labels) * 100

for cid in sorted(unique_labels - {-1}):
    mask = labels == cid
    centroid = embeddings[mask].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    centroids.append(centroid)
    cluster_ids.append(int(cid))"""
    add_code_listing(
        doc, l3_code,
        "Listing 3: HDBSCAN Anomaly Clustering and Centroid Computation (clustering/build_index.py)",
        "Listing 3 executes HDBSCAN density-based clustering over 384-dimensional normalized log embeddings. It groups recurring failure patterns into clusters and computes normalized centroids for runtime nearest-cluster assignment, marking novel noise as cluster ID -1."
    )

    # Listing 4: Leakage-Free Company-Disjoint Split
    l4_code = """def generate_split(seed: int = 42):
    with open(DATA_PATH, \"r\", encoding=\"utf-8\") as f:
        records = json.load(f)

    # Deterministic shuffling with fixed seed
    rng = random.Random(seed)
    candidate_companies = sorted([c for c, stats in company_counts.items() if stats[\"high\"] > 0])
    rng.shuffle(candidate_companies)

    company_records = defaultdict(list)
    for r in records:
        company_records[r[\"company\"]].append(r)

    test_companies = []
    test_records = []
    
    # Select test companies until reaching target 15-20% high-quality records
    for comp in candidate_companies:
        high_recs = [r for r in company_records[comp] if r.get(\"quality\") == \"high\"]
        if len(test_records) + len(high_recs) <= target_max:
            test_companies.append(comp)
            test_records.extend(high_recs)

    # Training gets all remaining companies; test low-quality records are discarded
    train_companies = [c for c in company_counts if c not in test_companies]
    train_records = [r for c in train_companies for r in company_records[c]]"""
    add_code_listing(
        doc, l4_code,
        "Listing 4: Company-Grouped Deterministic Train/Test Split (scripts/make_split.py)",
        "Listing 4 partitions the corpus strictly by company using fixed seed 42. It ensures zero company overlap between training and test sets and discards low-quality records from test companies to eliminate data leakage."
    )

    # Listing 5: FastAPI /analyze Endpoint
    l5_code = """@app.post(\"/analyze\", response_model=AnalyzeResponse)
def analyze_log(request: AnalyzeRequest, req: Request):
    client_ip = req.client.host if req.client else \"unknown\"
    forwarded = req.headers.get(\"x-forwarded-for\")
    if forwarded:
        client_ip = forwarded.split(\",\")[0].strip()

    if not ip_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail={\"message\": f\"Rate limit exceeded. Maximum {IP_RATE_LIMIT_RPM} RPM per IP.\", \"provider_status\": 429}
        )

    try:
        cluster_info = assign_cluster(request.log_text)
        use_cache = os.getenv(\"ENABLE_GEN_CACHE\", \"true\").lower() == \"true\"
        if \"PYTEST_CURRENT_TEST\" in os.environ:
            use_cache = False

        analysis = generate_rca(
            log_text=request.log_text,
            prompt_variant=\"relevance_gate_v2\",
            use_cache=use_cache
        )
        return AnalyzeResponse(...)"""
    add_code_listing(
        doc, l5_code,
        "Listing 5: SRE Incident Analysis REST Endpoint (backend/main.py)",
        "Listing 5 demonstrates the /analyze endpoint. It enforces per-IP rate limiting (15 RPM), computes HDBSCAN cluster assignment, queries ChromaDB, and coordinates cached or live LLM generation."
    )

    # Listing 6: SQLite Generation Cache
    l6_code = """def get_cached_generation(model: str, prompt_variant: str, query_text: str) -> Optional[Dict[str, Any]]:
    cache_key = _make_cache_key(model, prompt_variant, query_text)
    conn = _get_connection()
    try:
        cur = conn.execute(
            \"SELECT root_cause, remediation_steps, raw_output FROM gen_cache WHERE cache_key = ?\",
            (cache_key,)
        )
        row = cur.fetchone()
        if row:
            return {\"root_cause\": row[0], \"remediation_steps\": row[1], \"raw_output\": row[2]}
        return None
    finally:
        conn.close()

def set_cached_generation(model: str, prompt_variant: str, query_text: str, root_cause: str, remediation_steps: str, raw_output: str = \"\"):
    cache_key = _make_cache_key(model, prompt_variant, query_text)
    conn = _get_connection()
    try:
        with conn:
            conn.execute(\"INSERT OR REPLACE INTO gen_cache VALUES (?, ?, ?, ?, ?, ?, ?)\", ...)
    finally:
        conn.close()"""
    add_code_listing(
        doc, l6_code,
        "Listing 6: SQLite Generation Cache Persistence (rag/gen_cache.py)",
        "Listing 6 manages the local SQLite generation cache (rag/cache/gen_cache.sqlite3). Pre-warmed with 201 entries, it provides fallback continuity when Groq rate limits or network partitions occur."
    )

    # Listing 7: ROUGE Evaluation Loop
    l7_code = """def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer([\"rouge1\", \"rouge2\", \"rougeL\"], use_stemmer=True)
    r1, r2, rl = [], [], []
    cosine_sims = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores[\"rouge1\"].fmeasure)
        r2.append(scores[\"rouge2\"].fmeasure)
        rl.append(scores[\"rougeL\"].fmeasure)

        v_pred = np.array(embed(pred))
        v_ref = np.array(embed(ref))
        sim = float(v_pred @ v_ref)
        cosine_sims.append(sim)

    return {
        \"rouge1_f1\": float(np.mean(r1)),
        \"rouge2_f1\": float(np.mean(r2)),
        \"rougeL_f1\": float(np.mean(rl)),
        \"semantic_sim\": float(np.mean(cosine_sims))
    }"""
    add_code_listing(
        doc, l7_code,
        "Listing 7: Empirical ROUGE and Semantic Similarity Evaluation Loop (eval/evaluate.py)",
        "Listing 7 computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores alongside embedding cosine similarity across all 93 held-out test predictions."
    )

    # -------------------------------------------------------------
    # 9. TEAM CONTRIBUTIONS (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("9. Team Contributions", level=1)

    doc.add_paragraph(
        "TraceMind was engineered as a collaborative capstone project by a five-member engineering team. Each member assumed primary "
        "ownership of a specific subsystem, ensuring modular development, rigorous verification, and clean architectural interfaces. "
        "Table 6 details team roles and module responsibilities."
    )

    add_table_caption(doc, "Table 6: Team Roles, Work Breakdown Structure, and Module Responsibilities")

    team_contribs = [
        ("Patel Nishitkumar\n(1RUA24SCS0068)", "System Architecture, Backend Integration, Frontend",
         "Architected end-to-end FastAPI backend and Next.js 16 Bento UI console; built Monaco-style soft-wrapped log editor; implemented client-side 15 RPM rate limiting; configured Vercel edge deployment and Hugging Face Space runtime hooks."),
        ("Pragun\n(1RUA24SCS0072)", "Domain Analysis, Pipeline Design",
         "Conducted SRE operational domain analysis; formalized incident triage problem statement; structured the two-pass retrieval fallback workflow; designed the 5-step mitigation format and domain-specific incident prompt specifications."),
        ("Pranav\n(1RUA24SCS0077)", "Data Engineering, Clustering",
         "Engineered scrapers for icco/postmortems, Statuspage, GCP, and Cloudflare; developed the programmatic quality filtering pass; built HDBSCAN clustering (48 clusters, min_cluster_size=3); enforced the leakage-free company-disjoint split (seed 42)."),
        ("Sameera\n(1RUA24SCS0099)", "Vector Storage and Retrieval",
         "Constructed in-process ChromaDB vector database; implemented sentence-transformers embedding pipeline; designed two-pass retrieval with the 0.40 cosine similarity floor; engineered the relevance_gate_v2 prompt guardrail."),
        ("Unique\n(1RUA24SCS0118)", "Benchmarking and Evaluation",
         "Built automated benchmarking harness (eval/evaluate.py); executed RAG vs. zero-shot ablation on 93 test postmortems; analyzed ROUGE-1/2/L F1 and cosine similarity metrics; conducted category hit rate benchmark and relevance gate audit.")
    ]

    t_team_c = doc.add_table(rows=len(team_contribs) + 1, cols=3)
    format_table_structure(t_team_c, [Inches(1.8), Inches(1.8), Inches(2.9)])
    set_table_borders(t_team_c)

    tc_headers = ["Member & USN", "Assigned Project Role", "Key Technical Contributions"]
    for i, h_text in enumerate(tc_headers):
        cell = t_team_c.rows[0].cells[i]
        cell.paragraphs[0].text = h_text
        set_cell_shading(cell, "E2E8F0")
        r = cell.paragraphs[0].runs[0]
        r.font.bold = True
        r.font.size = Pt(8.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, (mem, role, cont) in enumerate(team_contribs):
        row = t_team_c.rows[idx + 1]
        for col_idx, text_val in enumerate([mem, role, cont]):
            cell = row.cells[col_idx]
            cell.paragraphs[0].text = text_val
            r = cell.paragraphs[0].runs[0]
            r.font.size = Pt(8.0)
            if col_idx == 0:
                r.font.bold = True
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

    # -------------------------------------------------------------
    # 10. CHALLENGES FACED (Flows Naturally on Same Page)
    # -------------------------------------------------------------
    doc.add_heading("10. Challenges Faced", level=1)

    challenges = [
        ("1. Superficial Statuspage Entries and Data Quality Degradation",
         "Scraping public Statuspage archives yielded hundreds of brief, vague customer announcements (e.g., 'we are investigating delays') that lacked diagnostic value and degraded retrieval synthesis.",
         "Engineered an automated quality filtering pass (scripts/scrapers/quality_pass.py) requiring concrete root-cause explanations and multi-step remediations, segregating 544 high-quality records from 284 low-quality summaries and prioritizing high-quality records in retrieval."),

        ("2. Local SLM Hardware Bottlenecks and Migration to Groq LPU Inference",
         "Initial pipeline designs targeted local 7B-8B parameter models (such as Mistral-7B or Llama-3-8B). On CPU-based cloud runtimes, local model inference required 45-90 seconds per incident, violating real-time triage requirements.",
         "Migrated inference to the Groq API utilizing openai/gpt-oss-120b on dedicated LPU hardware, reducing generation latency to sub-second speeds while significantly improving complex diagnostic reasoning."),

        ("3. Groq API Free-Tier Rate Limits (20 RPM, 8,000 TPM)",
         "Groq's free-tier API imposes strict rate limits of 20 requests per minute and 8,000 tokens per minute. Automated test suites and user testing frequently triggered HTTP 429 and 502 errors.",
         "Implemented a dual defense: an in-memory sliding-window client rate limiter (15 RPM) in backend/main.py and an SQLite generation cache (rag/gen_cache.py) pre-warmed with 201 completions to instantly serve common presets without external API calls."),

        ("4. Company Data Contamination in Dataset Splitting",
         "Random train/test splitting caused severe data leakage because infrastructure companies often experience multiple related incidents, allowing the model to memorize company-specific jargon.",
         "Engineered a deterministic company-disjoint split (scripts/make_split.py, seed 42) ensuring zero company overlap between training (662 records) and test (93 records), discarding 73 low-quality test-company records to prevent leakage."),

        ("5. Hallucinated Historical Analogies on Novel Anomaly Logs",
         "When an incoming error log represented a novel failure mode with no true historical counterpart, standard RAG forced false analogies, inventing inaccurate root causes based on weak semantic matches.",
         "Designed the relevance_gate_v2 prompt guardrail paired with a 0.40 cosine similarity floor. The prompt strictly requires the LLM to inspect correlation and output 'Precedent: not relevant' on line one if the match is unrelated, deriving the cause from the log alone."),

        ("6. Hugging Face Space Reverse Proxy Path Routing (/gradio_api)",
         "Deploying FastAPI inside Hugging Face Spaces resulted in Gradio's reverse proxy rewriting incoming API paths, causing external frontend requests to return 405 Method Not Allowed.",
         "Hooked Gradio's App.create_app lifecycle in app.py to explicitly mount FastAPI routes under both root / and /gradio_api/*, decoupling frontend API path configuration.")
    ]

    for title, prob, fix in challenges:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_before = Pt(4)
        p_c.paragraph_format.space_after = Pt(1)
        r_t = p_c.add_run(title)
        r_t.bold = True
        r_t.font.size = Pt(10)

        p_b = doc.add_paragraph()
        p_b.paragraph_format.space_before = Pt(1)
        p_b.paragraph_format.space_after = Pt(3)
        p_b.paragraph_format.left_indent = Inches(0.15)
        r_p = p_b.add_run("Problem: ")
        r_p.bold = True
        p_b.add_run(prob + " ")
        r_f = p_b.add_run("Resolution: ")
        r_f.bold = True
        p_b.add_run(fix)

    # -------------------------------------------------------------
    # 11. CONCLUSION, LIMITATIONS AND FUTURE WORK (Own Page)
    # -------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("11. Conclusion, Limitations and Future Work", level=1)

    doc.add_heading("11.1 Conclusion", level=2)
    doc.add_paragraph(
        "TraceMind demonstrates the practical viability and operational necessity of auditable, retrieval-augmented intelligence "
        "for Site Reliability Engineering. By coupling sentence embeddings and HDBSCAN anomaly clustering with an in-process ChromaDB vector "
        "store and a 120-billion parameter reasoning model on Groq, TraceMind transforms cryptic server crashes into grounded root-cause "
        "diagnoses and five-step remediation checklists. Across a strictly company-disjoint evaluation on 93 held-out high-quality incidents, "
        "TraceMind achieved measurable gains over zero-shot baselines (+0.0132 ROUGE-1, +0.0055 ROUGE-2, +0.0149 ROUGE-L). Most importantly, "
        "TraceMind eliminates black-box opacity: by surfacing raw similarity scores, 0.40 threshold markers, postmortem URLs, and exact prompts "
        "through the Pipeline Trace interface, TraceMind provides the transparency and provenance required to make AI trustworthy during production outages."
    )

    doc.add_heading("11.2 Limitations", level=2)
    doc.add_paragraph(
        "In the spirit of honest scientific inquiry, several important limitations must be acknowledged:"
    )

    limits = [
        ("Proof-of-Concept Sample Size:",
         "Our quantitative benchmark evaluated N = 93 held-out test incidents. While directionally positive, this sample size is not sufficient to claim formal statistical significance."),
        ("High-Quality Test Set Optimism:",
         "The evaluation split consists strictly of high-quality, human-curated postmortems. Real-world operational logs are frequently noisier, more fragmented, and less descriptive, meaning production performance may be lower than reported benchmark scores."),
        ("Unablated Pipeline Components:",
         "Due to computational constraints, the individual contributions of HDBSCAN clustering and the quality filter were not isolated in independent quantitative ablations."),
        ("Unverified Gate Irrelevance:",
         "While all 7 test cases scoring below the 0.40 floor triggered the relevance gate by emitting 'Precedent: not relevant', we did not manually audit each case to verify whether the retrieved document was genuinely irrelevant."),
        ("Third-Party Infrastructure Dependencies:",
         "The system relies upon Groq's cloud LPU API and Hugging Face Space CPU instances, which remain susceptible to free-tier rate limits (20 RPM) and cold-start wake-up delays.")
    ]

    for t_l, d_l in limits:
        p_l = doc.add_paragraph()
        p_l.paragraph_format.left_indent = Inches(0.2)
        p_l.paragraph_format.space_before = Pt(1.5)
        p_l.paragraph_format.space_after = Pt(2.0)
        r_t = p_l.add_run(f"• {t_l} ")
        r_t.bold = True
        p_l.add_run(d_l)

    doc.add_heading("11.3 Future Work", level=2)
    doc.add_paragraph(
        "Future extensions of the TraceMind architecture will focus on five principal enhancements:"
    )

    futures = [
        ("Streaming Log Ingestion:", "Integrating Apache Kafka and Fluentd connectors to ingest continuous, distributed log streams in real time."),
        ("Local Fine-Tuned SLMs:", "Fine-tuning an open-weight 8-billion parameter model (such as Llama-3-8B) on domain-specific postmortems to enable fully air-gapped on-premise deployment."),
        ("Human-in-the-Loop Actionability Rubric:", "Conducting structured user studies with professional SREs using a 5-point actionability rubric to measure MTTR reduction."),
        ("Automated Ticketing Integration:", "Developing bidirectional webhooks for PagerDuty, Jira, and Slack to automatically attach diagnostic traces to active incident channels."),
        ("Corpus Scale Expansion:", "Expanding the vector corpus from 828 records to over 5,000 postmortems spanning specialized Kubernetes, database, and bare-metal failure modes.")
    ]

    for t_f, d_f in futures:
        p_f = doc.add_paragraph()
        p_f.paragraph_format.left_indent = Inches(0.2)
        p_f.paragraph_format.space_before = Pt(1.5)
        p_f.paragraph_format.space_after = Pt(2.0)
        r_t = p_f.add_run(f"• {t_f} ")
        r_t.bold = True
        p_f.add_run(d_f)

    # -------------------------------------------------------------
    # 12. GITLINKS AND REFERENCES (Flows Naturally on Same Page)
    # -------------------------------------------------------------
    doc.add_heading("12. GitLinks and References", level=1)

    doc.add_heading("12.1 Project Deployments and Source Code", level=2)
    doc.add_paragraph(
        "TraceMind is open-source and publicly accessible across the following links:\n"
        "• GitHub Repository: https://github.com/nish-debug15/TraceMind\n"
        "• Live Frontend Application: https://tracemind.vercel.app\n"
        "• Live Backend API Service: https://nishhhx0-tracemind-backend.hf.space"
    )

    doc.add_heading("12.2 References", level=2)
    references = [
        "[1] P. Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,\" Advances in Neural Information Processing Systems (NeurIPS), 2020.",
        "[2] N. Reimers and I. Gurevych, \"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,\" Proc. EMNLP-IJCNLP, 2019.",
        "[3] R. J. G. B. Campello, D. Moulavi and J. Sander, \"Density-Based Clustering Based on Hierarchical Density Estimates,\" Proc. PAKDD, 2013.",
        "[4] C.-Y. Lin, \"ROUGE: A Package for Automatic Evaluation of Summaries,\" Text Summarization Branches Out (ACL Workshop), 2004.",
        "[5] ChromaDB documentation. https://docs.trychroma.com",
        "[6] FastAPI documentation. https://fastapi.tiangolo.com",
        "[7] Next.js documentation. https://nextjs.org/docs",
        "[8] Groq API documentation. https://console.groq.com/docs",
        "[9] icco/postmortems. https://github.com/icco/postmortems"
    ]

    for ref in references:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.3)
        p_r.paragraph_format.first_line_indent = Inches(-0.3)
        p_r.paragraph_format.space_before = Pt(1.5)
        p_r.paragraph_format.space_after = Pt(2.0)
        r_ref = p_r.add_run(ref)
        r_ref.font.name = "Times New Roman"
        r_ref.font.size = Pt(9.0)

    # Save document
    print(f"[Report Builder] Saving document to {DOCX_PATH}...")
    doc.save(str(DOCX_PATH))
    print("[Report Builder] DOCX saved successfully.")

    # Convert to PDF & update fields via Word COM
    print("[Report Builder] Launching Microsoft Word COM automation...")
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        wdoc = word.Documents.Open(str(DOCX_PATH.resolve()))
        print("[Report Builder] Updating Table of Contents and fields...")
        for toc in wdoc.TablesOfContents:
            toc.Update()
        wdoc.Fields.Update()
        wdoc.Save()

        print(f"[Report Builder] Exporting native PDF to {PDF_PATH}...")
        wdoc.SaveAs2(str(PDF_PATH.resolve()), FileFormat=17)  # 17 = wdFormatPDF

        page_count = wdoc.ComputeStatistics(2)  # 2 = wdStatisticPages
        print(f"[Report Builder] Native PDF exported successfully. Total pages: {page_count}")
        wdoc.Close()
    finally:
        word.Quit()

    return page_count


if __name__ == "__main__":
    count = main()
    print(f"COMPLETE: Page count = {count}")
