"""
scripts/build_tracemind_report.py
---------------------------------
Builds the TraceMind CIE-3 Project Report by editing SlackBoard_Report.docx in place.
Strictly adheres to all CIE-3 rules:
- BRIEF, plain, no extra detail. Target 13-16 pages.
- Cover page: exact text swaps keeping all formatting, logo, and styles.
- Body: Times New Roman 11pt justified, centered bold 18pt section titles.
- Captions: Table above, Figure below, Listing above.
- Zero parentheses or brackets in body prose outside references, tables, and code.
- Zero file paths in body prose outside Table 3 and listing captions.
- Zero in-text [n] citations.
- 4 verbatim code listings (max 20 lines).
- 3 UI screenshots without chrome.
- Exports DOCX and PDF, renders all pages to PNG for visual QA.
"""

import os
import re
import copy
import shutil
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import win32com.client
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCX = PROJECT_ROOT / "SlackBoard_Report.docx"
OUTPUT_DOCX = PROJECT_ROOT / "docs" / "TraceMind_Report.docx"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "TraceMind_Report.pdf"
QA_DIR = PROJECT_ROOT / "docs" / "qa_pages"

# Assets
FIG1_PATH = PROJECT_ROOT / "docs" / "fig1_architecture.png"
FIG2_PATH = PROJECT_ROOT / "docs" / "fig2_evaluation.png"
FIG3_PATH = PROJECT_ROOT / "docs" / "fig4_input_editor.png"
FIG4_PATH = PROJECT_ROOT / "docs" / "fig5_analysis_remediation.png"
FIG5_PATH = PROJECT_ROOT / "docs" / "pipeline_trace_demo.png"


def set_cell_margins(cell, top=45, bottom=45, left=70, right=70):
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
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def set_table_borders(table, color="CBD5E1", sz="4"):
    """Set clean horizontal borders on table."""
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement("w:tblBorders")
    for border_name in ["top", "bottom", "insideH"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), sz)
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), color)
        tblBorders.append(border)
    for border_name in ["left", "right", "insideV"]:
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), "none")
        tblBorders.append(border)
    tblPr.append(tblBorders)


def format_table(table, col_widths, top_pad=45, bot_pad=45):
    """Format table structure, cantSplit, tblHeader, cell margins."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    if len(table.rows) > 0:
        trPr0 = table.rows[0]._tr.get_or_add_trPr()
        tblHeader = OxmlElement("w:tblHeader")
        trPr0.append(tblHeader)

    for row_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        cantSplit = OxmlElement("w:cantSplit")
        trPr.append(cantSplit)
        for i, cell in enumerate(row.cells):
            cell.width = col_widths[i]
            set_cell_margins(cell, top=top_pad, bottom=bot_pad, left=70, right=70)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_idx == 0:
                set_cell_shading(cell, "F1F5F9")


def add_section_heading(doc, title_text, page_break_before=False):
    """Add centered bold 18pt section title."""
    p = doc.add_paragraph()
    if page_break_before:
        p.paragraph_format.page_break_before = True
        p.paragraph_format.space_before = Pt(0)
    else:
        p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title_text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(18)
    run.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_body_p(doc, text, bold_prefix=None, space_after=4):
    """Add justified body paragraph in Times New Roman 11pt."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(11)
        r_pre.bold = True
        r_pre.font.color.rgb = RGBColor(0, 0, 0)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_bullet_p(doc, text, space_after=2.5):
    """Add a clean one-sentence bullet point."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    r_sym = p.add_run("•  ")
    r_sym.font.name = "Times New Roman"
    r_sym.font.size = Pt(11)
    r_sym.font.color.rgb = RGBColor(0, 0, 0)
    r_txt = p.add_run(text)
    r_txt.font.name = "Times New Roman"
    r_txt.font.size = Pt(11)
    r_txt.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_table_caption(doc, caption_text, space_before=8):
    """Caption above table: Table N: Description."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.keep_with_next = True
    parts = caption_text.split(":", 1)
    r1 = p.add_run(parts[0] + ":")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(10)
    r1.bold = True
    r1.font.color.rgb = RGBColor(0, 0, 0)
    if len(parts) > 1:
        r2 = p.add_run(parts[1])
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(10)
        r2.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_figure_caption(doc, caption_text):
    """Caption below figure: Figure N: Description."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    parts = caption_text.split(":", 1)
    r1 = p.add_run(parts[0] + ":")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(10)
    r1.bold = True
    r1.font.color.rgb = RGBColor(0, 0, 0)
    if len(parts) > 1:
        r2 = p.add_run(parts[1])
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(10)
        r2.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_listing_caption(doc, caption_text):
    """Caption above listing: Listing N: Description."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.keep_with_next = True
    parts = caption_text.split(":", 1)
    r1 = p.add_run(parts[0] + ":")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(10)
    r1.bold = True
    r1.font.color.rgb = RGBColor(0, 0, 0)
    if len(parts) > 1:
        r2 = p.add_run(parts[1])
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(10)
        r2.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_code_listing(doc, code_str, caption_text, explanation_text):
    """Add bounded code listing box with caption above and explanation below."""
    add_listing_caption(doc, caption_text)

    # Single-cell table for shaded bounded code box
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
    set_cell_shading(cell, "F8FAFC")

    # Border
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for b_name in ["top", "left", "bottom", "right"]:
        b = OxmlElement(f"w:{b_name}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "CBD5E1")
        tcBorders.append(b)
    tcPr.append(tcBorders)

    # cantSplit
    trPr = table.rows[0]._tr.get_or_add_trPr()
    cantSplit = OxmlElement("w:cantSplit")
    trPr.append(cantSplit)

    cp = cell.paragraphs[0]
    cp.paragraph_format.space_before = Pt(0)
    cp.paragraph_format.space_after = Pt(0)
    cp.paragraph_format.line_spacing = 1.0
    run = cp.add_run(code_str.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(15, 23, 42)

    # Explanation note below
    add_body_p(doc, explanation_text, space_after=5)


def add_figure_image(doc, img_path, caption_text, width=Inches(4.8), page_break_before=False):
    """Add centered figure image with caption below."""
    p = doc.add_paragraph()
    if page_break_before:
        p.paragraph_format.page_break_before = True
        p.paragraph_format.space_before = Pt(0)
    else:
        p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(img_path), width=width)
    add_figure_caption(doc, caption_text)


def build_report():
    print("[Report Builder] Step 1: Copying sample docx to docs/TraceMind_Report.docx...")
    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE_DOCX, OUTPUT_DOCX)

    doc = docx.Document(str(OUTPUT_DOCX))

    print("[Report Builder] Step 2: Editing Cover Page text runs in place...")
    # Project title: P6
    p6 = doc.paragraphs[6]
    for r in p6.runs[1:]:
        r.text = ""
    p6.runs[0].text = "TraceMind: RAG-Powered Root Cause Analysis for SRE Logs\n"

    # P11: 'Submitted By\n'
    doc.paragraphs[11].runs[0].text = "Submitted By\n"
    for r in doc.paragraphs[11].runs[1:]:
        r.text = ""

    # Name/USN: P12 (trimmed trailing newlines so cover fits perfectly on Page 1)
    p12 = doc.paragraphs[12]
    names_text = (
        "Patel Nishitkumar, USN: 1RUA24SCS0068\n"
        "Pragun, USN: 1RUA24SCS0072\n"
        "Pranav, USN: 1RUA24SCS0077\n"
        "Sameera, USN: 1RUA24SCS0099\n"
        "Unique, USN: 1RUA24SCS0118"
    )
    for r in p12.runs[1:]:
        r.text = ""
    p12.runs[0].text = names_text

    # Guide: P15
    p15 = doc.paragraphs[15]
    for r in p15.runs[1:]:
        r.text = ""
    p15.runs[0].text = "Prof. Shanthala T N"

    # Designation: P16
    doc.paragraphs[16].text = "Professor"

    # Section 0 Footer text -> TraceMind
    footer0 = doc.sections[0].footer.paragraphs[0]
    footer0.text = "TraceMind"
    footer0.runs[0].font.name = "Times New Roman"
    footer0.runs[0].font.size = Pt(11)

    print("[Report Builder] Step 3: Deleting sample content after cover page...")
    # Delete sample tables
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    # Delete sample paragraphs after P24
    for p in list(doc.paragraphs[25:]):
        p._element.getparent().remove(p._element)

    print(f"[Report Builder] Cover preserved. Remaining paragraphs: {len(doc.paragraphs)}")

    print("[Report Builder] Step 4: Configuring section break on P24 for safe body margins...")
    # Attach sectPr to P24 so that Section 0 covers P0-P24 and Section 1 begins right after P24
    p24 = doc.paragraphs[24]
    pPr = p24._p.get_or_add_pPr()
    body_sectPr = doc._element.body.xpath("./w:sectPr")[0]
    pPr.append(copy.deepcopy(body_sectPr))

    # Save and reload so python-docx recognizes 2 sections
    doc.save(str(OUTPUT_DOCX))
    doc = docx.Document(str(OUTPUT_DOCX))

    body_sec = doc.sections[1]
    body_sec.top_margin = Pt(54)
    body_sec.bottom_margin = Pt(54)
    body_sec.left_margin = Pt(54)
    body_sec.right_margin = Pt(54)

    # Unlink footer and set to TraceMind
    body_sec.footer.is_linked_to_previous = False
    f_p = body_sec.footer.paragraphs[0]
    f_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    f_p.text = "TraceMind"
    f_p.runs[0].font.name = "Times New Roman"
    f_p.runs[0].font.size = Pt(10)
    f_p.runs[0].font.color.rgb = RGBColor(100, 116, 139)

    # -----------------------------------------------------------------------
    # PAGE 2: ABSTRACT (Starts on its own page)
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 2: ABSTRACT...")
    add_section_heading(doc, "ABSTRACT")

    add_body_p(
        doc,
        "Modern cloud applications produce high volumes of infrastructure logs during outages. "
        "Site reliability engineers must quickly identify root causes and mitigation steps under severe time pressure. "
        "Manual diagnosis across distributed microservices is slow and error-prone because engineers cannot readily recall similar historical failures. "
        "The objective of this project is to build TraceMind, an automated diagnostic system that applies natural language processing to correlate incoming error logs with past postmortems.",
        space_after=6
    )

    add_body_p(
        doc,
        "The pipeline uses sentence embeddings, density-based clustering for deduplication, and vector similarity search across historical incidents. "
        "An incoming log is matched against verified postmortems, and a relevance gate checks semantic alignment before prompting a large language model. "
        "On a company-disjoint test set of 93 incidents, Retrieval-Augmented Generation, or RAG, achieves a ROUGE-1 score of 0.1941, a ROUGE-2 score of 0.0439, and a ROUGE-L score of 0.1210 compared to 0.1809, 0.0384, and 0.1061 for zero-shot generation.",
        space_after=6
    )

    add_body_p(
        doc,
        "The mean top-1 retrieval similarity across test incidents is 0.5273, with 86 incidents exceeding the 0.40 quality floor. "
        "While retrieval grounding yields modest improvements in lexical recall, semantic similarity between generated and reference summaries remains flat at 0.5764 versus 0.5743. "
        "The system serves as an operational decision aid rather than an autonomous triage agent, providing engineers with structured root causes and remediation recommendations.",
        space_after=6
    )

    # -----------------------------------------------------------------------
    # PAGE 3: TABLE OF CONTENTS (Starts on its own page)
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 3: TABLE OF CONTENTS...")
    add_section_heading(doc, "TABLE OF CONTENTS", page_break_before=True)

    toc_titles = [
        "1. Introduction",
        "2. Problem Statement and Objectives",
        "3. Technologies Used",
        "4. Dataset",
        "5. System Architecture",
        "6. NLP Techniques Implemented",
        "7. Evaluation and Results",
        "8. Screenshots and Code",
        "9. Team Contributions",
        "10. Challenges Faced",
        "11. Conclusion",
        "12. GitLinks and References",
    ]
    for item in toc_titles:
        p_toc = doc.add_paragraph()
        p_toc.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_toc.paragraph_format.space_before = Pt(0)
        p_toc.paragraph_format.space_after = Pt(4)
        p_toc.paragraph_format.line_spacing = 1.15
        r_toc = p_toc.add_run(item)
        r_toc.font.name = "Times New Roman"
        r_toc.font.size = Pt(11)
        r_toc.font.color.rgb = RGBColor(0, 0, 0)

    # -----------------------------------------------------------------------
    # PAGE 4: SECTION 1 & SECTION 2
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 4: Section 1 & Section 2...")
    add_section_heading(doc, "1. Introduction", page_break_before=True)

    add_body_p(
        doc,
        "Infrastructure incidents in distributed production environments create complex operational challenges for site reliability engineering teams. "
        "When services experience cascading failures, error logs accumulate rapidly across container runtimes, database clusters, and networking interfaces. "
        "Engineers tasked with on-call duty must rapidly filter noisy tracebacks to diagnose underlying faults and restore normal operations. "
        "Most engineering organizations maintain extensive postmortem archives documenting previous outages, but engineers rarely have time to manually search these static archives during an active incident."
    )

    add_body_p(
        doc,
        "TraceMind addresses this operational challenge through an intelligent root cause analysis pipeline. "
        "The platform transforms raw log snippets into dense vector representations, matches them against indexed historical postmortems, and generates structured remediation advice using a large language model. "
        "By augmenting generative synthesis with empirical precedents, the system aims to reduce mean time to resolution and deliver consistent incident summaries. "
        "The platform features an interactive console that provides full pipeline visibility, including similarity scores and precedent relevance status."
    )

    add_section_heading(doc, "2. Problem Statement and Objectives", page_break_before=False)

    add_body_p(
        doc,
        "Site reliability engineers face high cognitive load during production outages, where identifying root causes from voluminous log streams requires substantial domain expertise. "
        "Existing observability tools visualize operational metrics but do not correlate novel error patterns with historical incident documentation. "
        "Without automated contextual retrieval, teams risk repeating diagnostic efforts and applying suboptimal mitigation steps. "
        "TraceMind resolves this limitation by providing an automated retrieval pipeline that connects raw logs to verified historical resolutions."
    )

    objectives = [
        "Construct a curated corpus of infrastructure postmortems categorized by failure mode and filtered for documentation quality.",
        "Generate dense vector representations for incident logs and apply density-based clustering to identify semantic incident groupings and deduplicate corpus entries.",
        "Implement a two-pass vector similarity retrieval mechanism with a relevance floor to fetch historical precedents from ChromaDB.",
        "Integrate a relevance gate prompt with a large language model to synthesize concise root cause explanations and actionable remediation steps.",
        "Evaluate model performance against an unaugmented zero-shot baseline across standard overlap metrics and embedding cosine similarity on a disjoint test set.",
    ]
    for obj in objectives:
        add_bullet_p(doc, obj)

    # -----------------------------------------------------------------------
    # PAGE 5: SECTION 3 & SECTION 4
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 5: Section 3 & Section 4...")
    p_s3 = add_section_heading(doc, "3. Technologies Used", page_break_before=True)
    p_s3.paragraph_format.space_before = Pt(0)
    p_s3.paragraph_format.space_after = Pt(3)

    add_table_caption(doc, "Table 1: System technologies and software versions", space_before=3)
    t1_data = [
        ["Layer", "Technology", "Version", "Purpose"],
        ["Backend API", "FastAPI", ">=0.104.0", "REST API routing and endpoint management"],
        ["Web Server", "Uvicorn", ">=0.24.0", "Asynchronous server execution runtime"],
        ["Data Validation", "Pydantic", ">=2.0", "Request and response schema validation"],
        ["Vector Embeddings", "sentence-transformers", ">=2.2.0", "Dense vector representation for log excerpts"],
        ["Deduplication", "HDBSCAN", ">=0.8.33", "Unsupervised density-based incident clustering"],
        ["Similarity Metrics", "scikit-learn", ">=1.3.0", "Cosine similarity calculation between embeddings"],
        ["Vector Database", "ChromaDB", ">=0.4.0", "Persistent vector storage and precedent search"],
        ["Metric Evaluation", "rouge-score", ">=0.1.2", "Text overlap scoring for evaluation summaries"],
        ["Web Framework", "Next.js", "16.3.5", "Server-rendered frontend interface framework"],
        ["UI Library", "React", "19.2.8", "Interactive component-based client application"],
    ]
    t1 = doc.add_table(rows=len(t1_data), cols=4)
    for r_idx, row in enumerate(t1_data):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(8.5)
            if r_idx == 0:
                r.bold = True
    format_table(t1, [Inches(1.4), Inches(1.5), Inches(1.1), Inches(2.5)], top_pad=35, bot_pad=35)
    set_table_borders(t1)

    p_s4 = add_section_heading(doc, "4. Dataset", page_break_before=False)
    p_s4.paragraph_format.space_before = Pt(8)
    p_s4.paragraph_format.space_after = Pt(3)

    add_body_p(
        doc,
        "The TraceMind evaluation benchmark is built from an incident corpus compiled from industry postmortems and public status pages. "
        "Each incident record contains a raw log excerpt, a verified root cause summary, remediation actions, and a binary documentation quality label. "
        "To prevent information leakage, the data is divided using a company-disjoint split initialized with random seed 42 so that no organization appears in both splits.",
        space_after=2.5
    )

    add_table_caption(doc, "Table 2: Corpus composition and dataset splits", space_before=3)
    t2_data = [
        ["Split", "Records", "Companies", "Quality Breakdown"],
        ["Full corpus", "828", "58", "544 high and 284 low quality"],
        ["Training set", "662", "18", "High and low quality"],
        ["Test set", "93", "40", "All high quality"],
        ["Excluded", "73", "40", "Low-quality test-company records"],
    ]
    t2 = doc.add_table(rows=len(t2_data), cols=4)
    for r_idx, row in enumerate(t2_data):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(8.5)
            if r_idx == 0:
                r.bold = True
    format_table(t2, [Inches(1.4), Inches(1.1), Inches(1.3), Inches(2.7)], top_pad=35, bot_pad=35)
    set_table_borders(t2)

    # -----------------------------------------------------------------------
    # PAGE 6: SECTION 5: SYSTEM ARCHITECTURE
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 6: Section 5: System Architecture...")
    add_section_heading(doc, "5. System Architecture", page_break_before=True)

    add_figure_image(
        doc,
        FIG1_PATH,
        "Figure 1: TraceMind system architecture and end-to-end pipeline flow",
        width=Inches(6.2)
    )

    stages = [
        ("Ingestion and Embedding: ", "Raw infrastructure logs are ingested via the web client and sent to the backend. The sentence transformer generates dense vector embeddings capturing semantic relationships."),
        ("Clustering and Deduplication: ", "HDBSCAN clusters the embeddings during preprocessing to identify recurring failure patterns. This step deduplicates corpus records and prevents redundant precedent indexing."),
        ("Two-Pass Vector Retrieval: ", "ChromaDB searches the index using cosine similarity against historical postmortems. The system prioritizes high-quality records above a similarity floor before evaluating all entries."),
        ("Relevance Gate and Prompt Assembly: ", "The retrieved precedent is combined with the input log into a structured prompt. The relevance gate instructs the language model to reject irrelevant precedents before generating inferences."),
        ("Generation and Presentation: ", "The language model synthesizes root cause explanations and prioritized remediation steps. Results are returned via the FastAPI service and displayed in the Next.js interface."),
    ]
    for prefix, body in stages:
        add_body_p(doc, body, bold_prefix=prefix, space_after=3)

    # -----------------------------------------------------------------------
    # PAGE 7: SECTION 6: NLP TECHNIQUES IMPLEMENTED
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 7: Section 6: NLP Techniques Implemented...")
    add_section_heading(doc, "6. NLP Techniques Implemented", page_break_before=True)

    add_body_p(
        doc,
        "The architecture incorporates multiple natural language processing techniques across embedding generation, offline corpus partitioning, vector similarity retrieval, prompt orchestration, and quantitative evaluation."
    )

    add_table_caption(doc, "Table 3: Natural language processing techniques and implementation mapping")
    t3_data = [
        ["Technique", "Where in the code", "How it is satisfied"],
        ["Sentence embeddings", "clustering/cluster.py", "The system transforms raw log excerpts into dense 384-dimensional vector embeddings using the all-MiniLM-L6-v2 model."],
        ["Density-based clustering", "clustering/build_index.py", "HDBSCAN groups normalized embedding vectors into clusters during offline preprocessing to identify semantic incident groupings and eliminate duplicate postmortems."],
        ["Vector similarity search", "rag/generate.py", "ChromaDB performs cosine similarity search on dense embeddings to identify the most relevant historical incidents matching the query log."],
        ["Two-pass retrieval", "rag/generate.py", "The retrieval pipeline queries high-quality postmortems with a 0.40 cosine similarity floor before falling back to the broader corpus."],
        ["RAG prompting with relevance gate", "rag/generate.py", "The system prompts the language model with retrieved context while explicitly requiring it to declare precedent irrelevance when semantic alignment is weak."],
        ["Evaluation metrics", "eval/evaluate.py", "Automated evaluation computes ROUGE overlap scores and embedding cosine similarity between model outputs and ground truth postmortems."],
    ]
    t3 = doc.add_table(rows=len(t3_data), cols=3)
    for r_idx, row in enumerate(t3_data):
        for c_idx, val in enumerate(row):
            cell = t3.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            if r_idx == 0:
                r.bold = True
    format_table(t3, [Inches(1.8), Inches(1.8), Inches(2.9)], top_pad=55, bot_pad=55)
    set_table_borders(t3)

    # -----------------------------------------------------------------------
    # PAGE 8: SECTION 7: EVALUATION AND RESULTS
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 8: Section 7: Evaluation and Results...")
    add_section_heading(doc, "7. Evaluation and Results", page_break_before=True)

    add_body_p(
        doc,
        "The quantitative evaluation benchmarks TraceMind using Retrieval-Augmented Generation, or RAG, against an unaugmented zero-shot baseline across the 93 held-out test incidents. "
        "Both configurations invoke the gpt-oss-120b model with identical inference parameters. "
        "Ground truth references are drawn from verified human-written postmortems in the test corpus."
    )

    add_table_caption(doc, "Table 4: Evaluation results comparing zero-shot and RAG generation")
    t4_data = [
        ["Metric", "Zero-Shot Baseline", "RAG TraceMind"],
        ["ROUGE-1", "0.1809", "0.1941"],
        ["ROUGE-2", "0.0384", "0.0439"],
        ["ROUGE-L", "0.1061", "0.1210"],
        ["Cosine Similarity", "0.5743", "0.5764"],
    ]
    t4 = doc.add_table(rows=len(t4_data), cols=3)
    for r_idx, row in enumerate(t4_data):
        for c_idx, val in enumerate(row):
            cell = t4.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            if r_idx == 0:
                r.bold = True
    format_table(t4, [Inches(2.5), Inches(2.0), Inches(2.0)], top_pad=45, bot_pad=45)
    set_table_borders(t4)

    add_figure_image(
        doc,
        FIG2_PATH,
        "Figure 2: Performance comparison across ROUGE metrics",
        width=Inches(4.8)
    )

    add_table_caption(doc, "Table 5: Retrieval quality and relevance gate behavior")
    t5_data = [
        ["Metric", "Value"],
        ["Mean top-1 similarity", "0.5273"],
        ["Test incidents above 0.40 floor", "86 of 93"],
        ["Gate triggered on below-floor incidents", "7 of 7"],
    ]
    t5 = doc.add_table(rows=len(t5_data), cols=2)
    for r_idx, row in enumerate(t5_data):
        for c_idx, val in enumerate(row):
            cell = t5.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            if r_idx == 0:
                r.bold = True
    format_table(t5, [Inches(3.8), Inches(2.7)], top_pad=45, bot_pad=45)
    set_table_borders(t5)

    add_body_p(
        doc,
        "The observed improvements in ROUGE metrics are modest, and no formal statistical significance tests were performed to confirm these deltas. "
        "Semantic cosine similarity between generated root causes and reference postmortems remains essentially flat at 0.5764 compared to 0.5743 for the baseline. "
        "The retrieval category hit rate of 33.3 percent falls below the majority class baseline of 35.5 percent, so category-level predictive accuracy is not claimed. "
        "In addition, automated gate firing across the seven low-similarity incidents was not manually verified by human domain experts to establish true irrelevance."
    )

    # -----------------------------------------------------------------------
    # PAGE 9: SECTION 8: SCREENSHOT 1 (EDITOR)
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 9: Screenshot 1 (Editor)...")
    add_section_heading(doc, "8. Screenshots and Code", page_break_before=True)

    add_body_p(
        doc,
        "The TraceMind web interface provides an operational diagnostic console for on-call engineers. "
        "The application supports log input, real-time root cause inference, remediation step display, and transparent pipeline inspection."
    )

    add_figure_image(
        doc,
        FIG3_PATH,
        "Figure 3: Incident log editor interface with pre-loaded SRE log excerpt",
        width=Inches(4.8)
    )

    # -----------------------------------------------------------------------
    # PAGE 10: SECTION 8: SCREENSHOT 2 (ANALYSIS RESULT)
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 10: Screenshot 2 (Analysis Result)...")
    add_figure_image(
        doc,
        FIG4_PATH,
        "Figure 4: Root cause analysis results with confidence gauge and remediation steps",
        width=Inches(4.8),
        page_break_before=True
    )

    # -----------------------------------------------------------------------
    # PAGE 11: SECTION 8: SCREENSHOT 3 (PIPELINE TRACE)
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 11: Screenshot 3 (Pipeline Trace)...")
    add_figure_image(
        doc,
        FIG5_PATH,
        "Figure 5: Pipeline trace panel displaying retrieved postmortem context and prompt construction",
        width=Inches(4.8),
        page_break_before=True
    )

    # -----------------------------------------------------------------------
    # PAGE 12: SECTION 8: CODE LISTINGS 1 & 2
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 12: Code Listings 1 & 2...")
    p_lst_header = doc.add_paragraph()
    p_lst_header.paragraph_format.page_break_before = True
    p_lst_header.paragraph_format.space_before = Pt(0)
    p_lst_header.paragraph_format.space_after = Pt(4)
    p_lst_header.paragraph_format.keep_with_next = True
    r_lh = p_lst_header.add_run("Core Implementation Code Listings")
    r_lh.font.name = "Times New Roman"
    r_lh.font.size = Pt(13)
    r_lh.bold = True
    r_lh.font.color.rgb = RGBColor(0, 0, 0)

    l1_code = """    query_embedding = embed(log_text)
    
    # Try high quality first
    results_high = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"quality": "high"}
    )
    ...
    if results_high['metadatas'] and results_high['metadatas'][0]:
        distance = results_high['distances'][0][0]
        similarity = 1.0 - (distance / 2.0)
        # Similarity floor for high quality
        if similarity >= 0.40:
            if return_pass:
                return results_high['metadatas'][0][0], similarity, 1
            return results_high['metadatas'][0][0], similarity
            
    # Fallback to any quality if no high quality match found or similarity < 0.40
    results_all = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    ...
    return results_all['metadatas'][0][0], similarity, 2"""

    add_code_listing(
        doc,
        l1_code,
        "Listing 1: Two-pass vector retrieval implementation (rag/generate.py)",
        "This function queries high-quality historical incidents above the similarity floor before falling back to the broader corpus."
    )

    l2_code = """    "relevance_gate_v2": {
        "system": (
            "You are an expert Site Reliability Engineer (SRE) assistant analyzing infrastructure incidents.\\n"
            "Analyze the incident log excerpt and correlate it with the provided historical incident.\\n"
            "The historical incident below may be unrelated to the current log. First judge whether it is clearly relevant.\\n\\n"
            "You must structure your response into exactly two sections labeled as follows:\\n\\n"
            "ROOT CAUSE:\\n"
            "The very first line of ROOT CAUSE must be exactly one of:\\n"
            "Precedent: relevant\\n"
            "Precedent: not relevant\\n\\n"
            "Followed by a concise, precise analysis of the technical root cause. If precedent is not relevant, derive the cause from the current log alone. "
            "Never claim the current incident 'mirrors' or 'matches' a historical one unless specific evidence in the log supports it.\\n\\n"
            "REMEDIATION STEPS:\\n"
            "At most 5 numbered steps, ordered by priority with immediate mitigation first, each exactly one sentence.\\n\\n"
            "Do not output internal chain-of-thought, reasoning scratchpads, or conversational preambles."
        ),
        ...
    },"""

    add_code_listing(
        doc,
        l2_code,
        "Listing 2: Relevance gate prompt configuration (rag/generate.py)",
        "The system prompt directs the language model to verify whether the historical reference is relevant before synthesizing root causes."
    )

    # -----------------------------------------------------------------------
    # PAGE 13: SECTION 8: CODE LISTINGS 3 & 4
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 13: Code Listings 3 & 4...")
    p_lst_header2 = doc.add_paragraph()
    p_lst_header2.paragraph_format.page_break_before = True
    p_lst_header2.paragraph_format.space_before = Pt(0)
    p_lst_header2.paragraph_format.space_after = Pt(4)
    p_lst_header2.paragraph_format.keep_with_next = True
    r_lh2 = p_lst_header2.add_run("Clustering and Evaluation Routines")
    r_lh2.font.name = "Times New Roman"
    r_lh2.font.size = Pt(13)
    r_lh2.bold = True
    r_lh2.font.color.rgb = RGBColor(0, 0, 0)

    l3_code = """    print(f"\\n[3/4] Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # L2 on normalized vectors ≈ cosine
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )

    t0 = time.time()
    labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - t0
    print(f"  Clustering completed in {elapsed:.1f}s")

    # ---- Compute statistics ----
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    noise_pct = n_noise / len(labels) * 100"""

    add_code_listing(
        doc,
        l3_code,
        "Listing 3: Density-based incident clustering routine (clustering/build_index.py)",
        "The offline indexing script executes density-based clustering to partition incident embeddings and compute cluster statistics."
    )

    l4_code = """def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    cosine_sims = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

        v_pred = np.array(embed(pred))
        v_ref = np.array(embed(ref))
        sim = float(v_pred @ v_ref)
        cosine_sims.append(sim)

    return {
        "rouge1_f1": float(np.mean(r1)),
        "rouge2_f1": float(np.mean(r2)),
        "rougeL_f1": float(np.mean(rl)),
        "semantic_sim": float(np.mean(cosine_sims))
    }"""

    add_code_listing(
        doc,
        l4_code,
        "Listing 4: Quantitative evaluation metrics loop (eval/evaluate.py)",
        "The evaluation loop computes token-level overlap and vector cosine similarity across model predictions and reference postmortems."
    )

    # -----------------------------------------------------------------------
    # PAGE 14: SECTION 9 & SECTION 10
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 14: Section 9 & Section 10...")
    add_section_heading(doc, "9. Team Contributions", page_break_before=True)

    add_table_caption(doc, "Table 6: Team member roles and contributions")
    t6_data = [
        ["Member", "Role", "Contribution"],
        ["Patel Nishitkumar", "System Architecture, Backend Integration, Frontend", "Implemented the FastAPI service architecture and Next.js user interface."],
        ["Pragun", "Domain Analysis, Pipeline Design", "Designed the end-to-end incident analysis pipeline and workflow specifications."],
        ["Pranav", "Data Engineering, Clustering", "Curated postmortem datasets and implemented offline density-based clustering."],
        ["Sameera", "Vector Storage and Retrieval", "Configured ChromaDB collections and implemented two-pass similarity retrieval."],
        ["Unique", "Benchmarking and Evaluation", "Built the quantitative evaluation harness and calculated benchmark metrics."],
    ]
    t6 = doc.add_table(rows=len(t6_data), cols=3)
    for r_idx, row in enumerate(t6_data):
        for c_idx, val in enumerate(row):
            cell = t6.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            if r_idx == 0:
                r.bold = True
    format_table(t6, [Inches(1.6), Inches(2.2), Inches(2.7)], top_pad=45, bot_pad=45)
    set_table_borders(t6)

    add_section_heading(doc, "10. Challenges Faced", page_break_before=False)

    challenges = [
        ("Low-Quality Status Page Entries",
         "Public status page records often contained brief placeholder messages lacking root cause details. The team implemented an automated quality filter to exclude records with insufficient narrative depth from the primary benchmark."),
        ("Model Migration and Rate Limits",
         "Transitioning from Mistral-7B to gpt-oss-120b on the Groq inference platform introduced strict free-tier request quotas. The pipeline was updated with request throttling and a persistent cache to ensure stable execution."),
        ("Data Leakage Across Splits",
         "Random sampling permitted incidents from the same parent organization to appear in both training and test sets. The split was restructured to partition records along company boundaries so that no organization overlaps."),
        ("False Historical Precedents on Novel Logs",
         "Outlier log inputs triggered high-similarity retrieval against unrelated postmortems, causing hallucinations during generative synthesis. The team implemented a relevance gate prompt instruction forcing the model to explicitly discard weak precedents."),
    ]
    for title, desc in challenges:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_before = Pt(5)
        p_c.paragraph_format.space_after = Pt(1.5)
        p_c.paragraph_format.keep_with_next = True
        r_ct = p_c.add_run(title)
        r_ct.font.name = "Times New Roman"
        r_ct.font.size = Pt(11)
        r_ct.bold = True
        r_ct.font.color.rgb = RGBColor(0, 0, 0)
        add_body_p(doc, desc, space_after=3.5)

    # -----------------------------------------------------------------------
    # PAGE 15: SECTION 11 & SECTION 12
    # -----------------------------------------------------------------------
    print("[Report Builder] Writing Page 15: Section 11 & Section 12...")
    p_c_head = add_section_heading(doc, "11. Conclusion", page_break_before=True)
    p_c_head.paragraph_format.space_before = Pt(0)
    p_c_head.paragraph_format.space_after = Pt(3)

    add_body_p(
        doc,
        "TraceMind demonstrates a practical natural language processing pipeline for site reliability engineering incident assistance. "
        "By integrating sentence embeddings, vector retrieval, and large language model synthesis, the platform provides automated root cause explanations and remediation recommendations from infrastructure logs. "
        "The inclusion of a relevance gate prevents false historical analogies when retrieval similarities fall below acceptable thresholds. "
        "The platform establishes an operational baseline for incident triage while maintaining transparency through interactive pipeline inspection.",
        space_after=2.5
    )

    p_lim_h = doc.add_paragraph()
    p_lim_h.paragraph_format.space_before = Pt(3)
    p_lim_h.paragraph_format.space_after = Pt(1)
    p_lim_h.paragraph_format.keep_with_next = True
    r_lh = p_lim_h.add_run("Limitations")
    r_lh.font.name = "Times New Roman"
    r_lh.font.size = Pt(10.5)
    r_lh.bold = True
    r_lh.font.color.rgb = RGBColor(0, 0, 0)

    limitations = [
        "The quantitative evaluation is conducted on a small benchmark containing 93 test incidents.",
        "The test set is restricted to high-quality records, which makes the reported performance optimistic compared to noisy operational environments.",
        "The isolated performance contributions of offline clustering and quality filtering were not independently ablated.",
        "Deployment on free-tier hosting platforms introduces request latency and cold start delays during live demonstrations.",
    ]
    for lim in limitations:
        add_bullet_p(doc, lim, space_after=1.5)

    p_fw_h = doc.add_paragraph()
    p_fw_h.paragraph_format.space_before = Pt(3)
    p_fw_h.paragraph_format.space_after = Pt(1)
    p_fw_h.paragraph_format.keep_with_next = True
    r_fh = p_fw_h.add_run("Future Work")
    r_fh.font.name = "Times New Roman"
    r_fh.font.size = Pt(10.5)
    r_fh.bold = True
    r_fh.font.color.rgb = RGBColor(0, 0, 0)

    future_work = [
        "Extend the ingestion pipeline to support continuous streaming log processing from distributed message queues.",
        "Train and fine-tune a smaller open-source language model specialized exclusively on infrastructure diagnostic corpora.",
        "Conduct a human subject evaluation study with operational engineers to assess the actionability of generated remediation recommendations.",
    ]
    for fw in future_work:
        add_bullet_p(doc, fw, space_after=1.5)

    p_ref_title = add_section_heading(doc, "12. GitLinks and References", page_break_before=False)
    p_ref_title.paragraph_format.space_before = Pt(5)
    p_ref_title.paragraph_format.space_after = Pt(2.5)

    p_gl_h = doc.add_paragraph()
    p_gl_h.paragraph_format.space_before = Pt(2.5)
    p_gl_h.paragraph_format.space_after = Pt(1)
    p_gl_h.paragraph_format.keep_with_next = True
    r_glh = p_gl_h.add_run("Project Repository Links")
    r_glh.font.name = "Times New Roman"
    r_glh.font.size = Pt(10.5)
    r_glh.bold = True
    r_glh.font.color.rgb = RGBColor(0, 0, 0)

    links = [
        "GitHub repository: https://github.com/nish-debug15/TraceMind",
        "Frontend web application: https://tracemind.vercel.app",
        "Backend API service: https://nishhhx0-tracemind-backend.hf.space",
    ]
    for lnk in links:
        add_bullet_p(doc, lnk, space_after=1.5)

    p_ref_h = doc.add_paragraph()
    p_ref_h.paragraph_format.space_before = Pt(2.5)
    p_ref_h.paragraph_format.space_after = Pt(1)
    p_ref_h.paragraph_format.keep_with_next = True
    r_rfh = p_ref_h.add_run("References")
    r_rfh.font.name = "Times New Roman"
    r_rfh.font.size = Pt(10.5)
    r_rfh.bold = True
    r_rfh.font.color.rgb = RGBColor(0, 0, 0)

    references = [
        '[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Advances in Neural Information Processing Systems (NeurIPS), 2020.',
        '[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," Proc. EMNLP-IJCNLP, 2019.',
        '[3] R. J. G. B. Campello, D. Moulavi and J. Sander, "Density-Based Clustering Based on Hierarchical Density Estimates," Proc. PAKDD, 2013.',
        '[4] C.-Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," Text Summarization Branches Out (ACL Workshop), 2004.',
        "[5] ChromaDB documentation. https://docs.trychroma.com",
        "[6] FastAPI documentation. https://fastapi.tiangolo.com",
        "[7] Next.js documentation. https://nextjs.org/docs",
        "[8] Groq API documentation. https://console.groq.com/docs",
        "[9] icco/postmortems. https://github.com/icco/postmortems",
    ]
    for ref in references:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_r.paragraph_format.left_indent = Inches(0.28)
        p_r.paragraph_format.first_line_indent = Inches(-0.28)
        p_r.paragraph_format.space_before = Pt(0)
        p_r.paragraph_format.space_after = Pt(1.5)
        p_r.paragraph_format.line_spacing = 1.05
        r_ref = p_r.add_run(ref)
        r_ref.font.name = "Times New Roman"
        r_ref.font.size = Pt(9.0)
        r_ref.font.color.rgb = RGBColor(0, 0, 0)

    doc.save(str(OUTPUT_DOCX))
    print(f"[Report Builder] Saved {OUTPUT_DOCX}")


def export_pdf():
    print("[Report Builder] Step 5: Exporting PDF via Microsoft Word COM...")
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(OUTPUT_DOCX.resolve()))
        doc.SaveAs(str(OUTPUT_PDF.resolve()), FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
        print(f"[Report Builder] Saved {OUTPUT_PDF}")
    finally:
        word.Quit()


def lint_document():
    print("[Report Builder] Step 6: Running strict text lint on body prose...")
    doc = docx.Document(str(OUTPUT_DOCX))

    # Lint body paragraphs (excluding cover paragraphs 0-24, references section, tables, listings)
    # References section starts after "References" under "12. GitLinks and References"
    in_references = False
    violations = []

    for idx, p in enumerate(doc.paragraphs[25:], start=26):
        text = p.text.strip()
        if not text:
            continue
        if text == "References":
            in_references = True
            continue

        if in_references:
            # References are allowed [1]-[9]
            continue

        # Check for listing captions (captions start with "Listing N:")
        is_listing_caption = text.startswith("Listing ")
        is_table_caption = text.startswith("Table ")
        is_figure_caption = text.startswith("Figure ")

        if is_listing_caption:
            # File paths and listing names are allowed in Listing captions
            continue

        # In body prose:
        # Check for '(' or ')'
        if "(" in text or ")" in text:
            violations.append(f"P{idx}: Parentheses found in body text -> {repr(text[:80])}")

        # Check for '[' or ']'
        if "[" in text or "]" in text:
            violations.append(f"P{idx}: Brackets found in body text -> {repr(text[:80])}")

        # Check for '.py', 'scripts/', 'rag/', 'eval/'
        for forbidden in [".py", "scripts/", "rag/", "eval/"]:
            if forbidden in text:
                violations.append(f"P{idx}: Forbidden path fragment '{forbidden}' in body text -> {repr(text[:80])}")

        # Check for citation numbers like [1], [2], etc.
        if re.search(r"\[\d+\]", text):
            violations.append(f"P{idx}: In-text citation found -> {repr(text[:80])}")

    # Also lint structured data tables (Tables 1, 2, 4, 5, 6 - Table 3 allows file paths in col 1)
    for t_idx, table in enumerate(doc.tables):
        if len(table.columns) == 1:
            # Code listing box
            continue
        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                cell_text = cell.text.strip()
                # Check for brackets in tables
                if "[" in cell_text or "]" in cell_text:
                    violations.append(f"Table {t_idx+1} R{r_idx}C{c_idx}: Brackets found -> {repr(cell_text[:60])}")
                # Check for in-text citations
                if re.search(r"\[\d+\]", cell_text):
                    violations.append(f"Table {t_idx+1} R{r_idx}C{c_idx}: Citation found -> {repr(cell_text[:60])}")
                # Table 3 col 1 allows file paths
                if t_idx == 2 and c_idx == 1:
                    continue
                for forbidden in [".py", "scripts/", "rag/", "eval/"]:
                    if forbidden in cell_text:
                        violations.append(f"Table {t_idx+1} R{r_idx}C{c_idx}: Forbidden path fragment '{forbidden}' -> {repr(cell_text[:60])}")

    if violations:
        print(f"[Lint FAILED] Found {len(violations)} violations:")
        for v in violations:
            print("  *", v)
    else:
        print("[Lint PASSED] Zero parentheses or brackets in body prose outside references, tables, and listing code.")
        print("[Lint PASSED] Zero file paths outside Table 3 and Listing captions.")
        print("[Lint PASSED] Zero in-text [n] citations.")
    return len(violations)


def render_qa_images():
    print("[Report Builder] Step 7: Rendering all PDF pages to PNG for QA inspection...")
    QA_DIR.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(str(OUTPUT_PDF))
    total_pages = len(pdf)
    print(f"[Report Builder] Total pages in PDF: {total_pages}")

    for i in range(total_pages):
        page = pdf[i]
        pix = page.get_pixmap(dpi=150)
        out_img = QA_DIR / f"page_{i+1:02d}.png"
        pix.save(str(out_img))
    pdf.close()
    print(f"[Report Builder] Rendered {total_pages} page images in {QA_DIR}")
    return total_pages


if __name__ == "__main__":
    build_report()
    export_pdf()
    violations = lint_document()
    total_pages = render_qa_images()
    print(f"\n[COMPLETE] Report built: {total_pages} pages, {violations} lint violations.")
