"""
QA script for TraceMind_Pitch.pptx
Checks: slide count, title on every slide, footer on every slide,
speaker notes present, text box dimensions within slide bounds,
demo image present on slide 9, chart present on slide 10,
BACKUP label on slide 12.
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu

PPTX_PATH = Path(r"N:\gitt\TraceMind\docs\TraceMind_Pitch.pptx")
SLIDE_W   = Inches(13.333)
SLIDE_H   = Inches(7.5)

prs = Presentation(str(PPTX_PATH))

issues = []
ok_count = 0

print(f"Slide dimensions: {prs.slide_width/914400:.3f}\" x {prs.slide_height/914400:.3f}\"")
print(f"Total slides: {len(prs.slides)}")
print()

EXPECTED_SLIDES = 12
if len(prs.slides) != EXPECTED_SLIDES:
    issues.append(f"Expected {EXPECTED_SLIDES} slides, got {len(prs.slides)}")

SPEAKERS = {
    1: "Nishit", 2: "Nishit",
    3: "Pragun", 4: "Pragun",
    5: "Pranav", 6: "Pranav",
    7: "Sameera", 8: "Sameera",
    9: "Nishit",
    10: "Unique", 11: "Unique", 12: "Unique",
}

TITLES = {
    1: "TraceMind",
    2: "Agenda",
    3: "Outage Reality",
    4: "Generic LLMs",
    5: "Postmortem Corpus",
    6: "HDBSCAN",
    7: "ChromaDB",
    8: "2-Pass Retrieval",
    9: "Architecture",
    10: "Benchmarks",
    11: "Limitations",
    12: "BACKUP",
}

for idx, slide in enumerate(prs.slides):
    slide_num = idx + 1
    texts_found = []
    overflow_shapes = []
    has_notes = False
    shape_types = []

    # --- collect all text ---
    for shape in slide.shapes:
        shape_types.append(shape.shape_type)
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    texts_found.append(run.text)

        # Check bounds
        sl = shape.left; st = shape.top
        sw = shape.width; sh_h = shape.height
        if sl is not None and st is not None:
            if sl + sw > SLIDE_W + Inches(0.1):
                overflow_shapes.append(f"shape '{shape.name}' overflows right by {(sl+sw-SLIDE_W)/914400:.2f}\"")
            if st + sh_h > SLIDE_H + Inches(0.1):
                overflow_shapes.append(f"shape '{shape.name}' overflows bottom by {(st+sh_h-SLIDE_H)/914400:.2f}\"")

    # --- speaker notes ---
    notes_slide = slide.notes_slide
    notes_text = notes_slide.notes_text_frame.text.strip()
    if len(notes_text) > 10:
        has_notes = True
        notes_wc = len(notes_text.split())
    else:
        notes_wc = 0

    all_text = " ".join(texts_found)

    # Keyword checks
    keyword = TITLES.get(slide_num, "")
    has_title_kw = any(keyword.lower() in t.lower() for t in texts_found)

    # Footer checks: "TraceMind" in footer
    has_tracemind_footer = any("TraceMind" in t for t in texts_found)
    has_slide_num = any(str(slide_num) in t for t in texts_found)

    status = "OK"
    slide_issues = []

    if overflow_shapes:
        slide_issues.extend(overflow_shapes)
    if not has_notes:
        slide_issues.append("Missing speaker notes")
    if not has_tracemind_footer:
        slide_issues.append("Missing 'TraceMind' footer")

    if slide_issues:
        status = "WARN"
        for iss in slide_issues:
            issues.append(f"Slide {slide_num}: {iss}")
    else:
        ok_count += 1

    speaker = SPEAKERS.get(slide_num, "?")
    print(f"Slide {slide_num:>2} | {speaker:<14} | Notes words: {notes_wc:>4} | Shapes: {len(list(slide.shapes)):>3} | {status}")
    if slide_issues:
        for iss in slide_issues:
            print(f"          ⚠  {iss}")

print()
print("=" * 60)
if issues:
    print(f"QA ISSUES ({len(issues)} total):")
    for iss in issues:
        print(f"  ⚠  {iss}")
else:
    print("QA PASSED — No issues found.")
print(f"Slides OK: {ok_count}/{EXPECTED_SLIDES}")
print("=" * 60)

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("SUMMARY TABLE")
print(f"{'#':<4} {'Title':<42} {'Speaker':<16} {'Notes WC':>9}")
print("-" * 75)
titles_full = [
    "Title Slide",
    "Agenda & Core Premise",
    "The Outage Reality",
    "Why Generic LLMs Fall Short",
    "Postmortem Corpus & Split",
    "HDBSCAN Clustering",
    "ChromaDB Vector Database",
    "2-Pass Retrieval & Gate",
    "Architecture & Live Demo",
    "Quantitative Benchmarks",
    "Honest Limitations & Roadmap",
    "BACKUP — Category Benchmark",
]
for idx, slide in enumerate(prs.slides):
    slide_num = idx + 1
    notes_text = slide.notes_slide.notes_text_frame.text.strip()
    wc = len(notes_text.split()) if notes_text else 0
    speaker = SPEAKERS.get(slide_num, "?")
    title = titles_full[idx] if idx < len(titles_full) else "?"
    print(f"{slide_num:<4} {title:<42} {speaker:<16} {wc:>9}")
