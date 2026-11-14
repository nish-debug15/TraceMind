import os
import re
from pathlib import Path

target_files = [
    "scripts/generate_pitch_deck.py",
    "scripts/build_report.py"
]

patterns = [
    r"deduplicat\w*",
    r"burst deduplication",
    r"recurring failure\w*"
]
combined_regex = re.compile("|".join(patterns), re.IGNORECASE)

print("=" * 80)
print("OCCURRENCES IN SCRIPTS (DO NOT EDIT)")
print("=" * 80)

for fpath in target_files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if combined_regex.search(line):
                # Clean line
                clean_line = line.strip()
                print(f"{fpath}:{line_num}: {clean_line}")

print("\n" + "=" * 80)
print("CHECKING DOCS/*.DOCX")
print("=" * 80)
import docx

docx_files = list(Path("docs").glob("*.docx"))
for doc_path in docx_files:
    doc = docx.Document(doc_path)
    for p_idx, p in enumerate(doc.paragraphs, 1):
        if combined_regex.search(p.text):
            print(f"{doc_path} (Paragraph {p_idx}): {p.text[:120]}...")
    for t_idx, table in enumerate(doc.tables, 1):
        for r_idx, row in enumerate(table.rows, 1):
            for c_idx, cell in enumerate(row.cells, 1):
                if combined_regex.search(cell.text):
                    print(f"{doc_path} (Table {t_idx}, Row {r_idx}, Col {c_idx}): {cell.text.strip()[:100]}...")
