"""
Two small fixes for rag_starter.py (run once, then re-index):

  1. A short PDF that spans several pages (like the 3-page paybill procedure) is kept as ONE chunk,
     instead of being cut page by page into pieces that are too small to be found.
  2. The prompt now tells the model to spell names exactly as written ("University of Embi" bug).

    ./venv/bin/python apply_fixes.py
    ./venv/bin/python rag_starter.py index

Turn fix 1 off or change its size by editing SHORT_DOC_MAX in rag_starter.py (0 switches it off).
A backup is saved as rag_starter_before_fixes.py.
"""
import re
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1] if len(sys.argv) > 1 else "rag_starter.py")
src = target.read_text(encoding="utf-8")

if "SHORT_DOC_MAX" in src:
    sys.exit(f"{target} already has these fixes. Nothing changed.")

LOOP_OLD = "        for page_number, page in enumerate(doc, start=1):\n"
LOOP_NEW = "        for page_number, page in doc_pages:\n"
EMPTY_OLD = "        empty_pages = 0\n"
EMPTY_NEW = '''        empty_pages = 0
        # A short document that spans several pages stays in one piece (e.g. a 3-page step-by-step procedure)
        page_texts = [pg.get_text() for pg in doc]
        non_empty = [t for t in page_texts if t.strip()]
        merged = normalize(" ".join(" ".join(non_empty).split()))
        if len(non_empty) > 1 and len(merged) <= SHORT_DOC_MAX:
            ids.append(f"{pdf.stem}-p1-c0")
            texts.append(merged)
            metas.append({"source": pdf.name, "page": 1})
            empty_pages = len(page_texts) - len(non_empty)
            doc_pages = []
        else:
            doc_pages = list(enumerate(doc, start=1))
'''
PROMPT_OLD = '        "Only mention offices or contacts that appear in the context.\\n"\n'
PROMPT_NEW = PROMPT_OLD + '        "Spell names exactly as they appear in the context (the institution is the University of Embu).\\n"\n'

missing = []
if LOOP_OLD not in src:
    missing.append("the page loop in index_documents()")
if EMPTY_OLD not in src:
    missing.append("the 'empty_pages = 0' line")
if PROMPT_OLD not in src:
    missing.append("the 'Only mention offices or contacts' prompt line")
if not re.search(r"^WHOLE_PAGE_MAX = .*$", src, flags=re.M):
    missing.append("the WHOLE_PAGE_MAX line")
if missing:
    sys.exit("Could not patch " + str(target) + ". Not found: " + "; ".join(missing) + ". Send me those lines and I will adjust it.")

shutil.copy(target, target.with_name(target.stem + "_before_fixes.py"))
src = src.replace(EMPTY_OLD, EMPTY_NEW, 1)
src = src.replace(LOOP_OLD, LOOP_NEW, 1)
src = src.replace(PROMPT_OLD, PROMPT_NEW, 1)
src = re.sub(r"^(WHOLE_PAGE_MAX = .*)$",
             lambda m: m.group(1) + "\nSHORT_DOC_MAX = 1500        # a PDF with several pages but less text than this stays in one chunk (0 = off)",
             src, count=1, flags=re.M)
target.write_text(src, encoding="utf-8")
print(f"Patched {target}. Backup: {target.stem}_before_fixes.py")
print("Now re-index:  ./venv/bin/python rag_starter.py index")
