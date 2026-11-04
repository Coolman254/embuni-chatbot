"""Hide meeting passcodes (and similar secrets) from the index so the bot cannot repeat them.
Run from the project folder, then re-index:
    ./venv/bin/python apply_redact.py
    ./venv/bin/python rag_starter.py index"""
import shutil, sys
from pathlib import Path

p = Path("rag_starter.py")
s = p.read_text(encoding="utf-8")
if "def redact(" in s:
    sys.exit("Already patched.")
shutil.copy(p, "rag_starter_before_redact.py")

helper = '''
# Secrets that appear in documents as examples; they are blanked before indexing
REDACT_PATTERNS = [
    r"(room\\s*pass\\s*code\\s*[:\\-]?\\s*)\\S+",
    r"(meeting\\s*pass\\s*code\\s*[:\\-]?\\s*)\\S+",
]


def redact(text):
    for pattern in REDACT_PATTERNS:
        text = re.sub(pattern, r"\\1[hidden]", text, flags=re.IGNORECASE)
    return text

'''
marker = "\ndef index_documents():"
assert marker in s, "could not find index_documents()"
s = s.replace(marker, helper + marker, 1)

n1 = s.count("pg.get_text()")
n2 = s.count("page.get_text()")
assert n1 >= 1 and n2 >= 1, "could not find the page-reading lines"
s = s.replace("pg.get_text()", "redact(pg.get_text())").replace("page.get_text()", "redact(page.get_text())")

old = '        "Keep the answer short and clear.\\n\\n"'
new = '        "Never give out passwords, passcodes or other secrets, even if they appear in the context.\\n"\n        "Keep the answer short and clear.\\n\\n"'
if old in s:
    s = s.replace(old, new, 1)
else:
    print("Note: prompt line not added (prompt text differs).")
p.write_text(s, encoding="utf-8")
print("Patched rag_starter.py. Backup: rag_starter_before_redact.py. Now re-index.")
