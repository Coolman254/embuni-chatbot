"""Expand student short forms (vc, dvc, uni, pg, ug ...) before searching and answering.
Run from the project folder:  ./venv/bin/python apply_abbrev.py   (no re-index)
Switch off for comparison with:  ABBREV=0"""
import shutil, sys
from pathlib import Path

p = Path("rag_starter.py")
s = p.read_text(encoding="utf-8")
if "def expand_abbreviations(" in s:
    sys.exit("Already patched.")
shutil.copy(p, "rag_starter_before_abbrev.py")

helper = '''
ABBREV = os.getenv("ABBREV", "1") == "1"          # ABBREV=0 leaves the question exactly as typed
_ABBREVIATIONS = [                                   # (short form, what the documents call it)
    (r"dvc", "Deputy Vice-Chancellor (DVC)"),
    (r"vc", "Vice-Chancellor (VC)"),
    (r"uni", "university"),
    (r"pg", "postgraduate"),
    (r"ug", "undergraduate"),
    (r"sup|supp", "supplementary"),
    (r"dos", "Dean of Students"),
    (r"elearning", "e-learning"),
]


def expand_abbreviations(question):
    """Write the short forms students type in full, so they match the words in the documents."""
    if not ABBREV:
        return question
    for short, full in _ABBREVIATIONS:
        question = re.sub(rf"\\b(?:{short})\\b", full, question, flags=re.IGNORECASE)
    return question

'''
marker = "\ndef answer(question):\n"
assert marker in s, "could not find answer()"
s = s.replace(marker, helper + marker + "    question = expand_abbreviations(question)\n", 1)
p.write_text(s, encoding="utf-8")
print("Patched rag_starter.py. Backup: rag_starter_before_abbrev.py. Restart the bot to use it.")
