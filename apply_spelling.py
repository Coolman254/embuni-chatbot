"""Always spell the university's name correctly in answers ("Embi" -> "Embu").
Run from the project folder:  ./venv/bin/python apply_spelling.py   (no re-index needed)"""
import shutil, sys
from pathlib import Path

p = Path("rag_starter.py")
s = p.read_text(encoding="utf-8")
if "def fix_spelling(" in s:
    sys.exit("Already patched.")
shutil.copy(p, "rag_starter_before_spelling.py")

helper = '''
def fix_spelling(text):
    """The model sometimes writes 'Embi'; the name is always 'Embu'."""
    text = re.sub(r"\\bEmbi\\b", "Embu", text)
    return re.sub(r"\\bEMBI\\b", "EMBU", text)

'''
marker = "\ndef answer(question):"
assert marker in s, "could not find answer()"
s = s.replace(marker, helper + marker, 1)

old = "    return text, pick_sources(relevant)\n"
assert old in s, "could not find the end of answer() (is apply_citations.py applied?)"
s = s.replace(old, "    return fix_spelling(text), pick_sources(relevant)\n", 1)
p.write_text(s, encoding="utf-8")
print("Patched rag_starter.py. Backup: rag_starter_before_spelling.py. Restart the bot to use it.")
