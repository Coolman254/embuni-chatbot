"""
Fix for questions that ask two things at once ("where is the university, and which courses do they offer?").

A two-part question is embedded as one blurred search, so one half can crowd out the other, and the model then says
"not provided" for the half that was missing. This patch:
  1. splits a question into its parts and searches for each part separately, then merges the results;
  2. tells the model to answer every part and to say "I do not know" only for a part the context does not cover.

Run after apply_hybrid.py (and the other patches). No re-index needed:
    ./venv/bin/python apply_multipart.py

Switch it off for a comparison run with:  MULTIPART=0 ./venv/bin/python evaluate.py run ...
Backup: rag_starter_before_multipart.py
"""
import re
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1] if len(sys.argv) > 1 else "rag_starter.py")
src = target.read_text(encoding="utf-8")

if "def retrieve_multi(" in src:
    sys.exit(f"{target} already has this fix. Nothing changed.")
if "def retrieve(" not in src:
    sys.exit("Run apply_hybrid.py first (this fix builds on it).")

FUNCS = '''MULTIPART = os.getenv("MULTIPART", "1") == "1"    # MULTIPART=0 searches the whole question as one piece
_SPLIT = re.compile(
    r"\\?+\\s*|\\s*,?\\s+and\\s+(?=(?:what|which|where|how|who|when|why)\\b)|\\s*,\\s*(?=(?:what|which|where|how|who|when|why)\\b)",
    re.IGNORECASE,
)


def split_question(question):
    """Cut 'where is X, which courses do they offer' into its separate questions."""
    parts = [p.strip(" ,.;") for p in _SPLIT.split(question) if p and p.strip(" ,.;")]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts if len(parts) > 1 else [question]


def retrieve_multi(question):
    parts = split_question(question) if MULTIPART else [question]
    if len(parts) == 1:
        return retrieve(question)
    per_part = TOP_K // len(parts) + 1
    docs, metas, dists, seen = [], [], [], set()
    for part in parts:
        d, m, x = retrieve(part, per_part)
        for doc, meta, dist in zip(d, m, x):
            if doc not in seen:
                seen.add(doc)
                docs.append(doc)
                metas.append(meta)
                dists.append(dist)
    return docs, metas, dists


'''
OLD_CALL = "    docs, metas, dists = retrieve(question)\n"
NEW_CALL = "    docs, metas, dists = retrieve_multi(question)\n"
OLD_PROMPT = '        "Keep the answer short and clear.\\n\\n"\n'
NEW_PROMPT = ('        "If the question has several parts, answer every part. Say you do not know only for a part '
              'the context does not cover.\\n"\n' + OLD_PROMPT)

missing = [n for n, s in (("the retrieve(question) call in answer()", OLD_CALL),
                          ("the 'Keep the answer short and clear.' prompt line", OLD_PROMPT)) if s not in src]
if "def answer(question):" not in src:
    missing.append("def answer(question)")
if missing:
    sys.exit("Could not patch " + str(target) + ". Not found: " + "; ".join(missing) + ". Send me those lines.")

shutil.copy(target, target.with_name(target.stem + "_before_multipart.py"))
src = src.replace(OLD_CALL, NEW_CALL, 1).replace(OLD_PROMPT, NEW_PROMPT, 1)
src = src.replace("def answer(question):", FUNCS + "def answer(question):", 1)
target.write_text(src, encoding="utf-8")
print(f"Patched {target}. Backup: {target.stem}_before_multipart.py. No re-index needed.")
