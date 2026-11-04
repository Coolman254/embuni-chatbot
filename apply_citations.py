"""Cite the chunks the search ranked first, instead of choosing by embedding distance.
Run from the project folder:  ./venv/bin/python apply_citations.py
Set CITE_MODE=distance to get the old behaviour back (for comparison)."""
import shutil, sys
from pathlib import Path

p = Path("rag_starter.py")
s = p.read_text(encoding="utf-8")
if "def pick_sources" in s:
    sys.exit("Already patched.")
shutil.copy(p, "rag_starter_before_citations.py")

old_cite = '''    # Cite only the pages closest to the best match, not every retrieved chunk
    best = min(dist for _, _, dist in relevant)
    close = [(m["source"], m["page"]) for _, m, dist in sorted(relevant, key=lambda r: r[2]) if dist <= best + SOURCE_MARGIN]
    sources = []
    for src in close:
        if src not in sources:
            sources.append(src)
    sources = sorted(sources[:MAX_SOURCES])
    return text, sources
'''
new_cite = '''    return text, pick_sources(relevant)
'''
assert old_cite in s, "could not find the source-selection block in answer()"
s = s.replace(old_cite, new_cite)

helper = '''
CITE_MODE = os.getenv("CITE_MODE", "rank")        # "rank": cite the first-ranked chunks; "distance": the old rule


def pick_sources(relevant):
    """relevant = [(document, metadata, distance)] in the order the search ranked them."""
    if CITE_MODE == "distance":
        best = min(dist for _, _, dist in relevant)
        close = [(m["source"], m["page"]) for _, m, dist in sorted(relevant, key=lambda r: r[2]) if dist <= best + SOURCE_MARGIN]
    else:
        close = [(m["source"], m["page"]) for _, m, _ in relevant]
    sources = []
    for src in close:
        if src not in sources:
            sources.append(src)
    return sorted(sources[:MAX_SOURCES])

'''
marker = "\ndef answer(question):"
assert marker in s
s = s.replace(marker, helper + marker, 1)

# several-part questions: alternate between the parts so each part gets cited
old_multi = '''    docs, metas, dists, seen = [], [], [], set()
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
new_multi = '''    results = [list(zip(*retrieve(part, per_part))) for part in parts]
    docs, metas, dists, seen = [], [], [], set()
    for rank in range(max(len(r) for r in results)):       # best of each part first, then the next best, ...
        for r in results:
            if rank < len(r) and r[rank][0] not in seen:
                doc, meta, dist = r[rank]
                seen.add(doc)
                docs.append(doc)
                metas.append(meta)
                dists.append(dist)
    return docs, metas, dists
'''
if old_multi in s:
    s = s.replace(old_multi, new_multi)
else:
    print("Note: question-splitting code not found or different; left as it was.")
p.write_text(s, encoding="utf-8")
print("Patched rag_starter.py. Backup: rag_starter_before_citations.py. No re-index needed.")
