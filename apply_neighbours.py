"""
Fix for answers that continue on the next page.

When a short chunk (the tail end of a page) is retrieved, the chunk that follows it in the document is added too,
so a rule that starts at the bottom of one page and finishes at the top of the next reaches the model in full.
(Example: the make-up CAT rule starts at the bottom of handbook page 39 and continues on page 40.)

Run after apply_hybrid.py. No re-index needed:
    ./venv/bin/python apply_neighbours.py

Switch it off or change it by editing NEIGHBOUR_MAX in rag_starter.py (0 = off). Backup: rag_starter_before_neighbours.py
"""
import re
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1] if len(sys.argv) > 1 else "rag_starter.py")
src = target.read_text(encoding="utf-8")

if "NEIGHBOUR_MAX" in src:
    sys.exit(f"{target} already has this fix. Nothing changed.")
if "def retrieve(" not in src:
    sys.exit("Run apply_hybrid.py first (this fix builds on it).")

OLD_LOAD = 'docs=data["documents"], metas=data["metadatas"], emb=emb,'
NEW_LOAD = 'docs=data["documents"], metas=data["metadatas"], emb=emb, nxt=_next_chunks(data["ids"], data["metadatas"]),'
OLD_RET = (
    '    best = sorted(near, key=lambda i: -fused(i))[:k]\n'
    '    return [idx["docs"][i] for i in best], [idx["metas"][i] for i in best], [float(dist[i]) for i in best]\n'
)
NEW_RET = '''    best = sorted(near, key=lambda i: -fused(i))[:k]
    out = []                                           # (chunk number, distance to report)
    for i in best:
        out.append((i, float(dist[i])))
        j = idx["nxt"].get(i)                          # a short page-end chunk brings the chunk that follows it
        if NEIGHBOUR_MAX and j is not None and len(idx["docs"][i]) < NEIGHBOUR_MAX and j not in best \\
                and all(j != o[0] for o in out):
            out.append((j, float(dist[i])))            # it inherits the distance of the chunk it continues
    return [idx["docs"][i] for i, _ in out], [idx["metas"][i] for i, _ in out], [d for _, d in out]
'''
HELPER = '''def _next_chunks(ids, metas):
    """For each PDF chunk, the chunk that follows it in the same document (same page or the next page)."""
    def key(n):
        m = re.search(r"-c(\\d+)$", ids[n])
        return (metas[n]["source"], metas[n].get("page", 0), int(m.group(1)) if m else 0)
    order = sorted(range(len(ids)), key=key)
    nxt = {}
    for a, b in zip(order, order[1:]):
        pa, pb = metas[a].get("page", 0), metas[b].get("page", 0)
        if metas[a]["source"] == metas[b]["source"] and pa > 0 and pb in (pa, pa + 1):
            nxt[a] = b
    return nxt


'''
missing = [name for name, s in (("the chunk loading line in _load_index()", OLD_LOAD),
                                ("the 'best = ...' block in retrieve()", OLD_RET)) if s not in src]
if missing:
    sys.exit("Could not patch " + str(target) + ". Not found: " + "; ".join(missing) + ". Send me those lines.")

shutil.copy(target, target.with_name(target.stem + "_before_neighbours.py"))
src = src.replace(OLD_LOAD, NEW_LOAD, 1).replace(OLD_RET, NEW_RET, 1)
src = src.replace("def _load_index():", HELPER + "def _load_index():", 1)
src = re.sub(r"^(RRF_K = .*)$",
             lambda m: m.group(1) + "\nNEIGHBOUR_MAX = 400                         # a retrieved chunk shorter than this brings the chunk after it (0 = off)",
             src, count=1, flags=re.M)
target.write_text(src, encoding="utf-8")
print(f"Patched {target}. Backup: {target.stem}_before_neighbours.py. No re-index needed.")
