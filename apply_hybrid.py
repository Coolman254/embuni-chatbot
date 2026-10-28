"""
Adds hybrid retrieval (embedding similarity + keyword/BM25 matching) to rag_starter.py.

    ./venv/bin/python apply_hybrid.py            # patches rag_starter.py in this folder (keeps a backup)
    ./venv/bin/python apply_hybrid.py other.py   # patch a different file

Afterwards you can switch it off for a comparison run, with no code change:
    HYBRID=0 ./venv/bin/python evaluate.py run f_dense ...
"""
import re
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1] if len(sys.argv) > 1 else "rag_starter.py")
src = target.read_text(encoding="utf-8")

if "def retrieve(" in src:
    sys.exit(f"{target} already has hybrid retrieval. Nothing changed.")

SETTINGS = '''
# --- Hybrid retrieval: combine embedding similarity with keyword (BM25) matching ---
HYBRID = os.getenv("HYBRID", "1") == "1"   # HYBRID=0 gives the original embedding-only search
RRF_K = 60                                  # how the two rankings are blended (60 is the usual value)
'''

FUNCTIONS = '''
_STOPWORDS = set(
    "a an and are as at be been by can could do does for from has have how i if in is it its me my of on or "
    "so than that the their them then there these they this to was we were what when where which who will "
    "with would you your".split()
)
_index_cache = {}


def _tokens(text):
    """Lower-case words without common filler words, with a very light plural strip (forms -> form)."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOPWORDS or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _load_index():
    """Read every chunk, its embedding and its word counts once, then keep them in memory."""
    if not _index_cache:
        data = get_collection().get(include=["documents", "metadatas", "embeddings"])
        emb = np.array(data["embeddings"], dtype=float)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        toks = [_tokens(d) for d in data["documents"]]
        df = Counter()
        for t in toks:
            df.update(set(t))
        _index_cache.update(
            docs=data["documents"], metas=data["metadatas"], emb=emb,
            counts=[Counter(t) for t in toks], lengths=[len(t) for t in toks], df=df,
            avg_len=(sum(len(t) for t in toks) / max(len(toks), 1)) or 1.0,
        )
    return _index_cache


def _bm25(query_tokens, idx, i, k1=1.5, b=0.75):
    n = len(idx["docs"])
    score = 0.0
    for term in set(query_tokens):
        tf = idx["counts"][i].get(term, 0)
        if not tf:
            continue
        df = idx["df"][term]
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * idx["lengths"][i] / idx["avg_len"]))
    return score


def retrieve(question, k=None):
    """Return (documents, metadatas, distances) for the best chunks, best first."""
    k = k or TOP_K
    if not HYBRID:
        q = embedder.encode([normalize(question)]).tolist()
        res = get_collection().query(query_embeddings=q, n_results=k)
        return res["documents"][0], res["metadatas"][0], res["distances"][0]

    idx = _load_index()
    text = normalize(question)
    qv = embedder.encode([text])[0]
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    dist = 1.0 - idx["emb"] @ qv                      # cosine distance to every chunk
    near = [i for i in range(len(dist)) if dist[i] <= MAX_DISTANCE]
    if not near:
        return [], [], []

    dense_rank = {i: r for r, i in enumerate(sorted(near, key=lambda i: dist[i]), start=1)}
    qt = _tokens(text)
    kw = {i: _bm25(qt, idx, i) for i in near}
    kw_rank = {i: r for r, i in enumerate(sorted((i for i in near if kw[i] > 0), key=lambda i: -kw[i]), start=1)}

    def fused(i):
        return 1 / (RRF_K + dense_rank[i]) + (1 / (RRF_K + kw_rank[i]) if i in kw_rank else 0)

    best = sorted(near, key=lambda i: -fused(i))[:k]
    return [idx["docs"][i] for i in best], [idx["metas"][i] for i in best], [float(dist[i]) for i in best]

'''

OLD_QUERY = (
    '    collection = get_collection()\n'
    '    q_embedding = embedder.encode([normalize(question)]).tolist()\n'
    '    results = collection.query(query_embeddings=q_embedding, n_results=TOP_K)\n'
    '\n'
    '    docs = results["documents"][0]\n'
    '    metas = results["metadatas"][0]\n'
    '    dists = results["distances"][0]\n'
)
NEW_QUERY = '    docs, metas, dists = retrieve(question)\n'

problems = []
if OLD_QUERY not in src:
    problems.append("the block at the start of answer() that queries the database")
if not re.search(r"^MAX_SOURCES = .*$", src, flags=re.M):
    problems.append("the MAX_SOURCES line")
if "def answer(question):" not in src:
    problems.append("def answer(question)")
if "from pathlib import Path" not in src:
    problems.append("the 'from pathlib import Path' import")
if problems:
    sys.exit("Could not patch " + str(target) + ". Not found: " + "; ".join(problems) +
             ". Send me that part of the file and I will adjust the patch.")

backup = target.with_name(target.stem + "_before_hybrid.py")
shutil.copy(target, backup)

src = src.replace("from pathlib import Path\n", "import math\nfrom collections import Counter\nfrom pathlib import Path\n", 1)
src = src.replace("import chromadb\n", "import chromadb\nimport numpy as np\n", 1)
src = re.sub(r"^(MAX_SOURCES = .*)$", lambda m: m.group(1) + "\n" + SETTINGS.rstrip("\n"), src, count=1, flags=re.M)
src = src.replace("def answer(question):", FUNCTIONS.lstrip("\n") + "\ndef answer(question):", 1)
src = src.replace(OLD_QUERY, NEW_QUERY, 1)
target.write_text(src, encoding="utf-8")
print(f"Patched {target}. Backup saved as {backup}.")
print("No re-index needed. Check it with:  ./venv/bin/python debug_retrieval.py \"Who can I email if I have trouble with portal fee payment?\" --k 8")
