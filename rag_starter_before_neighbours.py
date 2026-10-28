"""
Starter RAG chatbot for University of Embu documents.

Usage:
    python rag_starter.py index    # read PDFs in ./docs and build the vector database
    python rag_starter.py ask      # chat in the terminal

Before running:
    1. Put your PDFs in a folder called "docs" next to this file.
    2. Install Ollama and pull a model:  ollama pull llama3.2
    3. pip install -r requirements.txt
"""

import os
import re
import sys
import math
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
import pymupdf
import ollama
from sentence_transformers import SentenceTransformer

# ---------- Settings you can tune (good material for your evaluation chapter) ----------
DOCS_DIR = Path("docs")
WEB_DIR = Path("docs_web")   # text pages saved by fetch_pages.py
DB_DIR = "chroma_db"
COLLECTION = "uoem_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")   # any model you have pulled; override per run: LLM_MODEL=qwen2.5:7b
CHUNK_SIZE = 800            # characters per chunk
CHUNK_OVERLAP = 150         # characters shared between neighbouring chunks
TOP_K = int(os.getenv("TOP_K", "6"))   # how many chunks to give the model
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.85"))  # chunks further away than this are "not relevant"
# --- Which model writes the answer: local Ollama, or a hosted API (any OpenAI-compatible service) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")      # "ollama" or "api"
API_BASE_URL = os.getenv("LLM_BASE_URL", "")            # e.g. https://api.groq.com/openai/v1
API_KEY = os.getenv("LLM_API_KEY", "")                  # keep this secret, never put it in the code
API_MODEL = os.getenv("LLM_API_MODEL", "")              # e.g. llama-3.3-70b-versatile
WHOLE_PAGE_MAX = 1500       # pages shorter than this stay in one piece (keeps a table together with its title)
SHORT_DOC_MAX = 1500        # a PDF with several pages but less text than this stays in one chunk (0 = off)
SOURCE_MARGIN = 0.10        # only cite chunks this close to the best match
MAX_SOURCES = 3             # cite at most this many pages

# --- Hybrid retrieval: combine embedding similarity with keyword (BM25) matching ---
HYBRID = os.getenv("HYBRID", "1") == "1"   # HYBRID=0 gives the original embedding-only search
RRF_K = 60                                  # how the two rankings are blended (60 is the usual value)
# ---------------------------------------------------------------------------------------

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=DB_DIR)


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def normalize(text):
    """Write common spellings one way, so a question and a document match even if spelled differently."""
    return re.sub(r"\bph\.?\s*d\b\.?", "PhD (Doctor of Philosophy)", text, flags=re.IGNORECASE)


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut text into overlapping chunks. A short page stays whole."""
    text = normalize(" ".join(text.split()))  # tidy up whitespace and spelling
    if len(text) <= WHOLE_PAGE_MAX:
        return [text] if len(text.strip()) > 50 else []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def index_documents():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    web_pages = sorted(WEB_DIR.glob("*.txt")) if WEB_DIR.exists() else []
    if not pdfs and not web_pages:
        print(f"No PDFs found in '{DOCS_DIR}' and no pages in '{WEB_DIR}'. Add some and run again.")
        return

    # Start fresh so re-running does not create duplicates
    client.delete_collection(COLLECTION) if COLLECTION in [c.name for c in client.list_collections()] else None
    collection = get_collection()

    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        ids, texts, metas = [], [], []
        empty_pages = 0
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
        for page_number, page in doc_pages:
            page_text = page.get_text()
            if not page_text.strip():
                empty_pages += 1
                continue
            for i, chunk in enumerate(split_text(page_text)):
                ids.append(f"{pdf.stem}-p{page_number}-c{i}")
                texts.append(chunk)
                metas.append({"source": pdf.name, "page": page_number})

        if texts:
            embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
            collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

        print(f"{pdf.name}: {len(texts)} chunks indexed")
        if empty_pages:
            print(f"  Warning: {empty_pages} page(s) had no text. They may be scanned images (needs OCR).")

    # Web pages saved by fetch_pages.py: the page address is the source, page number 0 means "web page"
    for path in web_pages:
        raw = path.read_text(encoding="utf-8")
        header, _, body = raw.partition("\n\n")
        url = next((ln[len("SOURCE:"):].strip() for ln in header.splitlines() if ln.startswith("SOURCE:")), path.name)
        chunks = split_text(body)
        if chunks:
            ids = [f"{path.stem}-c{i}" for i in range(len(chunks))]
            embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
            collection.add(ids=ids, documents=chunks, embeddings=embeddings,
                           metadatas=[{"source": url, "page": 0} for _ in chunks])
        print(f"{url}: {len(chunks)} chunks indexed")

    print("Done. Run:  python rag_starter.py ask")


def generate(prompt):
    """Send the prompt to the chosen model and return its text."""
    if LLM_PROVIDER == "api":
        if not (API_BASE_URL and API_KEY and API_MODEL):
            raise RuntimeError("Set LLM_BASE_URL, LLM_API_KEY and LLM_API_MODEL to use the hosted model")
        from openai import OpenAI
        client_api = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=60)
        r = client_api.chat.completions.create(
            model=API_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return r.choices[0].message.content
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 4096},  # temperature 0 = repeatable results for evaluation
    )
    return response["message"]["content"]


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


def answer(question):
    docs, metas, dists = retrieve(question)

    # Keep only chunks that are close enough in meaning to the question
    relevant = [(d, m, dist) for d, m, dist in zip(docs, metas, dists) if dist <= MAX_DISTANCE]
    if not relevant:
        return "I could not find this in the university documents I have. Please contact the relevant office.", []

    context = "\n\n".join(
        f"[Source: {m['source']}{', page ' + str(m['page']) if m['page'] else ''}]\n{d}" for d, m, _ in relevant
    )
    prompt = (
        "You are a student support assistant for the University of Embu.\n"
        "Answer the question using ONLY the context below.\n"
        "If the context does not contain the answer, say you do not know and suggest "
        "contacting the relevant university office. Do not guess.\n"
        "If the context describes a procedure, list every step in order and do not stop early.\n"
        "Copy exact figures, dates and names from the context. Do not put page numbers in the answer.\n"
        "Only mention offices or contacts that appear in the context.\n"
        "Spell names exactly as they appear in the context (the institution is the University of Embu).\n"
        "Keep the answer short and clear.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    try:
        text = generate(prompt)
    except Exception as e:  # no internet, bad key, rate limit, Ollama not running
        return f"Sorry, the answer service is unavailable right now ({type(e).__name__}: {e}). Please try again later.", []
    # Cite only the pages closest to the best match, not every retrieved chunk
    best = min(dist for _, _, dist in relevant)
    close = [(m["source"], m["page"]) for _, m, dist in sorted(relevant, key=lambda r: r[2]) if dist <= best + SOURCE_MARGIN]
    sources = []
    for src in close:
        if src not in sources:
            sources.append(src)
    sources = sorted(sources[:MAX_SOURCES])
    return text, sources


def chat():
    model_name = API_MODEL if LLM_PROVIDER == "api" else LLM_MODEL
    print(f"University of Embu support chatbot (provider: {LLM_PROVIDER}, model: {model_name}, top_k={TOP_K}, max_distance={MAX_DISTANCE}). Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        text, sources = answer(question)
        print(f"\nBot: {text}")
        if sources:
            print("Sources: " + "; ".join(f"{s} (page {p})" if p else s for s, p in sources))
        print()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "index":
        index_documents()
    elif command == "ask":
        chat()
    else:
        print("Usage: python rag_starter.py index | ask")