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
import sys
from pathlib import Path

import chromadb
import pymupdf
import ollama
from sentence_transformers import SentenceTransformer

# ---------- Settings you can tune (good material for your evaluation chapter) ----------
DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "uoem_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")   # any model you have pulled; override per run: LLM_MODEL=qwen2.5:7b
CHUNK_SIZE = 800            # characters per chunk
CHUNK_OVERLAP = 150         # characters shared between neighbouring chunks
TOP_K = int(os.getenv("TOP_K", "6"))   # how many chunks to give the model
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.85"))  # chunks further away than this are "not relevant"
SOURCE_MARGIN = 0.10        # only cite chunks this close to the best match
MAX_SOURCES = 3             # cite at most this many pages
# ---------------------------------------------------------------------------------------

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=DB_DIR)


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut text into overlapping chunks."""
    text = " ".join(text.split())  # tidy up whitespace
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def index_documents():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in '{DOCS_DIR}'. Add some and run again.")
        return

    # Start fresh so re-running does not create duplicates
    client.delete_collection(COLLECTION) if COLLECTION in [c.name for c in client.list_collections()] else None
    collection = get_collection()

    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        ids, texts, metas = [], [], []
        empty_pages = 0
        for page_number, page in enumerate(doc, start=1):
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

    print("Done. Run:  python rag_starter.py ask")


def answer(question):
    collection = get_collection()
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=TOP_K)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Keep only chunks that are close enough in meaning to the question
    relevant = [(d, m, dist) for d, m, dist in zip(docs, metas, dists) if dist <= MAX_DISTANCE]
    if not relevant:
        return "I could not find this in the university documents I have. Please contact the relevant office.", []

    context = "\n\n".join(
        f"[Source: {m['source']}, page {m['page']}]\n{d}" for d, m, _ in relevant
    )
    prompt = (
        "You are a student support assistant for the University of Embu.\n"
        "Answer the question using ONLY the context below.\n"
        "If the context does not contain the answer, say you do not know and suggest "
        "contacting the relevant university office. Do not guess.\n"
        "If the context describes a procedure, list every step in order and do not stop early.\n"
        "Copy exact figures, dates and names from the context. Do not put page numbers in the answer.\n"
        "Only mention offices or contacts that appear in the context.\n"
        "Keep the answer short and clear.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 4096},  # temperature 0 = repeatable results for evaluation
    )
    # Cite only the pages closest to the best match, not every retrieved chunk
    best = min(dist for _, _, dist in relevant)
    close = [(m["source"], m["page"]) for _, m, dist in sorted(relevant, key=lambda r: r[2]) if dist <= best + SOURCE_MARGIN]
    sources = []
    for src in close:
        if src not in sources:
            sources.append(src)
    sources = sorted(sources[:MAX_SOURCES])
    return response["message"]["content"], sources


def chat():
    print(f"University of Embu support chatbot (model: {LLM_MODEL}, top_k={TOP_K}, max_distance={MAX_DISTANCE}). Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        text, sources = answer(question)
        print(f"\nBot: {text}")
        if sources:
            print("Sources: " + "; ".join(f"{s} (page {p})" for s, p in sources))
        print()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "index":
        index_documents()
    elif command == "ask":
        chat()
    else:
        print("Usage: python rag_starter.py index | ask")