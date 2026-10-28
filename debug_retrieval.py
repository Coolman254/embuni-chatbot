"""
Show what the retriever finds for a question: the top chunks, their distance and where they come from.
Use it on questions the bot got wrong, to see whether the right chunk was found at all.

    ./venv/bin/python debug_retrieval.py "What is the yearly tuition for a full-time Masters in a science-based course?"
    ./venv/bin/python debug_retrieval.py "How do I pay fees through the student portal?" --k 15
"""
import sys

import rag_starter
from rag_starter import embedder

args = [a for a in sys.argv[1:] if not a.startswith("--")]
k = 10
if "--k" in sys.argv:
    k = int(sys.argv[sys.argv.index("--k") + 1])
    args = [a for a in args if a != str(k)]
if not args:
    sys.exit(__doc__)
question = args[0]

retrieve = getattr(rag_starter, "retrieve", None)
if retrieve:
    docs, metas, dists = retrieve(question, k)
    mode = "hybrid (embeddings + keywords)" if getattr(rag_starter, "HYBRID", False) else "embeddings only"
else:   # an older rag_starter.py without hybrid search
    col = rag_starter.get_collection()
    q = embedder.encode([rag_starter.normalize(question)]).tolist()
    res = col.query(query_embeddings=q, n_results=k)
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    mode = "embeddings only"
print(f"Search mode: {mode}. The embedding model reads at most {embedder.max_seq_length} tokens of each chunk.\n")
for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
    page = f" p{meta['page']}" if meta.get("page") else ""
    print(f"#{rank}  distance {dist:.3f}  {meta['source']}{page}  ({len(doc)} characters)")
    print("    " + doc[:160].replace("\n", " ") + "...\n")
