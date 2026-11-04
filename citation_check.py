"""Compare citation rules on the handbook questions WITHOUT calling the language model (uses no Groq tokens).
Usage:  ./venv/bin/python citation_check.py [questions.csv]"""
import csv, re, sys
import rag_starter as rag

def cited_ok(sources, expected_page, expected_source=""):
    es = expected_source.strip().split("/")[-1].lower()
    if es and not es.startswith("student_handbook"):        # a non-handbook document: its name must be cited
        return any(es[:25] in s.lower() for s, _ in sources)
    pages = [p for s, p in sources if s.startswith("STUDENT_HANDBOOK") and p]
    nopage = any(s.startswith("STUDENT_HANDBOOK") and not p for s, p in sources)
    return expected_page in pages or nopage or any(abs(p - expected_page) == 1 for p in pages)

path = sys.argv[1] if len(sys.argv) > 1 else "test_questions.csv"
rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
        if r["in_documents"].strip() == "yes" and re.match(r"\d+", r["expected_source_page"].strip())]
score = {"distance": 0, "rank": 0}
for r in rows:
    docs, metas, dists = rag.retrieve_multi(r["question"])
    relevant = [(d, m, x) for d, m, x in zip(docs, metas, dists) if x <= rag.MAX_DISTANCE]
    if not relevant:
        continue
    e = int(re.match(r"\d+", r["expected_source_page"].strip()).group())
    res = {}
    for mode in ("distance", "rank"):
        rag.CITE_MODE = mode
        res[mode] = cited_ok(rag.pick_sources(relevant), e, r["expected_source"])
        score[mode] += res[mode]
    if res["distance"] != res["rank"]:
        print(f"#{r['id']} distance={'ok' if res['distance'] else 'MISS'} rank={'ok' if res['rank'] else 'MISS'}  {r['question'][:60]}")
n = len(rows)
print(f"\nCitations correct ({n} questions with an expected page):")
print(f"  by distance (old): {score['distance']}/{n} = {100*score['distance']/n:.1f}%")
print(f"  by rank (new):     {score['rank']}/{n} = {100*score['rank']/n:.1f}%")
