"""
Evaluate the chatbot on a set of test questions.

Step 1 - run every question through the chatbot (saves answers, sources and timings):
    python evaluate.py run baseline                      -> results_baseline.csv
    python evaluate.py run groq120b --pause 3            -> wait 3 seconds between questions (rate limits)
    python evaluate.py run groq120b --resume             -> carry on after a stopped run
    python evaluate.py run test --questions other.csv    -> use a different question file

Step 2 - grading.
    Rows that have a "must_contain" value are graded AUTOMATICALLY (column auto_correct):
      every item separated by ";" must appear in the answer (numbers ignore commas, "a|b" means a or b).
      Questions with expected_answer REFUSE are auto-graded by whether the bot refused.
    For the other rows, open the results file in a spreadsheet and fill in "correct" (1 or 0).
    A value you type in "correct" always overrides the automatic grade.
    "citation_correct" (1 or 0) is always filled in by hand.

Step 3 - calculate the scores:
    python evaluate.py score results_baseline.csv

Use a new label after each change (chunk800, top_k6, model7b ...) and keep every results file.
"""

import argparse
import csv
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REFUSAL_PHRASES = (
    "could not find this",
    "do not know",
    "don't know",
    "not mentioned",
    "no information",
    "contact the relevant",
    "do not have",
    "does not provide",
    "does not specify",
    "does not include",
    "does not contain",
    "not provided",
    "not specified",
)

FIELDS = [
    "id", "category", "question", "expected_answer", "expected_source_page", "expected_source", "in_documents",
    "must_contain", "bot_answer", "bot_sources", "seconds", "auto_refused", "auto_correct",
    "error", "correct", "citation_correct", "notes", "config",
]


def looks_like_refusal(text):
    t = text.lower()
    return any(p in t for p in REFUSAL_PHRASES)


def is_error(text):
    return text.startswith("Sorry, the answer service is unavailable")


def _norm(text):
    text = text.lower()
    text = re.sub(r"[\u00a0\u202f\u2009\u2007]", " ", text)    # special spaces the model sometimes uses
    text = text.replace("\u2011", "-").replace("\u2010", "-")     # non-breaking hyphens
    return re.sub(r"(?<=\d),(?=\d)", "", text)  # 150,503 -> 150503


def auto_grade(row, answer_text):
    """Return '1', '0' or '' (cannot grade automatically)."""
    if is_error(answer_text):
        return ""
    if row.get("expected_answer", "").strip() == "REFUSE":
        return "1" if looks_like_refusal(answer_text) else "0"
    spec = (row.get("must_contain") or "").strip()
    if not spec:
        return ""
    ans = _norm(answer_text)
    for item in spec.split(";"):
        options = [o.strip().lower() for o in item.split("|") if o.strip()]
        found = False
        for o in options:
            o = _norm(o)
            if re.fullmatch(r"\d+", o):
                found = re.search(rf"(?<![\d.]){o}(?!\d)", ans) is not None  # whole number, not part of a bigger one
            else:
                found = o in ans
            if found:
                break
        if not found:
            return "0"
    return "1"


def config_string(rag):
    provider = getattr(rag, "LLM_PROVIDER", "ollama")
    model = getattr(rag, "API_MODEL", "") if provider == "api" else getattr(rag, "LLM_MODEL", "")
    return (f"provider={provider}; model={model}; top_k={getattr(rag, 'TOP_K', '?')}; "
            f"max_distance={getattr(rag, 'MAX_DISTANCE', '?')}; chunk={getattr(rag, 'CHUNK_SIZE', '?')}; "
            f"whole_page_max={getattr(rag, 'WHOLE_PAGE_MAX', 'n/a')}; hybrid={getattr(rag, 'HYBRID', 'n/a')}")


def run(label, questions_file="test_questions.csv", pause=0.0, resume=False):
    import rag_starter  # imported here so "score" works without the heavy libraries
    from rag_starter import answer

    out_name = f"results_{label}.csv"
    with open(questions_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = set()
    mode = "w"
    if resume and Path(out_name).exists():
        with open(out_name, newline="", encoding="utf-8") as f:
            def _ok(r):   # a complete row: has an answer, no error, numeric id and seconds, valid in_documents
                try:
                    float(r.get("seconds") or "x")
                except ValueError:
                    return False
                return (bool(r.get("bot_answer")) and r.get("error") != "yes" and (r.get("id") or "").isdigit()
                        and (r.get("in_documents") or "").lower() in ("yes", "no"))
            kept = [r for r in csv.DictReader(f) if _ok(r)]
        done = {r["id"] for r in kept}
        with open(out_name, "w", newline="", encoding="utf-8") as f:   # drop failed rows so they can be retried
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
        mode = "a"
        print(f"Resuming: {len(done)} questions already answered in {out_name}")

    cfg = config_string(rag_starter)
    print(cfg + "\n")
    with open(out_name, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for i, r in enumerate(rows, start=1):
            if r["id"] in done:
                continue
            start = time.time()
            text, sources = answer(r["question"])
            seconds = round(time.time() - start, 2)
            writer.writerow({
                **{k: r.get(k, "") for k in FIELDS[:8]},
                "bot_answer": text.replace("\n", " "),
                "bot_sources": "; ".join((f"{s} p{p}" if p else s) for s, p in sources),
                "seconds": seconds,
                "auto_refused": "yes" if looks_like_refusal(text) else "no",
                "auto_correct": auto_grade(r, text),
                "error": "yes" if is_error(text) else "",
                "correct": "", "citation_correct": "", "notes": "",
                "config": cfg,
            })
            f.flush()
            flag = "  [ERROR]" if is_error(text) else ""
            print(f"[{i}/{len(rows)}] {seconds}s{flag}  {r['question'][:60]}")
            if pause:
                time.sleep(pause)
    print(f"\nSaved {out_name}.")
    print(f"Rows with a must_contain value or REFUSE are already graded (auto_correct). Fill in 'correct' for the rest, then run:")
    print(f"  python evaluate.py score {out_name}")


def score(results_file):
    with open(results_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def verdict(r):
        manual = r.get("correct", "").strip()
        if manual in ("0", "1"):
            return manual
        auto = r.get("auto_correct", "").strip()
        return auto if auto in ("0", "1") else ""

    errors = [r for r in rows if r.get("error", "").strip() == "yes"]
    graded = [r for r in rows if verdict(r) in ("0", "1") and r not in errors]
    if not graded:
        print("No graded rows yet. Fill in the 'correct' column with 1 or 0.")
        return

    in_docs = [r for r in graded if r["in_documents"].strip().lower() == "yes"]
    not_in = [r for r in graded if r["in_documents"].strip().lower() == "no"]
    cited = [r for r in in_docs if r["citation_correct"].strip() in ("0", "1")]
    times = [float(r["seconds"]) for r in rows if r["seconds"]]

    def pct(part, whole):
        return f"{100 * part / whole:.1f}%" if whole else "n/a"

    print(f"Results file: {results_file}")
    if rows and rows[0].get("config"):
        print(f"Settings: {rows[0]['config']}")
    print(f"Questions graded: {len(graded)} of {len(rows)}   (errors: {len(errors)}, ungraded: {len(rows) - len(graded) - len(errors)})")
    c_in = sum(verdict(r) == "1" for r in in_docs)
    print(f"Accuracy on questions IN the documents:        {c_in}/{len(in_docs)} = {pct(c_in, len(in_docs))}")
    c_out = sum(verdict(r) == "1" for r in not_in)
    print(f"Correct refusals on questions NOT in documents: {c_out}/{len(not_in)} = {pct(c_out, len(not_in))}")
    if cited:
        c_cit = sum(r["citation_correct"].strip() == "1" for r in cited)
        print(f"Correct citations (where answer was in docs):   {c_cit}/{len(cited)} = {pct(c_cit, len(cited))}")
    if times:
        print(f"Response time: mean {statistics.mean(times):.1f}s, median {statistics.median(times):.1f}s, max {max(times):.1f}s")

    by_cat = defaultdict(lambda: [0, 0])
    for r in in_docs:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += verdict(r) == "1"
    if by_cat:
        print("\nAccuracy by category (questions in the documents):")
        for cat, (ok, n) in sorted(by_cat.items()):
            print(f"  {cat:<18} {ok}/{n} = {pct(ok, n)}")

    if errors:
        print(f"\n{len(errors)} question(s) got an error from the model or network. Run again with --resume to retry them.")

    wrong = [r for r in graded if verdict(r) == "0"]
    if wrong:
        print("\nWrong answers to investigate:")
        for r in wrong:
            print(f"  #{r['id']} [{r['category']}] {r['question'][:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("command", nargs="?", choices=["run", "score"])
    ap.add_argument("target", nargs="?", help="label (run) or results file (score)")
    ap.add_argument("--questions", default="test_questions.csv")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--resume", action="store_true", help="continue a stopped run and retry errors")
    a = ap.parse_args()
    if a.command == "run" and a.target:
        run(a.target, a.questions, a.pause, a.resume)
    elif a.command == "score" and a.target:
        score(a.target)
    else:
        print(__doc__)