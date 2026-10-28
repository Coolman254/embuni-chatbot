"""
FastAPI backend for the University of Embu support chatbot.

Run from the project folder (the one that contains rag_starter.py, docs/ and chroma_db/):
    ./venv/bin/python -m pip install fastapi "uvicorn[standard]"
    ./venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000
"""

import os
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # so "import rag_starter" works from the project folder

from rag_starter import answer  # noqa: E402

DB_PATH = os.getenv("FEEDBACK_DB", str(ROOT / "feedback.db"))

app = FastAPI(title="University of Embu Student Support Chatbot")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS queries (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP,
               question TEXT, answer TEXT, sources TEXT, seconds REAL)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP,
               query_id INTEGER, rating INTEGER, comment TEXT)"""
    )
    return conn


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class Feedback(BaseModel):
    query_id: int
    rating: int = Field(ge=-1, le=1)  # 1 = helpful, -1 = not helpful
    comment: str = Field(default="", max_length=1000)


@app.post("/api/ask")
def ask(body: Ask):
    start = time.time()
    text, sources = answer(body.question.strip())
    seconds = round(time.time() - start, 2)
    src = []
    for s, p in sources:
        if str(s).startswith("http"):  # a web page saved by fetch_pages.py
            u = urlparse(s)
            src.append({"doc": (u.netloc + u.path).rstrip("/"), "page": 0, "url": s})
        else:
            src.append({"doc": Path(s).stem.replace("_", " ").title(), "page": int(p), "url": ""})
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO queries (question, answer, sources, seconds) VALUES (?,?,?,?)",
            (body.question, text, "; ".join(f"{x['doc']} p{x['page']}" if x["page"] else x["doc"] for x in src), seconds),
        )
        query_id = cur.lastrowid
    return {"id": query_id, "answer": text, "sources": src, "seconds": seconds}


@app.post("/api/feedback")
def feedback(body: Feedback):
    with db() as conn:
        conn.execute(
            "INSERT INTO feedback (query_id, rating, comment) VALUES (?,?,?)",
            (body.query_id, body.rating, body.comment),
        )
    return {"ok": True}


@app.get("/api/stats")
def stats():
    with db() as conn:
        asked = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=1").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating=-1").fetchone()[0]
    return {"questions": asked, "helpful": up, "not_helpful": down}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
