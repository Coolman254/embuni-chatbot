"""
Polite crawler for the University of Embu chatbot.

Reads the addresses in urls.txt (one per line) and, for each one:
  - checks the site's robots.txt and SKIPS anything the site does not allow
  - waits between requests so it never hammers a server
  - PDF   -> saved into docs/                (indexed like your other PDFs)
  - HTML  -> text saved into docs_web/*.txt  (indexed with the page address as the source)

It only visits the addresses you list. It does not follow links.

Usage:
    ./venv/bin/python -m pip install requests beautifulsoup4
    ./venv/bin/python fetch_pages.py                 # fetch everything in urls.txt
    ./venv/bin/python fetch_pages.py --refresh       # download again even if already saved
    ./venv/bin/python fetch_pages.py --urls my.txt   # use a different list

Then rebuild the index:
    ./venv/bin/python rag_starter.py index
"""

import argparse
import datetime
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

DOCS_DIR = Path("docs")
WEB_DIR = Path("docs_web")
DELAY_SECONDS = float(os.getenv("CRAWL_DELAY", "2"))
TIMEOUT = 30
MAX_BYTES = 30 * 1024 * 1024
MIN_TEXT_CHARS = 200

contact = os.getenv("CRAWLER_CONTACT", "")  # optional: put your student email here
USER_AGENT = "UoEmStudentChatbot/1.0 (university final-year project)" + (f" contact: {contact}" if contact else "")

_robots_cache = {}


def allowed_by_robots(url, session):
    """True if the site's robots.txt lets us fetch this address."""
    parts = urlparse(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin not in _robots_cache:
        rp = RobotFileParser()
        try:
            r = session.get(origin + "/robots.txt", timeout=TIMEOUT)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            elif 400 <= r.status_code < 500:
                rp.parse([])                      # no robots.txt: everything allowed
            else:
                rp.parse(["User-agent: *", "Disallow: /"])   # server trouble: be cautious
        except requests.RequestException:
            rp.parse(["User-agent: *", "Disallow: /"])       # cannot check: be cautious
        _robots_cache[origin] = rp
    return _robots_cache[origin].can_fetch(USER_AGENT, url)


def slug(url):
    parts = urlparse(url)
    base = re.sub(r"[^A-Za-z0-9]+", "_", (parts.netloc + parts.path + ("?" + parts.query if parts.query else ""))).strip("_")
    base = base[:80] or "page"
    return f"{base}_{hashlib.sha1(url.encode()).hexdigest()[:6]}"


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return title, "\n".join(lines)


def fetch(url, session):
    r = session.get(url, timeout=TIMEOUT, stream=True)
    r.raise_for_status()
    chunks, size = [], 0
    for part in r.iter_content(65536):
        size += len(part)
        if size > MAX_BYTES:
            raise ValueError("file is larger than 30 MB, skipped")
        chunks.append(part)
    return r, b"".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="urls.txt")
    ap.add_argument("--refresh", action="store_true", help="download again even if already saved")
    args = ap.parse_args()

    if not Path(args.urls).exists():
        sys.exit(f"Cannot find {args.urls}. Create it with one address per line.")
    urls = [ln.strip() for ln in Path(args.urls).read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    DOCS_DIR.mkdir(exist_ok=True)
    WEB_DIR.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    results = []

    for i, url in enumerate(urls, start=1):
        status = ""
        try:
            if not allowed_by_robots(url, session):
                status = "SKIPPED: robots.txt does not allow it"
            else:
                is_pdf_name = urlparse(url).path.lower().endswith(".pdf")
                if is_pdf_name:
                    target = DOCS_DIR / Path(unquote(urlparse(url).path)).name
                    if target.exists() and not args.refresh:
                        status = f"already have {target.name}"
                    else:
                        time.sleep(DELAY_SECONDS)
                        r, body = fetch(url, session)
                        if not body.startswith(b"%PDF"):
                            status = "FAILED: not a real PDF (got a web page instead)"
                        else:
                            target.write_bytes(body)
                            status = f"saved PDF -> {target}"
                else:
                    target = WEB_DIR / f"{slug(url)}.txt"
                    if target.exists() and not args.refresh:
                        status = f"already have {target.name}"
                    else:
                        time.sleep(DELAY_SECONDS)
                        r, body = fetch(url, session)
                        ctype = r.headers.get("Content-Type", "")
                        if "pdf" in ctype.lower() and body.startswith(b"%PDF"):
                            name = Path(unquote(urlparse(url).path)).name or f"{slug(url)}.pdf"
                            if not name.lower().endswith(".pdf"):
                                name += ".pdf"
                            (DOCS_DIR / name).write_bytes(body)
                            status = f"saved PDF -> {DOCS_DIR / name}"
                        else:
                            r.encoding = r.encoding or "utf-8"
                            title, text = clean_html(body.decode(r.encoding, errors="replace"))
                            if len(text) < MIN_TEXT_CHARS:
                                status = f"SKIPPED: only {len(text)} characters of text (page may need JavaScript)"
                            else:
                                header = f"SOURCE: {url}\nTITLE: {title}\nFETCHED: {datetime.date.today()}\n\n"
                                target.write_text(header + text, encoding="utf-8")
                                status = f"saved text ({len(text)} chars) -> {target}"
        except Exception as e:  # keep going if one address fails
            status = f"FAILED: {type(e).__name__}: {e}"
        results.append((url, status))
        print(f"[{i}/{len(urls)}] {url}\n        {status}")

    ok = sum(1 for _, s in results if s.startswith(("saved", "already")))
    print(f"\nDone: {ok} of {len(results)} addresses are available. Now run:  ./venv/bin/python rag_starter.py index")


if __name__ == "__main__":
    main()
