"""Build a large knowledge base from trusted menstrual-health pages.

Downloads each URL in knowledge/urls.txt, extracts the main article text, splits it into
sections (one entry per H2 heading), and writes knowledge/web_sources.json.
rag.KnowledgeBase loads it automatically alongside the curated knowledge/sources.json.

Usage:  python ingest.py            (re-run any time; then restart the app)
"""
import json, re, sys, time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
URLS = ROOT / "knowledge" / "urls.txt"
OUT = ROOT / "knowledge" / "web_sources.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Luna menstrual-health education bot; personal project)"}
JUNK = re.compile(r"(cookie|page last reviewed|next review due|share this|sign up|newsletter|"
                  r"was this page useful|feedback|javascript|subscribe|advertis)", re.I)
MIN_WORDS = 25


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def extract_sections(html, url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "button"]):
        tag.decompose()
    page_title = (soup.find("h1") or soup.title or soup.new_tag("x")).get_text(" ", strip=True) or url
    page_title = re.sub(r"\s*[-|–]\s*(NHS|MedlinePlus|ACOG|womenshealth\.gov).*$", "", page_title, flags=re.I)
    root = soup.find("main") or soup.find("article") or soup.body or soup
    sections, heading, buf = [], "Overview", []

    def flush():
        text = " ".join(buf).strip()
        if len(text.split()) >= MIN_WORDS:
            sections.append((heading, text))

    for el in root.find_all(["h2", "h3", "p", "li"]):
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not t or JUNK.search(t):
            continue
        if el.name == "h2":
            flush(); heading, buf = t, []
        elif el.name == "h3":
            buf.append(t.rstrip(".:") + ":")
        else:
            if el.name == "li" and el.find_parent(["nav", "footer"]):
                continue
            buf.append(t if t.endswith((".", "!", "?", ":")) else t + ".")
    flush()
    return page_title, sections


def main():
    urls = [u.strip() for u in URLS.read_text(encoding="utf-8").splitlines() if u.strip() and not u.startswith("#")]
    docs, failed = [], []
    for i, url in enumerate(urls, 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            title, sections = extract_sections(r.text, url)
            base = slug(re.sub(r"^https?://(www\.)?", "", url))
            for h, text in sections:
                docs.append({"id": f"{base}--{slug(h)}", "title": f"{title}: {h}" if h != "Overview" else title,
                             "url": url, "keywords": [], "text": text})
            print(f"[{i}/{len(urls)}] ok   {len(sections):2d} sections  {url}")
        except Exception as e:
            failed.append(url)
            print(f"[{i}/{len(urls)}] FAIL {url}  ({e})")
        time.sleep(1)  # be polite to the sites
    # de-duplicate ids
    seen = {}
    for d in docs:
        n = seen.get(d["id"], 0); seen[d["id"]] = n + 1
        if n: d["id"] += f"-{n}"
    OUT.write_text(json.dumps(docs, indent=1, ensure_ascii=False), encoding="utf-8")
    words = sum(len(d["text"].split()) for d in docs)
    print(f"\nWrote {len(docs)} sections (~{words:,} words) from {len(urls) - len(failed)} pages to {OUT}")
    if failed:
        print(f"{len(failed)} pages failed (site blocked or moved) - fix/remove them in {URLS.name}:")
        print("\n".join("  " + u for u in failed))
    print("Restart the app to re-index.")


if __name__ == "__main__":
    sys.exit(main())
