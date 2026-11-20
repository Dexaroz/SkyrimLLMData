"""Polite, cached, resumable client for the UESP MediaWiki API.

Pages are stored one JSON file per page under data/raw/uesp/<ns>/ with their
revision id and retrieval time, so builds are reproducible and re-runs only
download what is missing.  Licence of UESP content: CC BY-SA 2.5.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests

from .schemas import now_iso

API = "https://en.uesp.net/w/api.php"
WIKI = "https://en.uesp.net/wiki/"
UA = "SkyrimLLM-corpus-builder/1.0 (academic NLP research; respects robots & rate limits)"
NS = {"Lore": 130, "Skyrim": 134, "Dragonborn": 146}


def safe_name(title: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:120]
    return f"{base}_{hashlib.md5(title.encode()).hexdigest()[:8]}.json"


class UESPClient:
    def __init__(self, cache_dir: Path, delay: float = 1.0, timeout: int = 60):
        self.cache_dir = Path(cache_dir)
        self.delay = delay
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self._last = 0.0

    # ------------------------------------------------------------------ http
    def _get(self, **params) -> dict:
        params |= {"format": "json", "formatversion": 2, "maxlag": 5}
        for attempt in range(6):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.get(API, params=params, timeout=self.timeout)
                if r.status_code in (429, 503) or "maxlag" in r.text[:200]:
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"UESP API failed for {params}")

    # ------------------------------------------------------------- listings
    def list_pages(self, namespace: str) -> list[str]:
        """All non-redirect page titles in a namespace (cached)."""
        idx = self.cache_dir / f"_index_{namespace}.json"
        if idx.exists():
            return json.loads(idx.read_text(encoding="utf-8"))
        titles, cont = [], {}
        while True:
            j = self._get(action="query", list="allpages", apnamespace=NS[namespace],
                          apfilterredir="nonredirects", aplimit="max", **cont)
            titles += [p["title"] for p in j["query"]["allpages"]]
            if "continue" not in j:
                break
            cont = j["continue"]
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
        return titles

    def category_members(self, category: str) -> list[str]:
        out, cont = [], {}
        while True:
            j = self._get(action="query", list="categorymembers",
                          cmtitle=f"Category:{category}", cmlimit="max", **cont)
            out += [m["title"] for m in j["query"]["categorymembers"]]
            if "continue" not in j:
                return out
            cont = j["continue"]

    # ---------------------------------------------------------------- pages
    def path_for(self, title: str) -> Path:
        ns = title.split(":", 1)[0] if ":" in title else "Main"
        return self.cache_dir / ns / safe_name(title)

    def fetch(self, titles: Iterable[str], batch: int = 50, log_every: int = 10) -> int:
        todo = [t for t in dict.fromkeys(titles) if not self.path_for(t).exists()]
        n = 0
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            j = self._get(action="query", titles="|".join(chunk), redirects=1,
                          prop="revisions|categories|info", rvprop="content|ids|timestamp",
                          rvslots="main", cllimit="max", clshow="!hidden", inprop="url")
            redirects = {r["to"]: r["from"] for r in j["query"].get("redirects", [])}
            for p in j["query"].get("pages", []):
                if p.get("missing") or "revisions" not in p:
                    continue
                rev = p["revisions"][0]
                rec = {
                    "title": p["title"], "ns": p["ns"], "page_id": p["pageid"],
                    "revision_id": rev["revid"], "timestamp": rev["timestamp"],
                    "url": p.get("fullurl") or WIKI + p["title"].replace(" ", "_"),
                    "categories": [c["title"].removeprefix("Category:") for c in p.get("categories", [])],
                    "content": rev["slots"]["main"]["content"],
                    "retrieved_at": now_iso(),
                }
                for key in {p["title"], redirects.get(p["title"], p["title"])}:
                    path = self.path_for(key)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                n += 1
            if (i // batch) % log_every == 0:
                print(f"  fetched {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        return n

    def load(self, title: str) -> dict | None:
        p = self.path_for(title)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def iter_cached(self, namespace: str) -> Iterator[dict]:
        d = self.cache_dir / namespace
        seen = set()
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec["page_id"] in seen:     # redirect alias copies
                continue
            seen.add(rec["page_id"])
            yield rec
