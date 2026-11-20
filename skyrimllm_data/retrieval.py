"""Knowledge access filter + lexical retriever (SQLite FTS5 / BM25).

Used to build realistic RAG contexts for SFT examples. The *access filter*
(`visible`) is the contract the production RAG must also implement; the
retriever itself (BM25 here) can be swapped for hybrid dense+sparse search in
Phase 3 without changing training data format.
"""
from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from .common import CORPUS, DATA, read_jsonl
from .schemas import WorldState

INDEX = DATA / "index" / "chunks_fts.sqlite"
STOP = set("""a an the of and or to in on at for with by from is are was were be been am do does did
what who where when why how which that this these those you your yours me my mine i we our us it its
he she they them his her their there here about tell know any some can could would should will
please hey hello ok so just like have has had not no yes""".split())


def build_index(path: Path = INDEX) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id UNINDEXED, title, text, "
                "meta UNINDEXED, tokenize='porter unicode61')")
    n = 0
    rows = []
    for c in read_jsonl(CORPUS / "c1_world" / "chunks.jsonl"):
        if "meta" in c["scopes"]:
            continue
        meta = {k: c.get(k) for k in ("doc_id", "section", "scopes", "quest_gates", "spoiler",
                                      "entity_type", "n_tokens", "restricted_by")}
        rows.append((c["chunk_id"], c["title"], c["text"], json.dumps(meta)))
        n += 1
        if len(rows) >= 5000:
            con.executemany("INSERT INTO chunks VALUES (?,?,?,?)", rows)
            rows = []
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()
    return n


@lru_cache(maxsize=1)
def quest_participants() -> dict[str, set[str]]:
    out = {}
    p = CORPUS / "c3_quests" / "quests.jsonl"
    for q in read_jsonl(p):
        out[q["quest_id"]] = set(q.get("involved_npcs", []))
    return out


def gate_ok(g: dict, ws: WorldState) -> bool:
    qid = g["quest_id"]
    if g.get("completed"):
        return qid in ws.completed_quests
    if g.get("min_stage") is not None:
        return qid in ws.completed_quests or ws.quests.get(qid, -1) >= g["min_stage"]
    return qid in ws.completed_quests or qid in ws.quests   # "started"


def visible(meta: dict, npc: dict, ws: WorldState) -> bool:
    scopes = meta["scopes"]
    if "meta" in scopes:
        return False
    npc_scopes = set(npc["scopes"])
    ok = bool(npc_scopes.intersection(scopes))
    if not ok and "quest_participants" in scopes:
        ok = any(npc["npc_id"] in quest_participants().get(g["quest_id"], set())
                 for g in meta.get("quest_gates", []))
    if not ok:
        return False
    return all(gate_ok(g, ws) for g in meta.get("quest_gates", []))


def fts_query(text: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]+", text) if t.lower() not in STOP]
    toks = [t.replace("'", "").replace("-", " ") for t in toks]
    return " OR ".join(f'"{t}"' for t in toks[:16]) or '"skyrim"'


class Retriever:
    def __init__(self, path: Path = INDEX):
        if not path.exists():
            build_index(path)
        self.con = sqlite3.connect(path)

    def search(self, query: str, npc: dict | None = None, ws: WorldState | None = None,
               k: int = 4, pool: int = 200, invisible: bool = False,
               min_rel: float = 0.0) -> list[dict]:
        """Top-k chunks. With npc+ws, only chunks that NPC may know
        (or, with invisible=True, only chunks it may NOT know)."""
        rows = self.con.execute(
            "SELECT chunk_id, title, text, meta, bm25(chunks, 0, 4.0, 1.0) AS s FROM chunks "
            "WHERE chunks MATCH ? ORDER BY s LIMIT ?", (fts_query(query), pool)).fetchall()
        out = []
        for cid, title, text, meta, score in rows:
            meta = json.loads(meta)
            if npc is not None and ws is not None:
                if visible(meta, npc, ws) == invisible:
                    continue
            if out and -score < min_rel * out[0]["score"]:
                break
            out.append({"chunk_id": cid, "title": title, "text": text, "score": -score, **meta})
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    print("indexed", build_index())
