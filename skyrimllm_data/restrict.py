"""Apply curated secret rules to Corpus 1 chunks (knowledge restriction).

Runs after C1 and C3 (needs quest ids). Rewrites c1_world/chunks.jsonl in
place and writes c1_world/restriction_report.json listing every chunk whose
scopes were narrowed, so the change is auditable.
"""
from __future__ import annotations

import json
import re

from .build_c3_quests import resolve_gate
from .common import CORPUS, CURATION, load_yaml, read_jsonl, write_jsonl


def apply() -> dict:
    rules = load_yaml(CURATION / "secrets.yaml").get("rules", {})
    qidx = json.loads((CORPUS / "c3_quests" / "quest_index.json").read_text(encoding="utf-8"))
    compiled = {}
    for rid, r in rules.items():
        compiled[rid] = (
            re.compile("|".join(re.escape(t) for t in r["terms"]), re.I),
            [re.compile(re.escape(c), re.I) for c in r["context"]],
            r.get("require_all_context", False),
            r["scopes"],
            resolve_gate(r.get("gate"), qidx),
        )
    path = CORPUS / "c1_world" / "chunks.jsonl"
    out, report = [], {rid: [] for rid in rules}
    # C1 gates use raw editor ids; C3 disambiguates duplicates (CWSiegeObj@...).
    valid = set(qidx.values())
    by_raw: dict[str, list[str]] = {}
    for t, q in qidx.items():
        by_raw.setdefault(q.split("@")[0], []).append(t)
    for c in read_jsonl(path):
        for g in c.get("quest_gates", []):
            if g["quest_id"] in valid:
                continue
            titles = by_raw.get(g["quest_id"], [])
            if len(titles) == 1:
                g["quest_id"] = qidx[titles[0]]
            elif titles:
                pick = [t for t in titles if t.split(":", 1)[1] in (c["title"], c["section"])]
                g["quest_id"] = qidx[pick[0]] if pick else g["quest_id"]
        for rid, (terms, ctx, all_ctx, scopes, gate) in compiled.items():
            hits = [p.search(c["text"]) for p in ctx]
            ok_ctx = all(hits) if all_ctx else any(hits)
            if ok_ctx and terms.search(c["text"]):
                c["scopes"] = list(dict.fromkeys(
                    ([s for s in c["scopes"] if s.startswith("personal:")] if c.get("restricted_by") else [])
                    + scopes))
                if gate:
                    gates = c.get("quest_gates", [])
                    g = gate.model_dump(exclude_none=True)
                    if g not in gates:
                        gates.append(g)
                    c["quest_gates"] = gates
                c.setdefault("restricted_by", []).append(rid)
                c["spoiler"] = True
                report[rid].append(c["chunk_id"])
        out.append(c)
    write_jsonl(path, out)
    (CORPUS / "c1_world" / "restriction_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return {rid: len(v) for rid, v in report.items()}


if __name__ == "__main__":
    print(apply())
