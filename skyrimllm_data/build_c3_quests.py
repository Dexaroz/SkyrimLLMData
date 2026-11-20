"""Corpus 3 — Quests & World State (static definitions).

UESP quest pages -> quests.jsonl (QuestDefinition with stages from {{Journal
Entries}}), world_state_schema.json (variables + defaults) and
quest_index.json (UESP title -> editor id).  Dynamic per-playthrough state is
NOT stored here; it is represented by schemas.WorldState at runtime and
sampled for SFT scenarios.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import CONFIGS, CORPUS, CURATION, RAW, load_yaml, npc_id, read_jsonl, slug, write_jsonl
from .schemas import Provenance, QuestDefinition, QuestGate, QuestStage, WorldEffect
from .uesp import UESPClient
from . import wikitext as W


def split_list(v: str | None) -> list[str]:
    if not v:
        return []
    parts = re.split(r";|,|\bor\b|\band\b|<br>|\n", v)
    return [re.sub(r"\s*\(.*?\)", "", p).strip(" .") for p in parts if p.strip(" .")]


def quest_index(client: UESPClient) -> dict[str, str]:
    """UESP title -> quest id (editor id when present)."""
    idx = {}
    for rec in client.iter_cached("Skyrim"):
        if "{{Quest Header" not in rec["content"][:4000]:
            continue
        _, box = W.extract_infobox(__import__("mwparserfromhell").parse(rec["content"]))
        qid = re.split(r"[;,\s]+", (box.get("id") or "").strip())[0] or slug(rec["title"].split(":", 1)[1])
        if qid in idx.values():   # e.g. both Battle for Whiterun variants are CWSiegeObj
            qid = f"{qid}@{slug(rec['title'].split(':', 1)[1])}"
        idx[rec["title"]] = qid
    return idx


def resolve_gate(gate: dict | None, qidx: dict[str, str]) -> QuestGate | None:
    if not gate:
        return None
    g = dict(gate)
    g["quest_id"] = qidx.get(g["quest_id"], g["quest_id"])
    return QuestGate(**g)


def build(cfg_path: Path, out_dir: Path) -> dict:
    cfg = load_yaml(cfg_path)
    client = UESPClient(RAW / "uesp")
    qidx = quest_index(client)
    effects_cfg = load_yaml(CURATION / "world_effects.yaml")

    # npc names known in C1 (for involved_npcs)
    c1 = CORPUS / "c1_world" / "entities.jsonl"
    npc_titles = {}
    if c1.exists():
        for e in read_jsonl(c1):
            if e["type"] == "npc" and e["entity_id"].startswith("skyrim."):
                npc_titles["Skyrim:" + e["name"]] = npc_id(e["name"])

    scope_terms = {cfg["settlement"].lower(), cfg["hold"].lower(),
                   *[s.lower() for s in cfg.get("hold_settlements", [])]}

    effects_by_q: dict[str, list[WorldEffect]] = {}
    for e in effects_cfg.get("effects", []):
        qid = qidx.get(e["quest"], slug(e["quest"]))
        eff = WorldEffect(trigger=QuestGate(quest_id=qid, min_stage=e.get("min_stage"),
                                            completed=e.get("completed", False)),
                          variable=e["variable"], value=e["value"], description=e["description"],
                          affected_npcs=e.get("affected_npcs", []),
                          knowledge_scope=e.get("knowledge_scope", "common"))
        effects_by_q.setdefault(qid, []).append(eff)

    quests = []
    for rec in client.iter_cached("Skyrim"):
        title = rec["title"]
        if title not in qidx:
            continue
        parsed = W.parse_page(rec["content"])
        box = parsed["infobox"]
        stages_raw = "\n".join(s["raw"] for s in parsed["sections"] if s["kind"] == "stages")
        stages = [QuestStage(**s) for s in W.parse_journal(stages_raw or rec["content"])]
        involved = [npc_titles[l] for l in parsed["links"] if l in npc_titles]
        locs = split_list(box.get("loc"))
        givers = split_list(box.get("giver"))
        blob = " ".join(locs + givers + [box.get("description", "")]).lower()
        tags = [cfg["scope_id"]] if any(t in blob for t in scope_terms) else []
        if box.get("type"):
            tags.append(slug(box["type"]))
        qid = qidx[title]
        quests.append(QuestDefinition(
            quest_id=qid, name=title.split(":", 1)[1], type=box.get("type"),
            giver=givers, locations=locs, involved_npcs=list(dict.fromkeys(involved))[:40],
            prev=split_list(box.get("prev")), next=split_list(box.get("next")),
            description=box.get("description", ""), reward=box.get("reward"),
            stages=stages, world_effects=effects_by_q.get(qid, []), scope_tags=tags,
            provenance=Provenance(source="uesp", url=rec["url"], page_title=title,
                                  revision_id=rec["revision_id"], retrieved_at=rec["retrieved_at"],
                                  license="CC BY-SA 2.5", transform="Quest Header + Journal Entries parsed")))

    out_dir.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_dir / "quests.jsonl", quests)
    (out_dir / "quest_index.json").write_text(json.dumps(qidx, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "world_state_schema.json").write_text(json.dumps({
        "variables": effects_cfg.get("variables", {}),
        "doc": "Dynamic state = schemas.WorldState. quests: {quest_id: stage}; variables keys listed here.",
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    missing = [e["quest"] for e in effects_cfg.get("effects", []) if e["quest"] not in qidx]
    return {"quests": n, "with_stages": sum(bool(q.stages) for q in quests),
            "in_scope": sum(cfg["scope_id"] in q.scope_tags for q in quests),
            "effects": sum(len(v) for v in effects_by_q.values()), "unresolved_effect_quests": missing}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "scope_whiterun.yaml"))
    ap.add_argument("--out", default=str(CORPUS / "c3_quests"))
    a = ap.parse_args()
    print(build(Path(a.config), Path(a.out)))
