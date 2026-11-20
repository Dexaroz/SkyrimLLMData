"""Corpus 4 — scenario builder (deterministic, no LLM).

For every in-scope NPC × behaviour it samples a world-state preset, a player,
a player utterance and the RAG context the runtime would build (NPC-filtered
retrieval + optional distractor). Scenarios are the *specification* of each
SFT example; `generate_c4` turns them into conversations with the teacher.

Outputs c4_dialogue/scenarios.jsonl (+ context snapshot per scenario so the
teacher and the student see byte-identical system prompts).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .common import CONFIGS, CORPUS, CURATION, load_yaml, read_jsonl, short_hash, slug, write_jsonl
from .prompting import PROMPT_VERSION, active_events, build_system_prompt
from .retrieval import Retriever, gate_ok
from .schemas import WorldState

MEMORY_BANK = [
    ("conversation", "The traveller asked me about {topic} the other day; I told them what I knew.", 0.4, 0.1),
    ("observed_action", "The traveller paid me well and was polite when we last met.", 0.5, 0.6),
    ("observed_action", "The traveller insulted me in front of others two days ago.", 0.7, -0.7),
    ("witnessed_event", "I saw the traveller fight off a bandit near the city gates.", 0.6, 0.5),
    ("conversation", "The traveller promised to come back with news from Riverwood.", 0.5, 0.2),
    ("rumor", "I heard from a guard that the traveller was seen at Bleak Falls Barrow.", 0.4, 0.0),
    ("observed_action", "The traveller was caught stealing at the market last week.", 0.8, -0.8),
]


def make_world(preset: dict, qidx: dict, effects: list[dict], rng: random.Random, cfg: dict) -> WorldState:
    quests, done = {}, []
    for title, st in preset["quests"].items():
        qid = qidx.get(title, slug(title))
        if st == "done":
            done.append(qid)
        else:
            quests[qid] = int(st)
    ws = WorldState(game_time=preset["time"], quests=quests, completed_quests=done,
                    player={"race": rng.choice(cfg["player_races"]),
                            "gender": rng.choice(cfg["player_genders"]),
                            "name": rng.choice(cfg["player_names"])})
    for e in effects:
        if gate_ok(e["trigger"], ws):
            ws.variables[e["variable"]] = e["value"]
    return ws


class ScenarioBuilder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = random.Random(cfg["seed"])
        self.R = Retriever()
        self.npcs = {p["npc_id"]: p for p in read_jsonl(CORPUS / "c2_profiles" / "npcs.jsonl")}
        self.qidx = json.loads((CORPUS / "c3_quests" / "quest_index.json").read_text(encoding="utf-8"))
        self.quests = {q["quest_id"]: q for q in read_jsonl(CORPUS / "c3_quests" / "quests.jsonl")}
        self.effects = [e.model_dump() if hasattr(e, "model_dump") else e
                        for q in self.quests.values() for e in q.get("world_effects", [])]
        self.tpl = load_yaml(CURATION / "player_templates.yaml")
        self.presets = cfg["world_presets"]
        ents = list(read_jsonl(CORPUS / "c1_world" / "entities.jsonl"))
        self.lore_entities = [e["name"] for e in ents
                              if e["type"] in {"deity", "race", "faction", "event", "creature", "place", "lore"}
                              and e["entity_id"].startswith("lore.") and len(e["name"]) < 40]
        self.skyrim_places = [e["name"] for e in ents if e["type"] == "place"
                              and e["entity_id"].startswith("skyrim.") and len(e["name"]) < 40]

    # ------------------------------------------------------------ helpers
    def world(self) -> tuple[str, WorldState]:
        names = list(self.presets)
        name = self.rng.choices(names, weights=[self.presets[n].get("weight", 1) for n in names])[0]
        return name, make_world(self.presets[name], self.qidx, self.effects, self.rng, self.cfg)

    @staticmethod
    def query_for(behavior: str, utt: str, info: dict, npc: dict) -> str:
        """Retrieval query the runtime would issue (entity-focused, not the raw utterance)."""
        if info.get("entity"):
            return info["entity"]
        if info.get("about_npc"):
            return info["about_npc"]
        if info.get("place"):
            return info["place"]
        if behavior in {"self_identity", "occupation_service", "restricted_info", "memory_recall",
                        "rumor", "player_race_reaction", "multi_turn"}:
            return f"{npc['name']} {npc.get('workplace') or ''} {info.get('topic', '')}".strip()
        if info.get("topic"):
            return info["topic"]
        if behavior == "world_state_reaction":
            return f"Whiterun {utt}"
        return utt

    def context(self, npc: dict, ws: WorldState, query: str, k: int | None = None) -> list[dict]:
        k = k or self.cfg["retrieval_k"]
        ctx = self.R.search(query, npc, ws, k=k, min_rel=0.55)
        if self.rng.random() < self.cfg["distractor_prob"]:
            d = self.R.search(self.rng.choice(self.lore_entities + self.skyrim_places), npc, ws, k=1)
            if d and d[0]["chunk_id"] not in {c["chunk_id"] for c in ctx}:
                ctx.insert(self.rng.randrange(len(ctx) + 1), {**d[0], "distractor": True})
        return ctx

    def pick_entity(self, npc: dict, ws: WorldState, known: bool, tries: int = 25) -> str | None:
        pool = self.lore_entities if known or self.rng.random() < 0.6 else self.skyrim_places
        for _ in range(tries):
            ent = self.rng.choice(pool)
            vis = self.R.search(ent, npc, ws, k=1)
            hit = bool(vis) and ent.lower() in vis[0]["title"].lower()
            if known and hit:
                return ent
            if not known:
                inv = self.R.search(ent, npc, ws, k=1, invisible=True)
                if not hit and inv and ent.lower() in inv[0]["title"].lower():
                    return ent
        return None

    def fill(self, behavior: str, npc: dict, ws: WorldState) -> tuple[str, dict] | None:
        t = self.rng.choice(self.tpl[behavior])
        info: dict = {}
        others = [p for p in self.npcs.values() if p["npc_id"] != npc["npc_id"]]
        if "{npc}" in t:
            info["about_npc"] = self.rng.choice(others)["name"]
            t = t.replace("{npc}", info["about_npc"])
        if "{place}" in t:
            info["place"] = self.rng.choice(self.cfg["whiterun_places"])
            t = t.replace("{place}", info["place"])
        if "{workplace}" in t:
            t = t.replace("{workplace}", npc.get("workplace") or "your work")
        if "{opinion_topic}" in t:
            topics = list(npc.get("opinions", {})) or self.tpl["opinion_topics"]
            if self.rng.random() < 0.3:
                topics = self.tpl["opinion_topics"]
            info["topic"] = self.rng.choice(topics)
            t = t.replace("{opinion_topic}", info["topic"])
        if "{entity}" in t:
            ent = self.pick_entity(npc, ws, known=(behavior == "world_lore"))
            if not ent:
                return None
            info["entity"] = ent
            t = t.replace("{entity}", ent)
        if "{topic}" in t:
            if behavior == "restricted_info":
                topics = self.tpl["restricted_topics"].get(npc["npc_id"])
                if not topics:
                    return None
                info["topic"] = self.rng.choice(topics)
                t = t.replace("{topic}", info["topic"])
            elif behavior == "quest_aware":
                qs = [q for q in self.quests.values() if npc["npc_id"] in q.get("involved_npcs", [])
                      and "whiterun" in q.get("scope_tags", [])]
                if not qs:
                    return None
                q = self.rng.choice(qs)
                info["quest_id"] = q["quest_id"]
                ws.quests.setdefault(q["quest_id"], q["stages"][0]["stage"] if q["stages"] else 10)
                info["topic"] = q["name"]
                t = t.replace("{topic}", q["name"])
        return t, info

    # -------------------------------------------------------------- build
    def build_one(self, npc: dict, behavior: str, i: int) -> dict | None:
        preset, ws = self.world()
        filled = self.fill(behavior, npc, ws)
        if not filled:
            return None
        utt, info = filled
        memories = []
        if behavior == "memory_recall" and self.rng.random() < 0.65:
            kind, txt, imp, val = self.rng.choice(MEMORY_BANK)
            memories = [txt.format(topic=self.rng.choice(self.cfg["whiterun_places"]))]
            info["memory_kind"] = kind
        elif behavior == "memory_recall":
            info["memory_kind"] = "none"   # must NOT pretend to remember
        query = self.query_for(behavior, utt, info, npc)
        if behavior in {"smalltalk", "out_of_world", "hostile_player"}:
            ctx = []          # nothing useful to retrieve; teaches not to force facts
        else:
            ctx = self.context(npc, ws, query)
        events = active_events(ws, self.effects, npc)
        system = build_system_prompt(npc, ws, ctx, memories=memories, events=events,
                                     n_style_examples=self.cfg["style_examples"], rng=self.rng)
        secrets = [k["fact"] for k in npc.get("knowledge", []) if k.get("visibility") == "restricted"
                   and not (k.get("reveal_gate") and gate_ok(k["reveal_gate"], ws))]
        sid = f"{npc['npc_id']}.{behavior}.{i}.{short_hash(system, utt)}"
        return {
            "scenario_id": sid, "npc_id": npc["npc_id"], "behavior": behavior, "preset": preset,
            "world_state": ws.model_dump(), "player_turns": [utt], "retrieval_query": query,
            "context_chunk_ids": [c["chunk_id"] for c in ctx],
            "distractor_ids": [c["chunk_id"] for c in ctx if c.get("distractor")],
            "target_facts": [c["chunk_id"] for c in ctx if not c.get("distractor")][:2],
            "forbidden_facts": secrets, "memories": memories, "info": info,
            "expect_uncertainty": behavior == "unknown_info" or info.get("memory_kind") == "none",
            "system_prompt": system, "prompt_version": PROMPT_VERSION,
        }

    def assign_split(self, s: dict, tpl_idx: int) -> str:
        if s["npc_id"] in self.cfg["eval_holdout_npcs"]:
            return "eval"
        h = int(short_hash(s["scenario_id"], n=8), 16) / 16 ** 8
        if h < self.cfg["eval_fraction_seen"]:
            return "eval"
        if h < self.cfg["eval_fraction_seen"] + self.cfg["val_fraction"]:
            return "val"
        return "train"

    def run(self) -> list[dict]:
        out = []
        scale = self.cfg.get("scale", 1.0)
        for npc in self.npcs.values():
            for behavior, n in self.cfg["per_behavior"].items():
                for i in range(max(0, round(n * scale))):
                    s = self.build_one(npc, behavior, i)
                    if s:
                        s["split"] = self.assign_split(s, i)
                        out.append(s)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "c4_generation.yaml"))
    ap.add_argument("--out", default=str(CORPUS / "c4_dialogue" / "scenarios.jsonl"))
    ap.add_argument("--npc", nargs="*")
    a = ap.parse_args()
    b = ScenarioBuilder(load_yaml(Path(a.config)))
    if a.npc:
        b.npcs = {k: v for k, v in b.npcs.items() if k in a.npc}
    sc = b.run()
    n = write_jsonl(Path(a.out), sc)
    from collections import Counter
    print({"scenarios": n, "split": dict(Counter(s["split"] for s in sc)),
           "behaviors": dict(Counter(s["behavior"] for s in sc))})


if __name__ == "__main__":
    main()
