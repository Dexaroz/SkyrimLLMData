"""Corpus 4 — seed SFT examples from real in-game dialogue quoted on UESP.

Each quoted NPC line becomes a fully contextualised example: the condition
text on the wiki ("When Whiterun is under control of the Stormcloaks",
"If you are a Dunmer...") is mapped to a world-state preset / player race, the
player topic becomes the user turn, and the system prompt is built exactly as
for synthetic data (the line itself is removed from the style examples).

These are short, authentic style anchors. Licence: quoted Bethesda dialogue
-> redistributable=False (kept out of any public release).
"""
from __future__ import annotations

import json
import random
import re

from .build_c4_scenarios import ScenarioBuilder, make_world
from .common import CONFIGS, CORPUS, load_yaml, short_hash, write_jsonl
from .prompting import PROMPT_VERSION, active_events, build_system_prompt
from .quality import validate
from .schemas import ChatMessage, Provenance, SFTExample

SKIP = re.compile(r"combat|attack|follower|married|spouse|vampire|werewolf|if you (steal|kill|murder)|"
                  r"bleed|crime|bounty|sleep|sitting|kitchen|dead|dies|killed|brawl|pickpocket|"
                  r"wear|wearing|draw|weapon out|lockpick|trespass|shout|cast|spell", re.I)
GREETINGS = ["Hello.", "Hello there.", "Greetings.", "Good day.", "Hi.", "*approaches*"]
FAREWELLS = ["Goodbye.", "Farewell.", "I'll be going.", "See you around."]
RACE_WORDS = {"dunmer": "Dark Elf", "dark elf": "Dark Elf", "altmer": "High Elf", "high elf": "High Elf",
              "bosmer": "Wood Elf", "wood elf": "Wood Elf", "khajiit": "Khajiit", "argonian": "Argonian",
              "orc": "Orc", "orsimer": "Orc", "redguard": "Redguard", "breton": "Breton",
              "imperial": "Imperial", "nord": "Nord"}


def classify(cond: str | None, player: str | None) -> tuple[str, str] | None:
    c = (cond or "").lower()
    if cond and SKIP.search(cond):
        return None
    if player:
        return "game_topic", player
    if re.search(r"greet|hello|approach|first meet|talk to (him|her)", c):
        return "smalltalk", random.choice(GREETINGS)
    if re.search(r"ending|goodbye|farewell|leave|end the conversation|end conversation", c):
        return "smalltalk", random.choice(FAREWELLS)
    if re.search(r"rumou?r|gossip|job|work", c):
        return "rumor", random.choice(["Heard any rumours?", "Any news?", "Got any work?"])
    return None


def preset_for(cond: str | None) -> str | None:
    c = (cond or "").lower()
    if "stormcloak" in c and re.search(r"control|rule|captured|taken|side", c):
        return "stormcloak_whiterun"
    if "imperial" in c and re.search(r"control|rule|captured|taken|side", c):
        return "imperial_whiterun"
    if re.search(r"dragonborn|dragon rising|thane", c):
        return "dragonborn"
    if re.search(r"companion|circle", c):
        return "companion"
    return None


def build() -> dict:
    cfg = load_yaml(CONFIGS / "c4_generation.yaml")
    b = ScenarioBuilder(cfg)
    rng = random.Random(cfg["seed"] + 1)
    random.seed(cfg["seed"] + 1)
    out, skipped = [], 0
    for npc in b.npcs.values():
        for d in npc.get("dialogue_samples", []):
            cls = classify(d.get("condition"), d.get("player"))
            if not cls:
                skipped += 1
                continue
            behavior, user = cls
            pname = preset_for(d.get("condition")) or rng.choice(["arrival", "dragonborn"])
            ws = make_world(cfg["world_presets"][pname], b.qidx, b.effects, rng, cfg)
            low = (d.get("condition") or "").lower()
            for w, race in RACE_WORDS.items():
                if re.search(rf"\b(if|when) you('re| are)\b.*\b{w}\b", low):
                    ws.player["race"] = race
                    break
            npc_view = {**npc, "dialogue_samples": [x for x in npc["dialogue_samples"] if x["npc"] != d["npc"]]}
            query = d.get("player") or f"{npc['name']} {npc.get('workplace') or ''}"
            ctx = b.context(npc, ws, query) if behavior != "smalltalk" else []
            system = build_system_prompt(npc_view, ws, ctx, events=active_events(ws, b.effects, npc),
                                         n_style_examples=cfg["style_examples"], rng=rng)
            sc = {"behavior": behavior, "player_turns": [user], "forbidden_facts": []}
            val = validate(d["npc"], sc, system)
            split = "eval" if npc["npc_id"] in cfg["eval_holdout_npcs"] else (
                "val" if int(short_hash(d["npc"], n=6), 16) % 20 == 0 else "train")
            out.append(SFTExample(
                example_id=f"uesp.{npc['npc_id']}.{short_hash(d['npc'])}", npc_id=npc["npc_id"],
                behavior=behavior,
                messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=user),
                          ChatMessage(role="assistant", content=d["npc"])],
                context_chunk_ids=[c["chunk_id"] for c in ctx], origin="uesp_dialogue",
                quality={"flags": val["flags"], "condition": d.get("condition"), "preset": pname,
                         "split": split, "prompt_version": PROMPT_VERSION},
                provenance=[Provenance(**d["provenance"])]))
    path = CORPUS / "c4_dialogue" / "seed_uesp_dialogue.jsonl"
    n = write_jsonl(path, out)
    from collections import Counter
    return {"seed_examples": n, "skipped_lines": skipped,
            "split": dict(Counter(e.quality["split"] for e in out)),
            "behaviors": dict(Counter(e.behavior for e in out))}


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
