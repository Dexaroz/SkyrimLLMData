"""Automatic validation of NPC replies (Corpus 4 + evaluation).

Each check returns a flag; `validate` returns {"ok": bool, "flags": [...],
"scores": {...}}. Hard flags reject the example, soft flags are kept for
human review sampling.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .common import CORPUS, CURATION, load_yaml, read_jsonl

META = re.compile(r"\b(AI|artificial intelligence|language model|AI assistant|chatbot|NPCs?|the player|"
                  r"video ?games?|the game|this game|quest marker|quest log|level up|console command|save file|"
                  r"respawn|hit points|HP|XP|Bethesda|Skyrim Special Edition|instructions|"
                  r"notes? \[\d|\[\d+\])\b", re.I)
ANACHRONISM = re.compile(r"\b(internet|phone|computer|email|website|electricity|car|airplane|"
                         r"television|TV|gun|python|code|software|America|New York|Earth|"
                         r"okay|OK|cool|awesome|dude|guys)\b", re.I)
HEDGE = re.compile(r"(don't|do not|never|can't|cannot|couldn't|wouldn't) (know|say|tell|heard|recall|"
                   r"remember|be sure)|no idea|not sure|never heard|haven't heard|beyond me|"
                   r"ask (someone|the|at)|only (heard|rumou?rs)|not something I know|"
                   r"wouldn't know|can't help you there|don't believe we've met|don't recall|"
                   r"never met|first time I'?ve seen", re.I)
NARRATION = re.compile(r"\*[^*]{25,}\*|\([^)]{40,}\)")
SPEAKER_PREFIX = re.compile(r"^\s*[A-Z][\w' -]{1,30}:\s")

# Proper nouns always allowed (common Skyrim vocabulary) — extended at runtime
# with every entity name in Corpus 1.
BASE_ALLOWED = set("""I I'm I've I'll I'd Skyrim Tamriel Whiterun Nord Nords Imperial Imperials
Empire Stormcloak Stormcloaks Talos Kynareth Akatosh Mara Dibella Arkay Julianos Zenithar
Stendarr Divines Daedra Sovngarde Shor Ysmir Dragonborn Jarl Thane Thalmor Septim Septims
Gods Divine Aye Oh Ah Well Yes No Hmm Ha Hah Now Then So But And Still Listen Look""".split())


@lru_cache(maxsize=1)
def known_names() -> set[str]:
    names = set(BASE_ALLOWED)
    p = CORPUS / "c1_world" / "entities.jsonl"
    if p.exists():
        for e in read_jsonl(p):
            for a in e.get("aliases", []) + [e["name"]]:
                for w in re.findall(r"[A-Z][\w'-]+", a):
                    names.add(w)
    return names


@lru_cache(maxsize=1)
def secret_terms() -> dict[str, list[str]]:
    rules = load_yaml(CURATION / "secrets.yaml").get("rules", {})
    return {rid: (r["terms"], r["context"]) for rid, r in rules.items()}


def unknown_proper_nouns(reply: str, context: str) -> list[str]:
    caps = [re.sub(r"'s$", "", w) for w in re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z][\w'-]+)", reply)]
    ctx_words = set(re.findall(r"[A-Z][\w'-]+", context))
    known = known_names()
    return [w for w in dict.fromkeys(caps) if w not in ctx_words and w not in known]


def ngram_set(text: str, n: int = 3) -> set[tuple]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def jaccard(a: str, b: str) -> float:
    A, B = ngram_set(a), ngram_set(b)
    return len(A & B) / max(1, len(A | B))


def validate(reply: str, scenario: dict, system_prompt: str) -> dict:
    flags, soft = [], []
    words = len(reply.split())
    if words < 2:
        flags.append("too_short")
    if words > 140 and scenario.get("behavior") != "multi_turn":
        flags.append("too_long")
    if META.search(reply) and scenario.get("behavior") != "out_of_world":
        flags.append("meta_reference")
    if scenario.get("behavior") == "out_of_world" and re.search(r"\b(yes,? I am an AI|as an AI|language model)\b", reply, re.I):
        flags.append("broke_character")
    if ANACHRONISM.search(reply) and scenario.get("behavior") != "out_of_world":
        soft.append("anachronism")
    if SPEAKER_PREFIX.match(reply):
        flags.append("speaker_prefix")
    if NARRATION.search(reply):
        soft.append("long_narration")
    # knowledge restriction: forbidden facts / secret terms
    low = reply.lower()
    for fact in scenario.get("forbidden_facts", []):
        key = [w for w in re.findall(r"[A-Z][\w'-]{3,}", fact) if w.lower() not in {"she", "the", "her", "his"}]
        if any(k.lower() in low for k in key if k.lower() not in system_prompt.lower().split("## secrets")[0].lower()):
            flags.append("secret_leak")
            break
    for rid, (terms, ctx) in secret_terms().items():
        if any(t.lower() in low for t in terms) and any(c.lower() in low for c in ctx)                 and not any(t.lower() in system_prompt.lower() for t in terms):
            soft.append(f"possible_leak:{rid}")
    if scenario.get("expect_uncertainty") and not HEDGE.search(reply):
        flags.append("missing_uncertainty")
    unk = unknown_proper_nouns(reply, system_prompt + " " + " ".join(scenario.get("player_turns", [])))
    if unk:
        (flags if len(unk) >= 2 else soft).append("unsupported_names:" + ",".join(unk[:5]))
    return {"ok": not flags, "flags": flags + soft, "hard": flags}


def dedup(examples: list[dict], key=lambda e: e["messages"][-1]["content"], thr: float = 0.8) -> list[dict]:
    kept, by_npc = [], {}
    for e in examples:
        prev = by_npc.setdefault(e.get("npc_id"), [])
        txt = key(e)
        if any(jaccard(txt, p) >= thr for p in prev):
            continue
        prev.append(txt)
        kept.append(e)
    return kept
