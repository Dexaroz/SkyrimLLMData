"""Corpus 1 — World Knowledge.

UESP cache -> documents.jsonl, chunks.jsonl, entities.jsonl, relations.jsonl

Key decisions
* Meta sections (bugs, notes, stats, merchandise...) are dropped and
  game-mechanic sentences are filtered, so the RAG never feeds an NPC things
  like "disposition", "level 4" or "PC version".
* Every chunk gets knowledge *scopes* (who in-world could know it) and, for
  quest walkthroughs, *quest gates* + spoiler flag. The NPC knowledge filter
  in the RAG works purely on this metadata.
* Chunks are self-contained ("Title — Section: text") for better retrieval.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from .common import CONFIGS, CORPUS, RAW, entity_id, load_yaml, n_tokens, short_hash, slug, write_jsonl
from .schemas import Entity, KnowledgeChunk, Provenance, QuestGate, Relation, Section, WorldDocument
from .uesp import UESPClient
from . import wikitext as W

LICENSE = "CC BY-SA 2.5"

# -------------------------------------------------------- classification
TYPE_RULES = [  # (category regex, entity type) — first match wins
    (r"^Skyrim-(Dawnguard-|Dragonborn-)?NPCs$|^Skyrim-.*NPCs$", "npc"),
    (r"^Skyrim-(Dawnguard-|Dragonborn-)?Quests", "quest"),
    (r"^Skyrim-(Dawnguard-|Dragonborn-)?Places", "place"),
    (r"^Skyrim-(Dawnguard-|Dragonborn-)?Creatures", "creature"),
    (r"^Skyrim-(Skills|Perks|Gameplay|Magic|Spells|Shouts|Alchemy|Smithing|Enchanting|Effects|Crafting|Ingredients|Achievements)", "mechanic"),
    (r"^Skyrim-.*(Items|Armor|Weapons|Clothing|Jewelry|Keys|Food|Potions|Soul Gems)", "item"),
    (r"^Skyrim-(Dawnguard-|Dragonborn-)?(Items-)?Books|^Lore-Books", "book"),
    (r"^Lore-(Gods|Deities|Daedra|Aedra|Divines)", "deity"),
    (r"^Lore-Races|^Skyrim-Races", "race"),
    (r"^Lore-Factions|^Skyrim-(Dawnguard-|Dragonborn-)?Factions-Other$", "faction"),
    (r"^Lore-Places", "place"),
    (r"^Lore-People", "npc"),
    (r"^Lore-(Events|History|Wars)", "event"),
    (r"^Lore-Creatures", "creature"),
    (r"^Lore-Artifacts|^Lore-Items", "item"),
    (r"^Lore-", "lore"),
]

META_SENT = re.compile(
    r"\b(disposition|console|ref ?id|base ?id|form ?id|editor ?id|leveled|level \d+|"
    r"health|magicka|stamina|perk|bug|glitch|patch|PC|Xbox|PS3|PS4|Switch|Creation Kit|"
    r"script|npc|player character|quest marker|fast[- ]travel|save game|achievement|"
    r"trophy|respawn|loading screen|unofficial|in-game|the game|this game|the player|"
    r"follower slot|essential|protected|aggression|confidence|AI package|dialogue option|"
    r"chance|level-dependent|leveled|damage|armor rating|drops?|loot|spawns?|"
    r"hit points|DLC|Hearthfire|Anniversary Edition|Special Edition)\b|\d+ ?%",
    re.I)


def classify(title: str, cats: list[str], box: str | None) -> str:
    if box == "npc summary" or box == "lore people summary":
        return "npc"
    if box == "quest header":
        return "quest"
    if box in {"place summary", "lore place summary"}:
        return "place"
    if box in {"creature summary"}:
        return "creature"
    if box in {"book summary", "lore book summary"}:
        return "book"
    for pat, t in TYPE_RULES:
        if any(re.search(pat, c) for c in cats):
            return t
    return "lore" if title.startswith("Lore:") else "other"


def is_noise(text: str) -> bool:
    """Table residue / stat lists: many pipes or digits, few real words."""
    n = max(len(text), 1)
    pipes = text.count("|") / n
    digits = sum(ch.isdigit() for ch in text) / n
    words = re.findall(r"[A-Za-z]{3,}", text)
    return pipes > 0.02 or digits > 0.12 or len(words) < 8


def drop_meta_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    keep = [p for p in parts if not META_SENT.search(p)]
    return " ".join(keep).strip()


def hold_of(infobox: dict[str, str]) -> str | None:
    for k in ("region", "hold"):
        if v := infobox.get(k):
            v = re.sub(r"\s*\(.*?\)", "", v).split(",")[0].strip()
            if v.endswith("Hold") or v in {"The Pale", "The Reach", "The Rift", "Eastmarch",
                                           "Falkreath Hold", "Haafingar", "Hjaalmarch",
                                           "Winterhold", "Solstheim"}:
                return v
    return None


MAJOR_PLACES = {"Whiterun", "Solitude", "Windhelm", "Riften", "Markarth", "Winterhold",
                "Falkreath", "Morthal", "Dawnstar", "Skyrim", "Tamriel", "Cyrodiil",
                "High Hrothgar", "Throat of the World", "Sovngarde", "Helgen", "Riverwood",
                "Dragonsreach", "Jorrvaskr", "College of Winterhold", "Solstheim"}


# Lore every inhabitant of Skyrim plausibly knows (lead section only).
COMMON_LORE = {
    "Akatosh", "Arkay", "Dibella", "Julianos", "Kynareth", "Mara", "Stendarr", "Talos", "Zenithar",
    "Nine Divines", "Eight Divines", "Azura", "Boethiah", "Clavicus Vile", "Hermaeus Mora", "Hircine",
    "Malacath", "Mehrunes Dagon", "Mephala", "Meridia", "Molag Bal", "Namira", "Nocturnal", "Peryite",
    "Sanguine", "Sheogorath", "Vaermina", "Daedric Princes", "Daedra", "Aedra", "Shor", "Alduin",
    "Sovngarde", "Dragonborn", "Dragon", "Dragons", "Nord", "Imperial", "Breton", "Redguard", "Dark Elf",
    "High Elf", "Wood Elf", "Orc", "Khajiit", "Argonian", "Falmer", "Dwemer", "Giant", "Tamriel",
    "Skyrim", "Cyrodiil", "Hammerfell", "High Rock", "Morrowind", "Black Marsh", "Elsweyr", "Valenwood",
    "Summerset Isles", "Imperial City", "Aldmeri Dominion", "Thalmor", "Great War", "Red Year",
    "Oblivion Crisis", "Septim Dynasty", "Tiber Septim", "Ysgramor", "Companions", "Greybeards", "Blades",
    "Dark Brotherhood", "Thieves Guild", "College of Winterhold", "Stormcloaks", "Imperial Legion",
    "White-Gold Concordat", "Markarth Incident", "Ulfric Stormcloak", "Elder Scrolls", "Oblivion",
    "Aetherius", "Masser", "Secunda", "Vampire", "Werewolf", "Draugr", "Dragon Priest", "Atmora",
    "Solstheim", "Red Mountain", "Throat of the World", "High Hrothgar", "Thu'um", "Way of the Voice",
}


def scopes_for(etype: str, title: str, ns: str, infobox: dict, section_kind: str, is_lead: bool,
               cats: tuple = ()) -> list[str]:
    name = W.norm_title(title.split(":", 1)[-1])
    base = re.sub(r"\s*\(.*\)$", "", name)
    if etype in {"mechanic", "item"} and ns == "Skyrim":
        return ["meta"]
    if etype == "other" and ns == "Skyrim":
        # in-game notes, letters, recipes, mechanics, platform pages...
        if base.endswith(" People"):
            return [f"local:{slug(base.removesuffix(' People'))}"]
        return ["meta"]
    if etype == "book":
        return ["scholarly"]
    if ns == "Lore":
        skyrim_place = etype == "place" and (infobox.get("province", "").startswith("Skyrim")
                                             or any(c.endswith("-Skyrim") for c in cats))
        if is_lead and (base in COMMON_LORE or skyrim_place):
            return ["common"]
        return ["scholarly"]
    if etype == "place":
        if base in MAJOR_PLACES or infobox.get("type", "").lower() in {"city", "town", "hold"}:
            return ["common"]
        h = hold_of(infobox)
        return [f"regional:{slug(h)}"] if h else ["common"]
    if etype == "npc":
        if re.search(r"\bjarl\b|\bhigh king\b|\bemperor\b|\bulfric\b|\btullius\b",
                     (name + " " + infobox.get("class", "")).lower()):
            return ["common"]
        city = re.sub(r"\s*\(.*?\)", "", infobox.get("city", "")).split(",")[0].strip()
        return [f"local:{slug(city)}"] if city else ["common"]
    if etype == "creature":
        return ["common"]
    if etype == "quest":
        return ["common"] if section_kind != "walkthrough" else ["quest_participants"]
    return ["common"]


def chunk_text(text: str, max_tok: int, overlap_tok: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if n_tokens(p) > max_tok:  # split long paragraph by sentences
            sents = re.split(r"(?<=[.!?])\s+", p)
        else:
            sents = [p]
        for s in sents:
            if cur and n_tokens(cur) + n_tokens(s) > max_tok:
                chunks.append(cur.strip())
                tail = cur[-overlap_tok * 4:]
                cur = tail[tail.find(" ") + 1:] + " " + s if overlap_tok else s
            else:
                cur = (cur + "\n" + s) if cur else s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ---------------------------------------------------------------- relations
REL_FIELDS = {
    "npc": {"race": "race", "city": "lives_in", "store": "works_at", "class": "occupation",
            "faction": "member_of"},
    "place": {"region": "located_in", "location": "located_in", "hold": "located_in"},
    "quest": {"giver": "quest_giver", "loc": "takes_place_in", "prev": "preceded_by",
              "next": "followed_by", "id": "editor_id"},
}


def relations_for(doc: WorldDocument) -> list[Relation]:
    rels = []
    subj = doc.doc_id
    for field, pred in REL_FIELDS.get(doc.entity_type, {}).items():
        if not (v := doc.infobox.get(field)):
            continue
        for obj in re.split(r";|,| or |<br>", v):
            obj = obj.strip(" .")
            if obj and len(obj) < 80:
                rels.append(Relation(subject=subj, predicate=pred, object=obj, doc_id=subj,
                                     evidence=f"infobox.{field}"))
    if doc.entity_type == "place":
        for k, v in doc.infobox.items():
            if re.fullmatch(r"job\d+", k):
                rels.append(Relation(subject=subj, predicate="has_worker", object=v, doc_id=subj,
                                     evidence=f"infobox.{k}"))
    return rels


# ---------------------------------------------------------------------- main
def build(cfg_path: Path, out_dir: Path, limit: int | None = None) -> dict:
    cfg = load_yaml(cfg_path)
    client = UESPClient(RAW / "uesp")
    excl_t = [re.compile(p) for p in cfg.get("exclude_title_patterns", [])]
    excl_c = tuple(cfg.get("exclude_categories_prefix", []))
    max_tok, ovl = cfg.get("chunk_max_tokens", 350), cfg.get("chunk_overlap_tokens", 40)

    docs, chunks, ents, rels = [], [], [], []
    stats = Counter()
    for ns in cfg["namespaces"]:
        for i, rec in enumerate(client.iter_cached(ns)):
            if limit and i >= limit:
                break
            title, cats = rec["title"], rec["categories"]
            if any(p.search(title) for p in excl_t) or (cats and all(c.startswith(excl_c) for c in cats)):
                stats["excluded_title"] += 1
                continue
            if any(c.startswith(("Skyrim-Creation Club",)) for c in cats):
                stats["excluded_cc"] += 1
                continue
            parsed = W.parse_page(rec["content"])
            etype = classify(title, cats, parsed["infobox_type"])
            prov = Provenance(source="uesp", url=rec["url"], page_title=title,
                              revision_id=rec["revision_id"], retrieved_at=rec["retrieved_at"],
                              license=LICENSE, transform="wikitext->clean text; meta removed")
            did = entity_id(title)
            sections = []
            for s in parsed["sections"]:
                kind = s["kind"]
                if kind == "walkthrough" and etype != "quest":
                    kind = "content"
                elif kind == "content" and etype == "quest":
                    kind = "walkthrough"      # any quest section beyond the lead is a spoiler
                text = s["text"]
                if kind == "lead" and not text and parsed["infobox"].get("description"):
                    text = parsed["infobox"]["description"]
                    if etype == "quest":
                        text = f"{title.split(':', 1)[1]} is a quest. Objective: {text}"
                sections.append(Section(heading=s["heading"], level=s["level"], kind=kind, text=text))
            doc = WorldDocument(doc_id=did, title=title, namespace=ns, entity_type=etype,
                                categories=cats, infobox=parsed["infobox"], sections=sections,
                                links=parsed["links"], provenance=prov)
            docs.append(doc)
            stats[f"type:{etype}"] += 1
            name = W.norm_title(title.split(":", 1)[1])
            aliases = list({name, re.sub(r"\s*\(.*\)$", "", name)})
            ents.append(Entity(entity_id=did, name=re.sub(r"\s*\(.*\)$", "", name), type=etype,
                               aliases=aliases, doc_id=did,
                               attributes={k: v for k, v in parsed["infobox"].items()
                                           if k not in {"image", "imgdesc", "icon"}}))
            rels += relations_for(doc)

            # ---- chunks
            for s, raw in zip(sections, parsed["sections"]):
                if s.kind in {"meta", "stages", "dialogue"} or not s.text:
                    continue
                text = drop_meta_sentences(s.text) if ns == "Skyrim" else s.text
                if n_tokens(text) < 12:
                    continue
                scopes = scopes_for(etype, title, ns, parsed["infobox"], s.kind, s.kind == "lead", tuple(cats))
                personal_story = False
                if etype == "npc" and ns == "Skyrim" and s.kind != "lead":
                    # quest branches / dialogue told on NPC pages: only the NPC
                    # itself knows its own story; treated as spoiler.
                    quotes = len(re.findall(r'"[^"]{8,}"', text))
                    second = len(re.findall(r"\byou(r)?\b", text, re.I))
                    if quotes >= 1 or second >= 2:
                        scopes, personal_story = [f"personal:{slug(name)}"], True
                if scopes == ["meta"]:
                    stats["chunks_meta_skipped"] += 1
                    continue
                gates, spoiler = [], personal_story
                if s.kind == "walkthrough":
                    qid = parsed["infobox"].get("id") or slug(name)
                    gates, spoiler = [QuestGate(quest_id=qid)], True
                sec_links = [entity_id(l) for l in W.wiki_links(raw["raw"])
                             if l.startswith(("Skyrim:", "Lore:"))]
                for j, piece in enumerate(chunk_text(text, max_tok, ovl)):
                    if is_noise(piece):
                        stats["chunks_noise_skipped"] += 1
                        continue
                    header = f"{name} — {s.heading}" if s.heading != "Overview" else name
                    body = f"{header}: {piece}"
                    chunks.append(KnowledgeChunk(
                        chunk_id=f"{did}#{slug(s.heading)}-{j}-{short_hash(body, n=6)}",
                        doc_id=did, title=name, section=s.heading, text=body,
                        n_tokens=n_tokens(body), entity_type=etype,
                        entities=list(dict.fromkeys(sec_links))[:30], scopes=scopes,
                        quest_gates=gates, spoiler=spoiler, provenance=prov))
    # sections titled after a quest (on NPC/place pages) are quest spoilers
    quest_names = {}
    for d in docs:
        if d.entity_type == "quest":
            quest_names[W.norm_title(d.title.split(":", 1)[1])] = d.infobox.get("id") or slug(d.title.split(":", 1)[1])
    for c in chunks:
        q = quest_names.get(c.section)
        if q and c.entity_type != "quest":
            qid = re.split(r"[;,\s]+", q)[0]
            if not any(g.quest_id == qid for g in c.quest_gates):
                c.quest_gates.append(QuestGate(quest_id=qid))
            c.spoiler = True
            stats["chunks_quest_section_gated"] += 1

    # resolve relation objects to entity ids when possible
    by_name = {}
    for e in ents:
        for a in e.aliases:
            by_name.setdefault(a.lower(), e.entity_id)
    for r in rels:
        r.object = by_name.get(r.object.lower(), r.object)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "documents": write_jsonl(out_dir / "documents.jsonl", docs),
        "chunks": write_jsonl(out_dir / "chunks.jsonl", chunks),
        "entities": write_jsonl(out_dir / "entities.jsonl", ents),
        "relations": write_jsonl(out_dir / "relations.jsonl", rels),
    }
    scope_counts = Counter(s.split(":")[0] for c in chunks for s in c.scopes)
    return {**counts, **dict(stats), "scopes": dict(scope_counts),
            "chunk_tokens": sum(c.n_tokens for c in chunks)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "scope_whiterun.yaml"))
    ap.add_argument("--out", default=str(CORPUS / "c1_world"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(build(Path(a.config), Path(a.out), a.limit))
