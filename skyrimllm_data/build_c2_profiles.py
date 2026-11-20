"""Corpus 2 — NPC Profiles.

UESP NPC pages (+ Mantella bios as secondary evidence) + curated overlays
-> c2_profiles/npcs.jsonl.

Layers (later wins):  auto (UESP infobox/lead/dialogue)  <  curated overlay.
Each NPC gets knowledge *scopes* that match the chunk scopes of Corpus 1,
so the RAG filter is simply `chunk.scopes ∩ npc.scopes ≠ ∅` (+ quest gates).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import requests

from .build_c1_world import drop_meta_sentences
from .common import CONFIGS, CORPUS, CURATION, RAW, load_yaml, npc_id, slug, write_jsonl
from .schemas import DialogueLine, KnowledgeItem, NPCProfile, Provenance, Relationship
from .build_c3_quests import resolve_gate
from .uesp import UESPClient
from . import wikitext as W

MANTELLA_URL = "https://raw.githubusercontent.com/art-from-the-machine/Mantella/main/data/Skyrim/skyrim_characters.csv"

# Editor-id factions that are engine plumbing, not in-world affiliations.
FACTION_NOISE = re.compile(r"^(Crime|Town|Services|Favor|Job|Dialogue|Potential|Disliked|"
                           r"Player|Merchant|Is|Has|Bard|Vampire|Werewolf|Creature|Prey|"
                           r"Predator|Generic|MQ|MS|DA|DLC|CW|WE|WI|Bandit|.*Faction$)")
FACTION_MAP = {
    "CompanionsFaction": "Companions", "CompanionsCircle": "The Circle (Companions)",
    "CWImperialFaction": "Imperial Legion", "CWSonsFaction": "Stormcloaks",
    "GrayManeFaction": "Gray-Mane family", "BattleBornFaction": "Battle-Born family",
    "WhiterunJarlFaction": "Court of Whiterun", "TempleofKynarethFaction": "Temple of Kynareth",
    "ThalmorFaction": "Thalmor",
}
SCHOLARLY = re.compile(r"wizard|mage|priest|priestess|scholar|bard|steward|jarl|court|alchemist|"
                       r"spellsword|healer|sorcer|apothecar|librarian", re.I)


def get_scope_npcs(client: UESPClient, cfg: dict) -> list[str]:
    cache = RAW / "uesp" / f"_scope_{cfg['scope_id']}_npcs.json"
    if cache.exists():
        titles = json.loads(cache.read_text(encoding="utf-8"))
    else:
        titles = []
        for cat in cfg.get("npc_categories", []):
            titles += client.category_members(cat)
        cache.write_text(json.dumps(titles, indent=1), encoding="utf-8")
    titles += cfg.get("npc_extra", [])
    excl = set(cfg.get("npc_exclude", []))
    return [t for t in dict.fromkeys(titles) if t.startswith("Skyrim:") and t not in excl]


def load_mantella() -> dict[str, dict]:
    path = RAW / "mantella" / "skyrim_characters.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(MANTELLA_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    rows = csv.DictReader(path.open(encoding="utf-8-sig", errors="replace"))
    return {r["name"].strip().lower(): r for r in rows if r.get("name")}


FACTION_ALLOW = re.compile(r"Companions|Circle|Thieves|Dark Brotherhood|College|Bards|Blades|"
                           r"Greybeards|Silver Hand|Vigilant|Dawnguard|Thalmor|Legion|Stormcloak|"
                           r"Family|Dragonsreach|Temple|Priest|Court", re.I)


def readable_factions(raw: str | None) -> list[str]:
    """Keep only in-world affiliations (UESP lists many engine factions)."""
    out = []
    for f in re.split(r";", raw or ""):
        f = f.strip()
        if not f:
            continue
        if f in FACTION_MAP:
            out.append(FACTION_MAP[f])
        elif " " in f and FACTION_ALLOW.search(f) and not re.search(r"Government|Services|Vendor|alias|During|,|\.", f):
            out.append(f)
    return list(dict.fromkeys(out))


# Sentences in NPC summaries that describe game mechanics or the player.
OOC_SENT = re.compile(r"\byou(r|rself)?\b|\bif (s?he|they|\w+) (is|are) killed\b|\bdies\b|"
                      r"never actually|\bthe game\b|\bbugg?\w*|\bscript|\bquests?\b|\bappears?\b|"
                      r"\bcan (usually )?be found\b|\bmarker\b|\bfollower\b|\brespawn|\b\d{1,2} ?(am|pm)\b|"
                      r"\bgoes to bed\b|\bwakes up\b|\bmidnight\b|\bschedule\b", re.I)
BAD_OCC = re.compile(r"^(and|who|living|located|the)\b|\b(located|living|who)$")
RACES = r"(?:Nord|Imperial|Breton|Redguard|Dunmer|Dark Elf|Altmer|High Elf|Bosmer|Wood Elf|Orc|Orsimer|Khajiit|Argonian)"


def occupation_from_summary(summary: str) -> str | None:
    m = re.search(rf"\b(?:is|,) (?:an? |the )(?:(?:young|old|elderly|wealthy|poor|homeless) )?(?:{RACES} )?"
                  r"([a-z][a-z' -]{2,40}?)(?= who\b| of\b| in\b| at\b| and\b| living\b|,|\.)", summary)
    if not m:
        return None
    occ = re.sub(r"\s+(located|living|based)$", "", m.group(1).strip())
    return None if BAD_OCC.search(occ) else occ


def lead_summary(sections: list[dict]) -> str:
    lead = next((s["text"] for s in sections if s["kind"] == "lead"), "")
    text = drop_meta_sentences(lead.replace("\n", " "))
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    text = " ".join(x for x in sents if not OOC_SENT.search(x))
    # first ~900 chars, sentence-aligned
    if len(text) > 900:
        cut = text[:900].rfind(". ")
        text = text[:cut + 1] if cut > 200 else text[:900]
    return text


def build(cfg_path: Path, out_dir: Path) -> dict:
    cfg = load_yaml(cfg_path)
    client = UESPClient(RAW / "uesp")
    titles = get_scope_npcs(client, cfg)
    client.fetch(titles)
    overlays = load_yaml(CURATION / "npc_overlays.yaml")
    effects = load_yaml(CURATION / "world_effects.yaml").get("effects", [])
    qidx = json.loads((CORPUS / "c3_quests" / "quest_index.json").read_text(encoding="utf-8"))
    mantella = load_mantella()
    settle, hold = cfg["settlement"], cfg["hold"]

    profiles, missing = [], []
    for title in titles:
        rec = client.load(title)
        if not rec:
            missing.append(title)
            continue
        parsed = W.parse_page(rec["content"])
        box = parsed["infobox"]
        name = title.split(":", 1)[1]
        nid = npc_id(title)
        prov = [Provenance(source="uesp", url=rec["url"], page_title=title,
                           revision_id=rec["revision_id"], retrieved_at=rec["retrieved_at"],
                           license="CC BY-SA 2.5", transform="NPC Summary + lead + dialogue parsed")]
        # dialogue samples (quoted game lines)
        dlg = []
        for s in parsed["sections"]:
            if s["kind"] == "dialogue" and s["heading"].lower().startswith("dialogue"):
                for d in W.parse_dialogue(s["raw"]):
                    dlg.append(DialogueLine(**d, provenance=Provenance(
                        source="game_dialogue_quotes", url=rec["url"], page_title=title,
                        revision_id=rec["revision_id"], license="© Bethesda (quoted on UESP)",
                        redistributable=False)))
        related = []
        for s in parsed["sections"]:
            if s["heading"].lower().startswith("related quest"):
                related += [qidx[l] for l in W.wiki_links(s["raw"]) if l in qidx]
        summary = lead_summary(parsed["sections"])
        occupation = occupation_from_summary(summary) or box.get("class")
        city = re.sub(r"\s*\(.*?\)", "", box.get("city", "")).strip() or settle
        prof = NPCProfile(
            npc_id=nid, name=name,
            form_ids={k: box[k] for k in ("refid", "baseid", "class") if k in box},
            race=box.get("race"), gender=box.get("gender"), occupation=occupation,
            settlement=city, hold=hold, home=box.get("home"), workplace=box.get("store"),
            factions=readable_factions(box.get("faction")),
            essential=bool(box.get("essential", "").strip()),
            summary=summary,
            quests=list(dict.fromkeys(related)),
            dialogue_samples=dlg[:40], provenance=prov)
        if m := mantella.get(name.lower()):
            prof.bio_secondary = re.sub(r"\s+", " ", m.get("bio", "")).strip()[:1500] or None
            prof.provenance.append(Provenance(source="mantella_characters", url=MANTELLA_URL,
                                              page_title=name, license="AGPL-3.0"))
        # scopes
        scopes = ["common", f"local:{slug(city)}", f"regional:{slug(hold)}", f"personal:{nid}"]
        if SCHOLARLY.search(f"{occupation} {box.get('class', '')}"):
            scopes.append("scholarly")
        # curated overlay
        if ov := overlays.get(nid):
            prof.occupation = ov.get("occupation", prof.occupation)
            prof.personality = ov.get("personality", [])
            prof.speech_style = ov.get("speech_style")
            prof.opinions = ov.get("opinions", {})
            prof.relationships = [Relationship(**r) for r in ov.get("relationships", [])]
            prof.knowledge = [KnowledgeItem(fact=k["fact"], visibility=k.get("visibility", "public"),
                                            reveal_gate=resolve_gate(k.get("reveal_gate"), qidx),
                                            source_doc=ov.get("source"))
                              for k in ov.get("knowledge", [])]
            scopes += ov.get("extra_scopes", [])
            prof.review_status = "curated"
            prof.provenance.append(Provenance(source="curated", page_title=ov.get("source"),
                                              license="CC BY-SA 4.0"))
        for clan in ("Battle-Born", "Gray-Mane"):
            if name.endswith(clan) or name in {"Bergritte Battle-Born"}:
                prof.factions.append(f"{clan} family")
        for f in prof.factions:
            scopes.append(f"faction:{slug(f)}")
        prof.scopes = list(dict.fromkeys(scopes))
        # state variants from world effects that affect this NPC
        for e in effects:
            if nid in e.get("affected_npcs", []):
                prof.state_variants.append({
                    "condition": {"quest_id": qidx.get(e["quest"], e["quest"]),
                                  "completed": e.get("completed", False)},
                    "change": e["description"]})
        profiles.append(prof)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_dir / "npcs.jsonl", profiles)
    return {"npcs": n, "curated": sum(p.review_status == "curated" for p in profiles),
            "with_dialogue": sum(bool(p.dialogue_samples) for p in profiles),
            "dialogue_lines": sum(len(p.dialogue_samples) for p in profiles),
            "with_mantella_bio": sum(bool(p.bio_secondary) for p in profiles),
            "missing_pages": missing,
            "overlays_unmatched": sorted(set(overlays) - {p.npc_id for p in profiles})}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "scope_whiterun.yaml"))
    ap.add_argument("--out", default=str(CORPUS / "c2_profiles"))
    a = ap.parse_args()
    print(build(Path(a.config), Path(a.out)))
