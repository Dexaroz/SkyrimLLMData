"""Import dialogue extracted from the user's own Skyrim copy (style data).

Supported inputs
  --format alpaca   Mantella-LLM-Fine-Tuning `skyrim_alpaca_style_dataset.json`
                    ([{"instruction","input","output"}])
  --format csv      Any CSV export (xEdit / Lazy Voice Finder). Choose columns:
                    --prompt-col, --response-col, optional --speaker-col,
                    --voice-col.

Lines from in-scope NPCs get the full NPC context; others get a minimal
"generic Skyrim inhabitant" sheet (voice type), which teaches tone/style only.
Output: data/corpus/c4_dialogue/external/<name>.jsonl  (redistributable=False)

    python -m skyrimllm_data.import_external --format alpaca --input path/to/skyrim_alpaca_style_dataset.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from .common import CORPUS, npc_id, read_jsonl, short_hash, write_jsonl
from .prompting import RULES
from .quality import dedup
from .schemas import ChatMessage, Provenance, SFTExample

LICENSE = "© Bethesda Softworks — extracted from user's own copy; research use only"


def generic_sheet(voice: str | None) -> str:
    who = re.sub(r"(?<!^)(?=[A-Z])", " ", voice or "").strip().lower() or "inhabitant"
    return (f"You are a {who} living in Skyrim. Stay in character at all times.\n\n"
            + RULES.format(name="this person", era="4E 201"))


def clean(t: str) -> str:
    t = re.sub(r"\{[^}]*\}|\[[^\]]*\]", "", t or "")      # stage directions / tags
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["alpaca", "csv"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--prompt-col")
    ap.add_argument("--response-col")
    ap.add_argument("--speaker-col")
    ap.add_argument("--voice-col")
    ap.add_argument("--max", type=int, default=20000)
    a = ap.parse_args()
    npcs = {p["name"].lower(): p for p in read_jsonl(CORPUS / "c2_profiles" / "npcs.jsonl")}
    rows = []
    if a.format == "alpaca":
        for r in json.loads(Path(a.input).read_text(encoding="utf-8")):
            rows.append({"prompt": r.get("input", ""), "response": r.get("output", ""), "speaker": None, "voice": None})
    else:
        with open(a.input, encoding="utf-8-sig", errors="replace") as f:
            for r in csv.DictReader(f):
                rows.append({"prompt": r.get(a.prompt_col, ""), "response": r.get(a.response_col, ""),
                             "speaker": r.get(a.speaker_col) if a.speaker_col else None,
                             "voice": r.get(a.voice_col) if a.voice_col else None})
    out = []
    random.seed(0)
    for r in rows:
        p, resp = clean(r["prompt"]), clean(r["response"])
        if len(resp.split()) < 2 or len(p) < 2:
            continue
        npc = npcs.get((r["speaker"] or "").lower())
        system = generic_sheet(r["voice"]) if not npc else None
        if npc:   # minimal but correct identity for in-scope NPCs
            system = (f"You are {npc['name']}, {npc.get('occupation') or 'a resident'} in Whiterun.\n"
                      f"- {npc.get('summary', '')}\n\n" + RULES.format(name=npc["name"], era="4E 201"))
        out.append(SFTExample(
            example_id=f"game.{short_hash(p, resp)}", npc_id=npc_id(npc["name"]) if npc else None,
            behavior="game_style",
            messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=p),
                      ChatMessage(role="assistant", content=resp)],
            origin="game_dialogue", quality={"split": "train"},
            provenance=[Provenance(source="game_dialogue_export", license=LICENSE, redistributable=False,
                                   transform=f"import_external --format {a.format}")]).model_dump(mode="json"))
        if len(out) >= a.max:
            break
    out = dedup(out, key=lambda e: e["messages"][-1]["content"], thr=0.9)
    name = a.name or Path(a.input).stem
    n = write_jsonl(CORPUS / "c4_dialogue" / "external" / f"{name}.jsonl", out)
    print({"imported": n, "from_rows": len(rows)})


if __name__ == "__main__":
    main()
