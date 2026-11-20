"""Assemble the final Instruction Dataset (SFT) from all Corpus 4 sources.

    python -m skyrimllm_data.assemble_sft            # research build (all sources)
    python -m skyrimllm_data.assemble_sft --public   # only redistributable sources

Output: data/corpus/sft/{train,val}.jsonl in chat format
  {"messages": [...], "meta": {...}}  — directly loadable by TRL/Unsloth/Axolotl.
Mixing: game-dialogue style data is capped relative to contextual data so the
model does not learn to ignore the context.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from .common import CORPUS, read_jsonl, write_jsonl
from .quality import dedup


def load_all() -> list[dict]:
    d = CORPUS / "c4_dialogue"
    rows = []
    if (d / "seed_uesp_dialogue.jsonl").exists():
        rows += list(read_jsonl(d / "seed_uesp_dialogue.jsonl"))
    for split in ("train", "val"):
        p = d / "generated" / f"{split}.jsonl"
        if p.exists():
            rows += list(read_jsonl(p))
    for p in sorted((d / "external").glob("*.jsonl")) if (d / "external").exists() else []:
        rows += list(read_jsonl(p))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--max-style-ratio", type=float, default=0.25,
                    help="max share of context-free game_style examples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    rows = load_all()
    if a.public:
        rows = [r for r in rows if all(p.get("redistributable", True) for p in r.get("provenance", []))]
    rows = dedup(rows)
    style = [r for r in rows if r["origin"] == "game_dialogue"]
    ctx = [r for r in rows if r["origin"] != "game_dialogue"]
    cap = int(a.max_style_ratio / (1 - a.max_style_ratio) * len(ctx)) if ctx else len(style)
    rng.shuffle(style)
    rows = ctx + style[:cap]
    out = {"train": [], "val": []}
    for r in rows:
        split = r.get("quality", {}).get("split", "train")
        if split == "eval":
            continue                      # eval NPCs never enter training
        out["val" if split == "val" else "train"].append({
            "messages": r["messages"],
            "meta": {"id": r["example_id"], "npc": r.get("npc_id"), "behavior": r["behavior"],
                     "origin": r["origin"], "teacher": r.get("teacher_model")}})
    stats = {}
    for split, ex in out.items():
        rng.shuffle(ex)
        name = f"{split}{'_public' if a.public else ''}.jsonl"
        write_jsonl(CORPUS / "sft" / name, ex)
        stats[split] = {"n": len(ex), "origin": dict(Counter(e["meta"]["origin"] for e in ex)),
                        "behavior": dict(Counter(e["meta"]["behavior"] for e in ex))}
    print(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    main()
