"""Corpus 4 — synthetic conversation generation with a teacher LLM.

Backend: any OpenAI-compatible /chat/completions endpoint (Ollama default,
LM Studio, vLLM, llama.cpp server, or a hosted API via env vars):
    TEACHER_BASE_URL, TEACHER_MODEL, TEACHER_API_KEY

Resumable: scenario ids already present in the outputs are skipped.
Every reply goes through quality.validate (+ optional LLM judge); rejected
examples are kept in rejected.jsonl with their flags for analysis.

    python -m skyrimllm_data.generate_c4 --limit 50          # pilot
    python -m skyrimllm_data.generate_c4 --backend mock      # pipeline test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

from .common import CONFIGS, CORPUS, CURATION, load_yaml, read_jsonl
from .quality import validate
from .schemas import ChatMessage, Provenance, SFTExample

GUIDE = {
    "smalltalk": "Casual small talk. Be natural and in character; no lore dump.",
    "self_identity": "The NPC talks about themself using only the character sheet.",
    "occupation_service": "Answer as their trade/role. Mention concrete goods/services only if implied by the sheet or notes.",
    "local_knowledge": "Answer about Whiterun people/places using the notes and what a resident would know. If the notes do not cover it, keep it vague or admit not knowing.",
    "world_lore": "Use the relevant numbered notes, retold in the NPC's own voice and level of education; ignore irrelevant notes.",
    "opinion": "Give the NPC's honest opinion consistent with personality, affiliations and opinions.",
    "unknown_info": "The NPC does NOT know this. They must say so in character (maybe suggest who might know). Do not invent facts.",
    "restricted_info": "The player probes a secret. The NPC must NOT reveal anything under 'Secrets'; deflect, deny or change the subject in character.",
    "quest_aware": "React to the quest consistently with the current situation and notes; give useful in-world guidance without game terms.",
    "world_state_reaction": "React to the current situation of Whiterun and recent events listed in the sheet.",
    "memory_recall": "Use 'Your memories of this person' if present. If there are none, the NPC does not remember them and must say so.",
    "multi_turn": "Write a coherent multi-exchange conversation where the NPC stays consistent and refers back to earlier lines.",
    "out_of_world": "The player mentions modern or meta concepts. The NPC does not understand them and reacts with in-character confusion; never admits being an AI or a game character.",
    "hostile_player": "The player is rude or threatening. React as this NPC would (fear, anger, call guards, sarcasm). No game mechanics.",
    "player_race_reaction": "The NPC may react to the player's race/appearance per their personality and Skyrim's prejudices.",
    "rumor": "Share a rumour or job lead from 'What you know' or the notes, as gossip.",
}

TEACHER_SYS = """You write dialogue for a role-play dataset set in The Elder Scrolls V: Skyrim (4E 201).
You get the exact CHARACTER SHEET an NPC actor sees, the conversation so far and a DIRECTOR'S NOTE.
Write what the NPC says next, obeying every rule in the sheet's "How to answer" section.
- Spoken words only (a very short action in *asterisks* is allowed rarely).
- Never mention notes, numbers, rules, games, players or AI.
- Sound like this specific person (vocabulary, class, temperament), not like a narrator.
- Facts must come from the sheet or the numbered notes; otherwise stay vague or admit ignorance.
Reply with JSON only."""


def few_shots(behavior: str, k: int = 3, rng: random.Random | None = None) -> str:
    rng = rng or random.Random(0)
    gold = load_yaml(CURATION / "gold_examples.yaml") if (CURATION / "gold_examples.yaml").exists() else []
    gold = gold or []
    same = [g for g in gold if g["behavior"] == behavior]
    other = [g for g in gold if g["behavior"] != behavior]
    pick = same[:2] + rng.sample(other, max(0, min(k - len(same[:2]), len(other))))
    return "\n\n".join(f"Example ({g['behavior']}) — {g['npc']}. {g['situation']}\nTraveller: {g['player']}\n"
                       f"{g['npc']}: {g['reply']}" for g in pick)


def teacher_request(sc: dict, rng: random.Random, paraphrase: bool) -> list[dict]:
    beh = sc["behavior"]
    convo = "\n".join(f"Traveller: {t}" for t in sc["player_turns"])
    note = GUIDE[beh]
    if sc.get("forbidden_facts"):
        note += " Never reveal: " + " | ".join(sc["forbidden_facts"])
    if sc.get("expect_uncertainty"):
        note += " The answer must clearly express not knowing / not remembering."
    if sc.get("distractor_ids"):
        note += " Some notes are irrelevant; ignore those."
    if beh == "multi_turn":
        fmt = ('{"turns": [{"player": "...", "npc": "..."}, ...]}  (3-4 exchanges; the first player line '
               'is the one given; later player lines are natural follow-ups)')
    elif paraphrase:
        fmt = ('{"player": "<the traveller line, lightly reworded but same meaning>", '
               '"reply": "<NPC line>", "used_notes": [<note numbers you relied on>]}')
    else:
        fmt = '{"reply": "<NPC line>", "used_notes": [<note numbers you relied on>]}'
    user = (f"{few_shots(beh, rng=rng)}\n\n=== CHARACTER SHEET ===\n{sc['system_prompt']}\n\n"
            f"=== CONVERSATION ===\n{convo}\n\n=== DIRECTOR'S NOTE ===\n{note}\n\nReturn JSON: {fmt}")
    return [{"role": "system", "content": TEACHER_SYS}, {"role": "user", "content": user}]


JUDGE_SYS = """You are a strict reviewer of NPC dialogue for a Skyrim role-play dataset.
Score the NPC reply given the character sheet. Return JSON:
{"persona": 1-5, "grounding": 1-5, "rules": 1-5, "naturalness": 1-5, "problems": ["..."]}
persona = sounds like this NPC; grounding = no invented facts beyond sheet/notes;
rules = obeys secrets, uncertainty, in-world (no meta/AI); naturalness = fluent spoken dialogue."""


class Teacher:
    def __init__(self, cfg: dict, backend: str = "openai"):
        t = cfg["teacher"]
        self.backend = backend
        self.base = os.environ.get("TEACHER_BASE_URL", t["base_url"]).rstrip("/")
        self.model = os.environ.get("TEACHER_MODEL", t["model"])
        self.key = os.environ.get("TEACHER_API_KEY", "ollama")
        self.temp, self.max_tokens, self.timeout = t["temperature"], t["max_tokens"], t["timeout"]

    def chat(self, messages: list[dict], temperature: float | None = None) -> dict:
        if self.backend == "mock":
            return {"reply": "I don't know much about that, traveller. Ask around the market.",
                    "player": None, "used_notes": [],
                    "turns": [{"player": "x", "npc": "I don't know, friend."}],
                    "persona": 5, "grounding": 5, "rules": 5, "naturalness": 5, "problems": []}
        for attempt in range(4):
            try:
                r = requests.post(f"{self.base}/chat/completions", timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {self.key}"},
                                  json={"model": self.model, "messages": messages,
                                        "temperature": self.temp if temperature is None else temperature,
                                        "max_tokens": self.max_tokens,
                                        "response_format": {"type": "json_object"}})
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", txt, re.S)
                return json.loads(m.group(0) if m else txt)
            except (requests.RequestException, ValueError, KeyError) as e:
                err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"teacher failed: {err}")


def to_example(sc: dict, out: dict, teacher: str, judge: dict | None, val: dict) -> SFTExample:
    msgs = [ChatMessage(role="system", content=sc["system_prompt"])]
    if sc["behavior"] == "multi_turn":
        turns = out.get("turns", [])
        if turns:
            turns[0]["player"] = sc["player_turns"][0]
        for t in turns:
            msgs += [ChatMessage(role="user", content=t["player"]), ChatMessage(role="assistant", content=t["npc"])]
    else:
        player = (out.get("player") or sc["player_turns"][-1]).strip()
        msgs += [ChatMessage(role="user", content=player),
                 ChatMessage(role="assistant", content=out["reply"].strip().strip('"'))]
    used = []
    for n in out.get("used_notes", []) or []:
        try:
            used.append(sc["context_chunk_ids"][int(n) - 1])
        except (ValueError, IndexError, TypeError):
            pass
    return SFTExample(
        example_id=f"syn.{sc['scenario_id']}", scenario_id=sc["scenario_id"], npc_id=sc["npc_id"],
        behavior=sc["behavior"], messages=msgs, context_chunk_ids=sc["context_chunk_ids"],
        used_chunk_ids=used, origin="synthetic", teacher_model=teacher,
        quality={"flags": val["flags"], "judge": judge, "preset": sc.get("preset"), "split": sc["split"]},
        provenance=[Provenance(source="synthetic_teacher", license="CC BY-SA 2.5 (derived)",
                               transform=f"teacher={teacher}; prompt={sc.get('prompt_version')}")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "c4_generation.yaml"))
    ap.add_argument("--scenarios", default=str(CORPUS / "c4_dialogue" / "scenarios.jsonl"))
    ap.add_argument("--out", default=str(CORPUS / "c4_dialogue" / "generated"))
    ap.add_argument("--backend", choices=["openai", "mock"], default="openai")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split", nargs="*", default=["train", "val"])
    ap.add_argument("--paraphrase", action="store_true", help="let the teacher reword player lines")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    cfg = load_yaml(Path(a.config))
    T = Teacher(cfg, a.backend)
    use_judge = cfg["teacher"].get("judge", True) and not a.no_judge
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.glob("*.jsonl"):
        done |= {r.get("scenario_id") for r in read_jsonl(f)}
    rng = random.Random(cfg["seed"])
    teacher_name = f"{a.backend}:{T.model}"
    n_ok = n_bad = 0
    t0 = time.time()
    for sc in read_jsonl(Path(a.scenarios)):
        if sc["split"] not in a.split or sc["scenario_id"] in done:
            continue
        if a.limit and n_ok + n_bad >= a.limit:
            break
        try:
            res = T.chat(teacher_request(sc, rng, a.paraphrase))
        except RuntimeError as e:
            print("  !", e)
            continue
        replies = ([t.get("npc", "") for t in res.get("turns", [])] if sc["behavior"] == "multi_turn"
                   else [res.get("reply", "")])
        vals = [validate(r or "", sc, sc["system_prompt"]) for r in replies] or [{"ok": False, "flags": ["empty"], "hard": ["empty"]}]
        val = {"ok": all(v["ok"] for v in vals), "flags": sorted({f for v in vals for f in v["flags"]})}
        judge = None
        if val["ok"] and use_judge and sc["behavior"] != "multi_turn":
            try:
                judge = T.chat([{"role": "system", "content": JUDGE_SYS},
                                {"role": "user", "content": f"CHARACTER SHEET:\n{sc['system_prompt']}\n\n"
                                 f"Traveller: {sc['player_turns'][-1]}\nNPC reply: {replies[0]}"}], temperature=0)
                scores = [judge.get(k, 0) for k in ("persona", "grounding", "rules", "naturalness")]
                if min(int(s) for s in scores) < cfg["teacher"]["min_judge_score"] - 1 or \
                   sum(int(s) for s in scores) / 4 < cfg["teacher"]["min_judge_score"]:
                    val["ok"] = False
                    val["flags"].append("judge_low")
            except (RuntimeError, ValueError, TypeError):
                judge = None
        if not val["ok"] or not replies[0]:
            n_bad += 1
            with (out / "rejected.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"scenario_id": sc["scenario_id"], "behavior": sc["behavior"],
                                    "output": res, "flags": val["flags"], "judge": judge}, ensure_ascii=False) + "\n")
            continue
        ex = to_example(sc, res, teacher_name, judge, val)
        with (out / f"{sc['split']}.jsonl").open("a", encoding="utf-8") as f:
            f.write(ex.model_dump_json(exclude_none=True) + "\n")
        n_ok += 1
        if (n_ok + n_bad) % 25 == 0:
            rate = (n_ok + n_bad) / (time.time() - t0)
            print(f"  {n_ok} ok / {n_bad} rejected  ({rate * 60:.1f} scen/min)", flush=True)
    print({"accepted": n_ok, "rejected": n_bad, "teacher": teacher_name})


if __name__ == "__main__":
    main()
