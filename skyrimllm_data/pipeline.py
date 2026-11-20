"""End-to-end corpus build.

    python -m skyrimllm_data.pipeline              # everything except LLM generation
    python -m skyrimllm_data.pipeline --fetch      # also (re)download missing UESP pages
    python -m skyrimllm_data.pipeline --from c4    # resume from a step

Steps: fetch -> c1 -> c3 -> c2 -> restrict -> index -> c4 (scenarios) -> seed
       -> eval -> assemble -> report.   Synthetic generation (generate_c4) is a separate,
       long-running step that needs the teacher endpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .common import CONFIGS, CORPUS, RAW, load_yaml

STEPS = ["fetch", "c1", "c3", "c2", "restrict", "index", "c4", "seed", "eval", "assemble", "report"]


def run(step: str, cfg_path: Path) -> dict | int | None:
    if step == "fetch":
        from .uesp import UESPClient
        c = UESPClient(RAW / "uesp")
        n = 0
        for ns in load_yaml(cfg_path)["namespaces"]:
            n += c.fetch(c.list_pages(ns))
        return {"fetched": n}
    if step == "c1":
        from .build_c1_world import build
        return build(cfg_path, CORPUS / "c1_world")
    if step == "c3":
        from .build_c3_quests import build
        return build(cfg_path, CORPUS / "c3_quests")
    if step == "c2":
        from .build_c2_profiles import build
        return build(cfg_path, CORPUS / "c2_profiles")
    if step == "restrict":
        from .restrict import apply
        return apply()
    if step == "index":
        from .retrieval import build_index
        return {"indexed": build_index()}
    if step == "c4":
        from .build_c4_scenarios import ScenarioBuilder
        from .common import write_jsonl
        b = ScenarioBuilder(load_yaml(CONFIGS / "c4_generation.yaml"))
        sc = b.run()
        return {"scenarios": write_jsonl(CORPUS / "c4_dialogue" / "scenarios.jsonl", sc)}
    if step == "seed":
        from .build_c4_seed import build
        return build()
    if step == "eval":
        from .build_eval import build
        return build()
    if step == "assemble":
        from .assemble_sft import main as assemble
        return assemble([])
    if step == "report":
        from .report import build
        return build()
    raise ValueError(step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIGS / "scope_whiterun.yaml"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--from", dest="start", default="c1", choices=STEPS)
    ap.add_argument("--only", nargs="*", choices=STEPS)
    a = ap.parse_args()
    steps = a.only or STEPS[STEPS.index(a.start):]
    if a.fetch and "fetch" not in steps:
        steps = ["fetch"] + steps
    elif not a.fetch and "fetch" in steps and not a.only:
        steps.remove("fetch")
    log = {}
    for s in steps:
        t = time.time()
        print(f"== {s}", flush=True)
        res = run(s, Path(a.config))
        log[s] = {"result": res, "seconds": round(time.time() - t, 1)}
        print("  ", json.dumps(res, ensure_ascii=False)[:600], flush=True)
    (CORPUS / "build_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
