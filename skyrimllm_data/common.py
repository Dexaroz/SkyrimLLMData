from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus"
CONFIGS = ROOT / "configs"
CURATION = ROOT / "curation"


def slug(s: str) -> str:
    s = s.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def entity_id(title: str) -> str:
    """'Skyrim:Hulda' -> 'skyrim.hulda' ; 'Lore:Whiterun' -> 'lore.whiterun'."""
    ns, _, name = title.partition(":") if ":" in title else ("main", "", title)
    return f"{slug(ns)}.{slug(name)}"


def npc_id(title: str) -> str:
    return slug(title.split(":", 1)[-1])


def short_hash(*parts: str, n: int = 10) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:n]


def n_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            d = r.model_dump(mode="json", exclude_none=True) if isinstance(r, BaseModel) else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
