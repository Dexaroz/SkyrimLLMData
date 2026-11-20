"""Sanity tests for parsers, access control and validators.
Run: .venv/Scripts/python -m pytest -q   (or python tests/test_corpus.py)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyrimllm_data import wikitext as W
from skyrimllm_data.quality import validate
from skyrimllm_data.retrieval import gate_ok, visible
from skyrimllm_data.schemas import WorldState

DLG = """Greetings while she is working at the inn:
:''"What can I get you?"''
:'''Where can I learn more about magic?'''
:''"Magic is for the weak."''
"""
JOURNAL = """{{Journal Entries
|10:2||Go to [[Skyrim:Riverwood|Riverwood]].
{{Quest Objective|10|Talk to <Alias&#61;Friend> in Riverwood}}
|^||Alt text.
|30:2|fin|Done.
}}"""


def test_dialogue():
    d = W.parse_dialogue(DLG)
    assert d[0] == {"condition": "Greetings while she is working at the inn", "player": None, "npc": "What can I get you?"}
    assert d[1]["player"] == "Where can I learn more about magic?"


def test_journal():
    st = W.parse_journal(JOURNAL)
    assert [s["stage"] for s in st] == [10, 30]
    assert st[0]["objectives"] == ["Talk to [Friend] in Riverwood"]
    assert st[0]["journal"] == ["Go to Riverwood.", "Alt text."]
    assert st[1]["finishes"]


def test_gates_and_visibility():
    ws = WorldState(quests={"MS08": 10}, completed_quests=["MQ104"])
    assert gate_ok({"quest_id": "MS08"}, ws)
    assert not gate_ok({"quest_id": "MS08", "completed": True}, ws)
    assert gate_ok({"quest_id": "MQ104", "completed": True}, ws)
    hulda = {"npc_id": "hulda", "scopes": ["common", "local:whiterun", "personal:hulda"]}
    assert visible({"scopes": ["local:whiterun"]}, hulda, ws)
    assert not visible({"scopes": ["personal:saadia"]}, hulda, ws)
    assert not visible({"scopes": ["scholarly"]}, hulda, ws)
    assert not visible({"scopes": ["meta", "common"]}, hulda, ws)
    assert not visible({"scopes": ["local:whiterun"], "quest_gates": [{"quest_id": "C01", "completed": True}]}, hulda, ws)


def test_validator():
    sc = {"behavior": "unknown_info", "expect_uncertainty": True, "player_turns": ["Who is Vivec?"]}
    assert "missing_uncertainty" in validate("Vivec is a god of Morrowind.", sc, "")["hard"]
    assert validate("Never heard of them, friend. Ask at the College.", sc, "College")["ok"]
    sc2 = {"behavior": "smalltalk", "player_turns": ["hi"]}
    assert "meta_reference" in validate("As an AI language model I cannot.", sc2, "")["hard"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
