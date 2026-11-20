"""Data contracts for the five SkyrimLLM corpora.

Every record carries `Provenance` so any fact, chunk or training example can be
traced back to its source page/revision and licence. The same models are meant
to be imported by the runtime (RAG, NPC profile system, world-state manager),
so training data and inference use one schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Common
# --------------------------------------------------------------------------
class Provenance(BaseModel):
    source: str                       # id in configs/sources.yaml (e.g. "uesp")
    url: str | None = None
    page_title: str | None = None
    revision_id: int | None = None
    retrieved_at: str | None = None
    license: str                      # e.g. "CC BY-SA 2.5"
    redistributable: bool = True      # False for game-extracted text
    transform: str | None = None      # what we did (e.g. "wikitext->clean text")


# Knowledge-access scopes. A chunk is visible to an NPC only if one of the
# chunk's scopes is in the NPC's scopes (and its quest gates are satisfied).
#   common            any inhabitant of Skyrim
#   regional:<hold>   people living in that hold
#   local:<settlement> people of that settlement
#   faction:<id>      members of that faction
#   scholarly         scholars, mages, priests, court wizards, bards
#   personal:<npc_id> only that NPC (secrets, private history)
#   meta              game mechanics -> never shown to any NPC
Scope = str


class QuestGate(BaseModel):
    """Condition on world state: quest `quest_id` must be at stage >= min_stage
    (or completed if min_stage is None)."""
    quest_id: str
    min_stage: int | None = None
    completed: bool = False


# --------------------------------------------------------------------------
# Corpus 1 - World Knowledge (-> Knowledge RAG)
# --------------------------------------------------------------------------
EntityType = Literal[
    "npc", "place", "quest", "faction", "creature", "item", "book",
    "race", "deity", "event", "lore", "mechanic", "other",
]


class Section(BaseModel):
    heading: str
    level: int
    kind: Literal["lead", "content", "walkthrough", "dialogue", "stages", "meta"]
    text: str


class WorldDocument(BaseModel):
    doc_id: str
    title: str
    namespace: str                   # "Skyrim" | "Lore"
    entity_type: EntityType
    categories: list[str] = []
    infobox: dict[str, str] = {}
    sections: list[Section] = []
    links: list[str] = []            # outgoing wiki links (normalised titles)
    provenance: Provenance


class KnowledgeChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str                        # self-contained: "Title — Section: ..."
    n_tokens: int
    entity_type: EntityType
    entities: list[str] = []         # linked entity ids mentioned
    scopes: list[Scope] = ["common"]
    quest_gates: list[QuestGate] = []
    spoiler: bool = False
    provenance: Provenance


class Entity(BaseModel):
    entity_id: str
    name: str
    type: EntityType
    aliases: list[str] = []
    doc_id: str | None = None
    attributes: dict[str, Any] = {}


class Relation(BaseModel):
    subject: str
    predicate: str                   # works_at, lives_in, member_of, race, ...
    object: str
    doc_id: str
    evidence: str | None = None      # infobox field or sentence
    confidence: float = 1.0


# --------------------------------------------------------------------------
# Corpus 2 - NPC Profiles (-> NPC Profile System)
# --------------------------------------------------------------------------
class Relationship(BaseModel):
    target: str                      # npc_id or free text
    type: str                        # spouse, sibling, employer, rival, friend...
    note: str | None = None


class KnowledgeItem(BaseModel):
    fact: str
    visibility: Literal["public", "private", "restricted"] = "public"
    # public: will share with anyone; private: shares only with trusted people;
    # restricted: never reveals (secrets) unless a quest gate allows it.
    reveal_gate: QuestGate | None = None
    source_doc: str | None = None


class DialogueLine(BaseModel):
    player: str | None = None        # player topic (None = greeting/ambient)
    npc: str
    condition: str | None = None     # e.g. "When Whiterun is under Stormcloak control"
    provenance: Provenance


class NPCProfile(BaseModel):
    npc_id: str
    name: str
    form_ids: dict[str, str] = {}    # refid/baseid (meta; never in prompts)
    race: str | None = None
    gender: str | None = None
    occupation: str | None = None
    settlement: str | None = None
    hold: str | None = None
    home: str | None = None
    workplace: str | None = None
    factions: list[str] = []         # human-readable
    essential: bool = False
    summary: str = ""                # third-person background (UESP, cleaned)
    bio_secondary: str | None = None # Mantella bio (AGPL-3.0), secondary evidence
    personality: list[str] = []
    speech_style: str | None = None
    opinions: dict[str, str] = {}    # topic -> stance
    relationships: list[Relationship] = []
    knowledge: list[KnowledgeItem] = []
    scopes: list[Scope] = ["common"]
    quests: list[str] = []           # quest ids the NPC is involved in
    state_variants: list[dict[str, Any]] = []  # {"condition": ..., "change": ...}
    dialogue_samples: list[DialogueLine] = []
    review_status: Literal["auto", "llm_enriched", "curated", "reviewed"] = "auto"
    provenance: list[Provenance] = []


# --------------------------------------------------------------------------
# Corpus 3 - Quests & World State (-> World State Manager)
# --------------------------------------------------------------------------
class QuestStage(BaseModel):
    stage: int
    journal: list[str] = []          # alternative journal texts
    objectives: list[str] = []
    finishes: bool = False


class WorldEffect(BaseModel):
    """Static definition: what changes in the world when a quest hits a stage."""
    trigger: QuestGate
    variable: str                    # world-state variable, e.g. "whiterun.jarl"
    value: Any
    description: str
    affected_npcs: list[str] = []
    knowledge_scope: Scope = "common"  # who learns about the change


class QuestDefinition(BaseModel):
    quest_id: str                    # editor id (MQ102) or slug
    name: str
    type: str | None = None
    giver: list[str] = []
    locations: list[str] = []
    involved_npcs: list[str] = []
    prev: list[str] = []
    next: list[str] = []
    description: str = ""
    reward: str | None = None
    stages: list[QuestStage] = []
    world_effects: list[WorldEffect] = []
    scope_tags: list[str] = []       # e.g. "whiterun"
    provenance: Provenance


class WorldState(BaseModel):
    """Dynamic per-playthrough state (not training data; sampled for SFT)."""
    game_time: str = "4E 201, Morndas, morning"
    player: dict[str, Any] = {}      # name, race, gender, titles, known_as_dragonborn
    location: str | None = None
    quests: dict[str, int] = {}      # quest_id -> current stage
    completed_quests: list[str] = []
    variables: dict[str, Any] = {}   # e.g. {"whiterun.control": "Imperial"}
    npc_status: dict[str, str] = {}  # npc_id -> alive|dead|absent


# --------------------------------------------------------------------------
# Corpus 4 - Dialogue & Behaviour (-> SFT)
# --------------------------------------------------------------------------
Behavior = Literal[
    "smalltalk", "self_identity", "occupation_service", "local_knowledge",
    "world_lore", "opinion", "unknown_info", "restricted_info", "quest_aware",
    "world_state_reaction", "memory_recall", "multi_turn", "out_of_world",
    "hostile_player", "player_race_reaction", "rumor",
]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Scenario(BaseModel):
    scenario_id: str
    npc_id: str
    behavior: Behavior
    world_state: WorldState
    player_turns: list[str]          # 1..n player utterances (last one answered)
    retrieval_query: str | None = None
    target_facts: list[str] = []     # chunk ids / facts the answer should use
    forbidden_facts: list[str] = []  # facts the NPC must not reveal
    memories: list[str] = []         # memory ids injected into context
    notes: str | None = None
    split: Literal["train", "val", "eval"] = "train"


class SFTExample(BaseModel):
    example_id: str
    scenario_id: str | None = None
    npc_id: str | None = None
    behavior: str
    messages: list[ChatMessage]
    context_chunk_ids: list[str] = []
    used_chunk_ids: list[str] = []
    origin: Literal["synthetic", "uesp_dialogue", "game_dialogue", "curated"]
    teacher_model: str | None = None
    quality: dict[str, Any] = {}     # validator scores / flags
    provenance: list[Provenance] = []
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# Corpus 5 - NPC Memory & Events (runtime only; schema shared)
# --------------------------------------------------------------------------
class MemoryEvent(BaseModel):
    memory_id: str
    npc_id: str
    kind: Literal["conversation", "observed_action", "witnessed_event",
                  "rumor", "relationship_change", "self_reflection"]
    content: str                     # first-person, NPC point of view
    game_time: str
    location: str | None = None
    participants: list[str] = []
    importance: float = Field(0.5, ge=0, le=1)
    emotional_valence: float = Field(0.0, ge=-1, le=1)
    source_memory_ids: list[str] = []  # for rumors / summaries
    relationship_delta: dict[str, float] = {}  # npc_or_player -> delta
    created_at: str = Field(default_factory=now_iso)


# --------------------------------------------------------------------------
# Evaluation dataset
# --------------------------------------------------------------------------
class EvalCase(BaseModel):
    case_id: str
    dimension: Literal["factual", "character", "knowledge_restriction",
                       "context_state", "memory", "naturalness", "robustness"]
    npc_id: str
    world_state: WorldState
    memories: list[MemoryEvent] = []
    player_turns: list[str]
    must_include_any: list[str] = []   # lexical anchors (any one should appear)
    must_not_include: list[str] = []   # leaked secrets / meta / anachronisms
    expect_uncertainty: bool = False
    reference_answer: str | None = None
    rubric: str
    source: str
