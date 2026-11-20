"""UESP wikitext -> clean, structured text.

Handles: infobox extraction (NPC Summary, Place Summary, Quest Header, Lore
summaries), section splitting and classification, inline-template rendering,
NPC dialogue extraction and quest journal (stage) extraction.
"""
from __future__ import annotations

import html
import re

import mwparserfromhell as mwp

INFOBOXES = {
    "npc summary", "place summary", "quest header", "lore place summary",
    "lore people summary", "lore faction summary", "creature summary",
    "book summary", "lore book summary", "race summary", "item summary",
    "faction summary", "lore summary", "disease summary", "dlc npc summary",
}

# Inline templates whose first positional argument is the text to keep.
KEEP_ARG1 = {
    "year", "quest link", "faction", "lore link", "sr", "item link", "nowrap",
    "small", "ul", "hover", "tooltip", "about", "sic", "pc", "quote", "q",
    "cite book", "book link", "place link", "npc link", "nolink", "uc", "lc",
    "lc first", "uc first", "spell link", "lw", "spoiler", "il", "gold",
}
# Inline markers rendered to a short tag.
MARKERS = {"dg": "(Dawnguard)", "db": "(Dragonborn)", "hf": "(Hearthfire)",
           "cc": "(Creation Club)", "pc only": "", "xbox": "", "ps3": ""}

SECTION_KIND = [
    (r"bugs?|glitch|gallery|maps?|images?|references|credits|achievements?|"
     r"see also|external links|stats|statistics|leveled|inventory|merchandise|"
     r"(?:standard |displayed )?merchandise|items|concept art|miscellaneous|books|map|console|technical|ids?|trivia|"
     r"related quests|quick walkthrough|notes|unused dialogue|unused conversations|follower dialogue|"
     r"radiant quests|radiant options|rewards?|quest giver|gameplay|spells|skills|powers|abilities|"
     r"need title|other|\(creation club\)|quest-related dialogue|quest-related events?|"
     r"dawnguard|dragonborn|hearthfire|\(dawnguard\)|\(dragonborn\)", "meta"),
    (r"quest stages|journal", "stages"),
    (r"dialogue|conversations?|quotes|rumors|greetings", "dialogue"),
    (r"detailed walkthrough|walkthrough|first steps|outcome|choices", "walkthrough"),
]


def classify_section(heading: str) -> str:
    h = heading.lower().strip()
    for pat, kind in SECTION_KIND:
        if re.fullmatch(rf"(?:{pat})(?:\s.*)?", h):
            return kind
    return "content"


def norm_title(t: str) -> str:
    t = t.strip().replace("_", " ")
    return t[:1].upper() + t[1:] if t else t


# ----------------------------------------------------------------- templates
def _tname(t) -> str:
    return str(t.name).strip().lower().replace("_", " ")


def extract_infobox(code: mwp.wikicode.Wikicode) -> tuple[str | None, dict[str, str]]:
    for t in code.filter_templates(recursive=False):
        name = _tname(t)
        if name in INFOBOXES:
            data = {}
            for p in t.params:
                key = str(p.name).strip().lower()
                val = re.sub(r"\s*\n\s*", "; ", clean_inline(str(p.value)))
                if val:
                    data[key] = val
            return name, data
    return None, {}


def render_templates(code: mwp.wikicode.Wikicode) -> None:
    """Replace templates in-place with their readable form (or nothing)."""
    for t in code.filter_templates(recursive=False):
        name = _tname(t)
        try:
            if name in MARKERS:
                code.replace(t, MARKERS[name])
            elif name in KEEP_ARG1 and t.params:
                # Cite Book etc.: keep title; Faction: keep id; Year: keep year
                code.replace(t, str(t.params[0].value).strip())
            elif name in {"ref", "refname"} or name in INFOBOXES:
                code.remove(t)
            else:
                code.remove(t)
        except ValueError:
            pass


TABLE_RE = re.compile(r"\{\|.*?\n\|\}", re.S)


def _table_to_text(m: re.Match) -> str:
    rows = []
    for row in re.split(r"\n\|-[^\n]*", m.group(0)):
        cells = []
        for line in row.split("\n"):
            line = line.strip()
            if not line or line.startswith(("{|", "|}", "|+")):
                continue
            if line[0] in "!|":
                for cell in re.split(r"\|\||!!", line[1:]):
                    cell = cell.split("|", 1)[-1] if re.match(r"^\s*(class|style|rowspan|colspan|width|align)=", cell) else cell
                    cells.append(cell.strip())
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def clean_inline(text: str) -> str:
    """Clean a wikitext fragment to plain text."""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<noinclude>|</noinclude>|<onlyinclude>|</onlyinclude>|<includeonly>.*?</includeonly>", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:File|Image|Category|fr|de|es|it|pl|ru|pt|ja):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", text, flags=re.I)
    text = re.sub(r"__[A-Z]+__", "", text)
    text = TABLE_RE.sub(_table_to_text, text)
    code = mwp.parse(text)
    for _ in range(3):                  # nested templates
        if not code.filter_templates(recursive=False):
            break
        render_templates(code)
        code = mwp.parse(str(code))
    out = code.strip_code(normalize=True, collapse=True, keep_template_params=False)
    out = re.sub(r"<[^>]+>", "", out)
    out = re.sub(r"'{2,}", "", out)
    out = re.sub(r"^[:*#;]+\s*", "", out, flags=re.M)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n\s*\n+", "\n\n", out)
    return out.strip()


def wiki_links(text: str) -> list[str]:
    links = []
    for m in re.finditer(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]", text):
        t = m.group(1).strip()
        if not re.match(r"(File|Image|Category|[a-z]{2}):", t):
            links.append(norm_title(t))
    return list(dict.fromkeys(links))


# ------------------------------------------------------------------ sections
HEAD_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


def split_sections(wikitext: str) -> list[tuple[str, int, str]]:
    """Return [(heading, level, raw_text)], first item is the lead ('')."""
    out, pos, head, lvl = [], 0, "", 1
    for m in HEAD_RE.finditer(wikitext):
        out.append((head, lvl, wikitext[pos:m.start()]))
        head, lvl, pos = clean_inline(m.group(2)), len(m.group(1)), m.end()
    out.append((head, lvl, wikitext[pos:]))
    return out


def parse_page(wikitext: str) -> dict:
    code = mwp.parse(wikitext)
    box_name, infobox = extract_infobox(code)
    sections, parent_kind = [], {}
    for head, lvl, raw in split_sections(wikitext):
        kind = "lead" if head == "" else classify_section(head)
        # subsections inherit meta/walkthrough/dialogue from their parent
        parent = parent_kind.get(lvl - 1)
        if kind == "content" and parent in {"meta", "walkthrough", "dialogue", "stages"}:
            kind = parent
        parent_kind[lvl] = kind
        for k in [k for k in parent_kind if k > lvl]:
            del parent_kind[k]
        sections.append({"heading": head or "Overview", "level": lvl, "kind": kind,
                         "raw": raw, "text": clean_inline(raw) if kind not in {"stages"} else ""})
    return {"infobox_type": box_name, "infobox": infobox, "sections": sections,
            "links": wiki_links(wikitext)}


# ------------------------------------------------------------------ dialogue
PLAYER_RE = re.compile(r"^:+\s*'''(.+?)'''\s*$")
NPC_RE = re.compile(r"^:*\s*''\s*\"(.+?)\"\s*''")


def parse_dialogue(raw: str) -> list[dict]:
    """UESP NPC dialogue sections -> [{condition, player, npc}].

    Lines like `:'''Topic'''` are player prompts, `:''"..."''` are NPC lines and
    plain prose lines describe the condition under which the following lines
    are said.
    """
    out, condition, player = [], None, None
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("{|", "|", "!", "[[File")):
            continue
        if m := PLAYER_RE.match(s):
            player = clean_inline(m.group(1))
            continue
        if m := NPC_RE.match(s):
            npc = clean_inline(m.group(1))
            if len(npc) > 3:
                out.append({"condition": condition, "player": player, "npc": npc})
            continue
        if not s.startswith(":"):
            condition, player = clean_inline(s).rstrip(":").strip() or None, None
    return out


# ------------------------------------------------------------ quest journal
def parse_journal(raw: str) -> list[dict]:
    """{{Journal Entries |10:2||text {{Quest Objective|10|obj}} |^|fin|text }}"""
    m = re.search(r"\{\{Journal Entries(.*)", raw, re.S | re.I)
    if not m:
        return []
    body = m.group(1)
    stages: dict[int, dict] = {}
    current = None
    for part in re.split(r"\n\|(?=[\d^]+[^|\n]*\|)", "\n" + body):
        hm = re.match(r"([\d^]+)(?::\d+)?\|([^|\n]*)\|(.*)", part, re.S)
        if not hm:
            continue
        key, flags, rest = hm.groups()
        if key != "^":
            current = int(key)
        if current is None:
            continue
        st = stages.setdefault(current, {"stage": current, "journal": [], "objectives": [], "finishes": False})
        for om in re.finditer(r"\{\{Quest Objective\|\d+\|(.+?)\}\}", rest):
            obj = re.sub(r"<(?:Alias(?:=|&#61;))?([^<>]+)>", r"[\1]", om.group(1))
            obj = clean_inline(obj)
            if obj and obj not in st["objectives"]:
                st["objectives"].append(obj)
        text = re.sub(r"\{\{Quest Objective.*?\}\}", "", rest, flags=re.S)
        text = re.sub(r"\|missing=.*", "", text, flags=re.S)
        text = re.sub(r"<(?:Alias(?:=|&#61;))?([^<>]+)>", r"[\1]", text)
        text = clean_inline(text.split("}}\n")[0])
        if text and text not in st["journal"]:
            st["journal"].append(text)
        if "fin" in flags:
            st["finishes"] = True
    return [stages[k] for k in sorted(stages)]
