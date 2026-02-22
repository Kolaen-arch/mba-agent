"""
Paper structure manager.
Tracks the full paper outline, section status, adjacent context,
terminology glossary, and the red thread argument.
Persisted to YAML so it survives restarts and is human-editable.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Section:
    """A paper section with metadata."""
    id: str                     # e.g., "2.1"
    title: str                  # e.g., "Psychological Ownership Theory"
    parent_id: str = ""         # e.g., "2" for subsections
    status: str = "not_started" # not_started | outline | drafting | review | done
    word_count: int = 0
    target_words: int = 0
    docx_file: str = ""         # path to .docx if exists
    summary: str = ""           # 2-3 sentence summary of what this section argues
    key_sources: list = field(default_factory=list)  # main citations
    ends_with: str = ""         # last ~200 words (for transition context)
    starts_with: str = ""       # first ~200 words (for transition context)
    notes: str = ""             # your working notes
    order: int = 0              # sort position


@dataclass
class TermEntry:
    """A controlled term in the paper's glossary."""
    preferred: str              # the term to use
    alternatives: list = field(default_factory=list)  # variants to flag
    definition: str = ""
    first_used_in: str = ""     # section id
    source: str = ""            # originating citation


@dataclass
class PaperStructure:
    """The full paper structure."""
    title: str = ""
    research_question: str = ""
    red_thread: str = ""        # the core argument in 2-3 sentences
    methodology: str = ""
    sections: list = field(default_factory=list)
    glossary: list = field(default_factory=list)
    argument_chain: list = field(default_factory=list)  # ordered claim list


STRUCTURE_PATH = "./paper_structure.yaml"


def load_structure(path: str = STRUCTURE_PATH) -> PaperStructure:
    """Load paper structure from YAML."""
    if not os.path.exists(path):
        return PaperStructure()
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    ps = PaperStructure(
        title=data.get("title", ""),
        research_question=data.get("research_question", ""),
        red_thread=data.get("red_thread", ""),
        methodology=data.get("methodology", ""),
        argument_chain=data.get("argument_chain", []),
    )

    for s in data.get("sections", []):
        ps.sections.append(Section(**{k: v for k, v in s.items() if k in Section.__dataclass_fields__}))

    for t in data.get("glossary", []):
        ps.glossary.append(TermEntry(**{k: v for k, v in t.items() if k in TermEntry.__dataclass_fields__}))

    return ps


def save_structure(ps: PaperStructure, path: str = STRUCTURE_PATH) -> None:
    """Save paper structure to YAML."""
    data = {
        "title": ps.title,
        "research_question": ps.research_question,
        "red_thread": ps.red_thread,
        "methodology": ps.methodology,
        "argument_chain": ps.argument_chain,
        "sections": [asdict(s) for s in ps.sections],
        "glossary": [asdict(t) for t in ps.glossary],
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_section(ps: PaperStructure, section_id: str) -> Optional[Section]:
    for s in ps.sections:
        if s.id == section_id:
            return s
    return None


def get_adjacent_sections(ps: PaperStructure, section_id: str) -> dict:
    """
    Get the previous and next sections for transition context.
    Returns {prev: Section|None, current: Section, next: Section|None}
    """
    sorted_sections = sorted(ps.sections, key=lambda s: s.order)
    current = None
    prev_sec = None
    next_sec = None

    for i, s in enumerate(sorted_sections):
        if s.id == section_id:
            current = s
            if i > 0:
                prev_sec = sorted_sections[i - 1]
            if i < len(sorted_sections) - 1:
                next_sec = sorted_sections[i + 1]
            break

    return {"prev": prev_sec, "current": current, "next": next_sec}


def build_transition_context(ps: PaperStructure, section_id: str) -> str:
    """
    Build context string with adjacent section endings/beginnings
    so the agent knows what comes before and after.
    """
    adj = get_adjacent_sections(ps, section_id)
    parts = []

    if adj["prev"]:
        p = adj["prev"]
        parts.append(f"[PREVIOUS SECTION: {p.id} — {p.title}]")
        if p.summary:
            parts.append(f"Summary: {p.summary}")
        if p.ends_with:
            parts.append(f"Ends with: ...{p.ends_with}")
        parts.append("")

    if adj["current"]:
        c = adj["current"]
        parts.append(f"[CURRENT SECTION: {c.id} — {c.title}]")
        if c.summary:
            parts.append(f"Summary: {c.summary}")
        if c.notes:
            parts.append(f"Notes: {c.notes}")
        parts.append("")

    if adj["next"]:
        n = adj["next"]
        parts.append(f"[NEXT SECTION: {n.id} — {n.title}]")
        if n.summary:
            parts.append(f"Summary: {n.summary}")
        if n.starts_with:
            parts.append(f"Starts with: {n.starts_with}...")
        parts.append("")

    # Include red thread
    if ps.red_thread:
        parts.append(f"[RED THREAD — Core argument]: {ps.red_thread}")

    # Include argument chain context
    if ps.argument_chain:
        parts.append("[ARGUMENT CHAIN]:")
        for i, claim in enumerate(ps.argument_chain, 1):
            parts.append(f"  {i}. {claim}")

    return "\n".join(parts)


def build_glossary_context(ps: PaperStructure) -> str:
    """Build terminology glossary for consistency checking."""
    if not ps.glossary:
        return ""
    parts = ["[TERMINOLOGY GLOSSARY — use these terms consistently]:"]
    for t in ps.glossary:
        line = f"  • {t.preferred}"
        if t.alternatives:
            line += f" (NOT: {', '.join(t.alternatives)})"
        if t.definition:
            line += f" — {t.definition}"
        parts.append(line)
    return "\n".join(parts)


def compute_progress(ps: PaperStructure) -> dict:
    """Compute paper progress statistics."""
    total = len(ps.sections)
    if total == 0:
        return {"total": 0, "done": 0, "pct": 0, "words": 0, "target": 0, "by_status": {}}

    by_status = {}
    total_words = 0
    total_target = 0
    for s in ps.sections:
        by_status[s.status] = by_status.get(s.status, 0) + 1
        total_words += s.word_count
        total_target += s.target_words

    done = by_status.get("done", 0)
    pct = round(done / total * 100) if total > 0 else 0

    # Weighted progress (outline=10%, drafting=40%, review=70%, done=100%)
    weights = {"not_started": 0, "outline": 0.1, "drafting": 0.4, "review": 0.7, "done": 1.0}
    weighted = sum(weights.get(s.status, 0) for s in ps.sections)
    weighted_pct = round(weighted / total * 100) if total > 0 else 0

    return {
        "total_sections": total,
        "done": done,
        "completion_pct": pct,
        "weighted_pct": weighted_pct,
        "total_words": total_words,
        "target_words": total_target,
        "word_pct": round(total_words / total_target * 100) if total_target > 0 else 0,
        "page_estimate": total_words // 250,
        "by_status": by_status,
    }


def find_terminology_issues(ps: PaperStructure, text: str) -> list[dict]:
    """
    Scan text for terminology inconsistencies against the glossary.
    Returns list of {term, found, preferred, context}.
    """
    issues = []
    text_lower = text.lower()
    for entry in ps.glossary:
        for alt in entry.alternatives:
            if alt.lower() in text_lower:
                # Find context around the match
                idx = text_lower.find(alt.lower())
                start = max(0, idx - 40)
                end = min(len(text), idx + len(alt) + 40)
                context = text[start:end].strip()
                issues.append({
                    "found": alt,
                    "preferred": entry.preferred,
                    "context": f"...{context}...",
                })
    return issues


def generate_scaffold_yaml(
    title: str = "",
    rq: str = "",
    methodology: str = "Design Science Research",
) -> str:
    """
    Generate a starter paper_structure.yaml for the user to fill in.
    """
    template = PaperStructure(
        title=title or "MBA Final Paper",
        research_question=rq,
        methodology=methodology,
        red_thread="[Write your core argument in 2-3 sentences here]",
        argument_chain=[
            "[Claim 1: The problem statement — what gap exists]",
            "[Claim 2: Why existing theory doesn't address it]",
            "[Claim 3: Your proposed contribution]",
            "[Claim 4: How DSR validates it]",
            "[Claim 5: Implications for practice]",
        ],
        sections=[
            Section(id="1", title="Introduction", order=1, target_words=3000,
                    summary="", notes="Problem statement, RQ, contribution overview"),
            Section(id="1.1", title="Background and Motivation", parent_id="1", order=2, target_words=1500),
            Section(id="1.2", title="Research Question", parent_id="1", order=3, target_words=500),
            Section(id="1.3", title="Paper Structure", parent_id="1", order=4, target_words=500),
            Section(id="2", title="Theoretical Framework", order=5, target_words=8000),
            Section(id="2.1", title="Experience Economy", parent_id="2", order=6, target_words=2000,
                    key_sources=["Pine & Gilmore, 1998", "Pine & Gilmore, 2011"]),
            Section(id="2.2", title="Co-creation and Service-Dominant Logic", parent_id="2", order=7, target_words=2000,
                    key_sources=["Vargo & Lusch, 2004", "Prahalad & Ramaswamy, 2004"]),
            Section(id="2.3", title="Psychological Ownership", parent_id="2", order=8, target_words=2000,
                    key_sources=["Pierce et al., 2001", "Pierce et al., 2003"]),
            Section(id="2.4", title="Delegated Leadership", parent_id="2", order=9, target_words=2000),
            Section(id="2.5", title="Theoretical Synthesis", parent_id="2", order=10, target_words=1500,
                    notes="Where the four streams intersect — this is your contribution"),
            Section(id="3", title="Methodology", order=11, target_words=5000),
            Section(id="3.1", title="Design Science Research Approach", parent_id="3", order=12, target_words=2000),
            Section(id="3.2", title="Research Design", parent_id="3", order=13, target_words=1500),
            Section(id="3.3", title="Data Collection and Analysis", parent_id="3", order=14, target_words=1500),
            Section(id="4", title="Artifact Design and Development", order=15, target_words=5000),
            Section(id="5", title="Evaluation", order=16, target_words=4000),
            Section(id="6", title="Discussion", order=17, target_words=4000),
            Section(id="6.1", title="Theoretical Contributions", parent_id="6", order=18, target_words=2000),
            Section(id="6.2", title="Practical Implications", parent_id="6", order=19, target_words=2000),
            Section(id="7", title="Conclusion", order=20, target_words=2000,
                    notes="Revisit RQ, summarize contributions, limitations, future research"),
        ],
        glossary=[
            TermEntry(
                preferred="psychological ownership",
                alternatives=["psych ownership", "psychological sense of ownership", "PO"],
                definition="The state in which individuals feel as though the target of ownership is theirs (Pierce et al., 2001)",
                source="Pierce et al., 2001",
            ),
            TermEntry(
                preferred="co-creation",
                alternatives=["cocreation", "co creation", "value co-creation"],
                definition="Joint creation of value by the firm and the customer (Prahalad & Ramaswamy, 2004)",
                source="Prahalad & Ramaswamy, 2004",
            ),
            TermEntry(
                preferred="delegated leadership",
                alternatives=["distributed leadership", "shared leadership", "delegating leadership"],
                definition="[Define based on your framework]",
            ),
            TermEntry(
                preferred="experience economy",
                alternatives=["experiential economy", "experience-based economy"],
                definition="Economic offering based on staging memorable experiences (Pine & Gilmore, 1998)",
                source="Pine & Gilmore, 1998",
            ),
        ],
    )

    return yaml.dump(
        {
            "title": template.title,
            "research_question": template.research_question,
            "red_thread": template.red_thread,
            "methodology": template.methodology,
            "argument_chain": template.argument_chain,
            "sections": [asdict(s) for s in template.sections],
            "glossary": [asdict(t) for t in template.glossary],
        },
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
