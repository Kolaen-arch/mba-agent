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


def _match_heading_to_section(heading: str, ps: PaperStructure) -> Optional[Section]:
    """
    Fuzzy-match a DOCX heading to a structure section.
    Priority: exact title → section ID prefix → substring containment.
    """
    heading_clean = heading.strip().lower()
    sorted_sections = sorted(ps.sections, key=lambda s: s.order)

    # Pass 1: exact title match
    for s in sorted_sections:
        if s.title.strip().lower() == heading_clean:
            return s

    # Pass 2: heading starts with section ID (e.g. "2.1 Psychological Ownership")
    for s in sorted_sections:
        if heading_clean.startswith(s.id.lower()) and len(heading_clean) > len(s.id):
            # Check that the rest roughly matches the title
            rest = heading_clean[len(s.id):].strip().lstrip('.').strip()
            if rest and s.title.strip().lower().startswith(rest[:10]):
                return s
            # Even if rest doesn't match title, ID prefix is strong enough
            return s

    # Pass 3: substring containment (heading contains title or vice versa)
    for s in sorted_sections:
        title_lower = s.title.strip().lower()
        if len(title_lower) >= 4:
            if title_lower in heading_clean or heading_clean in title_lower:
                return s

    return None


def build_all_adjacent_pairs(ps: PaperStructure) -> list[dict]:
    """
    Build all adjacent section pairs with their content boundaries.
    Returns list of dicts with prev/next section info for red thread audit.
    """
    sorted_sections = sorted(ps.sections, key=lambda s: s.order)
    # Only include top-level or first-level subsections that have content
    pairs = []
    content_sections = [s for s in sorted_sections if s.ends_with or s.starts_with or s.summary]

    for i in range(len(content_sections) - 1):
        prev_sec = content_sections[i]
        next_sec = content_sections[i + 1]
        pairs.append({
            "prev_id": prev_sec.id,
            "prev_title": prev_sec.title,
            "prev_ends_with": prev_sec.ends_with,
            "prev_summary": prev_sec.summary,
            "next_id": next_sec.id,
            "next_title": next_sec.title,
            "next_starts_with": next_sec.starts_with,
            "next_summary": next_sec.summary,
        })
    return pairs


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


def validate_argument_chain(ps: PaperStructure) -> list[dict]:
    """Check that each claim in the argument chain maps to a section via keyword overlap."""
    stop_words = {"the", "a", "an", "is", "are", "of", "in", "to", "and", "that", "for",
                  "it", "on", "with", "as", "by", "this", "from", "or", "be", "at", "not",
                  "how", "what", "why", "can", "do", "does", "has", "have", "was", "were"}
    results = []
    for i, claim in enumerate(ps.argument_chain):
        claim_words = set(claim.lower().split()) - stop_words
        matching_sections = []
        for sec in ps.sections:
            if sec.summary:
                summary_words = set(sec.summary.lower().split()) - stop_words
                overlap = claim_words & summary_words
                if len(overlap) >= 2:
                    matching_sections.append(sec.id)
        results.append({
            "claim_index": i + 1,
            "claim": claim,
            "supported_by": matching_sections,
            "status": "supported" if matching_sections else "unsupported",
        })
    return results


def compute_word_budget(ps: PaperStructure) -> dict:
    """Detailed word budget analysis per section."""
    sections = sorted(ps.sections, key=lambda s: s.order)
    total_target = sum(s.target_words for s in sections)
    total_written = sum(s.word_count for s in sections)

    budget_items = []
    cumulative_target = 0
    cumulative_actual = 0

    for sec in sections:
        cumulative_target += sec.target_words
        cumulative_actual += sec.word_count
        pct = round(sec.word_count / sec.target_words * 100) if sec.target_words > 0 else 0
        budget_items.append({
            "id": sec.id,
            "title": sec.title,
            "target": sec.target_words,
            "actual": sec.word_count,
            "pct": pct,
            "status": "over" if pct > 120 else "on_track" if pct > 60 else "under" if pct > 0 else "empty",
            "cumulative_target": cumulative_target,
            "cumulative_actual": cumulative_actual,
        })

    over_budget = [b for b in budget_items if b["status"] == "over"]
    remaining_words = total_target - total_written
    remaining_sections = [b for b in budget_items if b["status"] in ("under", "empty")]

    return {
        "total_target": total_target,
        "total_written": total_written,
        "total_pct": round(total_written / total_target * 100) if total_target > 0 else 0,
        "remaining_words": remaining_words,
        "page_estimate": total_written // 250,
        "target_pages": total_target // 250,
        "sections": budget_items,
        "over_budget": over_budget,
        "avg_words_per_remaining_section": round(remaining_words / len(remaining_sections)) if remaining_sections else 0,
    }


def infer_section_type(title: str) -> str:
    """Guess section type from title for prompt fragment injection."""
    title_lower = title.lower()
    if any(w in title_lower for w in ["introduction", "indledning", "background", "baggrund"]):
        return "introduction"
    if any(w in title_lower for w in ["theory", "theoretical", "framework", "literature", "teori", "litteratur"]):
        return "theory"
    if any(w in title_lower for w in ["method", "metode", "design science", "research design", "forskningsdesign"]):
        return "methodology"
    if any(w in title_lower for w in ["finding", "result", "analysis", "artifact", "evaluation", "fund", "resultat", "analyse"]):
        return "findings"
    if any(w in title_lower for w in ["discussion", "diskussion", "implication", "contribution"]):
        return "discussion"
    if any(w in title_lower for w in ["conclusion", "konklusion"]):
        return "conclusion"
    return ""


def generate_scaffold_yaml(
    title: str = "",
    rq: str = "",
    methodology: str = "Design Science Research",
    language: str = "en",
) -> str:
    """
    Generate a starter paper_structure.yaml for the user to fill in.
    Supports Danish (da) and English (en) section names.
    Methodology templates: DSR, Case Study, Survey, Mixed Methods.
    """
    # Section templates by language
    _SECTIONS = {
        "en": {
            "intro": "Introduction",
            "background": "Background and Motivation",
            "rq": "Research Question",
            "structure": "Paper Structure",
            "theory": "Theoretical Framework",
            "lit_review": "Literature Review",
            "methodology": "Methodology",
            "research_design": "Research Design",
            "data_collection": "Data Collection and Analysis",
            "discussion": "Discussion",
            "theoretical_contributions": "Theoretical Contributions",
            "practical_implications": "Practical Implications",
            "conclusion": "Conclusion",
            # DSR-specific
            "dsr_approach": "Design Science Research Approach",
            "artifact_design": "Artifact Design and Development",
            "evaluation": "Evaluation",
            # Case Study-specific
            "case_selection": "Case Selection and Context",
            "within_case": "Within-Case Analysis",
            "cross_case": "Cross-Case Analysis",
            # Survey-specific
            "instrument_design": "Instrument Design",
            "sample_strategy": "Sampling Strategy",
            "statistical_analysis": "Statistical Analysis",
            # Mixed Methods-specific
            "qual_strand": "Qualitative Strand",
            "quant_strand": "Quantitative Strand",
            "integration": "Integration of Findings",
        },
        "da": {
            "intro": "Indledning",
            "background": "Baggrund og Motivation",
            "rq": "Forskningsspørgsmål",
            "structure": "Afhandlingens Struktur",
            "theory": "Teoretisk Grundlag",
            "lit_review": "Litteraturgennemgang",
            "methodology": "Metode",
            "research_design": "Forskningsdesign",
            "data_collection": "Dataindsamling og Analyse",
            "discussion": "Diskussion",
            "theoretical_contributions": "Teoretiske Bidrag",
            "practical_implications": "Praktiske Implikationer",
            "conclusion": "Konklusion",
            "dsr_approach": "Design Science Research Tilgang",
            "artifact_design": "Artefaktdesign og Udvikling",
            "evaluation": "Evaluering",
            "case_selection": "Caseudvælgelse og Kontekst",
            "within_case": "Inden-for-case Analyse",
            "cross_case": "Tværgående Caseanalyse",
            "instrument_design": "Instrumentdesign",
            "sample_strategy": "Stikprøvestrategi",
            "statistical_analysis": "Statistisk Analyse",
            "qual_strand": "Kvalitativt Spor",
            "quant_strand": "Kvantitativt Spor",
            "integration": "Integration af Fund",
        },
    }

    lang = language if language in _SECTIONS else "en"
    s = _SECTIONS[lang]
    meth = methodology.lower()

    # Build methodology-specific sections
    order = 0

    def sec(sid, title, parent="", target=0, **kwargs):
        nonlocal order
        order += 1
        return Section(id=sid, title=title, parent_id=parent, order=order, target_words=target, **kwargs)

    sections = [
        sec("1", s["intro"], target=3000, notes="Problem statement, RQ, contribution overview"),
        sec("1.1", s["background"], parent="1", target=1500),
        sec("1.2", s["rq"], parent="1", target=500),
        sec("1.3", s["structure"], parent="1", target=500),
        sec("2", s["theory"], target=8000),
    ]

    # Theory subsections are generic — user fills in their own topics
    sections.extend([
        sec("2.1", s["lit_review"] + " I", parent="2", target=2000),
        sec("2.2", s["lit_review"] + " II", parent="2", target=2000),
        sec("2.3", s["lit_review"] + " III", parent="2", target=2000),
    ])

    sections.append(sec("3", s["methodology"], target=5000))

    # Methodology-specific subsections
    if "dsr" in meth or "design science" in meth:
        sections.extend([
            sec("3.1", s["dsr_approach"], parent="3", target=2000),
            sec("3.2", s["research_design"], parent="3", target=1500),
            sec("3.3", s["data_collection"], parent="3", target=1500),
            sec("4", s["artifact_design"], target=5000),
            sec("5", s["evaluation"], target=4000),
        ])
    elif "case" in meth:
        sections.extend([
            sec("3.1", s["case_selection"], parent="3", target=2000),
            sec("3.2", s["research_design"], parent="3", target=1500),
            sec("3.3", s["data_collection"], parent="3", target=1500),
            sec("4", s["within_case"], target=4000),
            sec("5", s["cross_case"], target=4000),
        ])
    elif "survey" in meth:
        sections.extend([
            sec("3.1", s["instrument_design"], parent="3", target=2000),
            sec("3.2", s["sample_strategy"], parent="3", target=1500),
            sec("3.3", s["statistical_analysis"], parent="3", target=1500),
            sec("4", s["statistical_analysis"], target=5000),
        ])
    elif "mixed" in meth:
        sections.extend([
            sec("3.1", s["research_design"], parent="3", target=2000),
            sec("3.2", s["data_collection"], parent="3", target=1500),
            sec("4", s["qual_strand"], target=4000),
            sec("5", s["quant_strand"], target=4000),
            sec("5.1", s["integration"], parent="5", target=2000),
        ])
    else:
        # Generic methodology structure
        sections.extend([
            sec("3.1", s["research_design"], parent="3", target=2000),
            sec("3.2", s["data_collection"], parent="3", target=1500),
            sec("4", s["lit_review"], target=5000),
        ])

    # Common ending sections
    disc_id = str(order + 1)
    sections.extend([
        sec(disc_id, s["discussion"], target=4000),
        sec(f"{disc_id}.1", s["theoretical_contributions"], parent=disc_id, target=2000),
        sec(f"{disc_id}.2", s["practical_implications"], parent=disc_id, target=2000),
        sec(str(order + 1), s["conclusion"], target=2000,
            notes="Revisit RQ, summarize contributions, limitations, future research"),
    ])

    template = PaperStructure(
        title=title or ("MBA Afsluttende Projekt" if lang == "da" else "MBA Final Paper"),
        research_question=rq,
        methodology=methodology,
        red_thread="[Skriv dit kerneargument i 2-3 sætninger her]" if lang == "da" else "[Write your core argument in 2-3 sentences here]",
        argument_chain=[
            "[Claim 1: The problem statement]",
            "[Claim 2: Why existing theory doesn't address it]",
            "[Claim 3: Your proposed contribution]",
            "[Claim 4: How the methodology validates it]",
            "[Claim 5: Implications for practice]",
        ],
        sections=sections,
        glossary=[],
    )

    return yaml.dump(
        {
            "title": template.title,
            "research_question": template.research_question,
            "red_thread": template.red_thread,
            "methodology": template.methodology,
            "argument_chain": template.argument_chain,
            "language": lang,
            "sections": [asdict(s) for s in template.sections],
            "glossary": [asdict(t) for t in template.glossary],
        },
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
