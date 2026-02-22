"""
Citation manager: tracks, deduplicates, formats references,
and imports from BibTeX / Zotero JSON exports.
This is the single source of truth for the reference list —
the LLM never fabricates reference entries.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Citation:
    """A tracked citation."""
    key: str              # e.g., "Pine & Gilmore, 1999"
    bib_key: str = ""     # BibTeX key e.g., "pine1999experience"
    source_file: str = "" # PDF filename
    full_reference: str = ""  # Full APA 7th reference
    title: str = ""
    authors: list = field(default_factory=list)  # ["Pine, B. J.", "Gilmore, J. H."]
    year: str = ""
    journal: str = ""
    doi: str = ""
    used_in: list[str] = field(default_factory=list)
    from_bibtex: bool = False  # True if imported from .bib, not fabricated


class CitationManager:
    """Tracks citations, imports from BibTeX/Zotero, matches to PDFs."""

    def __init__(self, storage_path: str = "./output/citations.json"):
        self.storage_path = Path(storage_path)
        self.citations: dict[str, Citation] = {}
        self._bib_lookup: dict[str, str] = {}  # bib_key → citation key
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for key, val in data.items():
                    self.citations[key] = Citation(
                        key=val.get("key", key),
                        bib_key=val.get("bib_key", ""),
                        source_file=val.get("source_file", ""),
                        full_reference=val.get("full_reference", ""),
                        title=val.get("title", ""),
                        authors=val.get("authors", []),
                        year=val.get("year", ""),
                        journal=val.get("journal", ""),
                        doi=val.get("doi", ""),
                        used_in=val.get("used_in", []),
                        from_bibtex=val.get("from_bibtex", False),
                    )
                self._rebuild_bib_lookup()
            except (json.JSONDecodeError, TypeError):
                self.citations = {}

    def _rebuild_bib_lookup(self) -> None:
        self._bib_lookup = {}
        for key, c in self.citations.items():
            if c.bib_key:
                self._bib_lookup[c.bib_key] = key

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for k, v in self.citations.items():
            data[k] = {
                "key": v.key,
                "bib_key": v.bib_key,
                "source_file": v.source_file,
                "full_reference": v.full_reference,
                "title": v.title,
                "authors": v.authors,
                "year": v.year,
                "journal": v.journal,
                "doi": v.doi,
                "used_in": v.used_in,
                "from_bibtex": v.from_bibtex,
            }
        self.storage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def add(self, key: str, source_file: str = "", full_reference: str = "",
            section: str = "") -> None:
        if key in self.citations:
            if section and section not in self.citations[key].used_in:
                self.citations[key].used_in.append(section)
            if full_reference and not self.citations[key].full_reference:
                self.citations[key].full_reference = full_reference
        else:
            self.citations[key] = Citation(
                key=key, source_file=source_file,
                full_reference=full_reference,
                used_in=[section] if section else [],
            )
        self.save()

    # ── BibTeX Import ──

    def import_bibtex(self, bib_path: str) -> dict:
        """
        Import references from a .bib file.
        Returns stats: {imported, skipped, total}.
        """
        import bibtexparser

        with open(bib_path) as f:
            bib_db = bibtexparser.load(f)

        imported = 0
        skipped = 0

        for entry in bib_db.entries:
            bib_key = entry.get("ID", "")
            authors_raw = entry.get("author", "")
            year = entry.get("year", "")
            title = entry.get("title", "").strip("{}")
            journal = entry.get("journal", "") or entry.get("booktitle", "")
            doi = entry.get("doi", "")

            if not authors_raw or not year:
                skipped += 1
                continue

            # Parse authors into APA format
            authors = _parse_bib_authors(authors_raw)
            in_text_key = _make_in_text_key(authors, year)

            if in_text_key in self.citations:
                # Update existing with bib data
                c = self.citations[in_text_key]
                c.bib_key = bib_key
                c.authors = authors
                c.year = year
                c.title = title
                c.journal = journal
                c.doi = doi
                c.from_bibtex = True
                if not c.full_reference:
                    c.full_reference = _format_apa7(authors, year, title, journal, doi, entry)
            else:
                self.citations[in_text_key] = Citation(
                    key=in_text_key,
                    bib_key=bib_key,
                    authors=authors,
                    year=year,
                    title=title,
                    journal=journal,
                    doi=doi,
                    full_reference=_format_apa7(authors, year, title, journal, doi, entry),
                    from_bibtex=True,
                )
            imported += 1

        self._rebuild_bib_lookup()
        self.save()
        return {"imported": imported, "skipped": skipped, "total": len(bib_db.entries)}

    # ── Zotero JSON Import ──

    def import_zotero_json(self, json_path: str) -> dict:
        """
        Import from Zotero JSON export (File → Export Library → CSL JSON).
        """
        with open(json_path) as f:
            items = json.load(f)

        if isinstance(items, dict):
            items = items.get("items", [items])

        imported = 0
        skipped = 0

        for item in items:
            authors_data = item.get("author", [])
            year = ""
            issued = item.get("issued", {})
            if isinstance(issued, dict):
                parts = issued.get("date-parts", [[]])
                if parts and parts[0]:
                    year = str(parts[0][0])

            title = item.get("title", "")
            journal = item.get("container-title", "")
            doi = item.get("DOI", "")

            if not authors_data or not year:
                skipped += 1
                continue

            authors = []
            for a in authors_data:
                family = a.get("family", "")
                given = a.get("given", "")
                if family:
                    initials = ". ".join(n[0] for n in given.split() if n) + "." if given else ""
                    authors.append(f"{family}, {initials}" if initials else family)

            in_text_key = _make_in_text_key(authors, year)

            entry_stub = {"volume": item.get("volume", ""), "pages": item.get("page", "")}

            if in_text_key in self.citations:
                c = self.citations[in_text_key]
                c.authors = authors
                c.year = year
                c.title = title
                c.journal = journal
                c.doi = doi
                c.from_bibtex = True
                if not c.full_reference:
                    c.full_reference = _format_apa7(authors, year, title, journal, doi, entry_stub)
            else:
                self.citations[in_text_key] = Citation(
                    key=in_text_key, authors=authors, year=year,
                    title=title, journal=journal, doi=doi,
                    full_reference=_format_apa7(authors, year, title, journal, doi, entry_stub),
                    from_bibtex=True,
                )
            imported += 1

        self._rebuild_bib_lookup()
        self.save()
        return {"imported": imported, "skipped": skipped, "total": len(items)}

    # ── PDF Matching ──

    def match_pdfs_to_citations(self, pdf_filenames: list[str]) -> dict:
        """
        Try to match PDF filenames to citations using author/year patterns.
        Returns {citation_key: matched_filename}.
        """
        matches = {}
        for key, c in self.citations.items():
            if c.source_file:
                continue  # Already matched
            # Try matching by author surname + year in filename
            for fn in pdf_filenames:
                fn_lower = fn.lower()
                if c.year and c.year in fn_lower:
                    # Check if any author surname appears
                    for author in c.authors:
                        surname = author.split(",")[0].strip().lower()
                        if len(surname) > 2 and surname in fn_lower:
                            c.source_file = fn
                            matches[key] = fn
                            break
                if c.source_file:
                    break
        if matches:
            self.save()
        return matches

    # ── Extraction from text ──

    def extract_citations_from_text(self, text: str, section: str = "") -> list[str]:
        """Extract APA in-text citations from generated text."""
        pattern = r'\(([A-ZÆØÅÄÖa-z][a-zæøåäö]+(?:\s(?:&|and)\s[A-ZÆØÅÄÖa-z][a-zæøåäö]+)?(?:\s+et al\.)?,\s*\d{4}(?:[a-z])?(?:,\s*pp?\.?\s*\d+(?:-\d+)?)?)\)'
        matches = re.findall(pattern, text)

        for match in matches:
            key = re.sub(r',\s*p\.?\s*\d+(?:-\d+)?', '', match).strip()
            self.add(key=key, section=section)

        self.save()
        return matches

    # ── Output ──

    def generate_reference_list(self) -> str:
        """Generate a formatted reference list with clear provenance."""
        verified = []
        unverified = []

        for key in sorted(self.citations.keys()):
            c = self.citations[key]
            if c.from_bibtex and c.full_reference:
                verified.append(f"- {c.full_reference}")
            elif c.full_reference:
                unverified.append(f"- {c.full_reference} ⚠ *[not from BibTeX — verify]*")
            else:
                unverified.append(f"- {c.key} — *[reference entry missing]*")

        lines = ["# Reference List\n"]
        if verified:
            lines.append(f"## Verified ({len(verified)} from BibTeX/Zotero)\n")
            lines.extend(verified)
        if unverified:
            lines.append(f"\n## Needs Verification ({len(unverified)})\n")
            lines.extend(unverified)

        return "\n".join(lines)

    def find_missing_references(self) -> list[str]:
        return [k for k, v in self.citations.items() if not v.full_reference]

    def find_unverified_references(self) -> list[str]:
        return [k for k, v in self.citations.items() if not v.from_bibtex]

    def get_reference_for_key(self, key: str) -> str | None:
        """Get the full reference for an in-text citation key."""
        c = self.citations.get(key)
        if c and c.full_reference:
            return c.full_reference
        return None

    def build_context_for_agent(self) -> str:
        """
        Build citation context the agent can use when drafting.
        Only includes verified references so the LLM can't hallucinate entries.
        """
        lines = ["[VERIFIED REFERENCES — use only these for citations]:"]
        for key in sorted(self.citations.keys()):
            c = self.citations[key]
            if c.from_bibtex and c.full_reference:
                lines.append(f"  {c.key} → {c.full_reference}")
        return "\n".join(lines)

    @property
    def stats(self) -> dict:
        total = len(self.citations)
        verified = sum(1 for c in self.citations.values() if c.from_bibtex)
        with_ref = sum(1 for c in self.citations.values() if c.full_reference)
        return {
            "total": total,
            "verified_from_bibtex": verified,
            "with_reference": with_ref,
            "missing_reference": total - with_ref,
            "unverified": total - verified,
        }


# ── Helper functions ──

def _parse_bib_authors(raw: str) -> list[str]:
    """Parse BibTeX author string into APA-formatted list."""
    # BibTeX uses "and" to separate authors
    parts = re.split(r'\s+and\s+', raw)
    authors = []
    for part in parts:
        part = part.strip().strip("{}")
        if "," in part:
            # Already "Surname, Given"
            pieces = part.split(",", 1)
            surname = pieces[0].strip()
            given = pieces[1].strip() if len(pieces) > 1 else ""
            initials = ". ".join(n[0] for n in given.split() if n) + "." if given else ""
            authors.append(f"{surname}, {initials}" if initials else surname)
        elif " " in part:
            # "Given Surname" → "Surname, G."
            words = part.split()
            surname = words[-1]
            given = " ".join(words[:-1])
            initials = ". ".join(n[0] for n in given.split() if n) + "."
            authors.append(f"{surname}, {initials}")
        else:
            authors.append(part)
    return authors


def _make_in_text_key(authors: list[str], year: str) -> str:
    """Generate an in-text citation key like 'Pine & Gilmore, 1999'."""
    if not authors:
        return f"Unknown, {year}"

    surnames = [a.split(",")[0].strip() for a in authors]

    if len(surnames) == 1:
        return f"{surnames[0]}, {year}"
    elif len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]}, {year}"
    else:
        return f"{surnames[0]} et al., {year}"


def _format_apa7(authors: list[str], year: str, title: str,
                  journal: str, doi: str, entry: dict) -> str:
    """Format a reference in APA 7th edition style."""
    # Authors
    if len(authors) <= 20:
        if len(authors) == 1:
            auth_str = authors[0]
        elif len(authors) == 2:
            auth_str = f"{authors[0]} & {authors[1]}"
        else:
            auth_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
    else:
        auth_str = ", ".join(authors[:19]) + f", ... {authors[-1]}"

    # Title: sentence case
    title_clean = title.strip("{}")

    # Journal article
    volume = entry.get("volume", "")
    pages = entry.get("pages", "")

    parts = [f"{auth_str} ({year})."]

    if journal:
        parts.append(f"{title_clean}.")
        j_part = f"*{journal}*"
        if volume:
            j_part += f", *{volume}*"
        if pages:
            j_part += f", {pages}"
        j_part += "."
        parts.append(j_part)
    else:
        # Book or other
        parts.append(f"*{title_clean}*.")

    if doi:
        parts.append(f"https://doi.org/{doi}")

    return " ".join(parts)
