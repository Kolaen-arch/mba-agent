"""
PDF ingestion: extract text, clean, chunk, and prepare for embedding.
Handles English and Danish PDFs with proper character handling.
Filters out reference sections, headers/footers, figure captions, and noise.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf


@dataclass
class Chunk:
    """A text chunk with source metadata."""
    text: str
    source_file: str
    page_start: int
    page_end: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def source_tag(self) -> str:
        if self.page_start == self.page_end:
            return f"[SOURCE: {self.source_file}, page {self.page_start}]"
        return f"[SOURCE: {self.source_file}, pages {self.page_start}-{self.page_end}]"

    def to_context_str(self) -> str:
        return f"{self.source_tag}\n{self.text}"


# ── Cleaning ──

# Patterns for headers/footers (page numbers, journal names, DOIs)
HEADER_FOOTER_PATTERNS = [
    r'^\d+\s*$',                              # Bare page numbers
    r'^Page \d+ of \d+',                      # "Page X of Y"
    r'^\d+\s+[A-Z][a-z]+ et al\.',           # "42 Smith et al."
    r'^(Downloaded|Available) (from|at) http', # Download notices
    r'^https?://\S+$',                        # Bare URLs
    r'^DOI:?\s*\d',                           # DOI lines
    r'^\d{4}\s+\w+\s+\d+\s+\w+$',           # Date headers
    r'^©\s*\d{4}',                            # Copyright lines
    r'^ISSN\s',                               # ISSN
    r'^\[\d+\]$',                             # Bare reference numbers
    r'^Vol\.?\s*\d',                          # Volume headers
    r'^Journal of\s',                         # Journal name headers
    r'^This article\s',                       # Standard boilerplate
]
_header_re = [re.compile(p, re.IGNORECASE) for p in HEADER_FOOTER_PATTERNS]

# Patterns that indicate a references section has begun
REFS_START_PATTERNS = [
    r'^References?\s*$',
    r'^Bibliography\s*$',
    r'^Works Cited\s*$',
    r'^Litteratur(?:liste|fortegnelse)?\s*$',  # Danish
    r'^Kilder\s*$',                             # Danish: "Sources"
    r'^Referanser?\s*$',                        # Norwegian (close to Danish)
    r'^REFERENCES?\s*$',
]
_refs_start_re = [re.compile(p) for p in REFS_START_PATTERNS]

# Figure/table caption patterns
CAPTION_PATTERNS = [
    r'^(?:Figure|Fig\.?|Table|Tabel|Figur)\s+\d',  # "Figure 1", "Tabel 2"
    r'^(?:Source|Kilde|Note|Anm):',                 # Source/Note lines
]
_caption_re = [re.compile(p, re.IGNORECASE) for p in CAPTION_PATTERNS]


def _is_header_footer(line: str) -> bool:
    line = line.strip()
    if len(line) < 3:
        return True
    return any(r.match(line) for r in _header_re)


def _is_refs_start(line: str) -> bool:
    return any(r.match(line.strip()) for r in _refs_start_re)


def _is_caption(line: str) -> bool:
    return any(r.match(line.strip()) for r in _caption_re)


def _fix_encoding(text: str) -> str:
    """Fix common encoding artifacts in Danish/Swedish text."""
    fixes = {
        'Ã¦': 'æ', 'Ã¸': 'ø', 'Ã¥': 'å',
        'Ã†': 'Æ', 'Ã˜': 'Ø', 'Ã…': 'Å',
        'Ã¤': 'ä', 'Ã¶': 'ö', 'Ã¼': 'ü',
        'Ã„': 'Ä', 'Ã–': 'Ö', 'Ãœ': 'Ü',
        'â€™': "'", 'â€œ': '"', 'â€\x9d': '"',
        'â€"': '—', 'â€"': '–',
        'ï¬': 'fi', 'ï¬‚': 'fl',  # Ligature artifacts
        '\x00': '', '\ufffd': '',  # Null bytes, replacement chars
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)
    return text


def _clean_page_text(text: str) -> str:
    """Clean a single page's extracted text."""
    text = _fix_encoding(text)

    lines = text.split('\n')
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Skip empty/tiny lines
        if len(stripped) < 4:
            continue

        # Skip headers/footers
        if _is_header_footer(stripped):
            continue

        # Skip figure/table captions (they add noise, not substance)
        if _is_caption(stripped):
            continue

        # Fix hyphenation at line breaks (common in PDFs)
        if cleaned and cleaned[-1].endswith('-'):
            # Join hyphenated word
            prev = cleaned[-1][:-1]
            cleaned[-1] = prev + stripped
        else:
            cleaned.append(stripped)

    return '\n'.join(cleaned)


def extract_pdf_text(pdf_path: str, strip_references: bool = True) -> list[tuple[int, str]]:
    """
    Extract and clean text from PDF.
    Returns list of (page_number, text) tuples.
    Stops at the References section if strip_references=True.
    """
    doc = fitz.open(pdf_path)
    pages = []
    refs_started = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")

        # Clean
        text = _clean_page_text(raw_text)

        # Check if references section starts on this page
        if strip_references and not refs_started:
            for line in text.split('\n'):
                if _is_refs_start(line):
                    # Keep content before the references heading
                    idx = text.find(line)
                    if idx > 0:
                        text = text[:idx].strip()
                    else:
                        text = ""
                    refs_started = True
                    break

        if refs_started and not text:
            continue  # Skip pure reference pages

        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = text.strip()

        if text and len(text) > 50:
            pages.append((page_num + 1, text))

    doc.close()
    return pages


def extract_pdf_metadata(pdf_path: str) -> dict:
    """Extract metadata from PDF."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    page_count = len(doc)
    doc.close()

    return {
        "title": _fix_encoding(meta.get("title", "")),
        "author": _fix_encoding(meta.get("author", "")),
        "subject": _fix_encoding(meta.get("subject", "")),
        "filename": os.path.basename(pdf_path),
        "page_count": page_count,
    }


def _is_substantive_paragraph(text: str) -> bool:
    """Filter out non-substantive text blocks."""
    text = text.strip()
    if len(text) < 40:
        return False

    # Skip lines that are mostly numbers or symbols
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False

    # Skip very short "paragraphs" that are likely artifacts
    words = text.split()
    if len(words) < 8:
        return False

    return True


def chunk_pages(
    pages: list[tuple[int, str]],
    source_file: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Chunk extracted pages into overlapping segments.
    chunk_size and chunk_overlap are in characters.
    """
    chunks = []
    segments = []

    for page_num, text in pages:
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if _is_substantive_paragraph(para):
                segments.append((page_num, para))

    if not segments:
        return chunks

    current_text = ""
    current_page_start = segments[0][0]
    current_page_end = current_page_start
    chunk_idx = 0
    char_limit = chunk_size * 4  # Convert to chars

    for page_num, para in segments:
        test_text = current_text + "\n\n" + para if current_text else para

        if len(test_text) > char_limit and current_text:
            chunks.append(Chunk(
                text=current_text.strip(),
                source_file=source_file,
                page_start=current_page_start,
                page_end=current_page_end,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

            overlap_text = current_text[-(chunk_overlap * 4):]
            current_text = overlap_text + "\n\n" + para
            current_page_start = current_page_end
            current_page_end = page_num
        else:
            current_text = test_text
            current_page_end = page_num

    if current_text.strip():
        chunks.append(Chunk(
            text=current_text.strip(),
            source_file=source_file,
            page_start=current_page_start,
            page_end=current_page_end,
            chunk_index=chunk_idx,
        ))

    return chunks


def ingest_directory(
    papers_dir: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    strip_references: bool = True,
) -> tuple[list[Chunk], list[dict]]:
    """Ingest all PDFs from a directory."""
    papers_path = Path(papers_dir)
    pdf_files = sorted(papers_path.glob("**/*.pdf"))

    all_chunks = []
    all_metadata = []
    skipped = 0

    for pdf_file in pdf_files:
        filename = pdf_file.name
        try:
            pages = extract_pdf_text(str(pdf_file), strip_references=strip_references)
            if not pages:
                print(f"  ⚠ Skipped (no extractable text): {filename}")
                skipped += 1
                continue

            meta = extract_pdf_metadata(str(pdf_file))
            all_metadata.append(meta)

            chunks = chunk_pages(pages, filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
            print(f"  ✓ {filename}: {meta['page_count']}p → {len(pages)} content pages → {len(chunks)} chunks")

        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            skipped += 1

    if skipped:
        print(f"\n  ⚠ {skipped} files skipped (scanned/empty PDFs). Consider OCR for these.")

    return all_chunks, all_metadata
