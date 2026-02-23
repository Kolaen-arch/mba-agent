"""
LaTeX export handler: converts paper structure to a .tex file
with proper \\cite{} commands from BibTeX keys.
"""

import re
from pathlib import Path


def export_to_latex(
    sections: list[dict],
    citations,
    output_path: str,
    title: str = "",
    author: str = "",
    bib_file: str = "references.bib",
) -> str:
    """Export paper structure to a .tex file with \\cite{} references."""
    lines = [
        "\\documentclass[12pt,a4paper]{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{times}",
        "\\usepackage{setspace}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{natbib}",
        "\\usepackage{hyperref}",
        "\\doublespacing",
        "",
    ]

    if title:
        lines.append(f"\\title{{{_escape_latex(title)}}}")
    if author:
        lines.append(f"\\author{{{_escape_latex(author)}}}")

    lines.extend([
        "",
        "\\begin{document}",
    ])

    if title:
        lines.append("\\maketitle")
    lines.extend([
        "\\tableofcontents",
        "\\newpage",
        "",
    ])

    for sec in sorted(sections, key=lambda s: s.get("order", 0)):
        level = "subsection" if sec.get("parent_id") else "section"
        lines.append(f"\\{level}{{{_escape_latex(sec['title'])}}}")
        lines.append(f"\\label{{sec:{sec['id']}}}")

        content = sec.get("content", "")
        if content:
            content = _convert_citations_to_latex(content, citations)
            lines.append(content)
        else:
            lines.append("\\textit{[Section not yet written]}")
        lines.append("")

    lines.extend([
        "",
        "\\bibliographystyle{apalike}",
        f"\\bibliography{{{bib_file.replace('.bib', '')}}}",
        "",
        "\\end{document}",
    ])

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _convert_citations_to_latex(text: str, citations) -> str:
    """Convert (Author, Year) in-text citations to \\cite{bib_key}."""
    def replace_cite(match):
        in_text = match.group(1)
        # Strip page numbers to get the lookup key
        key = re.sub(r',\s*pp?\.?\s*\d+(?:-\d+)?', '', in_text).strip()
        cit = citations.citations.get(key)
        if cit and cit.bib_key:
            return f"\\cite{{{cit.bib_key}}}"
        return match.group(0)  # Keep original if no match

    pattern = r'\(([A-Z][a-z]+(?:\s(?:&|and)\s[A-Z][a-z]+)?(?:\s+et al\.)?,\s*\d{4}(?:[a-z])?(?:,\s*pp?\.?\s*\d+(?:-\d+)?)?)\)'
    return re.sub(pattern, replace_cite, text)


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters."""
    chars = {
        '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', '_': '\\_',
        '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
    }
    for old, new in chars.items():
        text = text.replace(old, new)
    return text
