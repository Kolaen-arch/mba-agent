# MBA Paper Agent

Local LLM agent for writing a ~100-page MBA paper. Uses Claude Opus + Sonnet via the Anthropic API.
Ingests 100+ PDFs (English + Danish), streams responses in real-time with extended thinking,
tracks paper structure, ensures section transitions are seamless, and prevents citation hallucination via BibTeX import.

## Setup

```bash
tar -xzf mba-agent.tar.gz && cd mba-agent
make install
cp config.example.yaml config.yaml   # add your Anthropic API key
```

Drop your files:
- PDFs → `papers/`
- .docx drafts → `current_draft/`
- BibTeX → `references.bib` (project root)

```bash
make ingest        # Process PDFs → vector store
make import-bib    # Import references from .bib or .json
make scaffold      # Generate paper_structure.yaml template
make serve         # http://localhost:5000 (gunicorn, async)
```

## Model Routing & Extended Thinking

Tasks are routed to the optimal model automatically:

| Mode | Model | Extended Thinking | Why |
|---|---|---|---|
| Draft | Opus | 10K tokens | Deep reasoning for academic writing |
| Synthesize | Opus | 12K tokens | Cross-source analysis |
| Review | Opus | 8K tokens | Critical evaluation |
| Transitions | Opus | 6K tokens | Structural analysis |
| Consistency | Opus | 6K tokens | Multi-dimensional checking |
| Structure | Opus | 8K tokens | Paper architecture |
| Chat | Sonnet | Off | Fast, cheap, interactive |
| Cite | Sonnet | Off | Lookup task, no deep reasoning |
| Edit DOCX | Sonnet | Off | Mechanical changes |

Configurable in `config.yaml`:
```yaml
model_routing:
  draft: "claude-opus-4-6"
  chat: "claude-sonnet-4-5-20250929"
thinking_budget:
  draft: 10000    # tokens for extended thinking
  chat: 0         # disabled
```

Also adjustable at runtime via `PATCH /api/config/models`.

## Streaming

All responses stream via Server-Sent Events. You see:
1. **Model badge** — which model is handling this request
2. **Thinking indicator** — collapsible box showing the model's reasoning (when enabled)
3. **Text streaming** — response appears word-by-word, no 45-second wait

The UI uses `fetch` + `ReadableStream` (not EventSource) to support POST requests with payloads.

## Citation System (No More Hallucination)

The #1 problem with LLM-assisted academic writing: fabricated references.

**Solution**: Import your actual bibliography from Zotero or BibTeX. The agent only uses verified references.

```bash
# Export from Zotero: File → Export Library → CSL JSON
# Or export as BibTeX (.bib)
make import-bib
```

What happens:
1. References are parsed into APA 7th format
2. PDF filenames are matched to citations (e.g., `pine_gilmore_1998.pdf` → `Pine & Gilmore, 1998`)
3. When drafting, only verified references are injected as context
4. The reference list separates verified (from BibTeX) vs unverified entries
5. Any citation the LLM generates that isn't in BibTeX gets flagged

Import via UI: click **📚 Import BibTeX / Zotero** in the sidebar.

## PDF Extraction Improvements

Academic PDFs are messy. The ingestion pipeline now:

- **Strips reference sections** — "References", "Bibliography", "Litteraturfortegnelse" detected and removed
- **Removes headers/footers** — page numbers, DOI lines, journal names, copyright notices
- **Filters figure/table captions** — "Figure 1: ..." adds noise, not substance
- **Fixes Danish/Swedish encoding** — `Ã¦` → `æ`, `Ã¸` → `ø`, `Ã¥` → `å`, ligature fixes
- **Filters non-substantive paragraphs** — blocks that are too short or mostly non-alpha
- **Fixes line-break hyphenation** — joins "theo-\nretical" → "theoretical"

## Eight Modes

| Mode | What it does |
|---|---|
| **Chat** | Freeform discussion (Sonnet, fast) |
| **Draft** | Write sections with adjacent context + glossary (Opus, thinking) |
| **Synthesize** | Literature synthesis across full library (Opus, thinking) |
| **Review** | Critique and improve text (Opus, thinking) |
| **Cite** | Find and format APA 7th citations (Sonnet) |
| **Transitions** | Bridge analysis between adjacent sections (Opus, thinking) |
| **Consistency** | Terminology, citation, argument alignment check (Opus, thinking) |
| **Structure** | Paper outline analysis, gap identification (Opus, thinking) |

## Persistence

Everything survives restarts:

| Data | Storage | Location |
|---|---|---|
| Chat sessions + messages | SQLite | `data/mba_agent.db` |
| Document versions | SQLite | `data/mba_agent.db` |
| Paper structure | YAML | `paper_structure.yaml` |
| Vector embeddings | ChromaDB | `.chroma_db/` |
| Citations (verified) | JSON | `output/citations.json` |

## Running with Gunicorn

`make serve` uses gunicorn with gevent workers:
```bash
gunicorn -k gevent -w 1 -b 0.0.0.0:5000 --timeout 300 "mba_agent.web.app:create_app()"
```

This means the UI doesn't freeze during 45-second API calls. SSE streaming works properly.
For development with auto-reload: `make serve-dev`.

## Context Overflow Protection

Each mode has a token budget for RAG context:

| Mode | Max context tokens |
|---|---|
| Synthesize | 100K |
| Draft | 80K |
| Review | 60K |
| Chat | 40K |
| Cite | 30K |

If retrieved context exceeds the budget, it's truncated from the end (less relevant chunks first).
Token estimation: ~3 chars per token for mixed English/Danish text.

## Project Structure

```
mba-agent/
├── config.yaml               ← API key, model routing, thinking budgets
├── paper_structure.yaml       ← your paper's brain (edit this)
├── references.bib             ← your BibTeX file
├── papers/                    ← 100+ PDFs
├── current_draft/             ← .docx files
├── output/                    ← agent outputs + citations.json
├── data/                      ← SQLite (auto-created)
├── mba_agent/
│   ├── agent.py               ← Claude API, streaming, model routing, thinking
│   ├── paper_structure.py     ← structure, transitions, glossary, progress
│   ├── prompts.py             ← 8 system prompts (customizable)
│   ├── store.py               ← ChromaDB vector store
│   ├── ingest.py              ← PDF extraction + cleaning + chunking
│   ├── citations.py           ← BibTeX/Zotero import, tracking, APA formatting
│   ├── database.py            ← SQLite persistence
│   ├── docx_handler.py        ← Word read/write/edit
│   ├── cli.py                 ← CLI commands
│   └── web/
│       ├── app.py             ← Flask + SSE streaming + all API routes
│       └── templates/
│           └── index.html     ← Web UI with streaming
└── .chroma_db/
```
