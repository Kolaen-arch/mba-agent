# MBA Paper Agent

Local LLM agent for writing ~100-page MBA papers. Uses Claude Opus + Sonnet via the Anthropic API.

## Commands
```bash
make install        # pip install -r requirements.txt
make ingest         # Process PDFs in papers/ → ChromaDB
make import-bib     # Import references from .bib or .json
make scaffold       # Generate paper_structure.yaml template
make serve          # http://localhost:5000 (gunicorn + gevent)
make serve-dev      # Flask dev mode with auto-reload
```

## Verify changes
```bash
python -c "from mba_agent.agent import MBAAgent; from mba_agent.store import PaperStore; from mba_agent.web.app import create_app; print('OK')"
```

## Architecture
```
User → Flask (web/app.py) → create_app()
         │
         ├── MBAAgent (agent.py) — Claude API + streaming + model routing
         │     ├── _stream() → SSE chunks (model/thinking/text/done)
         │     └── _call() → non-streaming fallback
         │
         ├── PaperStore (store.py) — ChromaDB vector store
         ├── CitationManager (citations.py) — BibTeX/Zotero import + APA 7th
         ├── PaperStructure (paper_structure.py) — YAML-based structure + progress
         ├── database (database.py) — SQLite: sessions, messages, doc versions
         └── Ingester (ingest.py) — PDF extraction + cleaning + chunking
```

## Module layout
```
mba_agent/
  agent.py            — Claude API, streaming, model routing, extended thinking
  paper_structure.py  — Structure, transitions, glossary, progress tracking
  prompts.py          — 8 system prompts (one per mode)
  store.py            — ChromaDB vector store (PaperStore)
  ingest.py           — PDF extraction + cleaning + chunking
  citations.py        — BibTeX/Zotero import, tracking, APA formatting
  database.py         — SQLite persistence (sessions, messages, doc versions)
  docx_handler.py     — Word read/write/edit
  cli.py              — CLI commands
  web/
    app.py            — Flask + SSE streaming + all API routes
    templates/
      index.html      — Full SPA with Academic Library design system
```

## 8 Modes (routed via model_routing in config.yaml)
| Mode | Model | Thinking | Purpose |
|------|-------|----------|---------|
| Chat | Sonnet | Off | Fast freeform discussion |
| Draft | Opus | 10K | Section writing with adjacent context |
| Synthesize | Opus | 12K | Cross-source literature synthesis |
| Review | Opus | 8K | Critique and improve text |
| Cite | Sonnet | Off | Find/format APA 7th citations |
| Transitions | Opus | 6K | Bridge analysis between sections |
| Consistency | Opus | 6K | Terminology/citation/argument alignment |
| Structure | Opus | 8K | Paper outline analysis, gap identification |

Mode routing lives in `config.yaml` → `model_routing` and `thinking_budget`. Adjustable at runtime via `PATCH /api/config/models`.

## Where to change what
- **Add/change system prompts** → `prompts.py` (8 prompts: CHAT_SYSTEM, DRAFT_SYSTEM, etc.)
- **Change mode routing** → `web/app.py` → `_route_mode()`
- **Change model selection** → `config.yaml` → `model_routing`
- **Change thinking budgets** → `config.yaml` → `thinking_budget`
- **Change PDF extraction** → `ingest.py` → `Ingester`
- **Change citation logic** → `citations.py` → `CitationManager`
- **Change paper structure** → `paper_structure.py` + `paper_structure.yaml`
- **Change vector store** → `store.py` → `PaperStore`
- **Change UI design** → `web/templates/index.html` (see `.interface-design/system.md`)
- **Change API routes** → `web/app.py` → `create_app()`

## Citation system (anti-hallucination)
- References imported from BibTeX/Zotero → `output/citations.json`
- PDF filenames matched to citations (e.g., `pine_gilmore_1998.pdf` → `Pine & Gilmore, 1998`)
- Only verified references injected into draft/synthesize context
- Unverified citations flagged in output
- Reference list separates verified vs unverified

## Persistence
| Data | Storage | Location |
|------|---------|----------|
| Chat sessions + messages | SQLite | `data/mba_agent.db` |
| Document versions | SQLite | `data/mba_agent.db` |
| Paper structure | YAML | `paper_structure.yaml` |
| Vector embeddings | ChromaDB | `.chroma_db/` |
| Citations (verified) | JSON | `output/citations.json` |

## Key files (not in repo, gitignored)
- `config.yaml` — API key + model config (copy from `config.example.yaml`)
- `papers/` — Source PDFs
- `current_draft/` — .docx working files
- `data/` — SQLite DB (auto-created)
- `.chroma_db/` — Vector store (auto-created after ingest)

## Design system
UI uses the "Academic Library" design direction. Tokens and patterns documented in `.interface-design/system.md`. Key principles:
- Warm cream/parchment palette (`#FAF8F4` base)
- Lora serif for agent content, Inter for UI chrome
- Readable CSS class names (`.sidebar`, `.mode-pill`, `.msg-content`)
- All design tokens in CSS `:root` custom properties

## Gotchas
1. **`config.yaml` required** — app crashes without it. Copy from `config.example.yaml` and add Anthropic API key.
2. **Gunicorn on Windows** — `make serve` uses gunicorn which doesn't work natively on Windows. Use `make serve-dev` (Flask dev server) instead.
3. **ChromaDB first run** — First `make ingest` downloads the embedding model (~500MB). Subsequent runs are fast.
4. **SSE streaming** — Uses `fetch` + `ReadableStream` (not EventSource) to support POST payloads.
5. **Database module** — Uses bare functions (`add_message`, `create_session`), not a class. Import as `from .. import database as db`.

## Language conventions
- Code: English
- Paper content: English (Scandinavian-influenced)
- Danish/Swedish sources: cited in original language
- Commit messages: English

## Git workflow
- `config.yaml` is gitignored (contains API key)
- Never commit `.chroma_db/`, `data/`, or `output/`
