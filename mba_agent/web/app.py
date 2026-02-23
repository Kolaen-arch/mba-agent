"""
Flask web application for the MBA Paper Agent.
Supports SSE streaming, BibTeX/Zotero import, model routing,
document editing, paper structure management, and full history.
"""

import difflib
import json
import logging
import os
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import markdown as md
import yaml
from flask import Flask, Response, request, jsonify, render_template, send_file

from ..agent import MBAAgent
from ..store import PaperStore
from ..citations import CitationManager
from ..docx_handler import (
    read_docx, write_docx_from_markdown, apply_changes_to_docx,
    list_docx_files, read_docx_section, export_full_paper,
)
from ..paper_structure import (
    load_structure, save_structure, get_section, get_adjacent_sections,
    build_transition_context, build_glossary_context, compute_progress,
    find_terminology_issues, generate_scaffold_yaml,
    _match_heading_to_section, build_all_adjacent_pairs,
)
from .. import database as db
from .. import prompts


def load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_app() -> Flask:
    cfg = load_config()

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["SECRET_KEY"] = os.urandom(24).hex()

    db.init_db()

    store = PaperStore(
        persist_dir=cfg.get("chroma_persist_dir", "./.chroma_db"),
        embedding_model=cfg.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )

    # Build model/thinking maps from config
    model_map = cfg.get("model_routing", None)
    thinking_map = cfg.get("thinking_budget", None)

    agent = MBAAgent(
        api_key=cfg.get("anthropic_api_key", ""),
        default_model=cfg.get("model", "claude-opus-4-6"),
        model_map=model_map,
        thinking_map=thinking_map,
        max_output_tokens=cfg.get("max_output_tokens", 16000),
    )

    citations = CitationManager(
        storage_path=f"{cfg.get('output_dir', './output')}/citations.json"
    )

    app.config["MBA_CFG"] = cfg
    app.config["MBA_STORE"] = store
    app.config["MBA_AGENT"] = agent
    app.config["MBA_CITATIONS"] = citations

    log = logging.getLogger("mba_agent")

    def _sanitize_error(e: Exception) -> str:
        """Log full traceback, return safe user-facing message."""
        error_id = str(uuid.uuid4())[:8]
        log.error(f"Error [{error_id}]: {type(e).__name__}: {e}\n{traceback.format_exc()}")

        import anthropic
        if isinstance(e, anthropic.APITimeoutError):
            return f"Request timed out. Please try again. (ref: {error_id})"
        if isinstance(e, anthropic.APIConnectionError):
            return f"Could not connect to Anthropic API. Check your network. (ref: {error_id})"
        if isinstance(e, anthropic.AuthenticationError):
            return f"Invalid API key. Check config.yaml. (ref: {error_id})"
        if isinstance(e, anthropic.RateLimitError):
            return f"Rate limited by Anthropic. Wait a moment and retry. (ref: {error_id})"
        if isinstance(e, anthropic.APIStatusError):
            return f"API error ({e.status_code}). (ref: {error_id})"

        # Generic — don't expose internals
        return f"Something went wrong. (ref: {error_id})"

    # ── Security: path validation ──

    def _validate_doc_path(path: str) -> bool:
        """Validate that path is under allowed directories (prevents path traversal)."""
        if not path:
            return False
        try:
            resolved = os.path.realpath(path)
            draft_dir = os.path.realpath(cfg.get("current_draft_dir", "./current_draft"))
            output_dir = os.path.realpath(cfg.get("output_dir", "./output"))
            # Allow paths under draft dir or output dir
            return resolved.startswith(draft_dir) or resolved.startswith(output_dir)
        except (ValueError, OSError):
            return False

    # ── Pages ──

    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Session API ──

    @app.route("/api/sessions", methods=["GET"])
    def api_list_sessions():
        return jsonify(db.list_sessions(limit=100))

    @app.route("/api/sessions", methods=["POST"])
    def api_create_session():
        data = request.json or {}
        title = data.get("title", f"Session {datetime.now().strftime('%b %d %H:%M')}")
        mode = data.get("mode", "chat")
        return jsonify(db.create_session(title, mode))

    @app.route("/api/sessions/<sid>", methods=["GET"])
    def api_get_session(sid):
        session = db.get_session(sid)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        return jsonify({"session": session, "messages": db.get_messages(sid)})

    @app.route("/api/sessions/<sid>", methods=["PATCH"])
    def api_update_session(sid):
        data = request.json or {}
        if "title" in data:
            db.update_session_title(sid, data["title"])
        return jsonify({"ok": True})

    @app.route("/api/sessions/<sid>", methods=["DELETE"])
    def api_delete_session(sid):
        db.delete_session(sid)
        return jsonify({"ok": True})

    @app.route("/api/sessions/<sid>/cost", methods=["GET"])
    def api_session_cost(sid):
        return jsonify(db.get_session_cost(sid))

    # ── Mode auto-inference (hidden modes routed from chat) ──

    _CITE_PATTERNS = [
        re.compile(r'\([A-Z][a-z]+.*\d{4}\)'),
        re.compile(r'\bcite\b', re.IGNORECASE),
        re.compile(r'\breference\b.*\bfor\b', re.IGNORECASE),
        re.compile(r'\bfind\s+source\b', re.IGNORECASE),
        re.compile(r'\bwhat\s+do\s+\w+\s+(and|&)\s+\w+\s+(argue|say|claim)\b', re.IGNORECASE),
    ]
    _STRUCTURE_PATTERNS = [
        re.compile(r'\b(outline|structure|gaps?)\b', re.IGNORECASE),
        re.compile(r'\bchapter\b.*\b(order|sequence|missing)\b', re.IGNORECASE),
        re.compile(r'\bsections?\b.*\b(reorganize|reorder|add|remove)\b', re.IGNORECASE),
    ]
    _DRAFT_PATTERNS = [
        re.compile(r'\b(write|draft)\s+section\b', re.IGNORECASE),
        re.compile(r'\b(write|draft)\s+\d+\.?\d*\b', re.IGNORECASE),
        re.compile(r'\b(write|draft)\s+(the\s+)?(introduction|methodology|discussion|conclusion)\b', re.IGNORECASE),
    ]
    _REVIEW_PATTERNS = [
        re.compile(r'\breview\s+(this|the|my)\b', re.IGNORECASE),
        re.compile(r'\bcheck\s+(this|the|my)\s+(paragraph|section|text|passage)\b', re.IGNORECASE),
        re.compile(r'\bfeedback\s+on\b', re.IGNORECASE),
    ]
    _SYNTH_PATTERNS = [
        re.compile(r'\bsynthesize\s+(the\s+)?literature\b', re.IGNORECASE),
        re.compile(r'\bliterature\s+(review|synthesis)\s+(on|about|for)\b', re.IGNORECASE),
    ]

    def _infer_mode(message: str, requested_mode: str) -> str:
        """Auto-infer hidden modes when user sends from chat."""
        if requested_mode != 'chat':
            return requested_mode
        # Check patterns in priority order
        for p in _DRAFT_PATTERNS:
            if p.search(message):
                return 'draft'
        for p in _REVIEW_PATTERNS:
            if p.search(message):
                return 'review'
        for p in _SYNTH_PATTERNS:
            if p.search(message):
                return 'synthesize'
        for p in _CITE_PATTERNS:
            if p.search(message):
                return 'cite'
        for p in _STRUCTURE_PATTERNS:
            if p.search(message):
                return 'structure'
        return 'chat'

    # ── Helper: build draft context from student's sections ──

    def _build_draft_context(ps, section_id):
        """
        Build context from the student's own written sections (~3K tokens).
        Includes summaries of prior/following sections plus red thread.
        """
        if not ps.sections or not section_id:
            return ""

        sorted_sections = sorted(ps.sections, key=lambda s: s.order)
        current_idx = None
        for i, s in enumerate(sorted_sections):
            if s.id == section_id:
                current_idx = i
                break
        if current_idx is None:
            return ""

        parts = []
        current = sorted_sections[current_idx]

        # Prior sections: summaries of what came before
        prior = [s for s in sorted_sections[:current_idx] if s.summary]
        if prior:
            parts.append("[PRIOR SECTIONS — what the reader has already read]:")
            for s in prior:
                parts.append(f"  {s.id} {s.title}: {s.summary}")
            parts.append("")

        # Current section context
        parts.append(f"[CURRENT SECTION: {current.id} {current.title}]")
        if current.notes:
            parts.append(f"Notes: {current.notes}")
        if current.target_words:
            parts.append(f"Target: {current.target_words} words (currently {current.word_count})")
        parts.append("")

        # Following sections: summaries of what comes after
        following = [s for s in sorted_sections[current_idx + 1:] if s.summary]
        if following:
            parts.append("[FOLLOWING SECTIONS — where the paper goes next]:")
            for s in following:
                parts.append(f"  {s.id} {s.title}: {s.summary}")
            parts.append("")

        # Red thread + argument chain
        if ps.red_thread:
            parts.append(f"[RED THREAD]: {ps.red_thread}")
        if ps.argument_chain:
            parts.append("[ARGUMENT CHAIN]:")
            for i, claim in enumerate(ps.argument_chain, 1):
                parts.append(f"  {i}. {claim}")

        return "\n".join(parts)

    # ── Helper: build context for a mode ──

    def _build_context(message, mode, data):
        """Build all context layers for a request."""
        search_query = message[:500]
        rag_context = store.build_context(
            search_query,
            n_results=cfg.get("max_retrieval_chunks", 25),
        )

        source_files = list(set(re.findall(r'\[SOURCE: (.+?),', rag_context)))

        # Add structure context for certain modes
        extra = ""
        section_id = data.get("section_id", "")
        ps = load_structure()

        if section_id and ps.sections:
            extra = build_transition_context(ps, section_id)
            gloss = build_glossary_context(ps)
            if gloss:
                extra += "\n\n" + gloss
            # Add draft context (student's own sections) for draft/review
            if mode in ("draft", "review"):
                draft_ctx = _build_draft_context(ps, section_id)
                if draft_ctx:
                    extra += "\n\n" + draft_ctx
        elif mode in ("draft", "review") and ps.sections and ps.red_thread:
            # Auto-inject red thread + argument chain for draft/review even without section_id
            parts = [f"[RED THREAD — Core argument]: {ps.red_thread}"]
            if ps.argument_chain:
                parts.append("[ARGUMENT CHAIN]:\n" + "\n".join(
                    f"{i+1}. {c}" for i, c in enumerate(ps.argument_chain)
                ))
            extra = "\n\n".join(parts)

        # Add verified citation context for draft/synthesize
        if mode in ("draft", "synthesize", "cite"):
            cit_ctx = citations.build_context_for_agent()
            if cit_ctx:
                extra += "\n\n" + cit_ctx

        if extra:
            rag_context = f"{extra}\n\n---\n\n{rag_context}"

        return rag_context, source_files, ps, section_id

    # ── Streaming Chat API (SSE) ──

    @app.route("/api/chat/stream", methods=["POST"])
    def api_chat_stream():
        """
        SSE streaming endpoint. Returns server-sent events:
        event: model       data: {"model": "...", "thinking_enabled": true}
        event: thinking    data: {"text": "..."}
        event: text        data: {"text": "..."}
        event: done        data: {"session_id": "...", "sources": [...], "citations": [...]}
        event: error       data: {"error": "..."}
        """
        data = request.json or {}
        session_id = data.get("session_id")
        message = data.get("message", "").strip()
        mode = _infer_mode(message, data.get("mode", "chat"))

        model_override = data.get("model_override", "")
        thinking_override = data.get("thinking_override")  # None = use default, int = override
        if thinking_override is not None:
            thinking_override = int(thinking_override)

        if not message:
            return jsonify({"error": "Empty message"}), 400

        def generate():
            nonlocal session_id
            try:
                if not session_id:
                    title = message[:60] + ("..." if len(message) > 60 else "")
                    session = db.create_session(title, mode)
                    session_id = session["id"]

                db.add_message(session_id, "user", message)

                rag_context, source_files, ps, section_id = _build_context(message, mode, data)

                # Pick the right system prompt and build the user message
                system_prompt, user_msg, use_history = _route_mode(
                    mode, message, data, rag_context, ps, section_id, agent
                )

                # Stream with per-request overrides
                for chunk in agent._stream(
                    system=system_prompt,
                    user_message=user_msg,
                    context=rag_context if mode not in ("transition", "consistency", "structure") else "",
                    use_history=use_history,
                    mode=mode,
                    model_override=model_override,
                    thinking_override=thinking_override,
                ):
                    if chunk["type"] == "model":
                        yield f"event: model\ndata: {json.dumps(chunk)}\n\n"
                    elif chunk["type"] == "thinking_start":
                        yield f"event: thinking_start\ndata: {{}}\n\n"
                    elif chunk["type"] == "thinking":
                        yield f"event: thinking\ndata: {json.dumps({'text': chunk['text']})}\n\n"
                    elif chunk["type"] == "thinking_done":
                        yield f"event: thinking_done\ndata: {{}}\n\n"
                    elif chunk["type"] == "text":
                        yield f"event: text\ndata: {json.dumps({'text': chunk['text']})}\n\n"
                    elif chunk["type"] == "done":
                        full_text = chunk["full_text"]
                        found_cites = citations.extract_citations_from_text(full_text, section=mode)
                        db.add_message(session_id, "assistant", full_text, context_sources=source_files)

                        # Track cost
                        usage = chunk.get("usage", {})
                        cost_usd = 0
                        if usage and session_id:
                            cost_entry = db.add_cost(
                                session_id=session_id,
                                mode=mode,
                                model=chunk.get("model", ""),
                                input_tokens=usage.get("input_tokens", 0),
                                output_tokens=usage.get("output_tokens", 0),
                                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                                cache_create_tokens=usage.get("cache_creation_input_tokens", 0),
                            )
                            cost_usd = cost_entry.get("estimated_usd", 0)

                        yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'sources': source_files[:10], 'citations': found_cites[:20], 'cost_usd': cost_usd, 'usage': usage})}\n\n"

            except Exception as e:
                error_msg = _sanitize_error(e)
                if session_id:
                    db.add_message(session_id, "assistant", error_msg)
                yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Non-streaming chat (fallback) ──

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.json or {}
        session_id = data.get("session_id")
        message = data.get("message", "").strip()
        mode = data.get("mode", "chat")

        if not message:
            return jsonify({"error": "Empty message"}), 400

        model_override = data.get("model_override", "")
        thinking_override = data.get("thinking_override")
        if thinking_override is not None:
            thinking_override = int(thinking_override)

        if not session_id:
            title = message[:60] + ("..." if len(message) > 60 else "")
            session = db.create_session(title, mode)
            session_id = session["id"]

        db.add_message(session_id, "user", message)

        try:
            rag_context, source_files, ps, section_id = _build_context(message, mode, data)

            system_prompt, user_msg, use_history = _route_mode(
                mode, message, data, rag_context, ps, section_id, agent
            )

            result = agent._call(
                system=system_prompt,
                user_message=user_msg,
                context=rag_context if mode not in ("transition", "consistency", "structure") else "",
                use_history=use_history,
                mode=mode,
                model_override=model_override,
                thinking_override=thinking_override,
            )

            found_cites = citations.extract_citations_from_text(result, section=mode)
            db.add_message(session_id, "assistant", result, context_sources=source_files)

            result_html = md.markdown(result, extensions=["tables", "fenced_code", "nl2br"])

            return jsonify({
                "session_id": session_id,
                "response": result,
                "response_html": result_html,
                "sources_used": source_files[:10],
                "citations_found": found_cites[:20] if found_cites else [],
                "model": agent.get_model(mode),
            })

        except Exception as e:
            error_msg = _sanitize_error(e)
            db.add_message(session_id, "assistant", error_msg)
            return jsonify({
                "session_id": session_id,
                "response": error_msg,
                "response_html": f"<p class='error'>{error_msg}</p>",
                "error": True,
            }), 500

    def _route_mode(mode, message, data, rag_context, ps, section_id, agent):
        """Route to the correct system prompt and build the user message."""
        use_history = False

        if mode == "draft":
            return prompts.DRAFT_SYSTEM, message, False
        elif mode == "synthesize":
            return prompts.SYNTHESIZE_SYSTEM, f"Synthesize the literature on: {message}", False
        elif mode == "review":
            return prompts.REVIEW_SYSTEM, f"Review the following draft section:\n\n{message}", False
        elif mode == "cite":
            return prompts.CITE_SYSTEM, f"Find and format citations for: {message}", False
        elif mode == "transition":
            adj = get_adjacent_sections(ps, section_id)
            prev_end = adj["prev"].ends_with if adj.get("prev") else "(previous section not yet written)"
            next_start = adj["next"].starts_with if adj.get("next") else "(next section not yet written)"
            msg = (
                f"PREVIOUS SECTION ENDING:\n{prev_end}\n\n"
                f"NEXT SECTION BEGINNING:\n{next_start}\n\n"
                f"RED THREAD:\n{ps.red_thread}\n\n"
                f"ARGUMENT CHAIN:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(ps.argument_chain))
                + f"\n\nUSER REQUEST:\n{message}"
            )
            return prompts.TRANSITION_SYSTEM, msg, False
        elif mode == "consistency":
            gloss_ctx = build_glossary_context(ps)
            msg = f"CHECK THIS SECTION FOR CONSISTENCY:\n\n{message}\n\n{gloss_ctx}"
            return prompts.CONSISTENCY_SYSTEM, msg, False
        elif mode == "structure":
            struct_ctx = yaml.dump(
                {"sections": [{"id": s.id, "title": s.title, "status": s.status,
                               "word_count": s.word_count, "target_words": s.target_words,
                               "summary": s.summary} for s in ps.sections],
                 "red_thread": ps.red_thread,
                 "argument_chain": ps.argument_chain},
                default_flow_style=False,
            )
            return prompts.OUTLINE_SYSTEM, f"{struct_ctx}\n\nUSER REQUEST:\n{message}", False
        elif mode == "edit_docx":
            doc_path = data.get("doc_path", "")
            doc_content = ""
            if doc_path and os.path.exists(doc_path):
                doc_data = read_docx(doc_path)
                doc_content = doc_data["full_text"]
            msg = f"INSTRUCTION: {message}\n\nCURRENT DOCUMENT CONTENT:\n\n{doc_content}"
            return prompts.EDIT_DOCX_SYSTEM, msg, False
        else:
            # Chat with history — rebuild from DB for session continuity across restarts
            sid = data.get("session_id", "")
            if sid:
                history = db.get_messages(sid)
                agent.conversation_history = []
                for msg in history[:-1]:  # Exclude the current message (already added)
                    if msg["role"] in ("user", "assistant"):
                        agent.conversation_history.append({
                            "role": msg["role"],
                            "content": msg["content"][:8000],  # Trim old messages for context budget
                        })
                # Keep last 20 turns (40 messages) to stay within context
                if len(agent.conversation_history) > 40:
                    agent.conversation_history = agent.conversation_history[-40:]
            else:
                agent.conversation_history = []
            return prompts.CHAT_SYSTEM, message, True

    # ── Document API ──

    @app.route("/api/documents", methods=["GET"])
    def api_list_documents():
        draft_dir = cfg.get("current_draft_dir", "./current_draft")
        return jsonify(list_docx_files(draft_dir))

    @app.route("/api/documents/read", methods=["POST"])
    def api_read_document():
        data = request.json or {}
        filepath = data.get("path", "")
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        if not _validate_doc_path(filepath):
            return jsonify({"error": "Access denied"}), 403
        try:
            return jsonify(read_docx(filepath))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/documents/apply-changes", methods=["POST"])
    def api_apply_changes():
        data = request.json or {}
        source_path = data.get("source_path", "")
        changes = data.get("changes", [])
        session_id = data.get("session_id")

        if not source_path or not os.path.exists(source_path):
            return jsonify({"error": "Source file not found"}), 404
        if not _validate_doc_path(source_path):
            return jsonify({"error": "Access denied"}), 403
        if not changes:
            return jsonify({"error": "No changes provided"}), 400

        try:
            original_data = read_docx(source_path)
            db.save_doc_version(
                filename=os.path.basename(source_path),
                content=original_data["full_text"],
                change_summary="Before changes",
                session_id=session_id,
            )

            stem = Path(source_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = cfg.get("output_dir", "./output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{stem}_{timestamp}.docx")

            change_log = apply_changes_to_docx(source_path, changes, output_path)

            new_data = read_docx(output_path)
            db.save_doc_version(
                filename=os.path.basename(output_path),
                content=new_data["full_text"],
                change_summary=change_log,
                session_id=session_id,
            )

            return jsonify({
                "output_path": output_path,
                "change_log": change_log,
                "word_count": new_data["metadata"]["word_count"],
            })
        except Exception as e:
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    @app.route("/api/documents/write", methods=["POST"])
    def api_write_document():
        data = request.json or {}
        content = data.get("content", "")
        filename = data.get("filename", "output.docx")
        title = data.get("title", "")

        if not content:
            return jsonify({"error": "No content"}), 400

        output_dir = cfg.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        try:
            write_docx_from_markdown(content, output_path, title=title)
            return jsonify({"path": output_path, "filename": filename})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/documents/download/<path:filename>")
    def api_download_document(filename):
        output_dir = os.path.abspath(cfg.get("output_dir", "./output"))
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        return send_file(filepath, as_attachment=True)

    @app.route("/api/documents/versions/<filename>")
    def api_doc_versions(filename):
        return jsonify(db.get_doc_versions(filename))

    @app.route("/api/documents/diff", methods=["POST"])
    def api_doc_diff():
        """Compute unified diff between two document versions."""
        data = request.json or {}
        version_a = data.get("version_a", "")
        version_b = data.get("version_b", "")

        if not version_a or not version_b:
            return jsonify({"error": "Both version_a and version_b required"}), 400

        va = db.get_doc_version(version_a)
        vb = db.get_doc_version(version_b)

        if not va:
            return jsonify({"error": f"Version {version_a} not found"}), 404
        if not vb:
            return jsonify({"error": f"Version {version_b} not found"}), 404

        diff_lines = list(difflib.unified_diff(
            va["content"].splitlines(keepends=True),
            vb["content"].splitlines(keepends=True),
            fromfile=f"{va['filename']} ({va['created_at'][:16]})",
            tofile=f"{vb['filename']} ({vb['created_at'][:16]})",
            lineterm="",
        ))

        # Count additions and deletions
        additions = sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))
        deletions = sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))

        return jsonify({
            "diff": "\n".join(diff_lines),
            "additions": additions,
            "deletions": deletions,
            "version_a": {"id": va["id"], "filename": va["filename"], "created_at": va["created_at"]},
            "version_b": {"id": vb["id"], "filename": vb["filename"], "created_at": vb["created_at"]},
        })

    # ── Citations / BibTeX API ──

    @app.route("/api/citations", methods=["GET"])
    def api_citations():
        return jsonify({
            "citations": {k: {
                "key": v.key, "full_reference": v.full_reference,
                "from_bibtex": v.from_bibtex, "used_in": v.used_in,
                "source_file": v.source_file,
            } for k, v in citations.citations.items()},
            "stats": citations.stats,
        })

    @app.route("/api/citations/import-bibtex", methods=["POST"])
    def api_import_bibtex():
        """Import references from a .bib file."""
        data = request.json or {}
        bib_path = data.get("path", "")

        # Check common locations
        if not bib_path:
            for candidate in ["./references.bib", "./bibliography.bib", "./papers/references.bib"]:
                if os.path.exists(candidate):
                    bib_path = candidate
                    break

        if not bib_path or not os.path.exists(bib_path):
            return jsonify({"error": "BibTeX file not found. Place references.bib in project root or papers/"}), 404

        try:
            result = citations.import_bibtex(bib_path)
            # Try to match PDFs
            sources = store.list_sources()
            matches = citations.match_pdfs_to_citations(sources)
            result["pdf_matches"] = len(matches)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/citations/import-zotero", methods=["POST"])
    def api_import_zotero():
        """Import from Zotero JSON export."""
        data = request.json or {}
        json_path = data.get("path", "")

        if not json_path:
            for candidate in ["./references.json", "./zotero.json", "./papers/references.json"]:
                if os.path.exists(candidate):
                    json_path = candidate
                    break

        if not json_path or not os.path.exists(json_path):
            return jsonify({"error": "Zotero JSON not found"}), 404

        try:
            result = citations.import_zotero_json(json_path)
            sources = store.list_sources()
            matches = citations.match_pdfs_to_citations(sources)
            result["pdf_matches"] = len(matches)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/citations/reference-list", methods=["GET"])
    def api_reference_list():
        return jsonify({"text": citations.generate_reference_list()})

    # ── Store / Status API ──

    @app.route("/api/status", methods=["GET"])
    def api_status():
        sources = store.list_sources()
        cit_stats = citations.stats
        return jsonify({
            "model": cfg.get("model", "claude-opus-4-6"),
            "model_routing": agent.model_map,
            "thinking_budget": agent.thinking_map,
            "chunk_count": store.count,
            "source_count": len(sources),
            "sources": sources,
            "citation_count": cit_stats["total"],
            "citation_stats": cit_stats,
            "paper_topic": cfg.get("paper_topic", ""),
        })

    @app.route("/api/sources", methods=["GET"])
    def api_sources():
        return jsonify(store.list_sources())

    # ── Paper Structure API ──

    @app.route("/api/structure", methods=["GET"])
    def api_get_structure():
        ps = load_structure()
        progress = compute_progress(ps)
        return jsonify({
            "title": ps.title,
            "research_question": ps.research_question,
            "red_thread": ps.red_thread,
            "methodology": ps.methodology,
            "argument_chain": ps.argument_chain,
            "sections": [
                {"id": s.id, "title": s.title, "parent_id": s.parent_id,
                 "status": s.status, "word_count": s.word_count,
                 "target_words": s.target_words, "summary": s.summary,
                 "docx_file": s.docx_file, "order": s.order,
                 "key_sources": s.key_sources, "notes": s.notes}
                for s in sorted(ps.sections, key=lambda x: x.order)
            ],
            "glossary": [
                {"preferred": t.preferred, "alternatives": t.alternatives,
                 "definition": t.definition, "source": t.source}
                for t in ps.glossary
            ],
            "progress": progress,
        })

    @app.route("/api/structure/section/<section_id>", methods=["PATCH"])
    def api_update_section(section_id):
        data = request.json or {}
        ps = load_structure()
        sec = get_section(ps, section_id)
        if not sec:
            return jsonify({"error": "Section not found"}), 404

        for fld in ["status", "word_count", "summary", "ends_with", "starts_with",
                     "docx_file", "notes", "target_words"]:
            if fld in data:
                setattr(sec, fld, data[fld])
        if "key_sources" in data:
            sec.key_sources = data["key_sources"]

        save_structure(ps)
        return jsonify({"ok": True})

    @app.route("/api/structure/scaffold", methods=["POST"])
    def api_generate_scaffold():
        data = request.json or {}
        content = generate_scaffold_yaml(
            title=data.get("title", ""),
            rq=data.get("research_question", ""),
            methodology=data.get("methodology", "Design Science Research"),
            language=data.get("language", "en"),
        )
        path = Path("paper_structure.yaml")
        overwrite = data.get("overwrite", False)
        if path.exists() and not overwrite:
            return jsonify({"error": "paper_structure.yaml already exists", "exists": True}), 409
        path.write_text(content)
        return jsonify({"ok": True, "path": str(path)})

    @app.route("/api/structure/terminology-check", methods=["POST"])
    def api_terminology_check():
        data = request.json or {}
        text = data.get("text", "")
        ps = load_structure()
        issues = find_terminology_issues(ps, text)
        return jsonify({"issues": issues, "count": len(issues)})

    @app.route("/api/structure/transition/<section_id>", methods=["GET"])
    def api_transition_context(section_id):
        ps = load_structure()
        adj = get_adjacent_sections(ps, section_id)
        return jsonify({
            "prev": {"id": adj["prev"].id, "title": adj["prev"].title,
                      "ends_with": adj["prev"].ends_with, "summary": adj["prev"].summary}
            if adj["prev"] else None,
            "current": {"id": adj["current"].id, "title": adj["current"].title,
                         "summary": adj["current"].summary}
            if adj["current"] else None,
            "next": {"id": adj["next"].id, "title": adj["next"].title,
                      "starts_with": adj["next"].starts_with, "summary": adj["next"].summary}
            if adj["next"] else None,
        })

    @app.route("/api/structure/sync-docx", methods=["POST"])
    def api_sync_docx():
        data = request.json or {}
        section_id = data.get("section_id", "")
        docx_path = data.get("docx_path", "")

        if not docx_path or not os.path.exists(docx_path):
            return jsonify({"error": "File not found"}), 404

        ps = load_structure()
        sec = get_section(ps, section_id)
        if not sec:
            return jsonify({"error": "Section not found"}), 404

        try:
            doc_data = read_docx(docx_path)
            full_text = doc_data["full_text"]
            words = full_text.split()

            sec.word_count = len(words)
            sec.docx_file = docx_path
            sec.starts_with = " ".join(words[:200]) if len(words) > 200 else full_text
            sec.ends_with = " ".join(words[-200:]) if len(words) > 200 else full_text

            # Generate summary if section has content but no summary
            summary_generated = False
            if not sec.summary and sec.word_count > 100:
                text_sample = sec.starts_with[:1500]
                try:
                    summary = agent._call(
                        system="You are a concise academic summarizer. Write a 2-sentence summary of what this section argues or covers. Be specific and academic.",
                        user_message=f"Section: {sec.title}\n\nExcerpt:\n{text_sample}",
                        mode="chat",
                    )
                    sec.summary = summary.strip()
                    summary_generated = True
                except Exception:
                    pass

            save_structure(ps)
            return jsonify({
                "ok": True,
                "word_count": sec.word_count,
                "starts_with_preview": sec.starts_with[:100] + "...",
                "ends_with_preview": "..." + sec.ends_with[-100:],
                "summary_generated": summary_generated,
                "summary": sec.summary,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Full paper export ──

    @app.route("/api/documents/export-full-paper", methods=["POST"])
    def api_export_full_paper():
        """Compile all sections into a single DOCX with APA formatting."""
        data = request.json or {}
        ps = load_structure()
        if not ps.sections:
            return jsonify({"error": "No paper structure loaded"}), 400

        # Build sections with content from DOCX files
        section_data = []
        for sec in sorted(ps.sections, key=lambda s: s.order):
            content = ""
            if sec.docx_file and os.path.exists(sec.docx_file):
                try:
                    doc_data = read_docx(sec.docx_file)
                    # Try to find the matching section content
                    for doc_sec in doc_data["sections"]:
                        matched = _match_heading_to_section(doc_sec["heading"], ps)
                        if matched and matched.id == sec.id:
                            content = "\n\n".join(doc_sec["paragraphs"])
                            break
                    if not content:
                        content = doc_data["full_text"]
                except Exception:
                    pass

            section_data.append({
                "id": sec.id,
                "title": sec.title,
                "parent_id": sec.parent_id,
                "content": content,
                "order": sec.order,
            })

        output_dir = cfg.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"full_paper_{timestamp}.docx")

        try:
            export_full_paper(
                sections=section_data,
                output_path=output_path,
                title=ps.title or data.get("title", ""),
                author=data.get("author", ""),
            )
            return jsonify({
                "ok": True,
                "path": output_path,
                "filename": os.path.basename(output_path),
                "download_url": f"/api/documents/download/{os.path.basename(output_path)}",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Auto-sync: extract structure from DOCX ──

    @app.route("/api/structure/auto-sync", methods=["POST"])
    def api_auto_sync():
        """
        Auto-extract structure from a DOCX file.
        Matches DOCX headings to paper_structure sections, updates
        word counts, starts_with/ends_with, and optionally generates
        Sonnet summaries for sections that lack them.
        """
        data = request.json or {}
        docx_path = data.get("docx_path", "")
        generate_summaries = data.get("generate_summaries", True)

        if not docx_path or not os.path.exists(docx_path):
            return jsonify({"error": "File not found"}), 404

        ps = load_structure()
        if not ps.sections:
            return jsonify({"error": "No paper structure loaded. Generate a scaffold first."}), 400

        try:
            doc_data = read_docx(docx_path)
            updated = []
            unmatched = []

            for doc_section in doc_data["sections"]:
                heading = doc_section["heading"]
                if heading == "Preamble":
                    continue

                sec = _match_heading_to_section(heading, ps)
                if not sec:
                    unmatched.append(heading)
                    continue

                # Update section from DOCX content
                text = "\n\n".join(doc_section["paragraphs"])
                words = text.split()
                sec.word_count = len(words)
                sec.docx_file = docx_path
                sec.starts_with = " ".join(words[:200]) if len(words) > 200 else text
                sec.ends_with = " ".join(words[-200:]) if len(words) > 200 else text

                # Update status based on word count vs target
                if sec.word_count == 0:
                    sec.status = "not_started"
                elif sec.target_words > 0 and sec.word_count >= sec.target_words * 0.8:
                    if sec.status in ("not_started", "outline"):
                        sec.status = "drafting"
                elif sec.word_count > 50:
                    if sec.status == "not_started":
                        sec.status = "outline"

                updated.append({
                    "id": sec.id,
                    "title": sec.title,
                    "matched_heading": heading,
                    "word_count": sec.word_count,
                    "status": sec.status,
                })

            # Generate Sonnet summaries for sections that need them
            summaries_generated = 0
            if generate_summaries:
                for item in updated:
                    sec = get_section(ps, item["id"])
                    if sec and not sec.summary and sec.word_count > 100:
                        # Use first 1500 chars for summary generation
                        text_sample = sec.starts_with[:1500] if sec.starts_with else ""
                        if text_sample:
                            try:
                                summary = agent._call(
                                    system="You are a concise academic summarizer. Given a section excerpt, write a 2-sentence summary of what this section argues or covers. Be specific and academic.",
                                    user_message=f"Section: {sec.title}\n\nExcerpt:\n{text_sample}",
                                    mode="chat",  # Uses Sonnet
                                )
                                sec.summary = summary.strip()
                                summaries_generated += 1
                            except Exception:
                                pass  # Skip summary on error

            save_structure(ps)

            return jsonify({
                "ok": True,
                "sections_updated": len(updated),
                "sections_unmatched": len(unmatched),
                "summaries_generated": summaries_generated,
                "updated": updated,
                "unmatched": unmatched,
            })

        except Exception as e:
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    # ── Red Thread Audit ──

    @app.route("/api/structure/red-thread-audit", methods=["GET"])
    def api_red_thread_audit():
        """
        Evaluate the red thread (core argument) flow across all section pairs.
        Uses Sonnet to rate each transition on coherence (1-5) and
        red thread visibility (visible/partial/lost).
        """
        ps = load_structure()
        if not ps.sections:
            return jsonify({"error": "No paper structure loaded"}), 400
        if not ps.red_thread:
            return jsonify({"error": "No red thread defined in paper structure"}), 400

        pairs = build_all_adjacent_pairs(ps)
        if not pairs:
            return jsonify({"error": "No section pairs with content found. Run auto-sync first."}), 400

        results = []
        for pair in pairs:
            prev_text = pair["prev_ends_with"] or pair["prev_summary"] or "(no content)"
            next_text = pair["next_starts_with"] or pair["next_summary"] or "(no content)"

            prompt = (
                f"RED THREAD: {ps.red_thread}\n\n"
                f"PREVIOUS SECTION ({pair['prev_id']} {pair['prev_title']}) ENDS WITH:\n"
                f"{prev_text[:800]}\n\n"
                f"NEXT SECTION ({pair['next_id']} {pair['next_title']}) STARTS WITH:\n"
                f"{next_text[:800]}\n\n"
                "Rate this transition:\n"
                "1. COHERENCE (1-5): How smoothly does one section flow into the next?\n"
                "2. RED_THREAD (visible/partial/lost): Is the core argument visible in this transition?\n"
                "3. SUGGESTION: One sentence on how to improve if score < 4.\n\n"
                "Reply in exactly this format:\n"
                "COHERENCE: [1-5]\n"
                "RED_THREAD: [visible|partial|lost]\n"
                "SUGGESTION: [text]"
            )

            try:
                result = agent._call(
                    system="You are an academic writing consultant evaluating argument flow between paper sections. Be specific and constructive.",
                    user_message=prompt,
                    mode="chat",  # Sonnet
                )

                # Parse response
                coherence = 3
                visibility = "partial"
                suggestion = ""

                for line in result.strip().split("\n"):
                    line = line.strip()
                    if line.upper().startswith("COHERENCE:"):
                        try:
                            coherence = int(line.split(":")[1].strip()[0])
                            coherence = max(1, min(5, coherence))
                        except (ValueError, IndexError):
                            pass
                    elif line.upper().startswith("RED_THREAD:"):
                        val = line.split(":")[1].strip().lower()
                        if val in ("visible", "partial", "lost"):
                            visibility = val
                    elif line.upper().startswith("SUGGESTION:"):
                        suggestion = line.split(":", 1)[1].strip()

                results.append({
                    "from_section": f"{pair['prev_id']} {pair['prev_title']}",
                    "to_section": f"{pair['next_id']} {pair['next_title']}",
                    "coherence": coherence,
                    "red_thread": visibility,
                    "suggestion": suggestion,
                })

            except Exception as e:
                results.append({
                    "from_section": f"{pair['prev_id']} {pair['prev_title']}",
                    "to_section": f"{pair['next_id']} {pair['next_title']}",
                    "coherence": 0,
                    "red_thread": "error",
                    "suggestion": str(e),
                })

        # Compute overall scores
        scored = [r for r in results if r["coherence"] > 0]
        avg_coherence = round(sum(r["coherence"] for r in scored) / len(scored), 1) if scored else 0
        weak_transitions = [r for r in results if r["coherence"] < 3 or r["red_thread"] == "lost"]

        return jsonify({
            "red_thread": ps.red_thread,
            "total_pairs": len(results),
            "average_coherence": avg_coherence,
            "weak_transitions": len(weak_transitions),
            "results": results,
        })

    # ── Config API ──

    @app.route("/api/config/models", methods=["GET"])
    def api_get_model_config():
        return jsonify({
            "model_routing": agent.model_map,
            "thinking_budget": agent.thinking_map,
        })

    @app.route("/api/config/models", methods=["PATCH"])
    def api_update_model_config():
        """Update model routing and thinking budgets at runtime."""
        data = request.json or {}
        if "model_routing" in data:
            agent.model_map.update(data["model_routing"])
        if "thinking_budget" in data:
            for k, v in data["thinking_budget"].items():
                agent.thinking_map[k] = int(v)
        return jsonify({"ok": True, "model_routing": agent.model_map, "thinking_budget": agent.thinking_map})

    return app
