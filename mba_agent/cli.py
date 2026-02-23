"""
CLI: Main entry point for the MBA paper agent.
Usage: python -m mba_agent <command> [options]
"""

import os
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        console.print("[red]Error: config.yaml not found.[/red]")
        console.print("Copy config.example.yaml to config.yaml and add your API key.")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_store(cfg: dict):
    """Initialize the vector store."""
    from .store import PaperStore
    return PaperStore(
        persist_dir=cfg.get("chroma_persist_dir", "./.chroma_db"),
        embedding_model=cfg.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )


def get_agent(cfg: dict):
    """Initialize the agent with available backends."""
    from .agent import MBAAgent
    from .claude_backend import ClaudeBackend

    backends = {}

    # Claude backends
    api_key = cfg.get("anthropic_api_key", "")
    if api_key and not api_key.startswith("sk-ant-your"):
        backends["claude-opus-4-6"] = ClaudeBackend(api_key, "claude-opus-4-6")
        backends["claude-sonnet-4-5-20250929"] = ClaudeBackend(api_key, "claude-sonnet-4-5-20250929")

    # Gemini backend (optional)
    gemini_key = cfg.get("gemini_api_key", "")
    if gemini_key:
        try:
            from .gemini_backend import GeminiBackend, HAS_GEMINI
            if HAS_GEMINI:
                gemini_model = cfg.get("gemini_model", "gemini-3.1-pro-preview")
                gemini_search = cfg.get("gemini_search", True)
                backends[gemini_model] = GeminiBackend(gemini_key, gemini_model, search=gemini_search)
        except ImportError:
            pass

    if not backends:
        console.print("[red]Error: No LLM backends configured. Set anthropic_api_key or gemini_api_key in config.yaml[/red]")
        sys.exit(1)

    return MBAAgent(
        backends=backends,
        default_model=cfg.get("model", "claude-opus-4-6"),
        model_map=cfg.get("model_routing"),
        thinking_map=cfg.get("thinking_budget"),
        max_output_tokens=cfg.get("max_output_tokens", 16000),
    )


def get_citations(cfg: dict):
    """Initialize citation manager."""
    from .citations import CitationManager
    output_dir = cfg.get("output_dir", "./output")
    return CitationManager(storage_path=f"{output_dir}/citations.json")


def save_output(text: str, filename: str, cfg: dict) -> Path:
    """Save agent output to file."""
    output_dir = Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


@click.group()
def cli():
    """MBA Paper Agent — Your academic writing partner."""
    pass


@cli.command()
@click.option("--clear", is_flag=True, help="Clear existing store before ingesting")
def ingest(clear):
    """Ingest all PDFs from the papers/ directory into the vector store."""
    cfg = load_config()
    store = get_store(cfg)

    if clear:
        console.print("[yellow]Clearing existing vector store...[/yellow]")
        store.clear()

    papers_dir = cfg.get("papers_dir", "./papers")
    console.print(f"\n[bold]Ingesting PDFs from {papers_dir}/[/bold]\n")

    from .ingest import ingest_directory
    chunks, metadata = ingest_directory(
        papers_dir,
        chunk_size=cfg.get("chunk_size", 1500),
        chunk_overlap=cfg.get("chunk_overlap", 200),
    )

    if not chunks:
        console.print("[red]No chunks extracted. Check your PDFs.[/red]")
        return

    console.print(f"\n[bold]Embedding and storing {len(chunks)} chunks...[/bold]\n")
    store.add_chunks(chunks)

    console.print(f"\n[green]Done. {store.count} total chunks in store.[/green]")
    console.print(f"Sources: {len(metadata)} PDFs processed.")


@cli.command()
def extract():
    """Extract and display cleaned text from all source files (Gemini full-context verification).

    Shows total character count and estimated token count for Gemini context window planning.
    """
    cfg = load_config()
    papers_dir = cfg.get("papers_dir", "./papers")
    papers_path = Path(papers_dir)

    console.print(f"\n[bold]Extracting text from {papers_dir}/[/bold]\n")

    total_chars = 0
    file_count = 0

    for pdf in sorted(papers_path.glob("**/*.pdf")):
        try:
            from .ingest import extract_pdf_text
            pages = extract_pdf_text(str(pdf), strip_references=True)
            chars = sum(len(t) for _, t in pages)
            total_chars += chars
            file_count += 1
            console.print(f"  [green]✓[/green] {pdf.name}: {len(pages)} pages, {chars:,} chars")
        except Exception as e:
            console.print(f"  [red]✗[/red] {pdf.name}: {e}")

    for docx in sorted(papers_path.glob("**/*.docx")):
        try:
            from docx import Document
            doc = Document(str(docx))
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            total_chars += len(text)
            file_count += 1
            console.print(f"  [green]✓[/green] {docx.name}: {len(text):,} chars")
        except Exception as e:
            console.print(f"  [red]✗[/red] {docx.name}: {e}")

    for ext in ("*.txt", "*.md"):
        for txt in sorted(papers_path.glob(f"**/{ext}")):
            try:
                content = txt.read_text(encoding="utf-8", errors="replace")
                total_chars += len(content)
                file_count += 1
                console.print(f"  [green]✓[/green] {txt.name}: {len(content):,} chars")
            except Exception as e:
                console.print(f"  [red]✗[/red] {txt.name}: {e}")

    est_tokens = total_chars // 4
    console.print(f"\n[bold]Total: {file_count} files, {total_chars:,} chars (~{est_tokens:,} tokens)[/bold]")

    if est_tokens > 900_000:
        console.print("[yellow]Warning: Exceeds Gemini's ~1M token context window. Some files may be truncated.[/yellow]")
    elif est_tokens > 500_000:
        console.print("[dim]Fits in Gemini 1M context with room for system prompt + output.[/dim]")
    else:
        console.print("[green]Fits comfortably in Gemini 1M context window.[/green]")


@cli.command()
@click.argument("instruction")
@click.option("--from-file", type=click.Path(exists=True), help="Existing draft file to revise")
@click.option("--save-as", default="", help="Output filename (default: auto)")
def draft(instruction, from_file, save_as):
    """Draft a paper section. Retrieves relevant context automatically.

    Example: python -m mba_agent draft "Write the theoretical framework section on psychological ownership"
    """
    cfg = load_config()
    store = get_store(cfg)
    agent = get_agent(cfg)
    cit = get_citations(cfg)

    console.print(f"\n[bold]Drafting:[/bold] {instruction}\n")
    console.print("Retrieving relevant sources...")

    context = store.build_context(
        instruction,
        n_results=cfg.get("max_retrieval_chunks", 25),
    )

    if from_file:
        existing = Path(from_file).read_text(encoding="utf-8")
        console.print(f"Using existing draft: {from_file}")
        result = agent.draft_with_existing(instruction, existing, context)
    else:
        result = agent.draft(instruction, context)

    # Track citations
    found = cit.extract_citations_from_text(result, section=instruction[:50])
    console.print(f"[dim]Tracked {len(found)} citations[/dim]")

    # Display
    console.print(Panel(Markdown(result), title="Draft Output", border_style="green"))

    # Save
    if not save_as:
        slug = instruction[:40].replace(" ", "_").lower()
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        save_as = f"draft_{slug}.md"
    path = save_output(result, save_as, cfg)
    console.print(f"\n[green]Saved to {path}[/green]")


@cli.command()
@click.argument("topic")
@click.option("--save-as", default="", help="Output filename")
def synthesize(topic, save_as):
    """Synthesize literature across your source library on a topic.

    Example: python -m mba_agent synthesize "co-creation in service-dominant logic"
    """
    cfg = load_config()
    store = get_store(cfg)
    agent = get_agent(cfg)

    console.print(f"\n[bold]Synthesizing:[/bold] {topic}\n")
    context = store.build_context(topic, n_results=30)
    result = agent.synthesize(topic, context)

    console.print(Panel(Markdown(result), title="Synthesis", border_style="blue"))

    if not save_as:
        slug = topic[:40].replace(" ", "_").lower()
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        save_as = f"synthesis_{slug}.md"
    path = save_output(result, save_as, cfg)
    console.print(f"\n[green]Saved to {path}[/green]")


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--focus", default="", help="Specific aspect to focus review on")
@click.option("--save-as", default="", help="Output filename")
def review(file_path, focus, save_as):
    """Review and critique a draft section.

    Example: python -m mba_agent review current_draft/chapter3.md --focus "argument coherence"
    """
    cfg = load_config()
    store = get_store(cfg)
    agent = get_agent(cfg)

    draft_text = Path(file_path).read_text(encoding="utf-8")
    console.print(f"\n[bold]Reviewing:[/bold] {file_path}\n")

    # Use both the draft content and topic for context retrieval
    search_query = draft_text[:2000]  # Use beginning of draft as search query
    context = store.build_context(search_query, n_results=20)

    result = agent.review(draft_text, context, focus=focus)

    console.print(Panel(Markdown(result), title="Review", border_style="yellow"))

    if not save_as:
        save_as = f"review_{Path(file_path).stem}.md"
    path = save_output(result, save_as, cfg)
    console.print(f"\n[green]Saved to {path}[/green]")


@cli.command()
@click.argument("query")
def cite(query):
    """Find and format citations from your library.

    Example: python -m mba_agent cite "Pine and Gilmore experience economy"
    """
    cfg = load_config()
    store = get_store(cfg)
    agent = get_agent(cfg)

    console.print(f"\n[bold]Finding citations for:[/bold] {query}\n")
    context = store.build_context(query, n_results=15)
    result = agent.cite(query, context)

    console.print(Panel(Markdown(result), title="Citations", border_style="magenta"))


@cli.command()
def chat():
    """Interactive chat session with your paper context.

    Start a freeform discussion about your paper, theory, or methodology.
    Type 'quit' or 'exit' to end. Type 'clear' to reset history.
    Type 'save' to save the conversation.
    """
    cfg = load_config()
    store = get_store(cfg)
    agent = get_agent(cfg)

    console.print(Panel(
        "[bold]MBA Paper Chat[/bold]\n"
        f"Model: {cfg.get('model', 'claude-opus-4-6')} | "
        f"Store: {store.count} chunks\n"
        "Commands: quit, clear, save",
        border_style="cyan",
    ))

    conversation_log = []

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            agent.clear_history()
            conversation_log = []
            console.print("[dim]History cleared.[/dim]")
            continue
        if user_input.lower() == "save":
            path = save_output(
                "\n\n---\n\n".join(conversation_log),
                "chat_session.md", cfg
            )
            console.print(f"[green]Saved to {path}[/green]")
            continue

        # Retrieve context based on user's message
        context = store.build_context(user_input, n_results=15)
        result = agent.chat(user_input, context)

        console.print(f"\n[bold green]Agent:[/bold green]")
        console.print(Markdown(result))

        conversation_log.append(f"**You:** {user_input}")
        conversation_log.append(f"**Agent:** {result}")


@cli.command()
def sources():
    """List all ingested sources in the vector store."""
    cfg = load_config()
    store = get_store(cfg)

    source_list = store.list_sources()
    console.print(f"\n[bold]Ingested Sources ({len(source_list)} files):[/bold]\n")
    for s in source_list:
        console.print(f"  • {s}")
    console.print(f"\n[dim]Total chunks in store: {store.count}[/dim]")


@cli.command()
def references():
    """Show all tracked citations and reference list."""
    cfg = load_config()
    cit = get_citations(cfg)

    ref_list = cit.generate_reference_list()
    console.print(Markdown(ref_list))

    missing = cit.find_missing_references()
    if missing:
        console.print(f"\n[yellow]Missing full references ({len(missing)}):[/yellow]")
        for m in missing:
            console.print(f"  ⚠ {m}")


@cli.command()
def status():
    """Show agent status: store size, sources, config."""
    cfg = load_config()
    store = get_store(cfg)
    cit = get_citations(cfg)

    source_list = store.list_sources()
    missing_refs = cit.find_missing_references()

    console.print(Panel(
        f"[bold]MBA Paper Agent Status[/bold]\n\n"
        f"Model: {cfg.get('model', 'claude-opus-4-6')}\n"
        f"Vector store: {store.count} chunks from {len(source_list)} sources\n"
        f"Citations tracked: {len(cit.citations)}\n"
        f"Missing references: {len(missing_refs)}\n"
        f"Paper topic: {cfg.get('paper_topic', 'Not set')}\n"
        f"Methodology: {cfg.get('methodology', 'Not set')}\n"
        f"Citation style: {cfg.get('citation_style', 'APA 7th')}",
        border_style="cyan",
    ))


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=5000, help="Port to bind to")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def serve(host, port, debug):
    """Start the web UI.

    Example: python -m mba_agent serve --port 5000
    """
    from .web.app import create_app
    app = create_app()
    console.print(Panel(
        f"[bold]MBA Paper Agent — Web UI[/bold]\n\n"
        f"Open: [cyan]http://{host}:{port}[/cyan]\n"
        f"Press Ctrl+C to stop.",
        border_style="green",
    ))
    app.run(host=host, port=port, debug=debug)


@cli.command()
@click.option("--title", default="", help="Paper title")
@click.option("--rq", default="", help="Research question")
@click.option("--methodology", default="Design Science Research", help="Methodology")
def scaffold(title, rq, methodology):
    """Generate a paper_structure.yaml template.

    Example: python -m mba_agent scaffold --title "My MBA Paper" --rq "How does X affect Y?"
    """
    from .paper_structure import generate_scaffold_yaml
    path = Path("paper_structure.yaml")
    if path.exists():
        console.print("[yellow]paper_structure.yaml already exists. Delete it first to regenerate.[/yellow]")
        return
    content = generate_scaffold_yaml(title=title, rq=rq, methodology=methodology)
    path.write_text(content)
    console.print(f"[green]Generated paper_structure.yaml[/green]")
    console.print("Edit this file to define your sections, red thread, and glossary.")


@cli.command(name="import-bib")
@click.argument("path", required=False, default="")
def import_bib(path):
    """Import references from a BibTeX (.bib) or Zotero JSON (.json) file.

    If no path given, looks for references.bib or references.json in project root.
    """
    cfg = load_config()
    cit = get_citations(cfg)

    # Auto-detect
    if not path:
        for candidate in ["references.bib", "bibliography.bib", "papers/references.bib",
                          "references.json", "zotero.json", "papers/references.json"]:
            if os.path.exists(candidate):
                path = candidate
                break

    if not path or not os.path.exists(path):
        console.print("[red]No .bib or .json file found.[/red]")
        console.print("Place references.bib or references.json in the project root, or specify a path.")
        return

    console.print(f"[bold]Importing from {path}[/bold]")

    if path.endswith(".bib"):
        result = cit.import_bibtex(path)
    elif path.endswith(".json"):
        result = cit.import_zotero_json(path)
    else:
        console.print("[red]Unsupported format. Use .bib or .json[/red]")
        return

    console.print(f"[green]Imported {result['imported']} references[/green]")
    if result['skipped']:
        console.print(f"[yellow]Skipped {result['skipped']} (missing author/year)[/yellow]")

    # Try to match PDFs
    store = get_store(cfg)
    sources = store.list_sources()
    if sources:
        matches = cit.match_pdfs_to_citations(sources)
        console.print(f"[dim]Matched {len(matches)} citations to PDF files[/dim]")

    stats = cit.stats
    console.print(f"\nTotal: {stats['total']} | Verified: {stats['verified_from_bibtex']} | Missing ref: {stats['missing_reference']}")


def main():
    cli()


if __name__ == "__main__":
    main()
