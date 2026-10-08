"""Local CLI for the Minecraft wiki RAG (no server)."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src import config, embed, evaluate, generate, store, wiki
from src import ingest as ingest_mod
from src.retrieve import search as retrieve_search

app = typer.Typer(help="Local Minecraft wiki RAG: ingest, search, ask, evaluate.")
console = Console()


def _highlight(hit) -> str:
    source = f"{hit.title} > {hit.section}"
    scores = f"cosine={hit.cosine:.3f}"
    if hit.rerank_score is not None:
        scores += f"  rerank={hit.rerank_score:.3f}"
    return f"[bold cyan]{source}[/]  [dim]{scores}[/]"


@app.command()
def ingest(
    scope: str = typer.Option(
        "categories", "--scope", help="'categories' (curated set) or 'all' (~18k pages)"
    ),
    categories: str | None = typer.Option(
        None,
        "--categories",
        help="Comma-separated category override (e.g. 'Machines,Food')",
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-fetch every page, ignoring the cache"
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Cap the number of records fetched"
    ),
    mode: str = typer.Option(
        config.FETCH_MODE,
        "--mode",
        help="Fetch mode: 'wikitext' (fast, 50/request) or 'extracts'",
    ),
):
    """Fetch wiki pages, chunk them, embed them, and upsert into Chroma."""
    if mode not in ("wikitext", "extracts"):
        console.print(f"[red]Unknown --mode {mode!r}; use 'wikitext' or 'extracts'.[/]")
        raise typer.Exit(2)
    if categories is not None:
        category_list = [c.strip() for c in categories.split(",") if c.strip()]
    elif scope == "all":
        category_list = None
    else:
        category_list = config.CRAWL_CATEGORIES

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("enumerating pages", total=None)

        def on_progress(done: int, total: int, title: str) -> None:
            progress.update(
                task,
                total=total,
                completed=done,
                description=f"fetching wiki pages ({title})",
            )

        records = list(
            wiki.crawl(
                category_list,
                force=force,
                limit=limit,
                mode=mode,
                on_progress=on_progress,
            )
        )

    if not records:
        console.print("[yellow]No records found — check your scope and cache.[/]")
        raise typer.Exit()

    unique: dict[int, dict] = {}
    for record in records:
        unique.setdefault(record["pageid"], record)
    records = list(unique.values())

    console.print(f"[green]Loaded {len(records)} articles.[/]")

    chunks = ingest_mod.chunk_records(records)
    console.print(f"[green]Chunked into {len(chunks)} chunks.[/]")

    embeddings = embed.embed_documents([c["embed_text"] for c in chunks])
    store.reset()
    store.upsert(chunks, embeddings)

    console.print(
        Panel(
            f"Ingested [bold]{len(records)}[/] articles as [bold]{len(chunks)}[/] chunks "
            f"into [bold]{config.COLLECTION_NAME}[/].",
            title="Done",
        )
    )


@app.command()
def search(
    query: str,
    k: int = typer.Option(config.RERANK_TOP_K, "--k", help="Results to show"),
    top_n: int = typer.Option(
        config.RETRIEVE_TOP_N, "--top-n", help="Candidates before reranking"
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Skip the cross-encoder stage"
    ),
):
    """Show retrieval results for a query, with scores."""
    hits = retrieve_search(query, k=k, top_n=top_n, rerank=not no_rerank)

    if not hits:
        console.print("[yellow]No results — run ingest first.[/]")
        raise typer.Exit()

    table = Table(title=query)
    table.add_column("#", justify="right")
    table.add_column("cosine:", justify="right")
    table.add_column("rerank:", justify="right")
    table.add_column("source")
    table.add_column("section")
    table.add_column("url", overflow="fold")

    for rank, hit in enumerate(hits, start=1):
        rerank_score = (
            f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "-"
        )
        table.add_row(
            str(rank),
            f"{hit.cosine:.3f}",
            rerank_score,
            f"[cyan]{hit.title}[/]",
            hit.section,
            f"[dim]{hit.url}[/]",
        )

    console.print(table)


@app.command()
def ask(query: str):
    """Retrieve context and answer with the local Ollama model."""
    hits = retrieve_search(query)

    if not hits:
        console.print("[yellow]No results — run ingest first.[/]")
        raise typer.Exit()

    context = "\n\n".join(
        f"[{i}] {hit.title} > {hit.section}\n{hit.content}\n({hit.url})"
        for i, hit in enumerate(hits, start=1)
    )

    console.print("[dim]Generating answer with llama3.1:8b...[/]")
    answer = generate.generate_answer(query, context)

    console.print(Panel(answer, title=f"Answer", border_style="green"))
    console.print("\n[bold]Sources:[/]")
    for rank, hit in enumerate(hits, start=1):
        console.print(f"  [{rank}] {_highlight(hit)}")
        console.print(f"      {hit.url}")


@app.command()
def eval():
    """Run the retrieval evaluation (Recall@1/5, MRR) on both stages."""
    evaluate.main()


@app.command()
def status():
    """Show index and cache stats."""
    cache = wiki.load_cache(config.WIKI_CACHE_PATH)
    titles = store.indexed_titles()
    console.print(
        Panel(
            f"Cache records:       [bold]{len(cache)}[/]\n"
            f"Indexed chunks:      [bold]{store.count()}[/]\n"
            f"Indexed pages:       [bold]{len(titles)}[/]\n"
            f"Embed model:         {config.EMBED_MODEL}\n"
            f"Rerank model:        {config.RERANK_MODEL}\n"
            f"Ollama model:        {config.OLLAMA_MODEL}\n"
            f"Chroma dir:          {config.CHROMA_DIR}\n"
            f"Wiki cache:          {config.WIKI_CACHE_PATH}",
            title="Status",
        )
    )


@app.command()
def reset():
    """Delete the Chroma index (keeps the wiki cache)."""
    store.reset()
    console.print("[green]Index cleared.[/]")


if __name__ == "__main__":
    app()
