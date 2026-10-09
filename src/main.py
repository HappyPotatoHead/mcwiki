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

from .app import App
from .config import (
    CATEGORIES,
    CHROMA_DIR,
    COLLECTION_NAME,
    DEVICE,
    EMBED_MODEL,
    FETCH_MODE,
    OLLAMA_MODEL,
    QUERY_INSTRUCTION,
    RERANK_MODEL,
    RERANK_TOP_K,
    RETRIEVE_TOP_N,
    WIKI_CACHE_PATH,
)
from .embedder import Embedder
from .evaluate import main as evaluate_main
from .generate import Generator
from .ingestion.wiki import load_cache
from .retrieve import Reranker
from .store import ChromaStore

app = typer.Typer()
console = Console()
embedder = Embedder(EMBED_MODEL, DEVICE)
reranker = Reranker(RERANK_MODEL, DEVICE)
database = ChromaStore(str(CHROMA_DIR), COLLECTION_NAME)
generator = Generator(OLLAMA_MODEL)

rag = App(generator, embedder, reranker, database, QUERY_INSTRUCTION)


def _highlight(match) -> str:
    source = f"{match.title} > {match.section}"
    scores = f"cosine={match.cosine:.3f}"
    if match.rerank_score is not None:
        scores += f"  rerank={match.rerank_score:.3f}"

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
        FETCH_MODE,
        "--mode",
        help="Fetch mode: 'wikitext' (fast, 50/request) or 'extracts'",
    ),
):
    if mode not in ("wikitext", "extracts"):
        console.print(f"[red]Unknown --mode {mode!r}; use 'wikitext' or 'extracts'.[/]")
        raise typer.Exit(2)
    if categories is not None:
        category_list = [
            category.strip() for category in categories.split(",") if category.strip()
        ]
    elif scope == "all":
        category_list = None
    else:
        category_list = CATEGORIES

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

        result = rag.ingest(
            str(CHROMA_DIR),
            COLLECTION_NAME,
            categories=category_list,
            force=force,
            limit=limit,
            mode=mode,
            on_progress=on_progress,
        )

    if result["articles"] == 0:
        console.print("[yellow]No records found — check your scope and cache.[/]")
        raise typer.Exit()

    console.print(
        Panel(
            f"Ingested [bold]{result['articles']}[/] articles "
            f"as [bold]{result['chunks']}[/] chunks "
            f"into [bold]{COLLECTION_NAME}[/].",
            title="Done",
        )
    )


@app.command()
def search(
    query: str,
    k: int = typer.Option(RERANK_TOP_K, "--k", help="Results to show"),
    top_n: int = typer.Option(
        RETRIEVE_TOP_N, "--top-n", help="Candidates before reranking"
    ),
    rerank: bool = typer.Option(
        False, "--no-rerank", help="Skip the cross-encoder stage"
    ),
):
    matches = rag.search(query, k, top_n, not rerank)

    table = Table(title=query)
    table.add_column("#", justify="right")
    table.add_column("cosine:", justify="right")
    table.add_column("rerank:", justify="right")
    table.add_column("source")
    table.add_column("section")
    table.add_column("url", overflow="fold")

    for rank, hit in enumerate(matches, start=1):
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
    matches = rag.search(query, RERANK_TOP_K, RETRIEVE_TOP_N, True)

    if not matches:
        console.print("[yellow]No results — run ingest first.[/]")
        raise typer.Exit()

    context = "\n\n".join(
        f"[{index}] {match.title} > {match.section}\n{match.content}\n({match.url})"
        for index, match in enumerate(matches, start=1)
    )

    console.print(f"[dim]Generating answer with {OLLAMA_MODEL}[/]")
    answer = rag.generate_answers(query, context)

    console.print(Panel(answer, title="Answer", border_style="green"))
    console.print("\n[bold]Sources:[/]")
    for rank, match in enumerate(matches, start=1):
        console.print(f"  [{rank}] {_highlight(match)}")
        console.print(f"      {match.url}")


@app.command()
def eval():
    evaluate_main()


@app.command()
def status():
    """Show index and cache stats."""
    cache = load_cache(WIKI_CACHE_PATH)
    titles = database.indexed_titles()
    console.print(
        Panel(
            f"Cache records:       [bold]{len(cache)}[/]\n"
            f"Indexed chunks:      [bold]{database.__len__()}[/]\n"
            f"Indexed pages:       [bold]{len(titles)}[/]\n"
            f"Embed model:         {EMBED_MODEL}\n"
            f"Rerank model:        {RERANK_MODEL}\n"
            f"Ollama model:        {OLLAMA_MODEL}\n"
            f"Chroma dir:          {CHROMA_DIR}\n"
            f"Wiki cache:          {WIKI_CACHE_PATH}",
            title="Status",
        )
    )


@app.command()
def reset():
    database.reset()
    console.print("[green]Index cleared.[/]")


if __name__ == "__main__":
    app()
