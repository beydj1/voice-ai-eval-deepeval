from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from voice_ai_eval.config import settings
from voice_ai_eval.evaluation.engine import EvaluationEngine
from voice_ai_eval.loaders.transcript_loader import LoadError, load_policy, load_transcripts
from voice_ai_eval.models import BatchResult
from voice_ai_eval.providers.factory import create_provider
from voice_ai_eval.reporting.report_generator import write_reports

app = typer.Typer(help="Evaluate voice-agent transcripts with deterministic checks and DeepEval.")
console = Console()


@app.command()
def validate(input_path: Path = typer.Argument(..., exists=True), policy_path: Path = typer.Option(Path("data/policies/default.yaml"), "--policy", exists=True)) -> None:
    """Validate transcript and policy schemas without invoking an LLM."""
    try:
        transcripts = load_transcripts(input_path)
        policy = load_policy(policy_path)
    except LoadError as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.print(f"[green]Valid[/green]: {len(transcripts)} transcript(s), policy {policy.name} v{policy.version}")


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True),
    policy_path: Path = typer.Option(Path("data/policies/default.yaml"), "--policy", exists=True),
    provider: str = typer.Option(settings.provider, "--provider"),
    model: str = typer.Option(settings.model, "--model"),
    threshold: float = typer.Option(settings.threshold, "--threshold", min=0.0, max=1.0),
    report_dir: Path = typer.Option(settings.report_dir, "--report-dir"),
    fail_below: float | None = typer.Option(None, "--fail-below", min=0.0, max=1.0),
) -> None:
    """Run batch evaluation and generate JSON, Markdown, and HTML reports."""
    try:
        loaded = load_transcripts(input_path)
        policy = load_policy(policy_path)
        judge = create_provider(provider, model, settings.local_model_base_url)
        engine = EvaluationEngine(judge, threshold=threshold, fail_on_critical=settings.fail_on_critical)
        batch = BatchResult(results=[engine.evaluate(transcript, policy, path) for path, transcript in loaded])
    except (LoadError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Evaluation failed:[/red] {exc}")
        raise typer.Exit(2) from exc

    paths = write_reports(batch, report_dir)
    table = Table(title="Voice AI Evaluation")
    table.add_column("Call")
    table.add_column("Score")
    table.add_column("Status")
    for result in batch.results:
        table.add_row(result.call_id, f"{result.overall_score:.2%}", "PASS" if result.passed else "FAIL")
    console.print(table)
    console.print(f"Reports: {paths['markdown']}, {paths['json']}, {paths['html']}")

    gate = fail_below if fail_below is not None else threshold
    if not batch.passed or batch.average_score < gate:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
