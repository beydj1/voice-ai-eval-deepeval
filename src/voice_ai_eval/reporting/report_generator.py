from __future__ import annotations

import html
from pathlib import Path

from voice_ai_eval.models import BatchResult


def write_reports(batch: BatchResult, report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "evaluation-results.json"
    markdown_path = report_dir / "evaluation-report.md"
    html_path = report_dir / "evaluation-report.html"
    batch.save_json(json_path)

    lines = [
        "# Voice AI Evaluation Report",
        "",
        f"- Calls evaluated: **{len(batch.results)}**",
        f"- Batch status: **{'PASS' if batch.passed else 'FAIL'}**",
        f"- Average score: **{batch.average_score:.2%}**",
        "",
    ]
    for result in batch.results:
        lines.extend([
            f"## {result.call_id} — {'PASS' if result.passed else 'FAIL'}",
            "",
            f"Scenario: {result.scenario}",
            f"Overall score: {result.overall_score:.2%}",
            f"Policy: {result.policy_name} v{result.policy_version}",
            "",
            "### Deterministic checks",
        ])
        lines.extend(f"- {'✅' if r.passed else '❌'} `{r.rule_id}` ({r.severity}): {r.message}" for r in result.deterministic_results)
        lines.append("\n### LLM metrics")
        lines.extend(f"- {'✅' if m.passed else '❌'} **{m.name}**: {m.score:.2%} — {m.reason}" for m in result.metric_results)
        lines.append("")
    markdown = "\n".join(lines)
    markdown_path.write_text(markdown, encoding="utf-8")

    escaped = html.escape(markdown)
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Voice AI Evaluation</title><style>body{{font-family:system-ui;max-width:1000px;margin:40px auto;padding:0 20px;line-height:1.45}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:20px;border-radius:8px}}</style></head><body><pre>{escaped}</pre></body></html>", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path, "html": html_path}
