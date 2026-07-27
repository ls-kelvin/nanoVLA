#!/usr/bin/env python3
"""Summarize RoboTwin eval logs under a robotwin_eval_logs run directory."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EVAL_LOG_PATTERN = re.compile(
    r"^(?P<task>.+)_(?P<config>demo_(?:clean|randomized))"
    r"_slot\d+_gpu\d+_port\d+_eval\.log$"
)
SUCCESS_RATE_PATTERN = re.compile(
    r"Success rate:\s*(?P<success>\d+)/(?P<total>\d+)\s*=>\s*(?P<percent>[\d.]+)%"
)
TASK_NAME_PATTERN = re.compile(r"^task_name:\s*(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class TaskEvalResult:
    task: str
    task_config: str
    success: int
    total: int
    accuracy: float
    log_path: Path

    @property
    def completed(self) -> int:
        return self.total


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def parse_task_from_filename(log_path: Path) -> tuple[str, str] | None:
    match = EVAL_LOG_PATTERN.match(log_path.name)
    if match is None:
        return None
    return match.group("task"), match.group("config")


def parse_latest_success_rate(log_path: Path) -> tuple[int, int, float] | None:
    text = strip_ansi(log_path.read_text(encoding="utf-8", errors="replace"))
    latest: tuple[int, int, float] | None = None
    for match in SUCCESS_RATE_PATTERN.finditer(text):
        latest = (
            int(match.group("success")),
            int(match.group("total")),
            float(match.group("percent")),
        )
    return latest


def parse_task_name_from_log(log_path: Path) -> str | None:
    text = strip_ansi(log_path.read_text(encoding="utf-8", errors="replace"))
    match = TASK_NAME_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1)


def collect_results(log_dir: Path) -> list[TaskEvalResult]:
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    results: list[TaskEvalResult] = []
    for log_path in sorted(log_dir.glob("*_eval.log")):
        parsed_name = parse_task_from_filename(log_path)
        if parsed_name is None:
            task_name = parse_task_name_from_log(log_path)
            task_config = "unknown"
            if task_name is None:
                print(f"Skip {log_path.name}: cannot parse task name", file=sys.stderr)
                continue
        else:
            task_name, task_config = parsed_name

        success_rate = parse_latest_success_rate(log_path)
        if success_rate is None:
            print(f"Skip {log_path.name}: no success rate found", file=sys.stderr)
            continue

        success, total, accuracy = success_rate
        results.append(
            TaskEvalResult(
                task=task_name,
                task_config=task_config,
                success=success,
                total=total,
                accuracy=accuracy,
                log_path=log_path,
            )
        )

    return results


def format_markdown(log_dir: Path, results: list[TaskEvalResult]) -> str:
    if not results:
        return f"# RoboTwin Eval Summary\n\nNo eval results found under `{log_dir}`.\n"

    task_configs = sorted({item.task_config for item in results})
    lines = [
        "# RoboTwin Eval Summary",
        "",
        f"Log directory: `{log_dir}`",
        f"Tasks: {len(results)}",
        "",
    ]

    for task_config in task_configs:
        section_rows = [item for item in results if item.task_config == task_config]
        if len(task_configs) > 1:
            lines.extend([f"## {task_config}", ""])

        lines.extend(
            [
                "| task | completed | success | accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
        )

        total_success = 0
        total_completed = 0
        for item in section_rows:
            total_success += item.success
            total_completed += item.total
            lines.append(
                f"| {item.task} | {item.completed} | {item.success} | {item.accuracy:.1f}% |"
            )

        if section_rows:
            overall = total_success / total_completed * 100 if total_completed else 0.0
            task_avg = sum(item.accuracy for item in section_rows) / len(section_rows)
            lines.extend(
                [
                    f"| **avg (per task)** |  |  | **{task_avg:.1f}%** |",
                    f"| **overall (weighted)** | **{total_completed}** | **{total_success}** | **{overall:.1f}%** |",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the latest success rate for each task from RoboTwin "
            "robotwin_eval_logs run directory."
        )
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Path to a robotwin_eval_logs run directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to write markdown output",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log_dir = args.log_dir.expanduser().resolve()

    try:
        results = collect_results(log_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    markdown = format_markdown(log_dir, results)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
