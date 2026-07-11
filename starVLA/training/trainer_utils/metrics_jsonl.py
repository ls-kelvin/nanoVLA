from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

METRICS_JSONL_FILENAME = "metrics.jsonl"


def to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    return float(value)


def metrics_jsonl_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / METRICS_JSONL_FILENAME


def append_metrics_jsonl(path: str | Path, step: int, metrics: Mapping[str, Any]) -> None:
    record = {"step": int(step)}
    for key, value in metrics.items():
        record[str(key)] = to_json_value(value)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def load_metrics_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics jsonl not found: {path}")

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "step" not in record:
                raise ValueError(f"Missing `step` field at {path}:{line_number}")
            records.append(record)
    return records


def dedupe_metrics_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for record in records:
        step = int(record["step"])
        by_step[step] = dict(record)
    return [by_step[step] for step in sorted(by_step)]
