#!/usr/bin/env python3
"""Summarize LIBERO-Plus raw eval JSONL by robustness category."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import OrderedDict
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from typing import Any

CATEGORY_COLUMNS = OrderedDict(
    [
        ("Camera Viewpoints", "Camera"),
        ("Robot Initial States", "Robot"),
        ("Language Instructions", "Language"),
        ("Light Conditions", "Light"),
        ("Background Textures", "Background"),
        ("Sensor Noise", "Noise"),
        ("Objects Layout", "Layout"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate RLinf LIBERO-Plus raw eval JSONL files into the "
            "README category table."
        )
    )
    parser.add_argument(
        "raw_paths",
        nargs="*",
        help="Raw JSONL files or glob patterns, for example '../results/*.jsonl'.",
    )
    parser.add_argument(
        "--raw-glob",
        action="append",
        default=[],
        help="Additional raw JSONL glob pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--classification-path",
        default=None,
        help=(
            "Optional task_classification.json path. By default this is loaded "
            "from the installed liberoplus package."
        ),
    )
    parser.add_argument(
        "--dedupe",
        choices=("first", "last", "none"),
        default="first",
        help="How to handle duplicate (suite, task_id, trial_id) records.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Number of decimal places for percentage output.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for a machine-readable summary JSON file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on raw records that are missing from task_classification.json.",
    )
    return parser.parse_args()


def resolve_task_classification_path(path_arg: str | None) -> Path:
    if path_arg is not None:
        path = Path(path_arg).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"task_classification.json not found: {path}")
        return path

    candidates: list[Path] = []
    try:
        candidates.append(
            Path(
                str(
                    files("liberoplus.liberoplus.benchmark").joinpath(
                        "task_classification.json"
                    )
                )
            )
        )
    except Exception:
        pass

    try:
        benchmark = import_module("liberoplus.liberoplus.benchmark")
        candidates.append(
            Path(benchmark.__file__).with_name("task_classification.json")
        )
    except Exception:
        pass

    candidates.append(
        Path.home()
        / "libs"
        / "libero_plus"
        / "liberoplus"
        / "liberoplus"
        / "benchmark"
        / "task_classification.json"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find LIBERO-Plus task_classification.json. "
        "Run `pip install -e ~/libs/libero_plus` or pass --classification-path."
    )


def load_classification(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    task_map: dict[tuple[str, int], dict[str, Any]] = {}
    for suite, entries in data.items():
        for index, entry in enumerate(entries):
            task_id = int(entry.get("id", index + 1)) - 1
            key = (str(suite), task_id)
            if key in task_map:
                raise ValueError(f"Duplicate classification entry for {key}")
            task_map[key] = {
                "name": entry.get("name"),
                "category": entry.get("category"),
            }
    return task_map


def expand_raw_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        matches = [Path(match) for match in glob.glob(str(expanded))]
        if not matches and expanded.is_file():
            matches = [expanded]
        paths.extend(matches)
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise FileNotFoundError("No raw JSONL files matched the provided paths.")
    return paths


def iter_raw_records(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_no}: {exc}"
                    ) from exc
                yield path, line_no, record


def coerce_success(record: dict[str, Any]) -> int | None:
    for key in ("success", "success_once", "is_success"):
        if key in record and record[key] is not None:
            return int(bool(record[key]))
    return None


def normalize_records(
    paths: list[Path], task_map: dict[tuple[str, int], dict[str, Any]], args
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[tuple[str, int]] = []

    for path, line_no, record in iter_raw_records(paths):
        try:
            suite = str(record["suite"])
            task_id = int(record["task_id"])
        except KeyError as exc:
            raise KeyError(f"Missing {exc} in {path}:{line_no}") from exc

        classification = task_map.get((suite, task_id))
        if classification is None:
            missing.append((suite, task_id))
            if args.strict:
                raise KeyError(
                    f"{path}:{line_no} has no classification for "
                    f"suite={suite!r}, task_id={task_id}"
                )
            continue

        success = coerce_success(record)
        if success is None:
            raise KeyError(f"Missing success field in {path}:{line_no}")

        records.append(
            {
                "suite": suite,
                "task_id": task_id,
                "trial_id": record.get("trial_id"),
                "success": success,
                "category": classification["category"],
                "task_name": classification["name"],
            }
        )

    if missing:
        unique_missing = sorted(set(missing))
        print(
            f"Skipped {len(missing)} records without classification "
            f"({len(unique_missing)} unique).",
            file=sys.stderr,
        )
    return dedupe_records(records, args.dedupe)


def dedupe_records(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "none":
        return records

    deduped: OrderedDict[tuple[str, int, Any], dict[str, Any]] = OrderedDict()
    for record in records:
        key = (record["suite"], record["task_id"], record["trial_id"])
        if mode == "first" and key in deduped:
            continue
        deduped[key] = record
    return list(deduped.values())


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    columns = OrderedDict(
        (column, {"success": 0, "count": 0}) for column in CATEGORY_COLUMNS.values()
    )
    total = {"success": 0, "count": 0}

    for record in records:
        category = record["category"]
        column = CATEGORY_COLUMNS.get(category)
        if column is None:
            continue
        columns[column]["success"] += record["success"]
        columns[column]["count"] += 1
        total["success"] += record["success"]
        total["count"] += 1

    return {"columns": columns, "total": total}


def rate(bucket: dict[str, int]) -> float | None:
    if bucket["count"] == 0:
        return None
    return 100.0 * bucket["success"] / bucket["count"]


def format_rate(bucket: dict[str, int], precision: int) -> str:
    value = rate(bucket)
    if value is None:
        return "nan"
    return f"{value:.{precision}f}"


def print_markdown_table(summary: dict[str, Any], precision: int):
    headers = list(summary["columns"].keys()) + ["Total"]
    values = [
        format_rate(bucket, precision) for bucket in summary["columns"].values()
    ] + [format_rate(summary["total"], precision)]

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    print("| " + " | ".join(values) + " |")
    print()
    print("Counts:")
    for column, bucket in summary["columns"].items():
        print(f"{column}: {bucket['success']}/{bucket['count']}")
    total = summary["total"]
    print(f"Total: {total['success']}/{total['count']}")


def write_summary_json(path: Path, summary: dict[str, Any], classification_path: Path):
    payload = {
        "classification_path": str(classification_path),
        "columns": {
            column: {
                **bucket,
                "success_rate": rate(bucket),
            }
            for column, bucket in summary["columns"].items()
        },
        "total": {
            **summary["total"],
            "success_rate": rate(summary["total"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    patterns = args.raw_glob + args.raw_paths
    if not patterns:
        raise SystemExit("Provide at least one raw JSONL path or --raw-glob pattern.")

    classification_path = resolve_task_classification_path(args.classification_path)
    task_map = load_classification(classification_path)
    raw_paths = expand_raw_paths(patterns)
    records = normalize_records(raw_paths, task_map, args)
    summary = aggregate(records)

    print(f"Classification: {classification_path}")
    print(f"Raw files: {len(raw_paths)}")
    print(f"Records after dedupe={args.dedupe}: {len(records)}")
    print()
    print_markdown_table(summary, args.precision)

    if args.output_json is not None:
        write_summary_json(
            Path(args.output_json).expanduser(), summary, classification_path
        )


if __name__ == "__main__":
    main()
