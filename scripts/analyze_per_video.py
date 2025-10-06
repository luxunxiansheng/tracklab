#!/usr/bin/env python3
"""Utility to extract per-video GS-HOTA metrics from a TrackLab run.

This script parses the TrackEval `person_detailed.csv` produced by the
`tracklab` evaluation pipeline and generates:

1. A console summary highlighting the best/worst sequences for HOTA/DetA/AssA.
2. A JSON artifact containing the per-sequence metrics for downstream analysis.

Usage options:
    python scripts/analyze_per_video.py \
        --experiment sn-offical \
        [--run-dir /path/to/specific/run]

When `--run-dir` is omitted the script automatically selects the most recent
run under `outputs/<experiment>/<YYYY-MM-DD>/<HH-MM-SS>`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Columns of interest in TrackEval detailed CSV files. Values are stored as
# ratios (e.g., 0.332 for 33.2%), so we scale them to percentage points.
VALUE_COLUMNS = {
    "HOTA___AUC": "hota",
    "DetA___AUC": "deta",
    "AssA___AUC": "assa",
    "DetRe___AUC": "det_re",
    "DetPr___AUC": "det_pr",
    "AssRe___AUC": "ass_re",
    "AssPr___AUC": "ass_pr",
    "LocA___AUC": "loca",
    "OWTA___AUC": "owta",
}

PERCENT_COLUMNS = {
    "MOTA": "mota",
    "MOTP": "motp",
    "CLR_Re": "clr_re",
    "CLR_Pr": "clr_pr",
    "MTR": "mtr",
    "PTR": "ptr",
    "MLR": "mlr",
    "IDF1": "idf1",
    "IDR": "idr",
    "IDP": "idp",
}

COUNT_COLUMNS = {
    "CLR_TP": "clr_tp",
    "CLR_FN": "clr_fn",
    "CLR_FP": "clr_fp",
    "IDSW": "idsw",
    "Frag": "fragments",
    "IDTP": "idtp",
    "IDFN": "idfn",
    "IDFP": "idfp",
}


@dataclass
class SequenceMetrics:
    seq: str
    metrics: Dict[str, float]

    @property
    def hota(self) -> float:
        return self.metrics.get("hota", float("nan"))

    @property
    def deta(self) -> float:
        return self.metrics.get("deta", float("nan"))

    @property
    def assa(self) -> float:
        return self.metrics.get("assa", float("nan"))

    @property
    def idswitches(self) -> float:
        return self.metrics.get("idsw", float("nan"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-video GS-HOTA metrics")
    parser.add_argument(
        "--experiment",
        default="sn-offical",
        help="Name of the experiment under outputs/ (default: sn-offical)",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory that contains experiment subfolders (default: outputs)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Optional explicit path to a TrackLab run directory",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Optional output path for JSON metrics. Defaults to <run_dir>/analysis/per_video_metrics.json",
    )
    return parser.parse_args()


def find_latest_run(outputs_root: Path, experiment: str) -> Path:
    experiment_dir = outputs_root / experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    date_dirs = sorted(
        (d for d in experiment_dir.iterdir() if d.is_dir()), reverse=True
    )
    for date_dir in date_dirs:
        time_dirs = sorted((d for d in date_dir.iterdir() if d.is_dir()), reverse=True)
        if time_dirs:
            return time_dirs[0]
    raise FileNotFoundError(f"No run directories found under {experiment_dir}")


def load_per_sequence_metrics(csv_path: Path) -> List[SequenceMetrics]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Detailed metrics file not found: {csv_path}")

    sequences: List[SequenceMetrics] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seq = row.get("seq")
            if not seq or seq.upper() == "COMBINED":
                continue

            metrics: Dict[str, float] = {}
            # Scale ratio-based metrics to percentage points.
            for column, key in VALUE_COLUMNS.items():
                if column in row and row[column]:
                    metrics[key] = float(row[column]) * 100.0
            for column, key in PERCENT_COLUMNS.items():
                if column in row and row[column]:
                    metrics[key] = float(row[column]) * 100.0

            # Count metrics remain integers in the JSON payload.
            for column, key in COUNT_COLUMNS.items():
                if column in row and row[column]:
                    metrics[key] = int(round(float(row[column])))

            sequences.append(SequenceMetrics(seq=seq, metrics=metrics))
    return sequences


def summarise_sequences(
    sequences: List[SequenceMetrics],
) -> Dict[str, List[Tuple[str, float]]]:
    def sort_metric(
        metric_key: str, descending: bool = True
    ) -> List[Tuple[str, float]]:
        filtered = [
            s
            for s in sequences
            if not math.isnan(s.metrics.get(metric_key, float("nan")))
        ]
        return [
            (seq.seq, seq.metrics[metric_key])
            for seq in sorted(
                filtered,
                key=lambda s: s.metrics.get(metric_key, float("nan")),
                reverse=descending,
            )
        ]

    return {
        "hota": sort_metric("hota"),
        "deta": sort_metric("deta"),
        "assa": sort_metric("assa"),
        "idf1": sort_metric("idf1"),
        "idsw": sort_metric(
            "idsw", descending=False
        ),  # For ID switches, lower is better.
    }


def save_metrics_json(sequences: List[SequenceMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort deterministically by sequence name before writing.
    payload = {seq.seq: seq.metrics for seq in sorted(sequences, key=lambda s: s.seq)}
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def print_console_report(run_dir: Path, sequences: List[SequenceMetrics]) -> None:
    summary = summarise_sequences(sequences)
    print("\nPer-video GS-HOTA analysis")
    print("Run directory:", run_dir)
    print("Total sequences:", len(sequences))

    def print_top(
        metric: str, toplist: List[Tuple[str, float]], reverse_label: str = "Top"
    ) -> None:
        if not toplist:
            return
        print(f"\n{reverse_label} 3 by {metric.upper()}:")
        for seq, value in toplist[:3]:
            print(f"  {seq:<10} {value:6.2f}")

    print_top("HOTA", summary["hota"])
    print_top("DetA", summary["deta"])
    print_top("AssA", summary["assa"])

    # Worst performers.
    print_top("HOTA", list(reversed(summary["hota"])), reverse_label="Bottom")
    print_top("AssA", list(reversed(summary["assa"])), reverse_label="Bottom")

    # Identity issues: sequences with most ID switches.
    idsw_sorted = sorted(summary["idsw"], key=lambda item: item[1])
    if idsw_sorted:
        print("\nLowest ID switch counts:")
        for seq, value in idsw_sorted[:3]:
            print(f"  {seq:<10} {value:6.0f}")
        print("Highest ID switch counts:")
        for seq, value in idsw_sorted[-3:][::-1]:
            print(f"  {seq:<10} {value:6.0f}")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(outputs_root, args.experiment)

    csv_path = (
        run_dir
        / "exports"
        / "results"
        / args.experiment.replace("-", "")
        / "person_detailed.csv"
    )
    if not csv_path.exists():
        # TrackEval exports use the tracker name (e.g., "tracklab") instead of experiment string.
        csv_path = run_dir / "exports" / "results" / "tracklab" / "person_detailed.csv"

    sequences = load_per_sequence_metrics(csv_path)
    if not sequences:
        raise RuntimeError(f"No per-sequence records found in {csv_path}")

    # Determine JSON export path.
    if args.export_json:
        json_path = args.export_json
    else:
        json_path = run_dir / "analysis" / "per_video_metrics.json"

    save_metrics_json(sequences, json_path)
    print_console_report(run_dir, sequences)
    print(f"\nSaved per-video metrics to {json_path}")


if __name__ == "__main__":
    main()
