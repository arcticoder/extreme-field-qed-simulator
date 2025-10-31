#!/usr/bin/env python3
"""
Anomaly scanner for sweep outputs (JSON or CSV).

Flags potential signatures of new physics by scanning sweep results for:
- Non-monotonic scaling of h_rms vs swept parameter (when expected monotonic)
- Outliers relative to a simple power-law fit (e.g., h ∝ E0^n)
- Separate grouping by boolean toggles like include_pair_losses

Usage:
  python scripts/anomaly_scanner.py --input path/to/sweep_results.json --param source.E0_Vpm
  python scripts/anomaly_scanner.py --input results/sweeps/test_mini/test_mini_sweep_summary.csv --param sweep_value --out results/anomaly_report.txt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ a * x^n in log space; return (a, n)."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return (np.nan, np.nan)
    logx = np.log10(x[mask])
    logy = np.log10(y[mask])
    n, loga = np.polyfit(logx, logy, 1)
    a = 10 ** loga
    return (a, n)


def analyze_group(df: pd.DataFrame, xcol: str, ycol: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()

    # Monotonicity (non-decreasing) check after sorting by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    monotonic = np.all(np.diff(y_sorted) >= -1e-18)  # allow tiny numerical slop
    report["monotonic_non_decreasing"] = bool(monotonic)

    # Power-law fit
    a, n = fit_power_law(x_sorted, y_sorted)
    report["power_law_a"] = float(a) if np.isfinite(a) else None
    report["power_law_n"] = float(n) if np.isfinite(n) else None

    # Outlier detection: factor deviation from fit
    if np.isfinite(a) and np.isfinite(n):
        y_fit = a * (x_sorted ** n)
        # Avoid divide-by-zero
        eps = 1e-300
        factors = np.maximum(y_sorted, eps) / np.maximum(y_fit, eps)
        outlier_idx = np.where((factors > 3.0) | (factors < 1/3.0))[0]
        report["outliers_indices"] = outlier_idx.tolist()
        report["outliers_points"] = [
            {xcol: float(x_sorted[i]), ycol: float(y_sorted[i]), "fit": float(y_fit[i]), "factor": float(factors[i])}
            for i in outlier_idx
        ]
    else:
        report["outliers_indices"] = []
        report["outliers_points"] = []

    return report


def make_report(df: pd.DataFrame, xcol: str, ycol: str, groupby: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {"groups": []}

    if groupby:
        grouped = df.groupby(groupby)
        for keys, group in grouped:
            analysis = analyze_group(group, xcol, ycol)
            results["groups"].append({"group_keys": (keys if isinstance(keys, tuple) else (keys,)), "analysis": analysis})
    else:
        analysis = analyze_group(df, xcol, ycol)
        results["groups"].append({"group_keys": (), "analysis": analysis})

    return results


def format_report(report: Dict[str, Any], xlabel: str, ylabel: str) -> str:
    lines = []
    lines.append("Anomaly Scanner Report")
    lines.append("======================\n")
    for g in report["groups"]:
        keys = g["group_keys"]
        header = "Group: " + ", ".join(str(k) for k in keys) if keys else "Group: (all)"
        lines.append(header)
        analysis = g["analysis"]
        lines.append(f"  Monotonic non-decreasing ({ylabel} vs {xlabel}): {analysis['monotonic_non_decreasing']}")
        n = analysis.get("power_law_n")
        a = analysis.get("power_law_a")
        if n is not None:
            lines.append(f"  Power-law fit: {ylabel} ≈ {a:.3e} * {xlabel}^{n:.2f}")
        else:
            lines.append("  Power-law fit: insufficient data")
        if analysis["outliers_points"]:
            lines.append("  Outliers (|factor| > 3x):")
            for p in analysis["outliers_points"]:
                lines.append(f"    - {xlabel}={p[xlabel]:.3e}, {ylabel}={p[ylabel]:.3e}, fit={p['fit']:.3e}, factor={p['factor']:.2f}")
        else:
            lines.append("  Outliers: none")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Scan sweep outputs for potential anomalies")
    ap.add_argument("--input", required=True, help="Path to sweep JSON or CSV")
    ap.add_argument("--param", required=True, help="Column for swept parameter (x)")
    ap.add_argument("--metric", default="h_rms", help="Metric column (y), defaults to h_rms")
    ap.add_argument("--groupby", nargs="*", default=None, help="Optional columns to group by (e.g., include_pair_losses)")
    ap.add_argument("--out", default=None, help="Optional path to write text report")
    args = ap.parse_args()

    path = Path(args.input)
    df = load_data(path)

    xcol = args.param
    ycol = args.metric
    groupby = args.groupby or []

    # Validate columns
    for col in [xcol, ycol] + groupby:
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in input. Available: {df.columns.tolist()}")

    report = make_report(df, xcol, ycol, groupby)
    text = format_report(report, xcol, ycol)
    print(text)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
