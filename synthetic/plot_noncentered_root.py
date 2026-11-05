#!/usr/bin/env python3
"""Plot Session-46 non-centered-root lambda sweeps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _apply_style() -> None:
    try:
        plt.style.use(["science", "notebook", "./custom.mplstyle"])
    except Exception:
        plt.style.use(["seaborn-v0_8-paper", "./custom.mplstyle"])
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )


def _load_shift_by_run_id(results_dir: Path | None) -> dict[str, float]:
    if results_dir is None:
        return {}
    shift_by_run_id = {}
    for path in results_dir.glob("mlruns_*/*/*/params/prototype_shift_norm"):
        try:
            shift_by_run_id[path.parent.parent.name] = float(path.read_text().strip())
        except ValueError:
            continue
    return shift_by_run_id


def _load_rows(path: Path, results_dir: Path | None = None) -> list[dict]:
    with path.open() as f:
        rows = json.load(f)
    shift_by_run_id = _load_shift_by_run_id(results_dir)
    normalized = []
    for row in rows:
        if "shift" in row:
            shift = float(row["shift"])
        elif "prototype_shift_norm" in row:
            shift = float(row["prototype_shift_norm"])
        else:
            run_id = row.get("run_id")
            if run_id not in shift_by_run_id:
                raise KeyError(
                    "shift not present in JSON and could not be recovered from results_dir "
                    f"for run_id={run_id}"
                )
            shift = shift_by_run_id[run_id]
        normalized.append(
            {
                "shift": shift,
                "lambda": float(row["lambda"]) if "lambda" in row else float(row["lambda_reg"]),
                "hier_viol": float(row["hier_viol"]) if "hier_viol" in row else float(row["hier_violation_mean"]),
                "proto_align_min": (
                    float(row["proto_align_min"])
                    if "proto_align_min" in row
                    else float(row.get("prototype_align_min_nonroot", np.nan))
                ),
                "cor_form_align_min": (
                    float(row["cor_form_align_min"])
                    if "cor_form_align_min" in row
                    else float(row.get("cor_form_align_min_nonroot", np.nan))
                ),
            }
        )
    return normalized


def _filter_shift(rows: list[dict], shift: float) -> list[dict]:
    filtered = [row for row in rows if np.isclose(float(row["shift"]), shift)]
    if not filtered:
        raise RuntimeError(f"No rows found for shift={shift:g}")
    return sorted(filtered, key=lambda row: float(row["lambda"]))


def _filter_lambda(rows: list[dict], lambda_reg: float) -> list[dict]:
    filtered = [row for row in rows if np.isclose(float(row["lambda"]), lambda_reg)]
    if not filtered:
        raise RuntimeError(f"No rows found for lambda={lambda_reg:g}")
    return sorted(filtered, key=lambda row: float(row["shift"]))


def _plot_shift(
    rows: list[dict],
    shift: float,
    output_dir: Path,
    align_key: str,
    align_label: str,
    stem_suffix: str,
    output_stem_suffix: str = "",
) -> None:
    lambdas = np.array([float(row["lambda"]) for row in rows], dtype=float)
    viol_pct = 100.0 * np.array([float(row["hier_viol"]) for row in rows], dtype=float)
    align = np.array([float(row[align_key]) for row in rows], dtype=float)

    left_color = "C1"
    right_color = "C0"

    fig, ax1 = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    ax1.plot(lambdas, viol_pct, "o-", color=left_color, linewidth=2, markersize=5)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\lambda_{\mathrm{reg}}$")
    ax1.set_ylabel(r"$\mathcal{HAS}$-Viol (%)", color=left_color)
    ax1.set_ylim(0.0, max(viol_pct.max() * 1.08, 0.02))

    ax2 = ax1.twinx()
    ax2.plot(lambdas, align, "s--", color=right_color, linewidth=2, markersize=5)
    ax2.set_ylabel(align_label, color=right_color)
    ax2.set_ylim(max(0.0, align.min() - 0.05), 1.01)

    stem = f"lambda_vs_hiervio_{stem_suffix}_shift_{shift:.0e}{output_stem_suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def _plot_fixed_lambda_shift(
    rows: list[dict],
    lambda_reg: float,
    output_dir: Path,
    output_stem_suffix: str = "",
) -> None:
    shifts = np.array([float(row["shift"]) for row in rows], dtype=float)
    viol_pct = 100.0 * np.array([float(row["hier_viol"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    ax.plot(shifts, viol_pct, "o-", linewidth=2, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Root Norm")
    ax.set_ylabel(r"$\mathcal{HAS}$-Viol (%)")
    ax.set_ylim(0.0, max(viol_pct.max() * 1.08, 0.01))

    stem = f"rootnorm_vs_hiervio_lambda_{lambda_reg:.0e}{output_stem_suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default="files/autosweep/46/branch_margin.json",
        help="Session-46 summary JSON.",
    )
    parser.add_argument(
        "--results_dir",
        default="files/analytic_model/sweep_autosweep_46_20260423_145627",
        help="Session-46 results dir used to recover shift labels when absent from JSON.",
    )
    parser.add_argument(
        "--output_dir",
        default="files/autosweep/46",
        help="Directory for plot outputs.",
    )
    parser.add_argument(
        "--shifts",
        nargs="+",
        type=float,
        default=[1e-4, 1e-2],
        help="Prototype shifts to plot.",
    )
    parser.add_argument(
        "--output_stem_suffix",
        default="",
        help="Optional suffix appended to output stems, e.g. _corrected.",
    )
    parser.add_argument(
        "--fixed_lambda",
        type=float,
        default=None,
        help="Optional fixed lambda for a Root Norm vs Violation Percentage plot.",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    results_dir = Path(args.results_dir) if args.results_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    rows = _load_rows(input_json, results_dir=results_dir)
    for shift in args.shifts:
        shift_rows = _filter_shift(rows, shift)
        _plot_shift(
            shift_rows,
            shift,
            output_dir,
            align_key="proto_align_min",
            align_label="Prototype Alignment",
            stem_suffix="protoalign",
            output_stem_suffix=args.output_stem_suffix,
        )
        _plot_shift(
            shift_rows,
            shift,
            output_dir,
            align_key="cor_form_align_min",
            align_label="Centroid Alignment",
            stem_suffix="centroidalign",
            output_stem_suffix=args.output_stem_suffix,
        )
    if args.fixed_lambda is not None:
        lambda_rows = _filter_lambda(rows, args.fixed_lambda)
        _plot_fixed_lambda_shift(
            lambda_rows,
            args.fixed_lambda,
            output_dir,
            output_stem_suffix=args.output_stem_suffix,
        )


if __name__ == "__main__":
    main()
