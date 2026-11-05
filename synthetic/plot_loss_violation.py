#!/usr/bin/env python3
"""Plot Session-40 loss/violation summaries from MLflow SQLite DBs.

Outputs (PNG + PDF):
  1) lambda_vs_mainloss_violation.*
     - x: lambda_reg (log scale)
     - y1: main_loss (loss without regularization, final step per run)
     - y2: avg violation rate over depths d1..d4 (final step per run)

  2) lambda_1e3_trace_mainloss_violation.*
     - x: training step/epoch for lambda=1e-3 run
     - y1: main_loss
     - y2: avg violation rate over depths d1..d4
  3) lambda_vs_full_loss_violation.*
     - x: lambda_reg (log scale)
     - y1: full loss (with regularization, final step per run)
     - y2: avg violation rate over depths d1..d4 (final step per run)

  4) lambda_1e3_trace_full_loss_violation.*
     - x: training step/epoch for lambda=1e-3 run
     - y1: full loss
     - y2: avg violation rate over depths d1..d4
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEPTHS = (1, 2, 3, 4)


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


def _fetch_runs(db_path: Path) -> list[tuple[str, float]]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT run_uuid, value FROM params WHERE key='lambda_reg'"
    ).fetchall()
    conn.close()
    out = []
    for run_id, lam in rows:
        try:
            out.append((run_id, float(lam)))
        except ValueError:
            continue
    return out


def _fetch_latest_metric(conn: sqlite3.Connection, run_id: str, key: str) -> float | None:
    row = conn.execute(
        "SELECT value FROM latest_metrics WHERE run_uuid=? AND key=?",
        (run_id, key),
    ).fetchone()
    return None if row is None else float(row[0])


def _aggregate_lambda_curve(db_paths: list[Path]) -> list[dict]:
    rows = []
    for db_path in db_paths:
        conn = sqlite3.connect(str(db_path))
        for run_id, lam in _fetch_runs(db_path):
            main_loss = _fetch_latest_metric(conn, run_id, "main_loss")
            full_loss = _fetch_latest_metric(conn, run_id, "loss")
            viols = []
            for d in DEPTHS:
                v = _fetch_latest_metric(conn, run_id, f"viol_rate/A/d{d}")
                if v is None:
                    break
                viols.append(v)
            if main_loss is None or full_loss is None or len(viols) != len(DEPTHS):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "lambda_reg": lam,
                    "main_loss": main_loss,
                    "full_loss": full_loss,
                    "avg_viol": float(np.mean(viols)),
                }
            )
        conn.close()

    by_lambda: dict[float, list[dict]] = {}
    for row in rows:
        by_lambda.setdefault(row["lambda_reg"], []).append(row)

    curve = []
    for lam in sorted(by_lambda):
        vals = by_lambda[lam]
        curve.append(
            {
                "lambda_reg": lam,
                "main_loss": float(np.mean([v["main_loss"] for v in vals])),
                "full_loss": float(np.mean([v["full_loss"] for v in vals])),
                "avg_viol": float(np.mean([v["avg_viol"] for v in vals])),
                "n_runs": len(vals),
            }
        )
    return curve


def _find_lambda_run(db_paths: list[Path], target_lambda: float) -> tuple[Path, str]:
    for db_path in db_paths:
        for run_id, lam in _fetch_runs(db_path):
            if np.isclose(lam, target_lambda):
                return db_path, run_id
    raise RuntimeError(f"Could not find run with lambda_reg={target_lambda:g}")


def _fetch_step_series(conn: sqlite3.Connection, run_id: str, key: str) -> dict[int, float]:
    rows = conn.execute(
        "SELECT step, value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step ASC",
        (run_id, key),
    ).fetchall()
    return {int(step): float(value) for step, value in rows}


def _fetch_trace(db_path: Path, run_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    main = _fetch_step_series(conn, run_id, "main_loss")
    full = _fetch_step_series(conn, run_id, "loss")
    viol_by_depth = [_fetch_step_series(conn, run_id, f"viol_rate/A/d{d}") for d in DEPTHS]
    conn.close()

    common_steps = sorted(set(main).intersection(set(full), *(set(v) for v in viol_by_depth)))
    trace = []
    for step in common_steps:
        trace.append(
            {
                "step": step,
                "main_loss": main[step],
                "full_loss": full[step],
                "avg_viol": float(np.mean([v[step] for v in viol_by_depth])),
            }
        )
    return trace


def _plot_lambda_curve(curve: list[dict], output_dir: Path, loss_key: str, out_prefix: str, y_label: str) -> None:
    lambdas = np.array([r["lambda_reg"] for r in curve], dtype=float)
    loss_vals = np.array([r[loss_key] for r in curve], dtype=float)
    avg_viol_pct = 100.0 * np.array([r["avg_viol"] for r in curve], dtype=float)

    left_color = "C0"
    right_color = "C1"
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    ax1.plot(lambdas, loss_vals, "o-", color=left_color)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\lambda_{\mathrm{reg}}$")
    ax1.set_ylabel(y_label, color=left_color)

    ax2 = ax1.twinx()
    ax2.plot(
        lambdas,
        avg_viol_pct,
        "s-",
        color=right_color,
        alpha=0.85,
    )
    ax2.set_ylabel("Violation Percentage (%)", color=right_color)

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{out_prefix}.{ext}", dpi=300)
    plt.close(fig)


def _plot_lambda_violation_only(curve: list[dict], output_dir: Path, out_prefix: str) -> None:
    lambdas = np.array([r["lambda_reg"] for r in curve], dtype=float)
    avg_viol_pct = 100.0 * np.array([r["avg_viol"] for r in curve], dtype=float)

    color = "C1"
    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    ax.plot(lambdas, avg_viol_pct, "s-", color=color, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{reg}}$")
    ax.set_ylabel("Violation Percentage (%)", color=color)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{out_prefix}.{ext}", dpi=300)
    plt.close(fig)


def _plot_lambda_trace(
    trace: list[dict],
    output_dir: Path,
    loss_key: str,
    out_prefix: str,
    y_label: str,
    trace_xmin: int | None = None,
    trace_xmax: int | None = None,
) -> None:
    steps = np.array([r["step"] for r in trace], dtype=int)
    loss_vals = np.array([r[loss_key] for r in trace], dtype=float)
    avg_viol_pct = 100.0 * np.array([r["avg_viol"] for r in trace], dtype=float)

    left_color = "C0"
    right_color = "C1"
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    ax1.plot(steps, loss_vals, "-", color=left_color)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel(y_label, color=left_color)
    if trace_xmin is not None or trace_xmax is not None:
        lo = steps.min() if trace_xmin is None else trace_xmin
        hi = steps.max() if trace_xmax is None else trace_xmax
        ax1.set_xlim(lo, hi)

    ax2 = ax1.twinx()
    ax2.plot(
        steps,
        avg_viol_pct,
        "-",
        color=right_color,
        alpha=0.85,
    )
    ax2.set_ylabel(r"$\mathcal{HAS}$-Viol (%)", color=right_color)

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{out_prefix}.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_paths",
        nargs="+",
        required=True,
        help="One or more MLflow SQLite DB paths",
    )
    parser.add_argument(
        "--trace_db_paths",
        nargs="+",
        default=None,
        help="Optional DB paths used only for the lambda trace plot (defaults to --db_paths).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="files/autosweep/40",
        help="Directory for plot outputs",
    )
    parser.add_argument(
        "--trace_lambda",
        type=float,
        default=1e-3,
        help="Lambda value for iteration trace plot",
    )
    parser.add_argument(
        "--trace_output_tag",
        type=str,
        default="",
        help="Optional suffix appended to trace output filenames (e.g., _sgd).",
    )
    parser.add_argument(
        "--trace_xmin",
        type=int,
        default=None,
        help="Optional minimum x-axis step for trace plots.",
    )
    parser.add_argument(
        "--trace_xmax",
        type=int,
        default=None,
        help="Optional maximum x-axis step for trace plots.",
    )
    args = parser.parse_args()

    db_paths = [Path(p) for p in args.db_paths]
    trace_db_paths = [Path(p) for p in (args.trace_db_paths if args.trace_db_paths else args.db_paths)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    curve = _aggregate_lambda_curve(db_paths)
    if not curve:
        raise RuntimeError("No valid runs found for lambda curve plot.")
    _plot_lambda_curve(
        curve,
        out_dir,
        loss_key="main_loss",
        out_prefix="lambda_vs_mainloss_violation",
        y_label="Main Loss",
    )
    _plot_lambda_curve(
        curve,
        out_dir,
        loss_key="full_loss",
        out_prefix="lambda_vs_full_loss_violation",
        y_label="Full loss (with regularization)",
    )
    _plot_lambda_violation_only(curve, out_dir, out_prefix="lambda_vs_violation_only")

    trace_db, trace_run = _find_lambda_run(trace_db_paths, args.trace_lambda)
    trace = _fetch_trace(trace_db, trace_run)
    if not trace:
        raise RuntimeError(f"No trace points found for lambda={args.trace_lambda:g}.")
    _plot_lambda_trace(
        trace,
        out_dir,
        loss_key="main_loss",
        out_prefix=f"lambda_1e3_trace_mainloss_violation{args.trace_output_tag}",
        y_label="Main Loss",
        trace_xmin=args.trace_xmin,
        trace_xmax=args.trace_xmax,
    )
    _plot_lambda_trace(
        trace,
        out_dir,
        loss_key="full_loss",
        out_prefix=f"lambda_1e3_trace_full_loss_violation{args.trace_output_tag}",
        y_label="Full loss (with regularization)",
        trace_xmin=args.trace_xmin,
        trace_xmax=args.trace_xmax,
    )

    print(f"Saved: {out_dir / 'lambda_vs_mainloss_violation.png'}")
    print(f"Saved: {out_dir / 'lambda_vs_mainloss_violation.pdf'}")
    print(f"Saved: {out_dir / 'lambda_1e3_trace_mainloss_violation.png'}")
    print(f"Saved: {out_dir / 'lambda_1e3_trace_mainloss_violation.pdf'}")
    print(f"Saved: {out_dir / 'lambda_vs_full_loss_violation.png'}")
    print(f"Saved: {out_dir / 'lambda_vs_full_loss_violation.pdf'}")
    print(f"Saved: {out_dir / 'lambda_vs_violation_only.png'}")
    print(f"Saved: {out_dir / 'lambda_vs_violation_only.pdf'}")
    print(f"Saved: {out_dir / f'lambda_1e3_trace_mainloss_violation{args.trace_output_tag}.png'}")
    print(f"Saved: {out_dir / f'lambda_1e3_trace_mainloss_violation{args.trace_output_tag}.pdf'}")
    print(f"Saved: {out_dir / f'lambda_1e3_trace_full_loss_violation{args.trace_output_tag}.png'}")
    print(f"Saved: {out_dir / f'lambda_1e3_trace_full_loss_violation{args.trace_output_tag}.pdf'}")
    print(f"Trace run: {trace_run} from {trace_db}")
    print(f"Trace points: {len(trace)}")


if __name__ == "__main__":
    main()
