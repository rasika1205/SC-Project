#!/usr/bin/env python3
"""Reproduce the SFpark LSTM / LSTM+ANFIS / LSTM+ANFIS+PSO evaluation.

This script evaluates SAVED predictions and saved candidate scores. It does not
retrain LSTM, ANFIS, or PSO. It recalculates forecast metrics, recommendation
metrics, a target-time-cluster bootstrap, selected-block CSVs, and paper figures.

Required inputs by default:
  inputs/anfis_all_predictions_with_labels.csv
  inputs/pso_test_candidate_scores.csv
  inputs/pso_optimized_weights.csv (optional; only used for the weight plot)

You can override the paths with command-line arguments. Use the saved results
from the same model run that generated the paper tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PRICE_COST_COEFFICIENT = 0.15
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261028
MIN_CANDIDATES = 2


def as_bool(s: pd.Series) -> pd.Series:
    """Read booleans safely if a CSV column is bool or string-valued."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        raise ValueError("No valid actual/predicted pairs to calculate metrics.")
    return {
        "N": int(len(actual)),
        "MAE_fraction": float(mean_absolute_error(actual, predicted)),
        "MAE_percentage_points": float(mean_absolute_error(actual, predicted) * 100),
        "RMSE_fraction": float(np.sqrt(mean_squared_error(actual, predicted))),
        "RMSE_percentage_points": float(np.sqrt(mean_squared_error(actual, predicted)) * 100),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "Mean_error_bias_percentage_points": float(np.mean(predicted - actual) * 100),
        "Mean_predicted_availability_pct": float(np.mean(predicted) * 100),
        "Mean_actual_availability_pct": float(np.mean(actual) * 100),
    }


def read_and_validate_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "BLOCK_ID", "TARGET_TIME", "PREDICTED_GMP_AVAILABILITY",
        "ANFIS_CALIBRATED_AVAILABILITY", "ACTUAL_TARGET_AVAILABILITY", "DATA_SPLIT",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Forecast/prediction file {path} is missing columns: {missing}")
    df["TARGET_TIME"] = pd.to_datetime(df["TARGET_TIME"], errors="coerce")
    if "HISTORY_QUALITY_OK" in df:
        df = df.loc[as_bool(df["HISTORY_QUALITY_OK"])].copy()
    if "ACTUAL_TARGET_QUALITY_OK" in df:
        df = df.loc[as_bool(df["ACTUAL_TARGET_QUALITY_OK"])].copy()
    df = df.loc[df["TARGET_TIME"].notna()].copy()
    for col in ["PREDICTED_GMP_AVAILABILITY", "ANFIS_CALIBRATED_AVAILABILITY", "ACTUAL_TARGET_AVAILABILITY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PREDICTED_GMP_AVAILABILITY", "ANFIS_CALIBRATED_AVAILABILITY", "ACTUAL_TARGET_AVAILABILITY"])
    return df


def calculate_forecast_metrics(predictions: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = predictions.loc[predictions["DATA_SPLIT"].astype(str).str.lower().eq("test")].copy()
    if test.empty:
        raise ValueError("No DATA_SPLIT == 'test' rows found in prediction file.")
    actual = test["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float)
    raw = test["PREDICTED_GMP_AVAILABILITY"].to_numpy(float)
    calibrated = test["ANFIS_CALIBRATED_AVAILABILITY"].to_numpy(float)
    rows = []
    for name, pred in [("LSTM (raw)", raw), ("LSTM + ANFIS (calibrated)", calibrated)]:
        row = {"Model": name, **regression_metrics(actual, pred)}
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "forecast_metrics.csv", index=False)

    # Paired target-time cluster bootstrap. Every row sharing a TARGET_TIME is
    # kept together during a resample. Seed makes this script reproducible.
    groups = []
    for _, g in test.groupby("TARGET_TIME", sort=True):
        groups.append((
            g["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float),
            g["PREDICTED_GMP_AVAILABILITY"].to_numpy(float),
            g["ANFIS_CALIBRATED_AVAILABILITY"].to_numpy(float),
        ))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {"MAE_reduction_pp": [], "RMSE_reduction_pp": [], "R2_difference": []}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_group_indices = rng.integers(0, len(groups), size=len(groups))
        a = np.concatenate([groups[j][0] for j in sampled_group_indices])
        p_raw = np.concatenate([groups[j][1] for j in sampled_group_indices])
        p_anfis = np.concatenate([groups[j][2] for j in sampled_group_indices])
        mae_red = (mean_absolute_error(a, p_raw) - mean_absolute_error(a, p_anfis)) * 100
        rmse_raw = np.sqrt(mean_squared_error(a, p_raw))
        rmse_anfis = np.sqrt(mean_squared_error(a, p_anfis))
        r2_diff = r2_score(a, p_anfis) - r2_score(a, p_raw)
        boot["MAE_reduction_pp"].append(mae_red)
        boot["RMSE_reduction_pp"].append((rmse_raw - rmse_anfis) * 100)
        boot["R2_difference"].append(r2_diff)
    point = {
        "MAE_reduction_pp": (mean_absolute_error(actual, raw) - mean_absolute_error(actual, calibrated)) * 100,
        "RMSE_reduction_pp": (np.sqrt(mean_squared_error(actual, raw)) - np.sqrt(mean_squared_error(actual, calibrated))) * 100,
        "R2_difference": r2_score(actual, calibrated) - r2_score(actual, raw),
    }
    bootstrap_rows = []
    for metric, vals in boot.items():
        bootstrap_rows.append({
            "Metric difference (positive favours ANFIS)": metric,
            "Point estimate": float(point[metric]),
            "Bootstrap 2.5%": float(np.percentile(vals, 2.5)),
            "Bootstrap 97.5%": float(np.percentile(vals, 97.5)),
            "Resampling unit": "TARGET_TIME group",
            "Replicates": BOOTSTRAP_REPLICATES,
            "Seed": BOOTSTRAP_SEED,
        })
    ci = pd.DataFrame(bootstrap_rows)
    ci.to_csv(outdir / "forecast_paired_cluster_bootstrap_ci.csv", index=False)
    test.to_csv(outdir / "forecast_test_observations.csv", index=False)
    return metrics, ci


def read_candidate_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "BLOCK_ID", "TARGET_TIME", "PREDICTED_GMP_AVAILABILITY",
        "ANFIS_CALIBRATED_AVAILABILITY", "PSO_SUITABILITY_SCORE",
        "ACTUAL_TARGET_AVAILABILITY", "RATE", "RATE_NORM", "DATA_SPLIT",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Candidate score file {path} is missing columns: {missing}")
    df["TARGET_TIME"] = pd.to_datetime(df["TARGET_TIME"], errors="coerce")
    df["BLOCK_ID"] = pd.to_numeric(df["BLOCK_ID"], errors="coerce")
    num_cols = ["PREDICTED_GMP_AVAILABILITY", "ANFIS_CALIBRATED_AVAILABILITY", "PSO_SUITABILITY_SCORE", "ACTUAL_TARGET_AVAILABILITY", "RATE", "RATE_NORM"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[df["DATA_SPLIT"].astype(str).str.lower().eq("test")].copy()
    if "HISTORY_QUALITY_OK" in df:
        df = df.loc[as_bool(df["HISTORY_QUALITY_OK"])].copy()
    if "ACTUAL_TARGET_QUALITY_OK" in df:
        df = df.loc[as_bool(df["ACTUAL_TARGET_QUALITY_OK"])].copy()
    df = df.dropna(subset=["TARGET_TIME", "BLOCK_ID", *num_cols])
    return df


def select_policy_rows(candidates: pd.DataFrame, policy: str, score_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one block per target time; retain only times with >=2 candidates."""
    counts = candidates.groupby("TARGET_TIME")["BLOCK_ID"].count()
    eligible_times = counts[counts >= MIN_CANDIDATES].index
    part = candidates.loc[candidates["TARGET_TIME"].isin(eligible_times)].copy()
    part = part.sort_values(["TARGET_TIME", "BLOCK_ID"], kind="mergesort")
    selected_idx = part.groupby("TARGET_TIME", sort=True)[score_col].idxmax()
    chosen = part.loc[selected_idx].copy().sort_values("TARGET_TIME").reset_index(drop=True)
    chosen["Policy"] = policy
    chosen["DECISION_COST"] = (1.0 - chosen["ACTUAL_TARGET_AVAILABILITY"]) + PRICE_COST_COEFFICIENT * chosen["RATE_NORM"]
    oracle_cost = part.assign(
        _candidate_cost=(1.0 - part["ACTUAL_TARGET_AVAILABILITY"]) + PRICE_COST_COEFFICIENT * part["RATE_NORM"]
    ).groupby("TARGET_TIME")["_candidate_cost"].min()
    chosen["ORACLE_COST"] = chosen["TARGET_TIME"].map(oracle_cost)
    chosen["REGRET"] = chosen["DECISION_COST"] - chosen["ORACLE_COST"]
    chosen["LOW_AVAILABILITY_PROXY"] = chosen["ACTUAL_TARGET_AVAILABILITY"] < 0.05
    return part, chosen


def summarize_policy(chosen: pd.DataFrame) -> dict:
    actual = chosen["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float)
    normalized_rate = chosen["RATE_NORM"].to_numpy(float)
    cost = chosen["DECISION_COST"].to_numpy(float)
    predicted_col = "PREDICTED_GMP_AVAILABILITY" if chosen["Policy"].iloc[0] == "LSTM" else "ANFIS_CALIBRATED_AVAILABILITY"
    return {
        "Policy": str(chosen["Policy"].iloc[0]),
        "Decision timestamps": int(chosen["TARGET_TIME"].nunique()),
        "Mean actual target availability (%)": float(actual.mean() * 100),
        "Mean predicted availability (%)": float(chosen[predicted_col].mean() * 100),
        "Mean RATE (dataset units)": float(chosen["RATE"].mean()),
        "Mean normalized RATE": float(normalized_rate.mean()),
        "Mean base decision cost (lower is better)": float(cost.mean()),
        "Mean regret vs hindsight oracle": float(chosen["REGRET"].mean()),
        "Target availability <5% (%)": float((actual < 0.05).mean() * 100),
        "Unique blocks selected": int(chosen["BLOCK_ID"].nunique()),
    }


def calculate_decision_metrics(candidates: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    policies = [
        ("LSTM", "PREDICTED_GMP_AVAILABILITY"),
        ("LSTM + ANFIS", "ANFIS_CALIBRATED_AVAILABILITY"),
        ("LSTM + ANFIS + PSO", "PSO_SUITABILITY_SCORE"),
    ]
    policy_recommendations = {}
    metric_rows = []
    for policy, col in policies:
        _, chosen = select_policy_rows(candidates, policy, col)
        policy_recommendations[policy] = chosen
        chosen.to_csv(outdir / (policy.lower().replace(" ", "_").replace("+", "and").replace("-", "_") + "_selected_blocks.csv"), index=False)
        metric_rows.append(summarize_policy(chosen))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(outdir / "decision_policy_metrics.csv", index=False)

    # One row per candidate test block for auditing/review.
    all_policy = []
    for policy, chosen in policy_recommendations.items():
        temp = chosen.copy()
        temp["POLICY"] = policy
        all_policy.append(temp)
    selected = pd.concat(all_policy, ignore_index=True)
    selected.to_csv(outdir / "all_policy_selected_blocks.csv", index=False)
    return metrics, policy_recommendations, selected


def read_weights(weights_path: Path | None) -> pd.DataFrame:
    if weights_path and weights_path.exists():
        w = pd.read_csv(weights_path)
        if {"Component", "Weight"}.issubset(w.columns):
            return w
    warnings.warn("PSO weight file unavailable; weights plot will be skipped.")
    return pd.DataFrame(columns=["Component", "Weight"])


def make_figures(test: pd.DataFrame, forecast_metrics: pd.DataFrame, decision_metrics: pd.DataFrame,
                 policy_recommendations: dict[str, pd.DataFrame], weights: pd.DataFrame, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    # Figure 1: Forecast errors (percentage points).
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x - width/2, forecast_metrics["MAE_percentage_points"], width, label="MAE")
    ax.bar(x + width/2, forecast_metrics["RMSE_percentage_points"], width, label="RMSE")
    ax.set_xticks(x, ["LSTM", "LSTM + ANFIS"])
    ax.set_ylabel("Error (percentage points)")
    ax.set_title("Forecast error comparison on held-out test data")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_forecast_error_comparison.png", dpi=300)
    plt.close(fig)

    # Figure 2: Actual vs predicted availability.
    actual = test["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float) * 100
    raw = test["PREDICTED_GMP_AVAILABILITY"].to_numpy(float) * 100
    calibrated = test["ANFIS_CALIBRATED_AVAILABILITY"].to_numpy(float) * 100
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(actual, raw, s=12, alpha=0.32, label="LSTM")
    ax.scatter(actual, calibrated, s=12, alpha=0.32, label="LSTM + ANFIS")
    ax.plot([0, 100], [0, 100], linestyle="--", label="Ideal prediction")
    ax.set_xlabel("Actual availability (%)")
    ax.set_ylabel("Predicted availability (%)")
    ax.set_title("Predicted vs actual availability")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_actual_vs_predicted.png", dpi=300)
    plt.close(fig)

    # Figure 3: Mean selected-block actual availability.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(decision_metrics["Policy"], decision_metrics["Mean actual target availability (%)"])
    ax.set_ylabel("Mean actual target availability (%)")
    ax.set_title("Observed availability of recommended blocks")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_recommendation_outcomes.png", dpi=300)
    plt.close(fig)

    # Figure 4: Cost and regret.
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(decision_metrics))
    width = 0.36
    bars1 = ax.bar(x - width/2, decision_metrics["Mean base decision cost (lower is better)"], width, label="Mean decision cost")
    bars2 = ax.bar(x + width/2, decision_metrics["Mean regret vs hindsight oracle"], width, label="Mean regret")
    ax.set_xticks(x, decision_metrics["Policy"])
    ax.set_ylabel("Cost (normalized units)")
    ax.set_title("Decision cost and hindsight regret")
    ax.tick_params(axis="x", rotation=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_decision_cost_and_regret.png", dpi=300)
    plt.close(fig)

    # Figure 5: Saved original PSO weights.
    if not weights.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        bars = ax.bar(weights["Component"], weights["Weight"])
        ax.set_ylabel("Optimized weight")
        ax.set_title("Original PSO optimized decision weights")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        fig.tight_layout()
        fig.savefig(figdir / "fig5_original_pso_weights.png", dpi=300)
        plt.close(fig)

    # Figure 6: absolute error distributions.
    raw_err = np.abs(test["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float) - test["PREDICTED_GMP_AVAILABILITY"].to_numpy(float)) * 100
    anfis_err = np.abs(test["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float) - test["ANFIS_CALIBRATED_AVAILABILITY"].to_numpy(float)) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    # Matplotlib 3.9 renamed boxplot labels -> tick_labels.
    box_label_kw = {"tick_labels": ["LSTM", "LSTM + ANFIS"]} if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 9) else {"labels": ["LSTM", "LSTM + ANFIS"]}
    ax.boxplot([raw_err, anfis_err], showfliers=False, **box_label_kw)
    ax.set_ylabel("Absolute error (percentage points)")
    ax.set_title("Distribution of absolute forecast errors")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "fig6_absolute_error_distribution.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--predictions", type=Path, default=base / "inputs" / "anfis_all_predictions_with_labels.csv",
                        help="CSV containing actual labels and raw/ANFIS predictions")
    parser.add_argument("--candidates", type=Path, default=base / "inputs" / "pso_test_candidate_scores.csv",
                        help="Candidate-level test CSV, including PSO_SUITABILITY_SCORE")
    parser.add_argument("--weights", type=Path, default=base / "inputs" / "pso_optimized_weights.csv",
                        help="Optional CSV with Component,Weight columns")
    parser.add_argument("--output-dir", type=Path, default=base / "evaluation_outputs",
                        help="Directory where tables and figures will be written")
    args = parser.parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"

    print(f"Reading prediction file: {args.predictions}")
    predictions = read_and_validate_forecasts(args.predictions)
    test = predictions.loc[predictions["DATA_SPLIT"].astype(str).str.lower().eq("test")].copy()
    forecast_metrics, bootstrap_ci = calculate_forecast_metrics(predictions, outdir)

    print(f"Reading candidate scores: {args.candidates}")
    candidates = read_candidate_scores(args.candidates)
    decision_metrics, policy_recommendations, selected = calculate_decision_metrics(candidates, outdir)
    weights = read_weights(args.weights)
    if not weights.empty:
        weights.to_csv(outdir / "pso_optimized_weights_used.csv", index=False)
    make_figures(test, forecast_metrics, decision_metrics, policy_recommendations, weights, figdir)

    summary = {
        "forecast_observations_test": int(len(test)),
        "forecast_target_timestamps_test": int(test["TARGET_TIME"].nunique()),
        "recommendation_timestamps_with_2plus_candidates": int(decision_metrics["Decision timestamps"].min()),
        "price_cost_coefficient": PRICE_COST_COEFFICIENT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "notes": [
            "This is an evaluation/reproduction script; it does not retrain LSTM, ANFIS, or PSO.",
            "Forecast metrics are computed from saved raw and ANFIS-calibrated predictions.",
            "Recommendation metrics are computed by selecting one block per target time from the same candidate set.",
            "The hindsight oracle uses future actual availability and is diagnostic only, not a deployable policy.",
            "This dataset does not directly measure driver-level parking success, road congestion, route distance, or search time.",
        ],
    }
    (outdir / "evaluation_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Forecast metrics ===")
    print(forecast_metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== Paired target-time cluster bootstrap ===")
    print(bootstrap_ci.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== Recommendation metrics ===")
    print(decision_metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nOutputs written to: {outdir}")
    print(f"Figures written to: {figdir}")


if __name__ == "__main__":
    main()
