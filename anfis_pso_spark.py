"""ANFIS calibration + PSO parking-block selection for SFpark.

Expected input files in the same directory as this script:
  1) LSTM_fuzzy_input.csv.xls
     Despite its extension, this file is CSV text.
  2) sfpark_feature_engineered_model_ready_v2_2_lookback3.csv

Method:
  - Join LSTM forecasts to historical features at LAST_HISTORY_TIME.
  - Join actual GMP availability at TARGET_TIME as the supervised target.
  - Train a small TSK-style ANFIS to calibrate LSTM availability forecasts.
  - Use PSO to learn weights for expected availability, prediction reliability,
    and affordability; tune weights on a separate chronological validation period.
  - Evaluate once on the final chronological test period.

Important:
  - The current files do not contain driver distance, road congestion, or actual
    search time. This version therefore does not fabricate/use those attributes.
  - Check that the LSTM forecasts are genuinely out-of-sample/chronological.
    If the LSTM was trained using target-period data, downstream results leak.
"""
from __future__ import annotations

from pathlib import Path
from itertools import product
import copy
import json
import random
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# EDIT THESE PATHS if your files are not in the same folder as this script.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FORECAST_PATH = BASE_DIR / "LSTM_fuzzy_input.csv.xls"
FEATURE_PATH = BASE_DIR / "sfpark_feature_engineered_model_ready_v2_2_lookback3.csv"
OUTPUT_DIR = BASE_DIR / "anfis_pso_outputs"

# Chronological split by unique TARGET_TIME, so a timestamp's candidate blocks
# cannot be split across train/validation/test.
TRAIN_FRACTION = 0.60
ANFIS_VALIDATION_FRACTION = 0.10
PSO_VALIDATION_FRACTION = 0.15
# The remainder (approximately 15%) is held out for final testing.

# ANFIS: 2 Gaussian membership functions per input -> 2^5 = 32 TSK rules.
N_MFS = 2
MAX_EPOCHS = 180
PATIENCE = 18
BATCH_SIZE = 512
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

# PSO configuration.
PSO_PARTICLES = 36
PSO_ITERATIONS = 70
PSO_INERTIA = 0.72
PSO_COGNITIVE = 1.45
PSO_SOCIAL = 1.45
# Fixed utility trade-off used when validating a recommendation policy.
# Run a sensitivity analysis for e.g. 0.05, 0.10, 0.15, 0.20 in the paper.
PRICE_COST_COEFFICIENT = 0.15

# These are available in the supplied forecast/history files.
ANFIS_INPUTS = [
    "PREDICTED_GMP_AVAILABILITY",
    "PREDICTION_UNCERTAINTY",
    "CURRENT_GMP_AVAILABILITY",
    "GMP_OCC_CHANGE_1",
    "RATE",
]


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_columns(df: pd.DataFrame, columns: list[str], file_label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def load_and_join_data() -> pd.DataFrame:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}\n"
            "Edit FORECAST_PATH near the top of this script."
        )
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature/label file not found: {FEATURE_PATH}\n"
            "Edit FEATURE_PATH near the top of this script."
        )

    # This forecast file is CSV text even though its filename ends in .csv.xls.
    forecast = pd.read_csv(FORECAST_PATH, low_memory=False)
    features = pd.read_csv(FEATURE_PATH, low_memory=False)

    require_columns(
        forecast,
        ["BLOCK_ID", "TARGET_TIME", "LAST_HISTORY_TIME", "RATE",
         "PREDICTED_GMP_AVAILABILITY", "PREDICTION_UNCERTAINTY"],
        "LSTM forecast file",
    )
    require_columns(
        features,
        ["BLOCK_ID", "START_TIME_DT", "GMP_AVAILABILITY", "GMP_QUALITY_OK",
         "GMP_OCC_CHANGE_1", "GMP_ROLLING_MEAN_3", "GMP_UNKNOWN_RATE",
         "EFFECTIVE_MONITORED_SPACES", "HOUR_SIN", "HOUR_COS",
         "DOW_SIN", "DOW_COS", "WEEKEND", "RATE_CHANGE"],
        "feature-engineered file",
    )

    for frame, cols in [
        (forecast, ["TARGET_TIME", "LAST_HISTORY_TIME"]),
        (features, ["START_TIME_DT"]),
    ]:
        for col in cols:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    forecast["BLOCK_ID"] = pd.to_numeric(forecast["BLOCK_ID"], errors="coerce")
    features["BLOCK_ID"] = pd.to_numeric(features["BLOCK_ID"], errors="coerce")
    forecast["RATE"] = pd.to_numeric(forecast["RATE"], errors="coerce")

    # Features available at prediction time; don't use any feature from TARGET_TIME
    # as an ANFIS input except a parking rate that is assumed to be a known tariff.
    history_cols = [
        "BLOCK_ID", "START_TIME_DT", "GMP_AVAILABILITY", "GMP_OCC_CHANGE_1",
        "GMP_ROLLING_MEAN_3", "GMP_UNKNOWN_RATE", "EFFECTIVE_MONITORED_SPACES",
        "HOUR_SIN", "HOUR_COS", "DOW_SIN", "DOW_COS", "WEEKEND",
        "RATE_CHANGE", "GMP_QUALITY_OK",
    ]
    history = features[history_cols].copy().rename(
        columns={
            "START_TIME_DT": "LAST_HISTORY_TIME",
            "GMP_AVAILABILITY": "CURRENT_GMP_AVAILABILITY",
            "GMP_QUALITY_OK": "HISTORY_QUALITY_OK",
        }
    )
    if history.duplicated(["BLOCK_ID", "LAST_HISTORY_TIME"]).any():
        raise ValueError("Feature file has duplicate BLOCK_ID + START_TIME_DT keys.")

    data = forecast.merge(
        history, on=["BLOCK_ID", "LAST_HISTORY_TIME"], how="left",
        validate="many_to_one", indicator="_history_merge"
    )
    missing_history = int((data["_history_merge"] != "both").sum())
    data = data.drop(columns=["_history_merge"])

    # True observed availability at the forecast target time is the supervised label.
    target = features[["BLOCK_ID", "START_TIME_DT", "GMP_AVAILABILITY", "GMP_QUALITY_OK"]].copy()
    target = target.rename(
        columns={
            "START_TIME_DT": "TARGET_TIME",
            "GMP_AVAILABILITY": "ACTUAL_TARGET_AVAILABILITY",
            "GMP_QUALITY_OK": "ACTUAL_TARGET_QUALITY_OK",
        }
    )
    if target.duplicated(["BLOCK_ID", "TARGET_TIME"]).any():
        raise ValueError("Feature file has duplicate BLOCK_ID + START_TIME_DT target keys.")
    data = data.merge(
        target, on=["BLOCK_ID", "TARGET_TIME"], how="left",
        validate="many_to_one", indicator="_target_merge"
    )
    missing_target = int((data["_target_merge"] != "both").sum())
    data = data.drop(columns=["_target_merge"])

    # Keep only rows with known, quality-checked observed target labels and inputs.
    data["ACTUAL_TARGET_AVAILABILITY"] = pd.to_numeric(
        data["ACTUAL_TARGET_AVAILABILITY"], errors="coerce"
    )
    data["PREDICTED_GMP_AVAILABILITY"] = pd.to_numeric(
        data["PREDICTED_GMP_AVAILABILITY"], errors="coerce"
    )
    data["PREDICTION_UNCERTAINTY"] = pd.to_numeric(
        data["PREDICTION_UNCERTAINTY"], errors="coerce"
    )
    data["CURRENT_GMP_AVAILABILITY"] = pd.to_numeric(
        data["CURRENT_GMP_AVAILABILITY"], errors="coerce"
    )
    data["GMP_OCC_CHANGE_1"] = pd.to_numeric(data["GMP_OCC_CHANGE_1"], errors="coerce")
    data["RATE"] = pd.to_numeric(data["RATE"], errors="coerce")

    before = len(data)
    valid = (
        data["TARGET_TIME"].notna()
        & data["LAST_HISTORY_TIME"].notna()
        & data["BLOCK_ID"].notna()
        & data["ACTUAL_TARGET_QUALITY_OK"].fillna(False).astype(bool)
        & data["HISTORY_QUALITY_OK"].fillna(False).astype(bool)
        & data[ANFIS_INPUTS].notna().all(axis=1)
        & data["ACTUAL_TARGET_AVAILABILITY"].between(0.0, 1.0)
        & data["PREDICTED_GMP_AVAILABILITY"].between(0.0, 1.0)
        & (data["PREDICTION_UNCERTAINTY"] >= 0.0)
    )
    data = data.loc[valid].copy()
    data["BLOCK_ID"] = data["BLOCK_ID"].astype(int)
    data = data.sort_values(["TARGET_TIME", "BLOCK_ID"]).reset_index(drop=True)

    print("\n--- Data join / quality checks ---")
    print(f"Forecast rows loaded: {len(forecast):,}")
    print(f"Missing history matches: {missing_history:,}")
    print(f"Missing actual target matches: {missing_target:,}")
    print(f"Rows retained after quality/input filtering: {len(data):,} of {before:,}")
    print(f"Unique blocks: {data['BLOCK_ID'].nunique():,}")
    print(f"Unique target timestamps: {data['TARGET_TIME'].nunique():,}")
    print(f"Target time range: {data['TARGET_TIME'].min()} to {data['TARGET_TIME'].max()}")

    if len(data) < 500:
        raise ValueError("Too few joined rows remain for credible ANFIS/PSO fitting.")
    if missing_target / max(len(forecast), 1) > 0.05:
        print("WARNING: More than 5% of forecasts could not be joined to observed target labels.")
    return data


def assign_chronological_splits(data: pd.DataFrame) -> pd.DataFrame:
    timestamps = np.array(sorted(data["TARGET_TIME"].dropna().unique()))
    n_times = len(timestamps)
    if n_times < 30:
        raise ValueError("Need at least 30 distinct target timestamps for temporal splits.")

    i_train = max(1, int(n_times * TRAIN_FRACTION))
    i_anfis_val = max(i_train + 1, int(n_times * (TRAIN_FRACTION + ANFIS_VALIDATION_FRACTION)))
    i_pso_val = max(i_anfis_val + 1, int(n_times * (TRAIN_FRACTION + ANFIS_VALIDATION_FRACTION + PSO_VALIDATION_FRACTION)))
    i_pso_val = min(i_pso_val, n_times - 1)

    train_boundary = pd.Timestamp(timestamps[i_train])
    anfis_val_boundary = pd.Timestamp(timestamps[i_anfis_val])
    pso_val_boundary = pd.Timestamp(timestamps[i_pso_val])

    result = data.copy()
    result["DATA_SPLIT"] = np.select(
        [
            result["TARGET_TIME"] < train_boundary,
            result["TARGET_TIME"] < anfis_val_boundary,
            result["TARGET_TIME"] < pso_val_boundary,
        ],
        ["train", "anfis_validation", "pso_validation"],
        default="test",
    )

    print("\n--- Chronological split by TARGET_TIME ---")
    for name, group in result.groupby("DATA_SPLIT", sort=False):
        print(f"{name:18s}: {len(group):6,} rows; {group['TARGET_TIME'].nunique():4,} timestamps; "
              f"{group['TARGET_TIME'].min()} -> {group['TARGET_TIME'].max()}")
    return result


class TSKANFIS(nn.Module):
    """Compact differentiable first-order Takagi-Sugeno ANFIS.

    Each input has `n_mfs` Gaussian memberships. A grid of rules is formed.
    Each rule has a linear consequent; normalized rule strengths aggregate output.
    A sigmoid keeps the predicted availability in [0, 1].
    """
    def __init__(self, n_inputs: int, n_mfs: int = 2, target_mean: float = 0.35):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        combinations = list(product(range(n_mfs), repeat=n_inputs))
        self.register_buffer("rule_index", torch.tensor(combinations, dtype=torch.long))
        self.n_rules = len(combinations)

        init_centers = torch.linspace(0.25, 0.75, n_mfs).repeat(n_inputs, 1)
        self.centers = nn.Parameter(init_centers + 0.015 * torch.randn_like(init_centers))
        # softplus(raw_sigma) initially approximately 0.30
        raw_sigma_init = np.log(np.exp(0.30) - 1.0)
        self.raw_sigmas = nn.Parameter(torch.full((n_inputs, n_mfs), float(raw_sigma_init)))
        self.consequents = nn.Parameter(torch.zeros(self.n_rules, n_inputs + 1))
        target_mean = float(np.clip(target_mean, 0.02, 0.98))
        initial_bias = np.log(target_mean / (1.0 - target_mean))
        with torch.no_grad():
            self.consequents[:, -1] = initial_bias
            self.consequents[:, :-1].normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_inputs]; memberships: [batch, n_inputs, n_mfs]
        sigmas = F.softplus(self.raw_sigmas) + 1e-3
        diff = (x.unsqueeze(-1) - self.centers.unsqueeze(0)) / sigmas.unsqueeze(0)
        memberships = torch.exp(-0.5 * diff.pow(2)).clamp_min(1e-8)

        # For each rule, multiply the corresponding membership of every input.
        rule_memberships = []
        for input_idx in range(self.n_inputs):
            rule_memberships.append(memberships[:, input_idx, self.rule_index[:, input_idx]])
        firing = torch.ones_like(rule_memberships[0])
        for vals in rule_memberships:
            firing = firing * vals
        normalized_firing = firing / firing.sum(dim=1, keepdim=True).clamp_min(1e-8)

        x_aug = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)], dim=1)
        rule_outputs = x_aug @ self.consequents.T
        raw_output = torch.sum(normalized_firing * rule_outputs, dim=1, keepdim=True)
        return torch.sigmoid(raw_output).squeeze(1)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return {
        "n": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": rmse,
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def train_anfis(data: pd.DataFrame):
    train = data.loc[data["DATA_SPLIT"] == "train"].copy()
    val = data.loc[data["DATA_SPLIT"] == "anfis_validation"].copy()
    if len(train) < 100 or len(val) < 30:
        raise ValueError("Train/ANFIS-validation split too small after filtering.")

    x_scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
    X_train = x_scaler.fit_transform(train[ANFIS_INPUTS]).astype(np.float32)
    X_val = x_scaler.transform(val[ANFIS_INPUTS]).astype(np.float32)
    y_train = train["ACTUAL_TARGET_AVAILABILITY"].to_numpy(dtype=np.float32)
    y_val = val["ACTUAL_TARGET_AVAILABILITY"].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSKANFIS(n_inputs=len(ANFIS_INPUTS), n_mfs=N_MFS, target_mean=float(y_train.mean())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    print(f"\n--- Training ANFIS on {device} ---")
    print(f"Inputs: {ANFIS_INPUTS}")
    print(f"Membership functions/input: {N_MFS}; TSK rules: {N_MFS ** len(ANFIS_INPUTS)}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(X_train_t.shape[0], device=device)
        train_losses = []
        for start in range(0, X_train_t.shape[0], BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            xb, yb = X_train_t[indices], y_train_t[indices]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(criterion(val_pred, y_val_t).cpu())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train MSE {np.mean(train_losses):.6f} | val MSE {val_loss:.6f}")
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"Best validation MSE: {best_val:.6f} at epoch {best_epoch}")

    X_all = x_scaler.transform(data[ANFIS_INPUTS]).astype(np.float32)
    with torch.no_grad():
        all_preds = []
        for start in range(0, len(X_all), 4096):
            xt = torch.tensor(X_all[start:start + 4096], dtype=torch.float32, device=device)
            all_preds.append(model(xt).detach().cpu().numpy())
    all_preds = np.clip(np.concatenate(all_preds), 0.0, 1.0)
    data = data.copy()
    data["ANFIS_CALIBRATED_AVAILABILITY"] = all_preds

    # Fit normalization for decision attributes using training data only.
    uncertainty_scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
    rate_scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
    train_after = data.loc[data["DATA_SPLIT"] == "train"]
    uncertainty_scaler.fit(train_after[["PREDICTION_UNCERTAINTY"]])
    rate_scaler.fit(train_after[["RATE"]])
    data["UNCERTAINTY_NORM"] = uncertainty_scaler.transform(data[["PREDICTION_UNCERTAINTY"]]).reshape(-1)
    data["RATE_NORM"] = rate_scaler.transform(data[["RATE"]]).reshape(-1)
    data["RELIABILITY_SCORE"] = 1.0 - data["UNCERTAINTY_NORM"]
    data["AFFORDABILITY_SCORE"] = 1.0 - data["RATE_NORM"]

    model_bundle = {
        "state_dict": model.state_dict(),
        "anfis_inputs": ANFIS_INPUTS,
        "n_mfs": N_MFS,
        "n_inputs": len(ANFIS_INPUTS),
        "best_epoch": best_epoch,
        "x_scaler": x_scaler,
        "uncertainty_scaler": uncertainty_scaler,
        "rate_scaler": rate_scaler,
    }
    return data, model_bundle


def pack_candidates(data: pd.DataFrame, min_candidates: int = 1) -> dict:
    """Pad per-target-time candidate rows to arrays for fast PSO evaluation."""
    source = data.sort_values(["TARGET_TIME", "BLOCK_ID"]).copy()
    groups = [(time, group) for time, group in source.groupby("TARGET_TIME", sort=True)
              if len(group) >= min_candidates]
    if not groups:
        raise ValueError("No target times meet the minimum candidate count for PSO.")

    max_candidates = max(len(group) for _, group in groups)
    n_groups = len(groups)
    X = np.zeros((n_groups, max_candidates, 3), dtype=np.float64)
    y_actual = np.zeros((n_groups, max_candidates), dtype=np.float64)
    rate_norm = np.zeros((n_groups, max_candidates), dtype=np.float64)
    uncertainty_norm = np.zeros((n_groups, max_candidates), dtype=np.float64)
    mask = np.zeros((n_groups, max_candidates), dtype=bool)
    block_ids = np.full((n_groups, max_candidates), -1, dtype=np.int64)
    actual_rows = np.empty((n_groups, max_candidates), dtype=object)
    target_times = []
    candidate_counts = []

    for group_idx, (time, group) in enumerate(groups):
        n = len(group)
        target_times.append(pd.Timestamp(time))
        candidate_counts.append(n)
        # X columns are positive desirability terms: calibrated availability,
        # reliability (=1-normalized uncertainty), affordability (=1-normalized rate).
        X[group_idx, :n, 0] = group["ANFIS_CALIBRATED_AVAILABILITY"].to_numpy(float)
        X[group_idx, :n, 1] = group["RELIABILITY_SCORE"].to_numpy(float)
        X[group_idx, :n, 2] = group["AFFORDABILITY_SCORE"].to_numpy(float)
        y_actual[group_idx, :n] = group["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float)
        rate_norm[group_idx, :n] = group["RATE_NORM"].to_numpy(float)
        uncertainty_norm[group_idx, :n] = group["UNCERTAINTY_NORM"].to_numpy(float)
        mask[group_idx, :n] = True
        block_ids[group_idx, :n] = group["BLOCK_ID"].to_numpy(np.int64)
        actual_rows[group_idx, :n] = group.index.to_numpy()

    return {
        "X": X,
        "y_actual": y_actual,
        "rate_norm": rate_norm,
        "uncertainty_norm": uncertainty_norm,
        "mask": mask,
        "block_ids": block_ids,
        "row_indices": actual_rows,
        "target_times": target_times,
        "candidate_counts": np.asarray(candidate_counts),
    }


def normalize_particles(particles: np.ndarray) -> np.ndarray:
    particles = np.maximum(particles, 1e-9)
    sums = particles.sum(axis=1, keepdims=True)
    return particles / np.maximum(sums, 1e-12)


def selected_indices(pack: dict, weights: np.ndarray) -> np.ndarray:
    X = pack["X"]
    scores = np.einsum("gcf,f->gc", X, weights)
    scores[~pack["mask"]] = -1e9
    return np.argmax(scores, axis=1)


def fitness_for_weights(pack: dict, weights: np.ndarray) -> float:
    # The learned weights choose a block. Fitness is then evaluated on what
    # actually happened at the target hour: low shortfall + a modest rate cost.
    chosen = selected_indices(pack, weights)
    group_ids = np.arange(len(chosen))
    actual_availability = pack["y_actual"][group_ids, chosen]
    rate_cost = pack["rate_norm"][group_ids, chosen]
    decision_cost = (1.0 - actual_availability) + PRICE_COST_COEFFICIENT * rate_cost
    return float(np.mean(decision_cost))


def run_pso(pack: dict) -> tuple[np.ndarray, float, list[dict]]:
    rng = np.random.default_rng(RANDOM_SEED)
    positions = rng.dirichlet(np.ones(3), size=PSO_PARTICLES)
    velocities = rng.normal(0.0, 0.04, size=(PSO_PARTICLES, 3))
    positions = normalize_particles(positions)

    pbest_positions = positions.copy()
    pbest_fitness = np.array([fitness_for_weights(pack, p) for p in positions])
    best_idx = int(np.argmin(pbest_fitness))
    gbest_position = pbest_positions[best_idx].copy()
    gbest_fitness = float(pbest_fitness[best_idx])
    history = [{"iteration": 0, "best_validation_cost": gbest_fitness,
                "w_availability": gbest_position[0],
                "w_reliability": gbest_position[1],
                "w_affordability": gbest_position[2]}]

    for iteration in range(1, PSO_ITERATIONS + 1):
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = (
            PSO_INERTIA * velocities
            + PSO_COGNITIVE * r1 * (pbest_positions - positions)
            + PSO_SOCIAL * r2 * (gbest_position[None, :] - positions)
        )
        positions = normalize_particles(np.maximum(positions + velocities, 1e-9))
        current_fitness = np.array([fitness_for_weights(pack, p) for p in positions])

        improved = current_fitness < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = current_fitness[improved]
        best_idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[best_idx] < gbest_fitness:
            gbest_position = pbest_positions[best_idx].copy()
            gbest_fitness = float(pbest_fitness[best_idx])

        history.append({
            "iteration": iteration,
            "best_validation_cost": gbest_fitness,
            "w_availability": gbest_position[0],
            "w_reliability": gbest_position[1],
            "w_affordability": gbest_position[2],
        })
        if iteration == 1 or iteration % 10 == 0:
            print(f"PSO iteration {iteration:03d} | best validation cost={gbest_fitness:.6f} "
                  f"| weights={gbest_position.round(3).tolist()}")

    return gbest_position, gbest_fitness, history


def make_recommendations(data: pd.DataFrame, weights: np.ndarray, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = data.loc[data["DATA_SPLIT"] == split].copy()
    if part.empty:
        raise ValueError(f"No rows in split {split}.")
    part["PSO_SUITABILITY_SCORE"] = (
        weights[0] * part["ANFIS_CALIBRATED_AVAILABILITY"]
        + weights[1] * part["RELIABILITY_SCORE"]
        + weights[2] * part["AFFORDABILITY_SCORE"]
    ).clip(0.0, 1.0)

    chosen_indices_list = []
    for _, group in part.groupby("TARGET_TIME", sort=True):
        chosen_indices_list.append(group["PSO_SUITABILITY_SCORE"].idxmax())
    part["SELECTED_BY_PSO"] = False
    part.loc[chosen_indices_list, "SELECTED_BY_PSO"] = True

    recommendations = part.loc[part["SELECTED_BY_PSO"]].copy()
    recommendations = recommendations.sort_values("TARGET_TIME").reset_index(drop=True)
    return part.sort_values(["TARGET_TIME", "BLOCK_ID"]).reset_index(drop=True), recommendations


def summarize_recommendations(recommendations: pd.DataFrame) -> dict:
    actual = recommendations["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float)
    rate_norm = recommendations["RATE_NORM"].to_numpy(float)
    raw_unc = recommendations["PREDICTION_UNCERTAINTY"].to_numpy(float)
    cost = (1.0 - actual) + PRICE_COST_COEFFICIENT * rate_norm
    return {
        "target_timestamps_recommended": int(recommendations["TARGET_TIME"].nunique()),
        "recommended_rows": int(len(recommendations)),
        "unique_blocks_recommended": int(recommendations["BLOCK_ID"].nunique()),
        "mean_actual_availability_of_recommended_block": float(np.mean(actual)),
        "recommendation_has_any_availability_rate": float(np.mean(actual > 0.0)),
        "recommendation_availability_above_5_percent_rate": float(np.mean(actual >= 0.05)),
        "recommended_block_completely_full_rate_le_1_percent": float(np.mean(actual <= 0.01)),
        "mean_predicted_availability_after_ANFIS": float(recommendations["ANFIS_CALIBRATED_AVAILABILITY"].mean()),
        "mean_prediction_uncertainty_raw": float(np.mean(raw_unc)),
        "mean_rate": float(recommendations["RATE"].mean()),
        "mean_normalized_rate": float(np.mean(rate_norm)),
        "mean_suitability_score": float(recommendations["PSO_SUITABILITY_SCORE"].mean()),
        "decision_cost_lower_is_better": float(np.mean(cost)),
        "price_cost_coefficient": PRICE_COST_COEFFICIENT,
    }


def main() -> None:
    set_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_join_data()
    data = assign_chronological_splits(data)

    # Fit ANFIS using only the earliest train period; use next period for early stopping.
    data, model_bundle = train_anfis(data)

    # One-time forecast/calibration evaluation on the final unseen period only.
    test = data.loc[data["DATA_SPLIT"] == "test"].copy()
    raw_metrics = regression_metrics(
        test["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float),
        test["PREDICTED_GMP_AVAILABILITY"].to_numpy(float),
    )
    anfis_metrics = regression_metrics(
        test["ACTUAL_TARGET_AVAILABILITY"].to_numpy(float),
        test["ANFIS_CALIBRATED_AVAILABILITY"].to_numpy(float),
    )

    # Tune PSO weights only on the dedicated PSO-validation period.
    pso_validation = data.loc[data["DATA_SPLIT"] == "pso_validation"].copy()
    pso_pack = pack_candidates(pso_validation, min_candidates=2)
    print(f"\nPSO tuning timestamps with >=2 candidate blocks: {len(pso_pack['target_times']):,}")
    best_weights, best_val_cost, pso_history = run_pso(pso_pack)

    # Final decision evaluation on the never-used test period.
    test_pack = pack_candidates(test, min_candidates=2)
    test_cost = fitness_for_weights(test_pack, best_weights)
    test_all_candidates, test_recommendations = make_recommendations(data, best_weights, "test")
    pso_metrics = summarize_recommendations(test_recommendations)
    pso_metrics["timestamps_with_2plus_candidates"] = int(len(test_pack["target_times"]))
    pso_metrics["best_pso_validation_cost"] = float(best_val_cost)
    pso_metrics["test_cost_on_timestamps_with_2plus_candidates"] = float(test_cost)
    pso_metrics["optimized_weights"] = {
        "availability": float(best_weights[0]),
        "reliability_low_uncertainty": float(best_weights[1]),
        "affordability_low_rate": float(best_weights[2]),
    }

    # Test forecast metrics and recommendation metrics are saved separately.
    test_forecast_metrics = {
        "LSTM_raw_prediction": raw_metrics,
        "ANFIS_calibrated_prediction": anfis_metrics,
        "interpretation": (
            "ANFIS is trained as a supervised availability calibrator, not as an unlabelled "
            "suitability scorer. The actual target is observed GMP availability at TARGET_TIME."
        ),
    }
    with open(OUTPUT_DIR / "forecast_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(test_forecast_metrics, f, indent=2)
    with open(OUTPUT_DIR / "pso_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(pso_metrics, f, indent=2)
    with open(OUTPUT_DIR / "pso_optimized_weights.json", "w", encoding="utf-8") as f:
        json.dump({
            "w_availability": float(best_weights[0]),
            "w_reliability": float(best_weights[1]),
            "w_affordability": float(best_weights[2]),
            "weights_sum": float(best_weights.sum()),
            "fitness_definition": "mean(1 - observed_target_availability + price_cost_coefficient * normalized_rate)",
            "price_cost_coefficient": PRICE_COST_COEFFICIENT,
            "pso_validation_best_cost": float(best_val_cost),
        }, f, indent=2)
    pd.DataFrame(pso_history).to_csv(OUTPUT_DIR / "pso_convergence_history.csv", index=False)

    # Store predictions for audit; test rows have no role in fitting/tuning.
    data.to_csv(OUTPUT_DIR / "anfis_all_predictions_with_labels.csv", index=False)
    test_all_candidates.to_csv(OUTPUT_DIR / "pso_test_candidate_scores.csv", index=False)
    test_recommendations.to_csv(OUTPUT_DIR / "pso_test_recommendations.csv", index=False)
    torch.save(model_bundle["state_dict"], OUTPUT_DIR / "anfis_model_state.pt")
    joblib.dump({
        "x_scaler": model_bundle["x_scaler"],
        "uncertainty_scaler": model_bundle["uncertainty_scaler"],
        "rate_scaler": model_bundle["rate_scaler"],
        "anfis_inputs": model_bundle["anfis_inputs"],
        "n_mfs": model_bundle["n_mfs"],
        "n_inputs": model_bundle["n_inputs"],
        "best_epoch": model_bundle["best_epoch"],
    }, OUTPUT_DIR / "anfis_scalers_and_config.joblib")

    print("\n--- TEST forecast metrics (lower MAE/RMSE is better; higher R2 is better) ---")
    print(json.dumps(test_forecast_metrics, indent=2))
    print("\n--- Optimized PSO weights ---")
    print(json.dumps(pso_metrics["optimized_weights"], indent=2))
    print("\n--- Test recommendation metrics ---")
    print(json.dumps({k: v for k, v in pso_metrics.items() if k != "optimized_weights"}, indent=2))
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
