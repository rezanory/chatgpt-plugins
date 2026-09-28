# SoilBin Q1/Q2 V6.1 - leakage-safe virtual sensing and synthetic augmentation
# Post-lock exploratory challenge. V5 remains frozen and is never overwritten.
from __future__ import annotations

import base64
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

SEED = 20260914  # exact V3/V5 seed
STABILITY_SEEDS = (20260914, 20260928, 20261005)
random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("SOILBIN_V61_OUTPUT_ROOT", "/kaggle/working/SOILBIN_V6_1"))
TABLE = ROOT / "tables"
FIG = ROOT / "figures"
STATE = ROOT / "state"
for directory in (ROOT, TABLE, FIG, STATE):
    directory.mkdir(parents=True, exist_ok=True)
ARCHIVE_BASE = os.environ.get(
    "SOILBIN_V61_ARCHIVE_BASE",
    "/kaggle/working/SoilBin_Q1_Q2_V6_1_VIRTUAL_SYNTHETIC_20260928",
)

EXPECTED_MODEL_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EXPECTED_MISSING = {"V1W1T1", "V2W3T2", "V3W2T1"}
TARGET_COLS = ["LC1_peak_magnitude_N_delta", "LC5_peak_magnitude_N_delta"]
TARGET_NAMES = ["LC1_proxy_N", "LC5_proxy_N"]
FROZEN_V5_JOINT_GROUP_MAE = 24.28878365273039
FROZEN_V5_LC1_GROUP_MAE = 12.785420282422
FROZEN_V5_LC5_GROUP_MAE = 35.79214702303878
FROZEN_V5_TASK = "D_delta_prev_peak"
RESPONSE_STATUS = "CALIBRATED_VERTICAL_FORCE_PROXY_N_PENDING_SENSOR_AREA_OR_STRESS_CALIBRATION"
DEPTH_STATUS = "PROVISIONAL_LC5_5CM_LC1_15CM_PENDING_LAYOUT_CONFIRMATION"


def phase(name: str) -> None:
    print(f"CGP_PHASE:{name}", flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            json_safe(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )


def decode_gz_b64(payload: str, expected_sha: str) -> str:
    raw = gzip.decompress(base64.b64decode(payload.strip()))
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{actual}:{expected_sha}")
    return raw.decode("utf-8")


def stable_seed(*parts) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little", signed=False)


def group_equal_mae(y, prediction, groups) -> float:
    frame = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=float),
            "prediction": np.asarray(prediction, dtype=float),
            "group": np.asarray(groups),
        }
    )
    return float(
        frame.assign(error=lambda data: (data.y - data.prediction).abs())
        .groupby("group", sort=True)
        .error.mean()
        .mean()
    )


def metrics_block(predictions: pd.DataFrame) -> dict:
    output = {
        "n": int(len(predictions)),
        "n_groups": int(predictions.Group_VW.nunique()),
    }
    target_group_maes = []
    for target in TARGET_NAMES:
        y = predictions[f"y_{target}"].to_numpy(float)
        p = predictions[f"p_{target}"].to_numpy(float)
        output[f"{target}_Group_MAE"] = group_equal_mae(y, p, predictions.Group_VW)
        output[f"{target}_MAE"] = float(mean_absolute_error(y, p))
        output[f"{target}_RMSE"] = float(mean_squared_error(y, p) ** 0.5)
        output[f"{target}_Bias"] = float(np.mean(p - y))
        output[f"{target}_R2"] = float(r2_score(y, p))
        target_group_maes.append(output[f"{target}_Group_MAE"])
    output["joint_Group_MAE"] = float(np.mean(target_group_maes))
    return output


def make_estimator(name: str, params: dict, seed: int = SEED) -> Pipeline:
    common = dict(
        n_estimators=250,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=int(seed),
        n_jobs=1,
        max_features=1.0,
    )
    if name == "RandomForest":
        model = RandomForestRegressor(**common)
    elif name == "ExtraTrees":
        model = ExtraTreesRegressor(**common)
    else:
        raise KeyError(name)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


# Exact winner-family catalog used by the V3/V5 reproduction.
CANDIDATES = [
    ("RandomForest", {"max_depth": None, "min_samples_leaf": 1}),
    ("RandomForest", {"max_depth": 4, "min_samples_leaf": 2}),
    ("ExtraTrees", {"max_depth": None, "min_samples_leaf": 1}),
    ("ExtraTrees", {"max_depth": 4, "min_samples_leaf": 2}),
]

phase("DATA_FREEZE")
model_raw = decode_gz_b64(MODEL_CSV_GZ_B64, EXPECTED_MODEL_SHA)
df = pd.read_csv(StringIO(model_raw))
if len(df) != 51 or df.Run_ID.nunique() != 51:
    raise RuntimeError("EXPECTED_51_RUNS")

df["Group_VW"] = df.Run_ID.str.extract(r"^(V\dW\d)")[0]
df["Speed_kmh"] = df["Speed_level"].astype(float)
df["Load_kN"] = df["Weight_level"].astype(float) + 1.0
df["Pass_T"] = df["Pass_T"].astype(int)
all_ids = {f"V{v}W{w}T{t}" for v in range(1, 4) for w in range(1, 4) for t in range(1, 7)}
if all_ids - set(df.Run_ID) != EXPECTED_MISSING:
    raise RuntimeError("DESIGN_SLOT_MISMATCH")
for column in TARGET_COLS:
    if column not in df or df[column].isna().any() or not np.isfinite(df[column]).all():
        raise RuntimeError(f"TARGET_INVALID:{column}")

# Exact 41 consecutive previous-pass pairs used by V3/V5.
previous = df.copy()
previous["Pass_T"] = previous["Pass_T"] + 1
rename = {column: f"prev_{column}" for column in TARGET_COLS + ["QC_status"]}
previous_small = previous[["Group_VW", "Pass_T"] + TARGET_COLS + ["QC_status"]].rename(columns=rename)
hist = df.merge(previous_small, on=["Group_VW", "Pass_T"], how="left", validate="one_to_one")
PREV_COLS = [f"prev_{column}" for column in TARGET_COLS]
hist = hist[hist[PREV_COLS[0]].notna() & hist[PREV_COLS[1]].notna()].copy().reset_index(drop=True)
if len(hist) != 41:
    raise RuntimeError(f"EXPECTED_41_HISTORY_PAIRS_GOT_{len(hist)}")
hist["row_id"] = np.arange(len(hist), dtype=int)
hist["pair_qc_ok"] = (
    (hist["QC_status"].fillna("") == "OK")
    & (hist["prev_QC_status"].fillna("") == "OK")
)

VIRTUAL_CHANNEL_SOURCES = {
    "VS_common_mean_N": "0.5 * (previous LC1 + previous LC5)",
    "VS_differential_N": "previous LC5 - previous LC1",
    "VS_normalized_imbalance": "(previous LC5 - previous LC1) / (abs(previous LC1) + abs(previous LC5) + eps)",
    "VS_ratio_LC5_LC1": "(abs(previous LC5) + eps) / (abs(previous LC1) + eps)",
    "VS_log_ratio_LC5_LC1": "log(VS_ratio_LC5_LC1)",
    "VS_vector_norm_N": "sqrt(previous LC1^2 + previous LC5^2)",
    "VS_geometric_mean_N": "sqrt(max(previous LC1 * previous LC5, 0))",
    "VS_angle_rad": "atan2(previous LC5, previous LC1)",
}


def add_virtual_channels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lc1 = result[PREV_COLS[0]].to_numpy(float)
    lc5 = result[PREV_COLS[1]].to_numpy(float)
    epsilon = 1e-9
    ratio = (np.abs(lc5) + epsilon) / (np.abs(lc1) + epsilon)
    result["VS_common_mean_N"] = 0.5 * (lc1 + lc5)
    result["VS_differential_N"] = lc5 - lc1
    result["VS_normalized_imbalance"] = (lc5 - lc1) / (np.abs(lc1) + np.abs(lc5) + epsilon)
    result["VS_ratio_LC5_LC1"] = ratio
    result["VS_log_ratio_LC5_LC1"] = np.log(ratio)
    result["VS_vector_norm_N"] = np.sqrt(lc1 ** 2 + lc5 ** 2)
    result["VS_geometric_mean_N"] = np.sqrt(np.maximum(lc1 * lc5, 0.0))
    result["VS_angle_rad"] = np.arctan2(lc5, lc1)
    for column in VIRTUAL_CHANNEL_SOURCES:
        if not np.isfinite(result[column]).all():
            raise RuntimeError(f"NONFINITE_VIRTUAL_CHANNEL:{column}")
    return result


hist = add_virtual_channels(hist)
BASE_FEATURES = ["Load_kN", "Speed_kmh", "Pass_T"] + PREV_COLS
VS_CORE_FEATURES = BASE_FEATURES + [
    "VS_common_mean_N",
    "VS_differential_N",
    "VS_normalized_imbalance",
    "VS_log_ratio_LC5_LC1",
]
VS_FULL_FEATURES = BASE_FEATURES + list(VIRTUAL_CHANNEL_SOURCES)
FEATURE_SETS = {
    "BASE": BASE_FEATURES,
    "VS_CORE": VS_CORE_FEATURES,
    "VS_FULL": VS_FULL_FEATURES,
}

# Synthetic rows are generated only from a current training partition.
# Evaluation partitions always contain real rows only.
EXPERIMENTS = {
    "BASELINE_REAL_ONLY": {"feature_set": "BASE", "augmentation": "none", "ratio": 0.0},
    "VS_CORE_REAL_ONLY": {"feature_set": "VS_CORE", "augmentation": "none", "ratio": 0.0},
    "VS_FULL_REAL_ONLY": {"feature_set": "VS_FULL", "augmentation": "none", "ratio": 0.0},
    "BASE_JITTER_050": {"feature_set": "BASE", "augmentation": "jitter", "ratio": 0.5},
    "VS_FULL_JITTER_050": {"feature_set": "VS_FULL", "augmentation": "jitter", "ratio": 0.5},
    "BASE_LOCAL_MIXUP_050": {"feature_set": "BASE", "augmentation": "local_mixup", "ratio": 0.5},
    "VS_FULL_LOCAL_MIXUP_050": {"feature_set": "VS_FULL", "augmentation": "local_mixup", "ratio": 0.5},
    "VS_FULL_LOCAL_MIXUP_100": {"feature_set": "VS_FULL", "augmentation": "local_mixup", "ratio": 1.0},
    "VS_FULL_LOCAL_MIXUP_200": {"feature_set": "VS_FULL", "augmentation": "local_mixup", "ratio": 2.0},
    "VS_FULL_HYBRID_100": {"feature_set": "VS_FULL", "augmentation": "hybrid", "ratio": 1.0},
}
FLAGSHIP_TASK = "VS_FULL_LOCAL_MIXUP_100"


def y_delta(data: pd.DataFrame) -> np.ndarray:
    return data[TARGET_COLS].to_numpy(float) - data[PREV_COLS].to_numpy(float)


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    standard = float(np.std(values, ddof=0))
    return max(mad, standard * 0.1, 1e-6)


LEAKAGE_STATS = {
    "augmentation_calls": 0,
    "synthetic_rows_generated_all_scopes": 0,
    "forbidden_source_intersections": 0,
    "synthetic_rows_in_evaluation": 0,
    "inner_validation_partitions_real_only": True,
    "outer_test_partitions_real_only": True,
    "current_pass_target_columns_in_feature_sets": False,
}
OUTER_PROVENANCE = []


def augment_training(
    train: pd.DataFrame,
    task_name: str,
    seed: int,
    forbidden_row_ids=None,
    context: dict | None = None,
    record_provenance: bool = False,
):
    if task_name not in EXPERIMENTS:
        raise KeyError(task_name)
    specification = EXPERIMENTS[task_name]
    feature_columns = FEATURE_SETS[specification["feature_set"]]
    real = add_virtual_channels(train.copy()).reset_index(drop=True)
    real_ids = set(int(item) for item in real.row_id)
    forbidden = set(int(item) for item in (forbidden_row_ids or []))
    if real_ids & forbidden:
        raise RuntimeError("TRAIN_EVAL_ROW_OVERLAP_BEFORE_AUGMENTATION")

    X_real = real[feature_columns].copy()
    y_real = y_delta(real)
    ratio = float(specification["ratio"])
    n_synthetic = int(round(len(real) * ratio))
    LEAKAGE_STATS["augmentation_calls"] += 1
    if specification["augmentation"] == "none" or n_synthetic == 0:
        return X_real, y_real, [], {
            "real_rows": int(len(real)),
            "synthetic_rows": 0,
            "augmentation": "none",
            "ratio": 0.0,
        }

    rng = np.random.default_rng(int(seed))
    y_real_delta = y_delta(real)
    synthetic_rows = []
    synthetic_targets = []
    provenance = []
    groups = real.Group_VW.astype(str).to_numpy()
    passes = real.Pass_T.to_numpy(int)
    distance_columns = ["Speed_kmh", "Load_kN"] + PREV_COLS
    distance_matrix = real[distance_columns].to_numpy(float)
    distance_scales = np.asarray([robust_scale(distance_matrix[:, index]) for index in range(distance_matrix.shape[1])])
    jitter_sigmas = {
        column: max(
            0.01 * float(np.median(np.abs(real[column].to_numpy(float)))),
            0.05 * robust_scale(real[column].to_numpy(float)),
            1e-6,
        )
        for column in PREV_COLS
    }

    for synthetic_index in range(n_synthetic):
        requested_kind = specification["augmentation"]
        if requested_kind == "hybrid":
            kind = "jitter" if synthetic_index % 2 == 0 else "local_mixup"
        else:
            kind = requested_kind
        anchor = int(rng.integers(0, len(real)))
        source_b = None
        mixing_lambda = None

        if kind == "local_mixup":
            candidates = np.flatnonzero((passes == passes[anchor]) & (np.arange(len(real)) != anchor))
            different_group = candidates[groups[candidates] != groups[anchor]]
            if len(different_group):
                candidates = different_group
            if len(candidates) == 0:
                kind = "jitter"
            else:
                standardized = (distance_matrix[candidates] - distance_matrix[anchor]) / distance_scales
                distances = np.sqrt(np.sum(standardized ** 2, axis=1))
                nearest = candidates[np.argsort(distances)[: min(3, len(candidates))]]
                source_b = int(rng.choice(nearest))
                mixing_lambda = float(np.clip(rng.beta(2.0, 2.0), 0.2, 0.8))

        row = real.iloc[anchor].copy()
        if kind == "jitter":
            for column in PREV_COLS:
                row[column] = max(0.0, float(row[column]) + float(rng.normal(0.0, jitter_sigmas[column])))
            synthetic_target = y_real_delta[anchor].copy()
        elif kind == "local_mixup" and source_b is not None and mixing_lambda is not None:
            other = real.iloc[source_b]
            for column in ["Speed_kmh", "Load_kN"] + PREV_COLS:
                row[column] = mixing_lambda * float(real.iloc[anchor][column]) + (1.0 - mixing_lambda) * float(other[column])
            row["Pass_T"] = int(real.iloc[anchor].Pass_T)
            synthetic_target = mixing_lambda * y_real_delta[anchor] + (1.0 - mixing_lambda) * y_real_delta[source_b]
        else:
            raise RuntimeError("UNREACHABLE_AUGMENTATION_KIND")

        row["Run_ID"] = f"SYN_{task_name}_{seed}_{synthetic_index}"
        row["Group_VW"] = "SYNTHETIC_TRAIN_ONLY"
        row["row_id"] = -1 - synthetic_index
        for target_column in TARGET_COLS:
            row[target_column] = np.nan
        synthetic_rows.append(row)
        synthetic_targets.append(np.asarray(synthetic_target, dtype=float))

        source_a_row_id = int(real.iloc[anchor].row_id)
        source_b_row_id = int(real.iloc[source_b].row_id) if source_b is not None else None
        source_ids = {source_a_row_id}
        if source_b_row_id is not None:
            source_ids.add(source_b_row_id)
        intersection = source_ids & forbidden
        if intersection:
            LEAKAGE_STATS["forbidden_source_intersections"] += 1
            raise RuntimeError(f"SYNTHETIC_SOURCE_LEAKAGE:{sorted(intersection)}")
        audit_row = {
            **(context or {}),
            "task": task_name,
            "augmentation_seed": int(seed),
            "synthetic_index": int(synthetic_index),
            "kind": kind,
            "source_a_row_id": source_a_row_id,
            "source_a_group": str(real.iloc[anchor].Group_VW),
            "source_b_row_id": source_b_row_id,
            "source_b_group": str(real.iloc[source_b].Group_VW) if source_b is not None else None,
            "mixing_lambda": mixing_lambda,
            "forbidden_intersection_count": 0,
        }
        provenance.append(audit_row)

    synthetic_frame = add_virtual_channels(pd.DataFrame(synthetic_rows))
    X_synthetic = synthetic_frame[feature_columns].copy()
    y_synthetic = np.vstack(synthetic_targets).astype(float)
    X_augmented = pd.concat([X_real, X_synthetic], ignore_index=True)
    y_augmented = np.vstack([y_real, y_synthetic])
    if len(X_augmented) != len(real) + n_synthetic or len(y_augmented) != len(X_augmented):
        raise RuntimeError("AUGMENTATION_ROW_COUNT_MISMATCH")
    if not np.isfinite(y_augmented).all():
        raise RuntimeError("NONFINITE_AUGMENTED_TARGET")
    LEAKAGE_STATS["synthetic_rows_generated_all_scopes"] += int(n_synthetic)
    if record_provenance:
        OUTER_PROVENANCE.extend(provenance)
    return X_augmented, y_augmented, provenance, {
        "real_rows": int(len(real)),
        "synthetic_rows": int(n_synthetic),
        "augmentation": specification["augmentation"],
        "ratio": ratio,
    }


def candidate_score(
    train: pd.DataFrame,
    task_name: str,
    model_name: str,
    params: dict,
    outer_label: str,
    candidate_index: int,
    search_rows: list,
    namespace: str,
) -> float:
    groups = train.Group_VW.to_numpy()
    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("INNER_GROUPS_LT_2")
    splitter = GroupKFold(n_splits=n_splits)
    predictions_delta = np.full((len(train), 2), np.nan, dtype=float)
    failed = None
    generated_total = 0
    for fold_index, (fit_indices, validation_indices) in enumerate(splitter.split(train, groups=groups)):
        fit_real = train.iloc[fit_indices].copy()
        validation_real = add_virtual_channels(train.iloc[validation_indices].copy())
        forbidden = set(int(item) for item in validation_real.row_id)
        try:
            X_fit, y_fit, _, augmentation_summary = augment_training(
                fit_real,
                task_name,
                stable_seed(namespace, outer_label, task_name, candidate_index, fold_index),
                forbidden_row_ids=forbidden,
                context={
                    "scope": "inner_fit",
                    "outer_label": str(outer_label),
                    "inner_fold": int(fold_index),
                },
                record_provenance=False,
            )
            generated_total += int(augmentation_summary["synthetic_rows"])
            X_validation = validation_real[FEATURE_SETS[EXPERIMENTS[task_name]["feature_set"]]]
            for target_index in range(2):
                estimator = make_estimator(model_name, params, SEED)
                estimator.fit(X_fit, y_fit[:, target_index])
                predictions_delta[validation_indices, target_index] = estimator.predict(X_validation)
        except Exception as exc:
            failed = f"{type(exc).__name__}:{str(exc)[:240]}"
            break

    if failed or np.isnan(predictions_delta).any():
        score = float("inf")
    else:
        final_predictions = predictions_delta + train[PREV_COLS].to_numpy(float)
        score = float(
            np.mean(
                [
                    group_equal_mae(
                        train[TARGET_COLS[target_index]],
                        final_predictions[:, target_index],
                        groups,
                    )
                    for target_index in range(2)
                ]
            )
        )
    search_rows.append(
        {
            "namespace": namespace,
            "outer_label": str(outer_label),
            "task": task_name,
            "candidate_index": int(candidate_index),
            "model": model_name,
            "params": json.dumps(params, sort_keys=True),
            "inner_joint_Group_MAE": score if math.isfinite(score) else None,
            "synthetic_rows_generated_across_inner_folds": int(generated_total),
            "failed": failed,
        }
    )
    return score


def select_model_for_task(
    train: pd.DataFrame,
    task_name: str,
    outer_label: str,
    search_rows: list,
    namespace: str,
):
    best = None
    for candidate_index, (model_name, params) in enumerate(CANDIDATES):
        score = candidate_score(
            train,
            task_name,
            model_name,
            params,
            outer_label,
            candidate_index,
            search_rows,
            namespace,
        )
        row = (score, candidate_index, model_name, params)
        if math.isfinite(score) and (best is None or row[:2] < best[:2]):
            best = row
    if best is None:
        raise RuntimeError(f"NO_VALID_MODEL:{task_name}:{outer_label}")
    return best


def select_task_and_model(
    train: pd.DataFrame,
    outer_label: str,
    search_rows: list,
):
    best = None
    combined_index = 0
    for task_order, task_name in enumerate(EXPERIMENTS):
        for model_order, (model_name, params) in enumerate(CANDIDATES):
            score = candidate_score(
                train,
                task_name,
                model_name,
                params,
                outer_label,
                combined_index,
                search_rows,
                "nested_task_model",
            )
            row = (score, task_order, model_order, task_name, model_name, params)
            if math.isfinite(score) and (best is None or row[:3] < best[:3]):
                best = row
            combined_index += 1
    if best is None:
        raise RuntimeError(f"NO_VALID_TASK_MODEL:{outer_label}")
    return best


def fit_predict_outer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    task_name: str,
    model_name: str,
    params: dict,
    augmentation_seed: int,
    context: dict,
):
    test_real = add_virtual_channels(test.copy())
    forbidden = set(int(item) for item in test_real.row_id)
    X_train, y_train, _, augmentation_summary = augment_training(
        train,
        task_name,
        augmentation_seed,
        forbidden_row_ids=forbidden,
        context=context,
        record_provenance=True,
    )
    features = FEATURE_SETS[EXPERIMENTS[task_name]["feature_set"]]
    X_test = test_real[features]
    prediction_delta = []
    for target_index in range(2):
        estimator = make_estimator(model_name, params, SEED)
        estimator.fit(X_train, y_train[:, target_index])
        prediction_delta.append(estimator.predict(X_test))
    final_prediction = np.column_stack(prediction_delta) + test_real[PREV_COLS].to_numpy(float)
    return final_prediction, augmentation_summary


phase("PAIRED_ABLATION_OUTER_CV")
outer_labels = list(pd.unique(hist.Group_VW))
ablation_predictions = []
baseline_search_rows = []
baseline_selection_rows = []
baseline_selected_by_outer = {}

for outer_index, outer_label in enumerate(outer_labels):
    test = hist[hist.Group_VW == outer_label].copy()
    train = hist[hist.Group_VW != outer_label].copy()
    if len(test) == 0 or train.Group_VW.nunique() != 8:
        raise RuntimeError(f"OUTER_SPLIT_INVALID:{outer_label}")
    best_score, candidate_index, model_name, params = select_model_for_task(
        train,
        "BASELINE_REAL_ONLY",
        f"VW:{outer_label}",
        baseline_search_rows,
        "baseline_model_selection",
    )
    baseline_selected_by_outer[str(outer_label)] = (model_name, params)
    baseline_selection_rows.append(
        {
            "outer_label": str(outer_label),
            "model": model_name,
            "params": json.dumps(params, sort_keys=True),
            "inner_joint_Group_MAE": float(best_score),
            "candidate_index": int(candidate_index),
        }
    )

    for task_name in EXPERIMENTS:
        final_prediction, augmentation_summary = fit_predict_outer(
            train,
            test,
            task_name,
            model_name,
            params,
            stable_seed("outer_ablation_fit", SEED, outer_label, task_name),
            {
                "scope": "outer_ablation_fit",
                "outer_label": str(outer_label),
                "inner_fold": None,
            },
        )
        for row_position, (_, row) in enumerate(test.iterrows()):
            output_row = {
                "row_id": int(row.row_id),
                "Run_ID": str(row.Run_ID),
                "Group_VW": str(row.Group_VW),
                "Speed_kmh": float(row.Speed_kmh),
                "Load_kN": float(row.Load_kN),
                "Pass_T": int(row.Pass_T),
                "task": task_name,
                "selected_task": task_name,
                "model": model_name,
                "params": json.dumps(params, sort_keys=True),
                "synthetic_rows_in_fit": int(augmentation_summary["synthetic_rows"]),
                "evaluation_row_is_real": True,
            }
            for target_index, target in enumerate(TARGET_NAMES):
                output_row[f"y_{target}"] = float(row[TARGET_COLS[target_index]])
                output_row[f"p_{target}"] = float(final_prediction[row_position, target_index])
            ablation_predictions.append(output_row)

ablation_predictions = pd.DataFrame(ablation_predictions)
for task_name in EXPERIMENTS:
    task_predictions = ablation_predictions[ablation_predictions.task == task_name]
    if task_predictions.row_id.nunique() != len(hist) or len(task_predictions) != len(hist):
        raise RuntimeError(f"INCOMPLETE_ABLATION_OOF:{task_name}")

phase("NESTED_TASK_MODEL_SELECTION")
nested_predictions = []
nested_search_rows = []
nested_selection_rows = []
for outer_label in outer_labels:
    test = hist[hist.Group_VW == outer_label].copy()
    train = hist[hist.Group_VW != outer_label].copy()
    best_score, task_order, model_order, selected_task, model_name, params = select_task_and_model(
        train,
        f"VW:{outer_label}",
        nested_search_rows,
    )
    final_prediction, augmentation_summary = fit_predict_outer(
        train,
        test,
        selected_task,
        model_name,
        params,
        stable_seed("outer_nested_fit", SEED, outer_label, selected_task, model_name, json.dumps(params, sort_keys=True)),
        {
            "scope": "outer_nested_fit",
            "outer_label": str(outer_label),
            "inner_fold": None,
        },
    )
    nested_selection_rows.append(
        {
            "outer_label": str(outer_label),
            "selected_task": selected_task,
            "model": model_name,
            "params": json.dumps(params, sort_keys=True),
            "inner_joint_Group_MAE": float(best_score),
            "task_order": int(task_order),
            "model_order": int(model_order),
            "synthetic_rows_in_outer_fit": int(augmentation_summary["synthetic_rows"]),
        }
    )
    for row_position, (_, row) in enumerate(test.iterrows()):
        output_row = {
            "row_id": int(row.row_id),
            "Run_ID": str(row.Run_ID),
            "Group_VW": str(row.Group_VW),
            "Speed_kmh": float(row.Speed_kmh),
            "Load_kN": float(row.Load_kN),
            "Pass_T": int(row.Pass_T),
            "task": "V61_NESTED_SELECTED",
            "selected_task": selected_task,
            "model": model_name,
            "params": json.dumps(params, sort_keys=True),
            "synthetic_rows_in_fit": int(augmentation_summary["synthetic_rows"]),
            "evaluation_row_is_real": True,
        }
        for target_index, target in enumerate(TARGET_NAMES):
            output_row[f"y_{target}"] = float(row[TARGET_COLS[target_index]])
            output_row[f"p_{target}"] = float(final_prediction[row_position, target_index])
        nested_predictions.append(output_row)

nested_predictions = pd.DataFrame(nested_predictions)
if nested_predictions.row_id.nunique() != len(hist) or len(nested_predictions) != len(hist):
    raise RuntimeError("INCOMPLETE_NESTED_OOF")

all_predictions = pd.concat([ablation_predictions, nested_predictions], ignore_index=True)
summary_rows = []
for task_name in list(EXPERIMENTS) + ["V61_NESTED_SELECTED"]:
    task_predictions = all_predictions[all_predictions.task == task_name].copy()
    metrics = metrics_block(task_predictions)
    metrics.update(
        {
            "task": task_name,
            "feature_set": EXPERIMENTS[task_name]["feature_set"] if task_name in EXPERIMENTS else "NESTED_SELECTED",
            "augmentation": EXPERIMENTS[task_name]["augmentation"] if task_name in EXPERIMENTS else "NESTED_SELECTED",
            "synthetic_ratio": EXPERIMENTS[task_name]["ratio"] if task_name in EXPERIMENTS else None,
        }
    )
    summary_rows.append(metrics)
summary_df = pd.DataFrame(summary_rows).sort_values("joint_Group_MAE").reset_index(drop=True)

baseline_metrics = summary_df[summary_df.task == "BASELINE_REAL_ONLY"].iloc[0]
baseline_drift_pct = float(
    100.0
    * (float(baseline_metrics.joint_Group_MAE) - FROZEN_V5_JOINT_GROUP_MAE)
    / FROZEN_V5_JOINT_GROUP_MAE
)
baseline_exact_match = bool(abs(float(baseline_metrics.joint_Group_MAE) - FROZEN_V5_JOINT_GROUP_MAE) <= 1e-9)
baseline_within_2pct = bool(abs(baseline_drift_pct) <= 2.0)
if not baseline_within_2pct:
    raise RuntimeError(
        "V5_BASELINE_REPRODUCTION_DRIFT_GT_2PCT:"
        f"{float(baseline_metrics.joint_Group_MAE)}:{FROZEN_V5_JOINT_GROUP_MAE}:{baseline_drift_pct}"
    )
print(
    "V5_BASELINE_REPRODUCED_WITHIN_2PCT",
    float(baseline_metrics.joint_Group_MAE),
    baseline_drift_pct,
    flush=True,
)


def paired_comparison(baseline: pd.DataFrame, challenger: pd.DataFrame, task_name: str):
    group_rows = []
    improvements = []
    groups_improved = 0
    for group in sorted(hist.Group_VW.unique()):
        base_group = baseline[baseline.Group_VW == group].sort_values("row_id")
        challenge_group = challenger[challenger.Group_VW == group].sort_values("row_id")
        if list(base_group.row_id) != list(challenge_group.row_id):
            raise RuntimeError(f"PAIRED_ROW_MISMATCH:{task_name}:{group}")
        baseline_group_mae = float(
            np.mean(
                [
                    mean_absolute_error(base_group[f"y_{target}"], base_group[f"p_{target}"])
                    for target in TARGET_NAMES
                ]
            )
        )
        challenger_group_mae = float(
            np.mean(
                [
                    mean_absolute_error(challenge_group[f"y_{target}"], challenge_group[f"p_{target}"])
                    for target in TARGET_NAMES
                ]
            )
        )
        improvement = baseline_group_mae - challenger_group_mae
        improvements.append(improvement)
        groups_improved += int(improvement > 0)
        group_rows.append(
            {
                "task": task_name,
                "Group_VW": group,
                "baseline_joint_MAE_N": baseline_group_mae,
                "challenger_joint_MAE_N": challenger_group_mae,
                "improvement_N": improvement,
            }
        )

    improvements = np.asarray(improvements, dtype=float)
    observed = float(np.mean(improvements))
    permutation = np.asarray(
        [
            np.mean(improvements * np.asarray(signs, dtype=float))
            for signs in itertools.product([-1.0, 1.0], repeat=len(improvements))
        ],
        dtype=float,
    )
    one_sided_p = float(np.sum(permutation >= observed - 1e-12) / len(permutation))
    bootstrap_rng = np.random.default_rng(stable_seed("paired_bootstrap", task_name))
    bootstrap = np.asarray(
        [
            np.mean(bootstrap_rng.choice(improvements, size=len(improvements), replace=True))
            for _ in range(20000)
        ],
        dtype=float,
    )
    baseline_summary = metrics_block(baseline)
    challenger_summary = metrics_block(challenger)
    lc1_worsening = float(
        100.0
        * (challenger_summary["LC1_proxy_N_Group_MAE"] - baseline_summary["LC1_proxy_N_Group_MAE"])
        / baseline_summary["LC1_proxy_N_Group_MAE"]
    )
    lc5_worsening = float(
        100.0
        * (challenger_summary["LC5_proxy_N_Group_MAE"] - baseline_summary["LC5_proxy_N_Group_MAE"])
        / baseline_summary["LC5_proxy_N_Group_MAE"]
    )
    target_guard = bool(lc1_worsening <= 5.0 and lc5_worsening <= 5.0)
    joint_improvement_pct = float(
        100.0
        * (baseline_summary["joint_Group_MAE"] - challenger_summary["joint_Group_MAE"])
        / baseline_summary["joint_Group_MAE"]
    )
    acceptance = bool(
        challenger_summary["joint_Group_MAE"] < baseline_summary["joint_Group_MAE"]
        and groups_improved >= 6
        and target_guard
        and one_sided_p <= 0.05
    )
    row = {
        "task": task_name,
        "joint_Group_MAE_N": float(challenger_summary["joint_Group_MAE"]),
        "relative_improvement_vs_runtime_baseline_pct": joint_improvement_pct,
        "relative_improvement_vs_frozen_V5_reference_pct": float(
            100.0
            * (FROZEN_V5_JOINT_GROUP_MAE - challenger_summary["joint_Group_MAE"])
            / FROZEN_V5_JOINT_GROUP_MAE
        ),
        "mean_group_paired_improvement_N": observed,
        "group_bootstrap_95CI_low_N": float(np.quantile(bootstrap, 0.025)),
        "group_bootstrap_95CI_high_N": float(np.quantile(bootstrap, 0.975)),
        "exact_signflip_one_sided_p": one_sided_p,
        "groups_improved": int(groups_improved),
        "groups_total": int(len(improvements)),
        "LC1_worsening_vs_runtime_baseline_pct": lc1_worsening,
        "LC5_worsening_vs_runtime_baseline_pct": lc5_worsening,
        "per_target_no_worse_than_5pct": target_guard,
        "meets_v61_exploratory_acceptance": acceptance,
        "beats_frozen_v5_reference": bool(challenger_summary["joint_Group_MAE"] < FROZEN_V5_JOINT_GROUP_MAE),
    }
    return row, group_rows


phase("PAIRED_INFERENCE")
baseline_predictions = all_predictions[all_predictions.task == "BASELINE_REAL_ONLY"].copy()
paired_rows = []
group_comparison_rows = []
for task_name in list(EXPERIMENTS) + ["V61_NESTED_SELECTED"]:
    if task_name == "BASELINE_REAL_ONLY":
        continue
    challenge_predictions = all_predictions[all_predictions.task == task_name].copy()
    paired_row, group_rows = paired_comparison(baseline_predictions, challenge_predictions, task_name)
    paired_rows.append(paired_row)
    group_comparison_rows.extend(group_rows)
paired_df = pd.DataFrame(paired_rows).sort_values("joint_Group_MAE_N").reset_index(drop=True)
group_comparison_df = pd.DataFrame(group_comparison_rows)

phase("FLAGSHIP_SEED_STABILITY")
stability_rows = []
stability_predictions = []
for stability_seed in STABILITY_SEEDS:
    seed_predictions = []
    for outer_label in outer_labels:
        test = hist[hist.Group_VW == outer_label].copy()
        train = hist[hist.Group_VW != outer_label].copy()
        model_name, params = baseline_selected_by_outer[str(outer_label)]
        final_prediction, augmentation_summary = fit_predict_outer(
            train,
            test,
            FLAGSHIP_TASK,
            model_name,
            params,
            stable_seed("stability_outer_fit", stability_seed, outer_label, FLAGSHIP_TASK),
            {
                "scope": "stability_outer_fit",
                "outer_label": str(outer_label),
                "inner_fold": None,
                "stability_seed": int(stability_seed),
            },
        )
        for row_position, (_, row) in enumerate(test.iterrows()):
            output_row = {
                "row_id": int(row.row_id),
                "Run_ID": str(row.Run_ID),
                "Group_VW": str(row.Group_VW),
                "task": FLAGSHIP_TASK,
                "stability_seed": int(stability_seed),
                "synthetic_rows_in_fit": int(augmentation_summary["synthetic_rows"]),
                "evaluation_row_is_real": True,
            }
            for target_index, target in enumerate(TARGET_NAMES):
                output_row[f"y_{target}"] = float(row[TARGET_COLS[target_index]])
                output_row[f"p_{target}"] = float(final_prediction[row_position, target_index])
            seed_predictions.append(output_row)
    seed_predictions = pd.DataFrame(seed_predictions)
    if seed_predictions.row_id.nunique() != len(hist):
        raise RuntimeError(f"INCOMPLETE_STABILITY_OOF:{stability_seed}")
    stability_predictions.append(seed_predictions)
    seed_metrics = metrics_block(seed_predictions)
    seed_pair, _ = paired_comparison(
        baseline_predictions,
        seed_predictions,
        f"{FLAGSHIP_TASK}_SEED_{stability_seed}",
    )
    stability_rows.append(
        {
            "stability_seed": int(stability_seed),
            **seed_metrics,
            "relative_improvement_vs_runtime_baseline_pct": seed_pair[
                "relative_improvement_vs_runtime_baseline_pct"
            ],
            "groups_improved": seed_pair["groups_improved"],
            "exact_signflip_one_sided_p": seed_pair["exact_signflip_one_sided_p"],
        }
    )
stability_df = pd.DataFrame(stability_rows)
stability_predictions_df = pd.concat(stability_predictions, ignore_index=True)
flagship_all_seeds_positive = bool(
    (stability_df.relative_improvement_vs_runtime_baseline_pct > 0).all()
)

# Fail closed if any evaluation row was synthetic or any target became a feature.
if not all_predictions.evaluation_row_is_real.all() or not stability_predictions_df.evaluation_row_is_real.all():
    LEAKAGE_STATS["synthetic_rows_in_evaluation"] += 1
    raise RuntimeError("SYNTHETIC_ROW_FOUND_IN_EVALUATION")
feature_union = set(itertools.chain.from_iterable(FEATURE_SETS.values()))
if feature_union & set(TARGET_COLS):
    LEAKAGE_STATS["current_pass_target_columns_in_feature_sets"] = True
    raise RuntimeError("CURRENT_PASS_TARGET_FEATURE_LEAKAGE")
if LEAKAGE_STATS["forbidden_source_intersections"] != 0:
    raise RuntimeError("FORBIDDEN_SOURCE_INTERSECTION_NONZERO")

nested_pair = paired_df[paired_df.task == "V61_NESTED_SELECTED"].iloc[0].to_dict()
flagship_pair = paired_df[paired_df.task == FLAGSHIP_TASK].iloc[0].to_dict()
best_descriptive = paired_df.iloc[0].to_dict()
nested_selection_df = pd.DataFrame(nested_selection_rows)
selection_counts = {
    "baseline_model_by_outer": dict(Counter(row["model"] for row in baseline_selection_rows)),
    "nested_task_by_outer": dict(Counter(nested_selection_df.selected_task)),
    "nested_model_by_outer": dict(Counter(nested_selection_df.model)),
}

leakage_audit = {
    "schema": "soilbin.q1.v6.1.leakage-audit",
    "passed": True,
    "outer_split": "leave-one-SpeedxLoad-group-out",
    "inner_split": "GroupKFold on remaining real groups",
    "synthetic_generation_scope": "training partition only",
    "evaluation_rows": "real observations only",
    "virtual_channels_are_observed_sensors": False,
    "virtual_channel_sources": VIRTUAL_CHANNEL_SOURCES,
    "current_pass_waveform_used": False,
    "current_pass_target_columns_used_as_features": False,
    "random_row_split_used": False,
    "external_labels_used_for_selection": False,
    "forbidden_source_intersections": int(LEAKAGE_STATS["forbidden_source_intersections"]),
    "synthetic_rows_in_evaluation": int(LEAKAGE_STATS["synthetic_rows_in_evaluation"]),
    "augmentation_calls": int(LEAKAGE_STATS["augmentation_calls"]),
    "synthetic_rows_generated_all_scopes": int(LEAKAGE_STATS["synthetic_rows_generated_all_scopes"]),
    "outer_provenance_rows_recorded": int(len(OUTER_PROVENANCE)),
}
dump(ROOT / "LEAKAGE_AUDIT_V6_1.json", leakage_audit)

# Persist evidence tables.
all_predictions.to_csv(TABLE / "oof_predictions_v6_1.csv", index=False)
pd.DataFrame(baseline_search_rows).to_csv(TABLE / "baseline_inner_search_v6_1.csv", index=False)
pd.DataFrame(nested_search_rows).to_csv(TABLE / "nested_task_model_search_v6_1.csv", index=False)
pd.DataFrame(baseline_selection_rows).to_csv(TABLE / "baseline_selection_v6_1.csv", index=False)
nested_selection_df.to_csv(TABLE / "nested_selection_v6_1.csv", index=False)
summary_df.to_csv(TABLE / "primary_summary_v6_1.csv", index=False)
paired_df.to_csv(TABLE / "paired_vs_baseline_v6_1.csv", index=False)
group_comparison_df.to_csv(TABLE / "paired_groups_v6_1.csv", index=False)
stability_df.to_csv(TABLE / "flagship_seed_stability_v6_1.csv", index=False)
stability_predictions_df.to_csv(TABLE / "flagship_seed_predictions_v6_1.csv", index=False)
pd.DataFrame(OUTER_PROVENANCE).to_csv(TABLE / "augmentation_provenance_outer_fits_v6_1.csv", index=False)

results = {
    "schema": "soilbin.q1.v6.1.virtual-synthetic",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "post_lock_exploratory": True,
    "does_not_supersede_v5": True,
    "source_model_sha256": EXPECTED_MODEL_SHA,
    "n_runs": 51,
    "n_history_pairs": 41,
    "n_groups": 9,
    "virtual_channels": VIRTUAL_CHANNEL_SOURCES,
    "experiment_matrix": EXPERIMENTS,
    "candidate_family": ["RandomForest", "ExtraTrees"],
    "candidate_config_count": len(CANDIDATES),
    "baseline_reproduction": {
        "task": "BASELINE_REAL_ONLY",
        "joint_Group_MAE_N": float(baseline_metrics.joint_Group_MAE),
        "LC1_Group_MAE_N": float(baseline_metrics.LC1_proxy_N_Group_MAE),
        "LC5_Group_MAE_N": float(baseline_metrics.LC5_proxy_N_Group_MAE),
        "frozen_v5_joint_Group_MAE_N": FROZEN_V5_JOINT_GROUP_MAE,
        "runtime_drift_pct": baseline_drift_pct,
        "exact_match_to_frozen_v5": baseline_exact_match,
        "within_2pct_of_frozen_v5": baseline_within_2pct,
    },
    "primary_nested_selected": nested_pair,
    "pre_registered_flagship": {
        **flagship_pair,
        "stability_seeds": list(STABILITY_SEEDS),
        "all_seed_improvements_positive": flagship_all_seeds_positive,
    },
    "best_descriptive_only": best_descriptive,
    "primary_summary": summary_df.to_dict(orient="records"),
    "paired_vs_runtime_baseline": paired_df.to_dict(orient="records"),
    "flagship_seed_stability": stability_df.to_dict(orient="records"),
    "selection_counts": selection_counts,
    "leakage_audit": leakage_audit,
    "guards": {
        "random_split_used": False,
        "external_labels_used_for_selection": False,
        "current_pass_waveform_used": False,
        "only_previous_pass_sensor_state_used_for_virtual_channels": True,
        "synthetic_rows_train_only": True,
        "evaluation_real_only": True,
        "response_status": RESPONSE_STATUS,
        "depth_mapping_status": DEPTH_STATUS,
    },
    "interpretation": (
        "Virtual channels are deterministic transforms of previous-pass LC1/LC5 and are not physical sensors. "
        "Synthetic rows are training-only interpolation/noise variants. V5 remains frozen regardless of V6.1 outcome "
        "until independent confirmation and an explicit new selection policy."
    ),
}
dump(ROOT / "RESULTS_V6_1.json", results)

# Figures.
plot_frame = summary_df.sort_values("joint_Group_MAE").reset_index(drop=True)
plt.figure(figsize=(13, 6))
plt.bar(np.arange(len(plot_frame)), plot_frame.joint_Group_MAE)
plt.axhline(FROZEN_V5_JOINT_GROUP_MAE, linestyle="--", linewidth=1)
plt.xticks(np.arange(len(plot_frame)), plot_frame.task, rotation=55, ha="right", fontsize=8)
plt.ylabel("Joint group-equal MAE (N)")
plt.tight_layout()
plt.savefig(FIG / "01_v6_1_primary_comparison.png", dpi=180)
plt.close()

focus = group_comparison_df[group_comparison_df.task.isin([FLAGSHIP_TASK, "V61_NESTED_SELECTED"])].copy()
if len(focus):
    pivot = focus.pivot(index="Group_VW", columns="task", values="improvement_N")
    pivot.plot(kind="bar", figsize=(11, 5))
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Improvement over real-only baseline (N)")
    plt.tight_layout()
    plt.savefig(FIG / "02_v6_1_group_improvements.png", dpi=180)
    plt.close()

report = [
    "# SoilBin V6.1 - Leakage-Safe Virtual Sensing and Synthetic Augmentation",
    "",
    "**Post-lock exploratory analysis; frozen V5 is not overwritten.**",
    "",
    f"- Frozen/runtime V5-family baseline: {float(baseline_metrics.joint_Group_MAE):.6f} N; drift={baseline_drift_pct:.6f}%.",
    f"- Primary nested-selected V6.1: {float(nested_pair['joint_Group_MAE_N']):.6f} N; improvement={float(nested_pair['relative_improvement_vs_runtime_baseline_pct']):.3f}%.",
    f"- Pre-registered flagship {FLAGSHIP_TASK}: {float(flagship_pair['joint_Group_MAE_N']):.6f} N; improvement={float(flagship_pair['relative_improvement_vs_runtime_baseline_pct']):.3f}%.",
    f"- Flagship positive across all augmentation seeds: {flagship_all_seeds_positive}.",
    f"- Leakage audit: PASS; forbidden-source intersections={LEAKAGE_STATS['forbidden_source_intersections']}; synthetic evaluation rows={LEAKAGE_STATS['synthetic_rows_in_evaluation']}.",
    "",
    "## Important interpretation",
    "- VS channels are mathematical transforms of previous LC1/LC5; they are not additional measured sensors.",
    "- Jitter and local mixup are generated only inside each training partition.",
    "- Validation and outer test partitions contain real observations only.",
    "- A better exploratory score does not supersede V5 without independent confirmation.",
    "",
    "## Experiment matrix",
]
for _, row in summary_df.iterrows():
    report.append(
        f"- {row.task}: joint Group-MAE={row.joint_Group_MAE:.6f} N; "
        f"LC1={row.LC1_proxy_N_Group_MAE:.6f}; LC5={row.LC5_proxy_N_Group_MAE:.6f}"
    )
report += ["", "## Paired comparisons"]
for _, row in paired_df.iterrows():
    report.append(
        f"- {row.task}: improvement={row.relative_improvement_vs_runtime_baseline_pct:.3f}%; "
        f"groups={int(row.groups_improved)}/9; p={row.exact_signflip_one_sided_p:.6f}; "
        f"acceptance={bool(row.meets_v61_exploratory_acceptance)}"
    )
(ROOT / "FINAL_REPORT_V6_1.md").write_text("\n".join(report), encoding="utf-8")

manifest = {
    "schema": "soilbin.q1.v6.1.run-manifest",
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "stability_seeds": list(STABILITY_SEEDS),
    "python": sys.version.split()[0],
    "source_sha256": EXPECTED_MODEL_SHA,
    "baseline_reproduction_exact": baseline_exact_match,
    "baseline_reproduction_within_2pct": baseline_within_2pct,
    "feature_sets": FEATURE_SETS,
    "virtual_channel_sources": VIRTUAL_CHANNEL_SOURCES,
    "experiments": EXPERIMENTS,
    "candidate_catalog": [{"model": name, "params": params} for name, params in CANDIDATES],
    "files": sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()),
}
dump(ROOT / "RUN_MANIFEST_V6_1.json", manifest)

shutil.make_archive(ARCHIVE_BASE, "zip", ROOT)
phase("COMPLETE")
print(
    "SOILBIN_V6_1_COMPLETE",
    json.dumps(
        {
            "baseline_joint_Group_MAE_N": float(baseline_metrics.joint_Group_MAE),
            "primary_nested_joint_Group_MAE_N": float(nested_pair["joint_Group_MAE_N"]),
            "primary_nested_improvement_pct": float(nested_pair["relative_improvement_vs_runtime_baseline_pct"]),
            "primary_nested_groups_improved": int(nested_pair["groups_improved"]),
            "flagship_joint_Group_MAE_N": float(flagship_pair["joint_Group_MAE_N"]),
            "flagship_all_seed_improvements_positive": flagship_all_seeds_positive,
            "best_descriptive_task": str(best_descriptive["task"]),
            "best_descriptive_mae_N": float(best_descriptive["joint_Group_MAE_N"]),
            "leakage_audit_passed": True,
        },
        sort_keys=True,
    ),
    flush=True,
)
