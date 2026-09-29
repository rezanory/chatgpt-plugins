from __future__ import annotations
import base64, gc, hashlib, json, math, os, re, shutil, zlib
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(os.environ.get("M07_EXTERNAL_WORK_ROOT", "/kaggle/working")).resolve()
INPUT = Path(os.environ.get("M07_EXTERNAL_INPUT_ROOT", "/kaggle/input")).resolve()
DATA_ROOT = Path(os.environ.get("M07_EXTERNAL_DATA_ROOT", str(INPUT))).resolve()
STATE_INPUT_ROOT = Path(os.environ.get("M07_EXTERNAL_STATE_ROOT", str(INPUT))).resolve()

SEED = 42
IMAGE_SIZE = int(os.environ.get("M07_EXTERNAL_RESOLUTION", "224"))
if IMAGE_SIZE not in {224, 320, 384}:
    raise RuntimeError(f"M07_EXTERNAL_RESOLUTION_INVALID={IMAGE_SIZE}")
OUT = WORK / f"M07_RSNA_PEDIATRIC_EXTERNAL_R{IMAGE_SIZE}_V1"
OUT.mkdir(parents=True, exist_ok=True)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 260927
STATE_HANDLES = {
    224: "rezanory/m07-final-5fold-fix2-d260914d",
    320: "trickermark/m07-gate-r320-state-v1-7",
    384: "trickermark/m07-gate-r384-state-v1-7",
}
STATE_HANDLE = os.environ.get("M07_EXTERNAL_STATE_HANDLE", STATE_HANDLES[IMAGE_SIZE])
if STATE_HANDLE != STATE_HANDLES[IMAGE_SIZE]:
    raise RuntimeError(
        f"M07_EXTERNAL_STATE_HANDLE_DRIFT resolution={IMAGE_SIZE} handle={STATE_HANDLE}"
    )
EVAL_BATCH_SIZE = int(
    os.environ.get(
        "M07_EXTERNAL_BATCH_SIZE",
        str({224: 12, 320: 6, 384: 4}[IMAGE_SIZE]),
    )
)
if EVAL_BATCH_SIZE < 1:
    raise RuntimeError("M07_EXTERNAL_BATCH_SIZE_INVALID")
EXPECTED_SPLIT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
EXPECTED_RECIPE = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
EXPECTED_RSNA_MANIFEST_SHA256 = "3450a101e626a09535d98e5aa9ca52dbf64ded584c2952f7613e3c22981cf5cd"
EXPECTED_RSNA_COUNTS = {
    "expanded_images": 1099,
    "expanded_patients": 553,
    "expanded_negative": 578,
    "expanded_positive": 521,
    "primary_images": 282,
    "primary_patients": 158,
    "primary_negative": 146,
    "primary_positive": 136,
}
RSNA_DATASET_REF = "nih-chest-xrays/data"
RSNA_POSITIVE_LABEL = "Adjudicated Lung Opacity"
RSNA_NEGATIVE_LABEL = "Normal"

print("CGP_PHASE:M07_RSNA_EXTERNAL_BOOT", flush=True)
print(json.dumps({
    "data_root": str(DATA_ROOT),
    "external_dataset": RSNA_DATASET_REF,
    "state_input_root": str(STATE_INPUT_ROOT),
    "work_root": str(WORK),
    "resolution": IMAGE_SIZE,
}, sort_keys=True), flush=True)


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def locate_state_root():
    roots = []
    if IMAGE_SIZE == 224:
        for marker in STATE_INPUT_ROOT.rglob("PERSISTENCE_MANIFEST_V1_4.json"):
            root = marker.parent.resolve()
            if all(
                (
                    root
                    / f"M07_FOLD_{fold}_RECOVERY"
                    / "FINAL_5FOLD"
                    / f"fold_{fold}"
                    / "best.weights.h5"
                ).is_file()
                for fold in range(1, 6)
            ):
                roots.append(root)
    else:
        expected_artifacts = {f"FOLD_{fold}_RECOVERY.zip" for fold in range(1, 6)}
        for marker in STATE_INPUT_ROOT.rglob("CAMPAIGN_STATE.json"):
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                continue
            contract = payload.get("run_contract") or {}
            artifacts = payload.get("artifact_sha256") or {}
            valid = (
                payload.get("schema") == "phase2.state.v2"
                and payload.get("status") == "COMPLETE"
                and payload.get("model_id") == "M07"
                and int(payload.get("resolution", -1)) == IMAGE_SIZE
                and contract.get("split_fingerprint") == EXPECTED_SPLIT
                and ((contract.get("extra") or {}).get("recipe_fingerprint"))
                    == EXPECTED_RECIPE
                and set(artifacts) == expected_artifacts
                and all(
                    re.fullmatch(r"[0-9a-f]{64}", str(v or ""))
                    for v in artifacts.values()
                )
            )
            if not valid:
                continue
            root = marker.parent.resolve()
            if all(
                (
                    root
                    / f"FOLD_{fold}_RECOVERY"
                    / "FOLDS"
                    / f"fold_{fold}"
                    / "COMPLETED.json"
                ).is_file()
                and (
                    root
                    / f"FOLD_{fold}_RECOVERY"
                    / "FOLDS"
                    / f"fold_{fold}"
                    / "final_selected.weights.h5"
                ).is_file()
                for fold in range(1, 6)
            ):
                roots.append(root)

    roots = sorted(set(roots), key=str)
    if len(roots) != 1:
        raise RuntimeError(
            f"M07_STATE_ROOT_COUNT={len(roots)} resolution={IMAGE_SIZE} handle={STATE_HANDLE}"
        )
    return roots[0]


def build_manifest(state_root):
    if "EMBEDDED_MANIFEST_B64" not in globals():
        raise RuntimeError("RSNA_EMBEDDED_MANIFEST_MISSING")
    try:
        payload = zlib.decompress(base64.b64decode(EMBEDDED_MANIFEST_B64.encode("ascii")))
        rows = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"RSNA_MANIFEST_DECODE_FAILED={type(exc).__name__}") from exc

    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    manifest_sha = hashlib.sha256(canonical).hexdigest()
    if manifest_sha != EXPECTED_RSNA_MANIFEST_SHA256:
        raise RuntimeError("RSNA_MANIFEST_SHA_MISMATCH")

    manifest = pd.DataFrame(rows)
    required_columns = {
        "img_id", "StudyInstanceUID", "SOPInstanceUID", "patient_id", "age",
        "sex", "view", "label", "class_name", "primary_peds_lt10",
        "expanded_peds_le18",
    }
    if not required_columns.issubset(set(manifest.columns)):
        raise RuntimeError(
            "RSNA_MANIFEST_COLUMNS_MISSING="
            + json.dumps(sorted(required_columns - set(manifest.columns)))
        )
    if len(manifest) != EXPECTED_RSNA_COUNTS["expanded_images"]:
        raise RuntimeError(f"RSNA_MANIFEST_N={len(manifest)}")
    if manifest.img_id.astype(str).duplicated().any():
        raise RuntimeError("RSNA_MANIFEST_DUPLICATE_IMAGE_ID")

    manifest["label"] = manifest["label"].astype(int)
    manifest["age"] = manifest["age"].astype(int)
    manifest["patient_id"] = manifest["patient_id"].astype(int)
    manifest["primary_peds_lt10"] = manifest["primary_peds_lt10"].astype(bool)
    manifest["expanded_peds_le18"] = manifest["expanded_peds_le18"].astype(bool)
    manifest["image_id"] = manifest["img_id"].astype(str)

    counts = {
        "expanded_images": int(len(manifest)),
        "expanded_patients": int(manifest.patient_id.nunique()),
        "expanded_negative": int((manifest.label == 0).sum()),
        "expanded_positive": int((manifest.label == 1).sum()),
        "primary_images": int(manifest.primary_peds_lt10.sum()),
        "primary_patients": int(
            manifest.loc[manifest.primary_peds_lt10, "patient_id"].nunique()
        ),
        "primary_negative": int(
            ((manifest.primary_peds_lt10) & (manifest.label == 0)).sum()
        ),
        "primary_positive": int(
            ((manifest.primary_peds_lt10) & (manifest.label == 1)).sum()
        ),
    }
    if counts != EXPECTED_RSNA_COUNTS:
        raise RuntimeError(
            "RSNA_MANIFEST_COUNTS_MISMATCH=" + json.dumps(counts, sort_keys=True)
        )

    required_ids = set(manifest.image_id.astype(str))
    image_paths = {}
    for path in DATA_ROOT.rglob("*.png"):
        if path.name not in required_ids:
            continue
        rp = path.resolve()
        if state_root in rp.parents:
            continue
        image_paths.setdefault(path.name, []).append(rp)
    missing = sorted(required_ids - set(image_paths))
    ambiguous = {
        key: len(value) for key, value in image_paths.items() if len(value) != 1
    }
    if missing or ambiguous:
        raise RuntimeError(
            "RSNA_IMAGE_RESOLUTION_MISMATCH="
            + json.dumps(
                {"resolved": len(image_paths), "missing": missing[:25], "ambiguous": ambiguous},
                sort_keys=True,
            )
        )

    metadata_files = [
        p.resolve()
        for p in DATA_ROOT.rglob("Data_Entry_2017.csv")
        if state_root not in p.resolve().parents
    ]
    if len(metadata_files) != 1:
        raise RuntimeError(f"RSNA_NIH_METADATA_COUNT={len(metadata_files)}")
    metadata_path = metadata_files[0]
    metadata = pd.read_csv(metadata_path)
    rename = {
        "Image Index": "img_id",
        "Patient Age": "age",
        "Patient ID": "patient_id",
        "View Position": "view",
        "Patient Gender": "sex",
    }
    if not set(rename).issubset(set(metadata.columns)):
        raise RuntimeError("RSNA_NIH_METADATA_SCHEMA_INVALID")
    metadata = metadata.rename(columns=rename)
    metadata = metadata[metadata.img_id.astype(str).isin(required_ids)].copy()
    if len(metadata) != len(manifest) or metadata.img_id.astype(str).duplicated().any():
        raise RuntimeError(f"RSNA_NIH_METADATA_MATCH_COUNT={len(metadata)}")

    check = manifest[
        ["img_id", "age", "patient_id", "view", "sex"]
    ].merge(
        metadata[["img_id", "age", "patient_id", "view", "sex"]],
        on="img_id",
        how="left",
        suffixes=("_manifest", "_nih"),
        validate="one_to_one",
    )
    for key in ("age", "patient_id", "view", "sex"):
        left = check[f"{key}_manifest"].astype(str)
        right = check[f"{key}_nih"].astype(str)
        if not left.equals(right):
            raise RuntimeError(f"RSNA_NIH_METADATA_DRIFT={key}")

    manifest["filepath"] = [
        str(image_paths[image_id][0]) for image_id in manifest.image_id.astype(str)
    ]
    manifest["sha256_external_image"] = [
        sha256_file(Path(path)) for path in manifest.filepath
    ]
    if manifest.sha256_external_image.duplicated().any():
        raise RuntimeError("RSNA_EXTERNAL_DUPLICATE_IMAGE_SHA")

    internal_hashes = set()
    oof_files = list(state_root.rglob("M07_OOF_PREDICTIONS.csv"))
    if oof_files:
        oof = pd.read_csv(sorted(oof_files, key=lambda p: len(str(p)))[0])
        if "sha256" in oof:
            internal_hashes = set(oof.sha256.dropna().astype(str))
    overlap = set(manifest.sha256_external_image) & internal_hashes
    if overlap:
        raise RuntimeError(f"RSNA_INTERNAL_EXTERNAL_SHA_OVERLAP={len(overlap)}")

    meta = {
        "dataset_ref": RSNA_DATASET_REF,
        "manifest_sha256": manifest_sha,
        "metadata_file": metadata_path.relative_to(DATA_ROOT).as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "counts": counts,
        "exact_internal_external_sha_overlap": 0,
        "unique_external_image_sha256": int(manifest.sha256_external_image.nunique()),
        "label_policy": {
            "source": "RSNA Pneumonia Detection Challenge calculated/adjudicated labels",
            "negative": RSNA_NEGATIVE_LABEL,
            "positive": RSNA_POSITIVE_LABEL,
            "excluded": ["No Lung Opacity / Not Normal", "Unlabeled"],
            "primary_age": "1-9 years",
            "expanded_age": "1-18 years",
            "frozen_before_inference": True,
            "phenotype_note": (
                "RSNA positive endpoint is adjudicated lung opacity, a pneumonia-related "
                "radiographic phenotype; it is not an exact pneumonia diagnosis label."
            ),
        },
    }
    return manifest.sort_values("image_id").reset_index(drop=True), meta


def preprocess_manifest(manifest):
    out = manifest.copy()
    patient_nonempty = int(out.patient_id.notna().sum())
    study_nonempty = int(out.StudyInstanceUID.astype(str).str.strip().ne("").sum())
    return out, {
        "policy": (
            "NIH ChestX-ray14 PNG as distributed by nih-chest-xrays/data -> "
            f"canonical M07 resize_with_pad {IMAGE_SIZE} bilinear antialias; no "
            "external calibration, enhancement fitting, or label-dependent preprocessing"
        ),
        "identity_evidence": {
            "patient_id_nonempty": patient_nonempty,
            "patient_id_unique_nonempty": int(out.patient_id.nunique()),
            "study_instance_uid_nonempty": study_nonempty,
            "study_instance_uid_unique_nonempty": int(out.StudyInstanceUID.nunique()),
            "total": int(len(out)),
        },
    }


import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, fbeta_score,
    matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
mixed_precision.set_global_policy("mixed_float16")


@tf.keras.utils.register_keras_serializable(package="PneumoniaAI")
class EdgeBlock(tf.keras.layers.Layer):
    def __init__(self, filters=16, max_gate=0.25, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.max_gate = int(filters), float(max_gate)
        self.conv1 = tf.keras.layers.Conv2D(
            self.filters, 3, padding="same", activation="gelu", dtype="float32"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            3, 1, padding="same", activation="tanh", dtype="float32"
        )

    def build(self, input_shape):
        self.raw_gate = self.add_weight(
            name="raw_gate", shape=(),
            initializer=tf.keras.initializers.Constant(-4.0),
            trainable=True, dtype="float32",
        )
        super().build(input_shape)

    @staticmethod
    def norm(t):
        lo = tf.stop_gradient(tf.reduce_min(t, axis=[1, 2, 3], keepdims=True))
        hi = tf.stop_gradient(tf.reduce_max(t, axis=[1, 2, 3], keepdims=True))
        return tf.math.divide_no_nan(t - lo, tf.maximum(hi - lo, tf.constant(1e-3, tf.float32)))

    def call(self, inputs):
        dtype = inputs.dtype
        x = tf.cast(inputs, tf.float32)
        gray = tf.image.rgb_to_grayscale(x) / 255.0
        sobel = tf.image.sobel_edges(gray)
        kernel = tf.reshape(
            tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], tf.float32),
            [3, 3, 1, 1],
        )
        lap = tf.abs(tf.nn.conv2d(gray, kernel, strides=1, padding="SAME"))
        edge = tf.concat([self.norm(tf.abs(sobel[..., 1])),
                          self.norm(tf.abs(sobel[..., 0])),
                          self.norm(lap)], axis=-1)
        delta = tf.cast(self.conv2(self.conv1(edge)), tf.float32) * 255.0
        gate = tf.nn.sigmoid(tf.cast(self.raw_gate, tf.float32)) * self.max_gate
        return tf.cast(tf.clip_by_value(x + gate * delta, 0.0, 255.0), dtype)


@tf.keras.utils.register_keras_serializable(package="PneumoniaAI")
class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction=16, spatial_kernel=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction, self.spatial_kernel = int(reduction), int(spatial_kernel)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(channels // self.reduction, 8)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu", use_bias=True, dtype="float32")
        self.fc2 = tf.keras.layers.Dense(channels, use_bias=True, dtype="float32")
        self.spatial = tf.keras.layers.Conv2D(
            1, self.spatial_kernel, padding="same", activation="sigmoid",
            use_bias=False, dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        dtype = inputs.dtype
        x = tf.cast(inputs, tf.float32)
        ca = tf.nn.sigmoid(
            self.fc2(self.fc1(tf.reduce_mean(x, axis=[1, 2])))
            + self.fc2(self.fc1(tf.reduce_max(x, axis=[1, 2])))
        )[:, None, None, :]
        x = x * ca
        sa = self.spatial(tf.concat(
            [tf.reduce_mean(x, axis=-1, keepdims=True),
             tf.reduce_max(x, axis=-1, keepdims=True)], axis=-1
        ))
        return tf.cast(x * sa, dtype)


def augmentation_block(seed):
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.025, fill_mode="reflect", seed=seed + 101),
        tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="reflect", seed=seed + 102),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05),
            fill_mode="reflect", seed=seed + 103,
        ),
        tf.keras.layers.RandomContrast(0.08, seed=seed + 104),
        tf.keras.layers.ReLU(max_value=255.0, name="clip_augmented_pixels"),
    ], name="augmentation")


def build_m07(params, seed):
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32, name="image")
    x = augmentation_block(seed)(inputs)
    x = EdgeBlock(int(params["edge_filters"]), float(params["edge_max_gate"]), name="edge_block")(x)
    import inspect
    kwargs = dict(include_top=False, weights=None, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling=None)
    if "include_preprocessing" in inspect.signature(tf.keras.applications.ConvNeXtTiny).parameters:
        kwargs["include_preprocessing"] = True
    backbone = tf.keras.applications.ConvNeXtTiny(**kwargs)
    x = backbone(x)
    x = CBAM(int(params["cbam_reduction"]), 7, name="cbam")(x)
    x = tf.keras.layers.Activation("linear", name="aez_spatial_features")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32", name="head_norm")(x)
    x = tf.keras.layers.Dropout(float(params["dropout"]), name="head_dropout")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(x)
    return tf.keras.Model(inputs, out, name="M07")


AUTOTUNE = tf.data.AUTOTUNE


def decode_resize(path, label):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(
        image, IMAGE_SIZE, IMAGE_SIZE, method="bilinear", antialias=True
    )
    return tf.clip_by_value(image, 0.0, 255.0), tf.cast(label, tf.float32)


def build_eval_dataset(frame, batch_size):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = tf.data.Dataset.from_tensor_slices(
        (frame.filepath.astype(str).to_numpy(), frame.label.astype(np.float32).to_numpy())
    ).with_options(opts)
    ds = ds.map(decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds.batch(int(batch_size), drop_remainder=False).prefetch(AUTOTUNE)


def load_folds(state_root):
    params0, thresholds, weights, receipts = None, {}, {}, {}
    for fold in range(1, 6):
        if IMAGE_SIZE == 224:
            root = state_root / f"M07_FOLD_{fold}_RECOVERY/FINAL_5FOLD/fold_{fold}"
            receipt = json.loads((root / "COMPLETED.json").read_text(encoding="utf-8"))
            weight = root / "best.weights.h5"
            valid = (
                receipt.get("schema") == "m07.final.fold.v1.6"
                and receipt.get("status") == "COMPLETED"
                and int(receipt.get("fold_id", -1)) == fold
                and receipt.get("split_fingerprint") == EXPECTED_SPLIT
                and receipt.get("hpo_recipe_fingerprint") == EXPECTED_RECIPE
                and receipt.get("locked_test_used") is False
                and int(((receipt.get("run_contract") or {}).get("resolution", -1))) == 224
            )
            params = receipt.get("params") or {}
            fixed = receipt.get("m07_fixed") or {}
            metrics_payload = receipt.get("metrics") or {}
            expected_weight_sha = (receipt.get("artifact_sha256") or {}).get("weights")
        else:
            root = state_root / f"FOLD_{fold}_RECOVERY/FOLDS/fold_{fold}"
            receipt = json.loads((root / "COMPLETED.json").read_text(encoding="utf-8"))
            weight = root / "final_selected.weights.h5"
            valid = (
                receipt.get("schema") == "pneumonia.phase2.fold.v1.7"
                and receipt.get("status") == "COMPLETED"
                and receipt.get("model_id") == "M07"
                and int(receipt.get("resolution", -1)) == IMAGE_SIZE
                and int(receipt.get("fold_id", -1)) == fold
                and receipt.get("split_fingerprint") == EXPECTED_SPLIT
                and receipt.get("confirmed_m07_recipe_fingerprint") == EXPECTED_RECIPE
                and receipt.get("locked_test_used_for_training") is False
                and receipt.get("external_used_for_training") is False
            )
            params = receipt.get("shared_training_params") or {}
            fixed = receipt.get("fixed_identity") or {}
            metrics_payload = receipt.get("validation_metrics") or {}
            expected_weight_sha = (
                receipt.get("artifact_sha256") or {}
            ).get("final_selected.weights.h5")

        if not weight.is_file():
            raise RuntimeError(f"FOLD_{fold}_WEIGHTS_MISSING_R{IMAGE_SIZE}")
        if not valid:
            raise RuntimeError(f"FOLD_{fold}_RECEIPT_INVALID_R{IMAGE_SIZE}")
        actual_sha = sha256_file(weight)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(expected_weight_sha or ""))
            or actual_sha != expected_weight_sha
        ):
            raise RuntimeError(f"FOLD_{fold}_WEIGHT_SHA_MISMATCH_R{IMAGE_SIZE}")
        if params0 is None:
            params0 = params
        elif params != params0:
            raise RuntimeError(f"FOLD_{fold}_PARAM_DRIFT_R{IMAGE_SIZE}")
        if [
            fixed.get("edge_filters"),
            fixed.get("edge_max_gate"),
            fixed.get("cbam_reduction"),
            fixed.get("cbam_spatial_kernel"),
        ] != [16, 0.25, 32, 7]:
            raise RuntimeError(f"FOLD_{fold}_M07_FIXED_DRIFT_R{IMAGE_SIZE}")
        t = float(metrics_payload.get("threshold"))
        if not 0.0 < t < 1.0:
            raise RuntimeError(f"FOLD_{fold}_THRESHOLD_INVALID_R{IMAGE_SIZE}")
        thresholds[fold], weights[fold] = t, weight
        receipts[fold] = {
            "schema": receipt.get("schema"),
            "resolution": IMAGE_SIZE,
            "threshold": t,
            "validation_metrics": metrics_payload,
            "weights_file": weight.name,
            "weights_sha256": actual_sha,
            "receipt_sha256": receipt.get("receipt_sha256"),
            "run_fingerprint": receipt.get("run_fingerprint"),
        }
    return params0, thresholds, weights, receipts


def logit_np(x):
    x = np.clip(np.asarray(x, float), 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def metrics(y, pred, score):
    y, pred, score = np.asarray(y, int), np.asarray(pred, int), np.asarray(score, float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_normal": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_normal": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1_normal": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "precision_positive": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc": float(roc_auc_score(y, score)),
        "auprc_positive": float(average_precision_score(y, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap(frame):
    y = frame.label.to_numpy(int)
    pred = frame.prediction_primary_normalized.to_numpy(int)
    score = frame.normalized_ensemble_score.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    keys = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "f2", "mcc", "auroc", "auprc_positive",
        "precision_normal", "recall_normal", "precision_positive", "recall_positive",
    ]
    vals = {k: [] for k in keys}

    cluster_column = None
    unit = "EXAM_IMAGE"
    for candidate, candidate_unit in (
        ("patient_id", "PATIENT_CLUSTER"),
        ("study_instance_uid", "STUDY_CLUSTER"),
    ):
        if candidate not in frame.columns:
            continue
        values = frame[candidate].fillna("").astype(str).str.strip()
        if len(values) == len(frame) and values.ne("").all():
            cluster_column = candidate
            unit = candidate_unit
            break

    if cluster_column:
        groups = frame[cluster_column].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        group_indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
        for _ in range(BOOTSTRAPS):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_indices[g] for g in sampled_groups])
            if len(np.unique(y[idx])) < 2:
                continue
            m = metrics(y[idx], pred[idx], score[idx])
            for k in keys:
                vals[k].append(m[k])
    else:
        for _ in range(BOOTSTRAPS):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            m = metrics(y[idx], pred[idx], score[idx])
            for k in keys:
                vals[k].append(m[k])

    return {
        "unit": unit,
        "cluster_column": cluster_column,
        "patient_cluster_bootstrap": cluster_column == "patient_id",
        "study_cluster_bootstrap": cluster_column == "study_instance_uid",
        "requested": BOOTSTRAPS,
        "valid": len(next(iter(vals.values()))),
        "95ci": {
            k: {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))}
            for k, v in vals.items() if v
        },
    }


def ece15(y, p):
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 0, 1)
    edges, total, rows = np.linspace(0, 1, 16), 0.0, []
    for i in range(15):
        mask = (p >= edges[i]) & (p <= edges[i + 1] if i == 14 else p < edges[i + 1])
        if not mask.any():
            continue
        conf, obs = float(p[mask].mean()), float(y[mask].mean())
        total += float(mask.mean()) * abs(conf - obs)
        rows.append({"bin": i, "n": int(mask.sum()), "mean_probability": conf, "observed": obs})
    return float(total), rows


def evaluate(frame, name, positive_label_name="Positive"):
    y = frame.label.to_numpy(int)
    pred = frame.prediction_primary_normalized.to_numpy(int)
    score = frame.normalized_ensemble_score.to_numpy(float)
    result = metrics(y, pred, score)
    result["brier_mean_probability"] = float(brier_score_loss(y, frame.mean_probability))
    ece, bins = ece15(y, frame.mean_probability)
    result["ece15_mean_probability"] = ece
    ci = bootstrap(frame)

    fpr, tpr, rt = roc_curve(y, score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": rt}).to_csv(
        OUT / f"{name}_ROC.csv", index=False
    )
    precision, recall, pt = precision_recall_curve(y, score)
    pr = pd.DataFrame({"precision": precision, "recall": recall})
    pr["threshold"] = np.nan
    if len(pt):
        pr.loc[:len(pt)-1, "threshold"] = pt
    pr.to_csv(OUT / f"{name}_PR.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / f"{name}_CALIBRATION_BINS.csv", index=False)

    import matplotlib.pyplot as plt
    cm = np.asarray(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], [RSNA_NEGATIVE_LABEL, positive_label_name])
    ax.set_yticks([0, 1], [RSNA_NEGATIVE_LABEL, positive_label_name])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(OUT / f"{name}_CONFUSION_MATRIX.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"AUROC={result['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} ROC"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / f"{name}_ROC.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, label=f"AUPRC={result['auprc_positive']:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{name} Precision-Recall"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / f"{name}_PR.png", dpi=180); plt.close(fig)
    return {"metrics": result, "bootstrap": ci}


state_root = locate_state_root()
params, thresholds, weights, fold_receipts = load_folds(state_root)
manifest, manifest_meta = build_manifest(state_root)
manifest, presentation = preprocess_manifest(manifest)
manifest.to_csv(OUT / "M07_RSNA_PEDIATRIC_INFERENCE_MANIFEST.csv", index=False)

primary_manifest = manifest[manifest.primary_peds_lt10].copy()
primary_counts = {
    int(k): int(v)
    for k, v in primary_manifest.label.value_counts().sort_index().to_dict().items()
}
expanded_counts = {
    int(k): int(v)
    for k, v in manifest.label.value_counts().sort_index().to_dict().items()
}
if (
    len(primary_manifest) != EXPECTED_RSNA_COUNTS["primary_images"]
    or primary_counts != {
        0: EXPECTED_RSNA_COUNTS["primary_negative"],
        1: EXPECTED_RSNA_COUNTS["primary_positive"],
    }
):
    raise RuntimeError(
        "RSNA_PRIMARY_MANIFEST_MISMATCH="
        + json.dumps({"n": len(primary_manifest), "counts": primary_counts}, sort_keys=True)
    )
if (
    len(manifest) != EXPECTED_RSNA_COUNTS["expanded_images"]
    or expanded_counts != {
        0: EXPECTED_RSNA_COUNTS["expanded_negative"],
        1: EXPECTED_RSNA_COUNTS["expanded_positive"],
    }
):
    raise RuntimeError(
        "RSNA_EXPANDED_MANIFEST_MISMATCH="
        + json.dumps({"n": len(manifest), "counts": expanded_counts}, sort_keys=True)
    )
primary_manifest.to_csv(OUT / "M07_RSNA_PEDIATRIC_PRIMARY_LT10_MANIFEST.csv", index=False)

integrity_receipt = {
    "schema": "m07.external.rsna_pediatric.pre_inference_integrity.v1",
    "status": "PASS_PREINFERENCE_INTEGRITY",
    "resolution": IMAGE_SIZE,
    "manifest_sha256": manifest_meta["manifest_sha256"],
    "counts": manifest_meta["counts"],
    "external_exact_duplicate_image_sha": 0,
    "internal_external_exact_sha_overlap": manifest_meta["exact_internal_external_sha_overlap"],
    "metadata_sha256": manifest_meta["metadata_sha256"],
    "training_performed": False,
    "hpo_performed": False,
    "inference_started": False,
    "external_threshold_tuning": False,
    "external_adaptation": False,
}
(OUT / "M07_RSNA_PREINFERENCE_INTEGRITY_RECEIPT.json").write_text(
    json.dumps(safe_json(integrity_receipt), indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("CGP_PHASE:M07_RSNA_INFERENCE", flush=True)
ds = build_eval_dataset(manifest, EVAL_BATCH_SIZE)
probs = []
for fold in range(1, 6):
    print(f"CGP_PHASE:M07_RSNA_FOLD {fold}/5", flush=True)
    tf.keras.backend.clear_session(); gc.collect()
    model = build_m07(params, SEED + fold)
    model.load_weights(weights[fold])
    p = model.predict(ds, verbose=0).reshape(-1).astype(float)
    if len(p) != len(manifest) or not np.isfinite(p).all():
        raise RuntimeError(f"FOLD_{fold}_PREDICTIONS_INVALID")
    probs.append(p)
    manifest[f"probability_fold_{fold}"] = p
    manifest[f"threshold_fold_{fold}"] = thresholds[fold]
    manifest[f"vote_fold_{fold}"] = (p >= thresholds[fold]).astype(int)
    del model
    tf.keras.backend.clear_session(); gc.collect()

P = np.vstack(probs)
scores = np.vstack([logit_np(P[i]) - logit_np(thresholds[i + 1]) for i in range(5)])
votes = np.vstack([(P[i] >= thresholds[i + 1]).astype(int) for i in range(5)])
manifest["mean_probability"] = P.mean(axis=0)
manifest["normalized_ensemble_score"] = scores.mean(axis=0)
manifest["prediction_primary_normalized"] = (
    manifest.normalized_ensemble_score >= 0
).astype(int)
manifest["prediction_majority_vote"] = (votes.sum(axis=0) >= 3).astype(int)
manifest.to_csv(OUT / "M07_RSNA_PEDIATRIC_EXTERNAL_PREDICTIONS.csv", index=False)

primary_frame = manifest[manifest.primary_peds_lt10].copy()
expanded_frame = manifest.copy()
primary = evaluate(
    primary_frame, "PRIMARY_RSNA_PEDIATRIC_LT10", RSNA_POSITIVE_LABEL
)
expanded = evaluate(
    expanded_frame, "EXPANDED_RSNA_PEDIATRIC_LE18", RSNA_POSITIVE_LABEL
)
primary_frame.to_csv(
    OUT / "M07_RSNA_PEDIATRIC_PRIMARY_LT10_PREDICTIONS.csv", index=False
)
expanded_frame.to_csv(
    OUT / "M07_RSNA_PEDIATRIC_EXPANDED_LE18_PREDICTIONS.csv", index=False
)

oof_metrics = None
oof_files = list(state_root.rglob("M07_OOF_PRIMARY_METRICS.json"))
if oof_files:
    oof_metrics = json.loads(
        sorted(oof_files, key=lambda p: len(str(p)))[0].read_text(encoding="utf-8")
    )

report = {
    "schema": "m07.external.rsna_pediatric.resolution.v1",
    "status": "SCIENTIFIC_RECEIPT_PASS",
    "model": "M07 Final Gate - ConvNeXt-Tiny + EdgeBlock + CBAM + FLSD-53",
    "resolution": IMAGE_SIZE,
    "source_state": STATE_HANDLE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "split_fingerprint": EXPECTED_SPLIT,
    "hpo_recipe_fingerprint": EXPECTED_RECIPE,
    "external_dataset": RSNA_DATASET_REF,
    "external_manifest_sha256": EXPECTED_RSNA_MANIFEST_SHA256,
    "training_performed": False,
    "hpo_performed": False,
    "external_threshold_tuning": False,
    "external_adaptation": False,
    "external_calibration_fitting": False,
    "aggregation_primary": (
        "mean(logit(p_fold)-logit(validation_threshold_fold)); decision >= 0"
    ),
    "aggregation_secondary": (
        "majority vote using the same five frozen internal-validation thresholds"
    ),
    "fold_thresholds": {str(k): float(v) for k, v in thresholds.items()},
    "fold_receipts": fold_receipts,
    "manifest": manifest_meta,
    "presentation": presentation,
    "primary_pediatric_lt10": primary,
    "expanded_pediatric_le18": expanded,
    "internal_training_oof_reference": oof_metrics,
    "limitations": [
        (
            "The RSNA positive endpoint is adjudicated Lung Opacity rather than an exact "
            "pneumonia diagnosis. This is a prespecified independent radiographic "
            "domain-shift validation and must not be described as exact pneumonia-label "
            "replication."
        ),
        (
            "Primary cohort (age 1-9) and expanded cohort (age 1-18) were frozen before "
            "inference using official RSNA mappings/adjudicated calculated labels plus "
            "NIH age metadata."
        ),
        (
            "No external labels are used for training, model selection, threshold "
            "selection, adaptation, or calibration fitting."
        ),
        (
            "All five frozen fold weights and their original internal validation "
            "thresholds are reused unchanged."
        ),
    ],
}
(OUT / "M07_RSNA_PEDIATRIC_EXTERNAL_REPORT.json").write_text(
    json.dumps(safe_json(report), indent=2, ensure_ascii=False), encoding="utf-8"
)
(OUT / "M07_INTERNAL_RSNA_EXTERNAL_COMPARISON.json").write_text(
    json.dumps(
        safe_json(
            {
                "internal_training_oof_reference": oof_metrics,
                "external_primary_pediatric_lt10": primary["metrics"],
                "comparison_warning": (
                    "Do not interpret metric differences as a same-endpoint generalization "
                    "gap because RSNA uses adjudicated Lung Opacity rather than exact "
                    "pneumonia diagnosis."
                ),
            }
        ),
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

model_card = (
    "# M07 External Validation Addendum - RSNA Pediatric Cohort\n\n"
    f"Resolution: {IMAGE_SIZE}\n\n"
    f"Frozen recipe fingerprint: {EXPECTED_RECIPE}\n\n"
    f"External dataset: {RSNA_DATASET_REF}\n\n"
    "Training/HPO/adaptation/calibration/threshold tuning on external: NO\n\n"
    "Primary prespecified cohort: age 1-9, adjudicated Lung Opacity vs Normal\n\n"
    f"Primary N: {len(primary_frame)} "
    f"({primary_counts[0]} Normal, {primary_counts[1]} Lung Opacity)\n\n"
    "Expanded prespecified cohort: age 1-18, adjudicated Lung Opacity vs Normal\n\n"
    f"Expanded N: {len(expanded_frame)} "
    f"({expanded_counts[0]} Normal, {expanded_counts[1]} Lung Opacity)\n\n"
    "Important: Lung Opacity is a pneumonia-related radiographic phenotype, not an "
    "exact pneumonia diagnosis label.\n\n"
    "Primary metrics:\n\n"
    + json.dumps(safe_json(primary["metrics"]), indent=2)
    + "\n"
)
(OUT / "M07_MODEL_CARD_RSNA_EXTERNAL_ADDENDUM.md").write_text(
    model_card, encoding="utf-8"
)

hashes = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        hashes[p.relative_to(OUT).as_posix()] = sha256_file(p)

receipt = {
    "schema": "m07.external.rsna_pediatric.terminal.v1",
    "status": "SCIENTIFIC_RECEIPT_PASS",
    "resolution": IMAGE_SIZE,
    "primary_n": len(primary_frame),
    "primary_normal": int(primary_counts[0]),
    "primary_lung_opacity": int(primary_counts[1]),
    "expanded_n": len(expanded_frame),
    "expanded_normal": int(expanded_counts[0]),
    "expanded_lung_opacity": int(expanded_counts[1]),
    "training_performed": False,
    "hpo_performed": False,
    "external_threshold_tuning": False,
    "external_adaptation": False,
    "external_calibration_fitting": False,
    "source_state": STATE_HANDLE,
    "split_fingerprint": EXPECTED_SPLIT,
    "hpo_recipe_fingerprint": EXPECTED_RECIPE,
    "external_dataset": RSNA_DATASET_REF,
    "external_manifest_sha256": EXPECTED_RSNA_MANIFEST_SHA256,
    "artifact_sha256": hashes,
}
body = json.dumps(safe_json(receipt), sort_keys=True, separators=(",", ":")).encode()
receipt["receipt_sha256"] = hashlib.sha256(body).hexdigest()
(OUT / "M07_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)

zip_path = Path(
    shutil.make_archive(
        str(WORK / f"M07_RSNA_PEDIATRIC_EXTERNAL_R{IMAGE_SIZE}_V1_COMPLETE"),
        "zip",
        root_dir=OUT,
    )
)
print("CGP_PHASE:M07_RSNA_EXTERNAL_COMPLETE", flush=True)
print(
    json.dumps(
        {
            "status": "SCIENTIFIC_RECEIPT_PASS",
            "resolution": IMAGE_SIZE,
            "primary_n": len(primary_frame),
            "expanded_n": len(expanded_frame),
            "zip": str(zip_path),
            "zip_sha256": sha256_file(zip_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        sort_keys=True,
    ),
    flush=True,
)
