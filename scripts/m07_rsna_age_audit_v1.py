from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import random
import statistics
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Iterable

READ_ENDPOINT = (
    "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/"
    "control-plane/v3/read/kaggle"
)
DEFAULT_ACCOUNT = "kg-03"
DEFAULT_KERNEL = "rezanory/m07-rsna-pediatric-external-r224-v1-36572524789"
PREDICTIONS_SUFFIX = "M07_RSNA_PEDIATRIC_EXTERNAL_PREDICTIONS.csv"
REQUIRED_COLUMNS = {
    "age",
    "patient_id",
    "label",
    "prediction_primary_normalized",
    "normalized_ensemble_score",
    "mean_probability",
}


def post_json(payload: dict, token: str, timeout: int = 120) -> dict:
    request = urllib.request.Request(
        READ_ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-rsna-age-audit-v1/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        out = json.loads(response.read().decode("utf-8", "replace") or "{}")
    if out.get("ok") is not True or out.get("read_only") is not True:
        raise RuntimeError("read broker did not return ok/read_only")
    result = out.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("read broker result missing")
    return result


def find_prediction_url(account_id: str, kernel_ref: str, token: str) -> str:
    owner, slug = kernel_ref.split("/", 1)
    result = post_json(
        {
            "action": "raw_read",
            "account_id": account_id,
            "service": "kernels.KernelsApiService",
            "method": "ListKernelSessionOutput",
            "body": {
                "userName": owner,
                "kernelSlug": slug,
                "page": 1,
                "pageSize": 100,
            },
        },
        token,
    )
    files = result.get("files")
    if not isinstance(files, list):
        raise RuntimeError("kernel output inventory missing files")
    hits = [
        item
        for item in files
        if isinstance(item, dict)
        and str(item.get("fileName") or "").endswith(PREDICTIONS_SUFFIX)
        and item.get("url")
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one prediction CSV, found {len(hits)}")
    url = str(hits[0]["url"])
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host.endswith(".kaggleusercontent.com")
        or host.endswith(".googleusercontent.com")
        or host == "storage.googleapis.com"
        or host in {"api.kaggle.com", "www.kaggle.com"}
    ):
        raise RuntimeError("prediction URL host not allowlisted")
    return url


def download_rows(url: str) -> list[dict]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "m07-rsna-age-audit-v1/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise RuntimeError("prediction CSV has no header")
    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise RuntimeError("prediction CSV missing columns: " + ",".join(sorted(missing)))
    rows = []
    for raw in reader:
        age = int(float(raw["age"]))
        label = int(float(raw["label"]))
        pred = int(float(raw["prediction_primary_normalized"]))
        score = float(raw["normalized_ensemble_score"])
        prob = float(raw["mean_probability"])
        if age < 1 or age > 18 or label not in {0, 1} or pred not in {0, 1}:
            raise RuntimeError("prediction row failed age/label/pred validation")
        if not math.isfinite(score) or not math.isfinite(prob):
            raise RuntimeError("prediction row has non-finite score")
        rows.append(
            {
                "age": age,
                "patient_id": str(raw["patient_id"]),
                "label": label,
                "pred": pred,
                "score": score,
                "prob": prob,
            }
        )
    if len(rows) != 1099:
        raise RuntimeError(f"expected 1099 external rows, found {len(rows)}")
    return rows


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    rank = 1
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        rank += j - i
        i = j
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    ap = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            tp += 1
            ap += tp / rank
    return ap / positives


def metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty subgroup")
    tn = sum(r["label"] == 0 and r["pred"] == 0 for r in rows)
    fp = sum(r["label"] == 0 and r["pred"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["pred"] == 0 for r in rows)
    tp = sum(r["label"] == 1 and r["pred"] == 1 for r in rows)
    n = len(rows)
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision_pos = tp / (tp + fp) if tp + fp else float("nan")
    precision_neg = tn / (tn + fn) if tn + fn else float("nan")
    f1_pos = (
        2 * precision_pos * sensitivity / (precision_pos + sensitivity)
        if precision_pos + sensitivity
        else 0.0
    )
    f1_neg = (
        2 * precision_neg * specificity / (precision_neg + specificity)
        if precision_neg + specificity
        else 0.0
    )
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    return {
        "n": n,
        "patients": len({r["patient_id"] for r in rows}),
        "negative": tn + fp,
        "positive": tp + fn,
        "accuracy": (tn + tp) / n,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_positive": precision_pos,
        "f1_positive": f1_pos,
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "auroc": _auc(labels, scores),
        "auprc_positive": _average_precision(labels, scores),
        "brier_mean_probability": statistics.fmean(
            (r["prob"] - r["label"]) ** 2 for r in rows
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def percentile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def patient_cluster_bootstrap(
    rows: list[dict], *, draws: int = 1000, seed: int = 20260929
) -> dict:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    patient_ids = sorted(by_patient)
    rng = random.Random(seed)
    tracked = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auroc")
    values = {name: [] for name in tracked}
    valid = 0
    for _ in range(draws):
        sample = []
        for pid in rng.choices(patient_ids, k=len(patient_ids)):
            sample.extend(by_patient[pid])
        result = metrics(sample)
        if not all(math.isfinite(result[name]) for name in tracked):
            continue
        valid += 1
        for name in tracked:
            values[name].append(float(result[name]))
    return {
        "unit": "PATIENT_CLUSTER",
        "requested": draws,
        "valid": valid,
        "95ci": {
            name: {
                "lo": percentile(series, 0.025),
                "hi": percentile(series, 0.975),
            }
            for name, series in values.items()
        },
    }


def select(rows: Iterable[dict], lo: int, hi: int) -> list[dict]:
    return [row for row in rows if lo <= int(row["age"]) <= hi]


def build_audit(rows: list[dict], bootstraps: int) -> dict:
    groups = {
        "age_1_5_training_matched": select(rows, 1, 5),
        "age_6_9_near_shift": select(rows, 6, 9),
        "age_10_18_older_shift": select(rows, 10, 18),
        "age_1_9_primary": select(rows, 1, 9),
        "age_1_18_expanded": select(rows, 1, 18),
    }
    result = {}
    for index, (name, subset) in enumerate(groups.items()):
        result[name] = {
            "metrics": metrics(subset),
            "bootstrap": patient_cluster_bootstrap(
                subset, draws=bootstraps, seed=20260929 + index
            ),
        }
    matched = result["age_1_5_training_matched"]["metrics"]
    near = result["age_6_9_near_shift"]["metrics"]
    older = result["age_10_18_older_shift"]["metrics"]
    result["comparisons"] = {
        "accuracy_pp_6_9_minus_1_5": 100.0 * (near["accuracy"] - matched["accuracy"]),
        "accuracy_pp_10_18_minus_1_5": 100.0 * (older["accuracy"] - matched["accuracy"]),
        "specificity_pp_6_9_minus_1_5": 100.0 * (
            near["specificity"] - matched["specificity"]
        ),
        "specificity_pp_10_18_minus_1_5": 100.0 * (
            older["specificity"] - matched["specificity"]
        ),
        "auroc_pp_6_9_minus_1_5": 100.0 * (near["auroc"] - matched["auroc"]),
        "auroc_pp_10_18_minus_1_5": 100.0 * (older["auroc"] - matched["auroc"]),
    }
    return {
        "schema": "m07.external.rsna.age_audit.v1",
        "status": "PASS",
        "source_kernel": DEFAULT_KERNEL,
        "source_resolution": 224,
        "training_age_domain": "1-5 years",
        "training_performed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "external_adaptation": False,
        "groups": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-id", default=DEFAULT_ACCOUNT)
    parser.add_argument("--kernel-ref", default=DEFAULT_KERNEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")
    url = find_prediction_url(args.account_id, args.kernel_ref, token)
    rows = download_rows(url)
    audit = build_audit(rows, args.bootstraps)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_AGE_AUDIT=" + json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
