from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import urllib.request

import m07_rsna_age_audit_v1 as base
from m07_rsna_resolution_reconcile_v1 import RUNS, output_inventory, safe_output_url

RESOLUTIONS = (224, 320, 384)
PREDICTIONS = "M07_RSNA_PEDIATRIC_EXTERNAL_PREDICTIONS.csv"
N_FOLDS = 5
SEED_TAG = "m07-rsna-fold-consensus-v1"


def download_vote_rows(url: str) -> list[dict]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "m07-rsna-fold-consensus-v1/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    required = {
        "image_id",
        "patient_id",
        "age",
        "label",
        *(f"vote_fold_{fold}" for fold in range(1, 6)),
    }
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        missing = sorted(required - set(reader.fieldnames or []))
        raise RuntimeError("vote prediction CSV schema mismatch: " + ",".join(missing))
    rows = []
    for raw in reader:
        votes = [int(float(raw[f"vote_fold_{fold}"])) for fold in range(1, 6)]
        if any(vote not in {0, 1} for vote in votes):
            raise RuntimeError("fold vote is not binary")
        rows.append(
            {
                "image_id": str(raw["image_id"]),
                "patient_id": str(raw["patient_id"]),
                "age": int(float(raw["age"])),
                "label": int(float(raw["label"])),
                "votes": votes,
                "vote_count": sum(votes),
            }
        )
    if len(rows) != 1099:
        raise RuntimeError(f"expected 1099 rows, found {len(rows)}")
    return rows


def fetch_all(token: str) -> dict[int, list[dict]]:
    outputs = {}
    for resolution in RESOLUTIONS:
        spec = RUNS[resolution]
        files = output_inventory(spec["account_id"], spec["kernel_ref"], token)
        outputs[resolution] = download_vote_rows(safe_output_url(files, PREDICTIONS))
    return outputs


def align(outputs: dict[int, list[dict]]) -> list[dict]:
    indexed = {
        resolution: {row["image_id"]: row for row in rows}
        for resolution, rows in outputs.items()
    }
    ids = set(indexed[224])
    if any(set(indexed[r]) != ids for r in RESOLUTIONS):
        raise RuntimeError("vote image sets differ")
    merged = []
    for image_id in sorted(ids):
        refs = [indexed[r][image_id] for r in RESOLUTIONS]
        identity = {(r["patient_id"], r["age"], r["label"]) for r in refs}
        if len(identity) != 1:
            raise RuntimeError(f"identity mismatch for {image_id}")
        patient_id, age, label = next(iter(identity))
        row = {
            "image_id": image_id,
            "patient_id": patient_id,
            "age": age,
            "label": label,
        }
        for resolution, ref in zip(RESOLUTIONS, refs):
            row[f"votes_{resolution}"] = ref["vote_count"]
        row["votes_total"] = sum(ref["vote_count"] for ref in refs)
        merged.append(row)
    return merged


def patient_fold(patient_id: str) -> int:
    digest = hashlib.sha256((SEED_TAG + "|" + patient_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % N_FOLDS


def rule_catalog() -> list[dict]:
    rules = []
    for k in range(1, 16):
        rules.append({"family": "total_15", "k": k})
    for k in range(1, 6):
        rules.append({"family": "at_least_2_resolutions", "k": k})
        rules.append({"family": "all_3_resolutions", "k": k})
    for resolution in RESOLUTIONS:
        for k in range(1, 6):
            rules.append({"family": f"single_{resolution}", "k": k})
    return rules


def predict(row: dict, rule: dict) -> int:
    family = rule["family"]
    k = int(rule["k"])
    if family == "total_15":
        return int(row["votes_total"] >= k)
    if family == "at_least_2_resolutions":
        count = sum(row[f"votes_{r}"] >= k for r in RESOLUTIONS)
        return int(count >= 2)
    if family == "all_3_resolutions":
        return int(all(row[f"votes_{r}"] >= k for r in RESOLUTIONS))
    if family.startswith("single_"):
        resolution = int(family.split("_", 1)[1])
        return int(row[f"votes_{resolution}"] >= k)
    raise ValueError(f"unknown rule family {family}")


def metric_rows(rows: list[dict], rule: dict) -> list[dict]:
    output = []
    for row in rows:
        pred = predict(row, rule)
        output.append(
            {
                "patient_id": row["patient_id"],
                "age": row["age"],
                "label": row["label"],
                "pred": pred,
                "score": float(row["votes_total"]),
                "prob": row["votes_total"] / 15.0,
            }
        )
    return output


def compact(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "n",
            "patients",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision_positive",
            "f1_positive",
            "macro_f1",
            "tn",
            "fp",
            "fn",
            "tp",
        )
    }


def evaluate(rows: list[dict], rule: dict) -> dict:
    return compact(base.metrics(metric_rows(rows, rule)))


def sweep(rows: list[dict]) -> list[dict]:
    results = []
    for rule in rule_catalog():
        results.append({"rule": rule, "metrics": evaluate(rows, rule)})
    return results


def choose_train_rule(rows: list[dict]) -> tuple[dict, dict]:
    best = None
    for item in sweep(rows):
        metrics = item["metrics"]
        if metrics["sensitivity"] + 1e-12 < 0.90:
            continue
        key = (
            metrics["precision_positive"],
            metrics["specificity"],
            metrics["balanced_accuracy"],
            metrics["accuracy"],
            -metrics["fp"],
        )
        if best is None or key > best[0]:
            best = (key, item["rule"], metrics)
    if best is None:
        raise RuntimeError("no consensus rule satisfies train sensitivity >= 0.90")
    return best[1], best[2]


def crossfit(rows: list[dict]) -> dict:
    outputs = []
    receipts = []
    for fold in range(N_FOLDS):
        train = [r for r in rows if patient_fold(r["patient_id"]) != fold]
        test = [r for r in rows if patient_fold(r["patient_id"]) == fold]
        train_patients = {r["patient_id"] for r in train}
        test_patients = {r["patient_id"] for r in test}
        if train_patients & test_patients:
            raise RuntimeError("patient leakage")
        rule, train_metrics = choose_train_rule(train)
        outputs.extend(metric_rows(test, rule))
        receipts.append(
            {
                "fold": fold,
                "rule": rule,
                "train_patients": len(train_patients),
                "test_patients": len(test_patients),
                "train_precision_positive": train_metrics["precision_positive"],
                "train_sensitivity": train_metrics["sensitivity"],
                "train_specificity": train_metrics["specificity"],
            }
        )
    return {
        "metrics": compact(base.metrics(outputs)),
        "fold_receipts": receipts,
        "patient_leakage": False,
    }


def descriptive_frontier(rows: list[dict]) -> dict:
    items = sweep(rows)
    feasible = [
        item
        for item in items
        if item["metrics"]["sensitivity"] + 1e-12 >= 0.90
    ]
    feasible.sort(
        key=lambda item: (
            item["metrics"]["precision_positive"],
            item["metrics"]["specificity"],
            item["metrics"]["accuracy"],
        ),
        reverse=True,
    )
    target_hits = [
        item
        for item in feasible
        if item["metrics"]["precision_positive"] >= 0.85
    ]
    return {
        "best_precision_with_sensitivity_ge_90": feasible[0] if feasible else None,
        "any_rule_hits_precision85_sensitivity90": bool(target_hits),
        "target_hit_count": len(target_hits),
        "top5_feasible": feasible[:5],
    }


def analyze(rows: list[dict]) -> dict:
    conventional = {
        "r224_majority_3of5": evaluate(rows, {"family": "single_224", "k": 3}),
        "r320_majority_3of5": evaluate(rows, {"family": "single_320", "k": 3}),
        "r384_majority_3of5": evaluate(rows, {"family": "single_384", "k": 3}),
        "all15_majority_8of15": evaluate(rows, {"family": "total_15", "k": 8}),
    }
    return {
        "schema": "m07.external.rsna.fold_consensus.v1",
        "status": "PASS",
        "n": len(rows),
        "patients": len({r["patient_id"] for r in rows}),
        "model_weights_changed": False,
        "retraining_performed": False,
        "raw_external_results_preserved": True,
        "conventional_label_free_rules": conventional,
        "descriptive_full_external_frontier_not_for_selection": descriptive_frontier(rows),
        "patient_level_crossfit_selection": {
            "constraint": "train-fold sensitivity >= 0.90",
            "objective": "maximize train-fold positive precision",
            **crossfit(rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")
    rows = align(fetch_all(token))
    result = analyze(rows)
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_FOLD_CONSENSUS=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
