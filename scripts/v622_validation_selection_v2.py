from __future__ import annotations

import csv
import dataclasses
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/selection-artifact"
EXPECTED_SELECT_SHA = "f6be9ddf5060dbea5fab233ee0d349820ef679c64662eec23dd01c2929aec9b2"
ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-selection-v2-runtime"
EXP = ROOT / "experiments"
SRC = ROOT / "governance"
OUT = ROOT / "selection-output"
EXP.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SOURCE_HASHES = {
    "select_champion.py": EXPECTED_SELECT_SHA,
    "auto_ensemble_selection.py": "39502bdf29fe696e510f44895e9d8d35959fce8e16e2ec0928a5f31dcb451311",
    "backbone_ensemble_selection.py": "9657bc631b5c1b67da15b47562fd10d2c9e3831acb3dcedea59b108fec95f979",
}
SOURCE_NAMES = [
    "select_champion.py",
    "auto_ensemble_selection.py",
    "backbone_ensemble_selection.py",
    "balanced_metrics.py",
    "backbone_comparison_contracts.py",
    "source_package_integrity.py",
    "v6_fingerprint_contracts.py",
]


def _request(worker_id: str, artifact_name: str, timeout: int = 120) -> tuple[dict, str]:
    token = os.environ["V622_SELECTION_TOKEN"]
    payload = json.dumps({"worker_id": worker_id, "artifact_name": artifact_name}).encode("utf-8")
    req = urllib.request.Request(
        BASE,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "v622-self-hosted-selection-v2",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if value.get("project") != "PNEUMONIA V6.2.2" or value.get("purpose") != "validation_only_selection_artifact":
        raise RuntimeError(f"Unexpected artifact response for {worker_id}/{artifact_name}: {value}")
    content = value.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Artifact content missing for {worker_id}/{artifact_name}")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != value.get("sha256"):
        raise RuntimeError(f"Artifact transport hash mismatch for {worker_id}/{artifact_name}")
    return value, content


def preflight_canonical_governance() -> dict:
    last_error: Exception | None = None
    for attempt in range(1, 21):
        try:
            value, _ = _request("W01", "select_champion.py", timeout=45)
            if value.get("sha256") != EXPECTED_SELECT_SHA:
                raise RuntimeError(
                    f"stale/noncanonical bridge response: {value.get('sha256')} != {EXPECTED_SELECT_SHA}; "
                    f"file={value.get('file_name')}"
                )
            if value.get("governance_source_package_pinned") is not True:
                raise RuntimeError("bridge response does not confirm canonical governance package pinning")
            return {
                "attempt": attempt,
                "file_name": value.get("file_name"),
                "sha256": value.get("sha256"),
                "governance_source_package_pinned": True,
            }
        except Exception as exc:
            last_error = exc
            print(f"Preflight attempt {attempt}/20 not ready: {exc}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"Canonical governance bridge preflight failed after retries: {last_error}")


def rebind_prediction_paths(value, local_prediction: pathlib.Path):
    if isinstance(value, dict):
        return {k: rebind_prediction_paths(v, local_prediction) for k, v in value.items()}
    if isinstance(value, list):
        return [rebind_prediction_paths(v, local_prediction) for v in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if normalized.endswith("/val_predictions.csv") or normalized == "val_predictions.csv":
            return str(local_prediction)
    return value


def compact(value, depth: int = 0):
    if depth > 6:
        return "<depth-limited>"
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            value = value.tolist()
        except Exception:
            value = str(value)
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if str(key) in {"oof_probability", "oof_prediction", "oof_raw_probability", "oof_fold"}:
                continue
            out[str(key)] = compact(child, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > 30:
            encoded = json.dumps([str(v) for v in value], sort_keys=True).encode("utf-8")
            return {"length": len(value), "sha256": hashlib.sha256(encoded).hexdigest()}
        return [compact(v, depth + 1) for v in value]
    if isinstance(value, pathlib.Path):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def post_issue_comment(markdown: str) -> None:
    payload = json.dumps({"body": markdown}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/issues/18/comments",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "v622-self-hosted-selection-v2",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status not in (200, 201):
            raise RuntimeError(f"Issue receipt publication failed HTTP {response.status}")


def main() -> None:
    preflight = preflight_canonical_governance()

    source_manifest = []
    for name in SOURCE_NAMES:
        value, content = _request("W01", name)
        path = SRC / name
        path.write_text(content, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = EXPECTED_SOURCE_HASHES.get(name)
        if expected and digest != expected:
            raise RuntimeError(f"Recovered governance hash mismatch for {name}: {digest} != {expected}")
        source_manifest.append(
            {
                "name": name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "file_name": value.get("file_name"),
            }
        )

    prediction_schema = None
    prediction_rows: dict[str, int] = {}
    transport_manifest = []

    for worker_number in range(1, 37):
        worker_id = f"W{worker_number:02d}"
        run_dir = EXP / worker_id
        run_dir.mkdir(parents=True, exist_ok=True)

        report_meta, report_text = _request(worker_id, "final_report.json")
        pred_meta, pred_text = _request(worker_id, "val_predictions.csv")

        report_original = run_dir / "final_report.original.json"
        prediction_path = run_dir / "val_predictions.csv"
        report_original.write_text(report_text, encoding="utf-8")
        prediction_path.write_text(pred_text, encoding="utf-8")

        report_value = json.loads(report_text)
        rebound = rebind_prediction_paths(report_value, prediction_path.resolve())
        report_path = run_dir / "final_report.json"
        report_path.write_text(json.dumps(rebound, indent=2, ensure_ascii=False), encoding="utf-8")

        with prediction_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            row_count = sum(1 for _ in reader)

        if prediction_schema is None:
            prediction_schema = header
        elif header != prediction_schema:
            raise RuntimeError(f"Prediction schema mismatch at {worker_id}")
        prediction_rows[worker_id] = row_count
        transport_manifest.append(
            {
                "worker_id": worker_id,
                "report_sha256": report_meta.get("sha256"),
                "prediction_sha256": pred_meta.get("sha256"),
                "prediction_rows": row_count,
                "transport_rebinding_only": True,
            }
        )

    if len(set(prediction_rows.values())) != 1:
        raise RuntimeError(f"Prediction row-count mismatch across canonical runs: {prediction_rows}")

    single_out = OUT / "single"
    single_out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SRC / "select_champion.py"),
            "--experiments-root",
            str(EXP),
            "--output-dir",
            str(single_out),
        ],
        check=True,
        cwd=str(SRC),
    )

    leaderboard_path = single_out / "validation_leaderboard.csv"
    champion_path = single_out / "validation_champion_candidate.json"
    if not leaderboard_path.exists() or not champion_path.exists():
        raise RuntimeError("Recovered select_champion.py did not create required outputs")

    import pandas as pd

    leaderboard = pd.read_csv(leaderboard_path)
    if len(leaderboard) != 36:
        raise RuntimeError(f"Validation leaderboard must contain 36 rows, got {len(leaderboard)}")
    if leaderboard["run_id"].astype(str).nunique() != 36:
        raise RuntimeError("Validation leaderboard run_id uniqueness mismatch")
    champion = json.loads(champion_path.read_text(encoding="utf-8"))

    sys.path.insert(0, str(SRC))
    import auto_ensemble_selection as aes

    best_runs = aes.best_run_per_model(leaderboard)
    if len(best_runs) != 12:
        raise RuntimeError(f"best_run_per_model must return 12 model families, got {len(best_runs)}")

    signature = inspect.signature(aes.select_automatic_ensemble)
    kwargs = {}
    unsupported = []
    ensemble_out = OUT / "ensemble"
    ensemble_out.mkdir(parents=True, exist_ok=True)
    for name, parameter in signature.parameters.items():
        if name in {"leaderboard", "validation_leaderboard"}:
            kwargs[name] = leaderboard
        elif name in {"selection_size", "top_k"}:
            kwargs[name] = 3
        elif name in {"output_dir", "output"}:
            kwargs[name] = ensemble_out
        elif name == "seed":
            kwargs[name] = 2026
        elif name in {"folds", "n_splits"}:
            kwargs[name] = 5
        elif name == "weight_step":
            kwargs[name] = 0.05
        elif name == "minimum_mean_weight":
            kwargs[name] = 0.05
        elif name in {"allowed", "allowed_models", "allowed_model_ids"}:
            kwargs[name] = None
        elif parameter.default is inspect.Parameter.empty:
            unsupported.append(name)
    if unsupported:
        raise RuntimeError(f"Unsupported required select_automatic_ensemble parameters: {unsupported}; signature={signature}")

    ensemble_result = aes.select_automatic_ensemble(**kwargs)

    best_model_runs = []
    for run in best_runs:
        best_model_runs.append(
            {
                "model_id": str(getattr(run, "model_id", "")),
                "run_id": str(getattr(run, "run_id", "")),
                "validation_rank": int(getattr(run, "validation_rank", 0)),
                "screening_minutes": float(getattr(run, "screening_minutes", 0.0)),
            }
        )

    receipt = {
        "project": "PNEUMONIA V6.2.2",
        "stage": "validation_only_model_and_ensemble_selection_v2",
        "canonical_runs": 36,
        "model_families": 12,
        "locked_test_used": False,
        "external_cohort_used": False,
        "hpo_repeated": False,
        "confirmation_repeated": False,
        "preflight": preflight,
        "prediction_schema": prediction_schema,
        "prediction_rows_per_run": next(iter(prediction_rows.values())),
        "governance_sources": source_manifest,
        "select_automatic_ensemble_signature": str(signature),
        "single_champion_candidate": compact(champion),
        "best_run_per_model": best_model_runs,
        "ensemble_selection": compact(ensemble_result),
        "freeze_status": "NOT_FROZEN_PENDING_BACKBONE_COMPARISON_GATE",
        "run_url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}",
        "transport_note": "Original final_report.json hashes retained; only val_predictions.csv path strings were rebound locally for transport. Metrics/predictions were not modified.",
    }

    receipt_path = ROOT / "selection-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = (
        "## V6.2.2 validation-only model + ensemble selection receipt v2\n\n"
        "> Recovered governance executed on the 36 canonical validation artifacts. Locked official test and external cohorts remained unopened.\n\n"
        "```json\n" + json.dumps(receipt, indent=2, ensure_ascii=False) + "\n```\n"
    )
    if len(markdown.encode("utf-8")) > 60000:
        raise RuntimeError("Selection journal receipt exceeds safety size budget")
    post_issue_comment(markdown)
    print(json.dumps({"selection_complete": True, "runs": 36, "models": 12, "freeze_status": receipt["freeze_status"]}))


if __name__ == "__main__":
    main()
