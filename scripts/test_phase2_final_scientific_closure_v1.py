from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest
import zipfile

import phase2_final_scientific_closure_v1 as closure


def write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class FinalScientificClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.freeze_path = self.root / "SELECTION_FREEZE.json"
        self.full_path = self.root / "PHASE2_FULL_RESULTS_MATRIX_V1.json"
        self.external_root = self.root / "external"
        self.external_root.mkdir()

        self.freeze = {
            "schema": "pneumonia.phase2.selection_freeze.v1",
            "status": "FROZEN_FOR_REPORT_ONLY_CONFIRMATION",
            "selection_basis": {
                "requested_metrics": ["macro_precision", "macro_recall"],
                "source_dual_run_id": "37125310017",
            },
            "frozen_candidate_set": [
                {
                    "model_id": "M09",
                    "resolution": 384,
                    "n": 5209,
                    "macro_precision": 0.9791,
                    "macro_recall": 0.9798,
                    "tn": 1387,
                    "fp": 41,
                    "fn": 44,
                    "tp": 3737,
                },
                {
                    "model_id": "M10",
                    "resolution": 320,
                    "n": 5209,
                    "macro_precision": 0.9794,
                    "macro_recall": 0.9796,
                    "tn": 1386,
                    "fp": 42,
                    "fn": 43,
                    "tp": 3738,
                },
            ],
            "freeze_rule": "PARETO_NONDOMINATED_SET_UNDER_MACRO_PRECISION_AND_MACRO_RECALL",
            "single_winner_declared": False,
            "confirmatory_superiority_claim": False,
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "locked_test_may_reselect": False,
            "external_may_reselect": False,
            "thresholds_refit_after_freeze": False,
            "cross_model_ensemble_selected": False,
            "cross_model_ensemble_fitted": False,
        }
        write_json(self.freeze_path, self.freeze)
        self.freeze_sha = hashlib.sha256(self.freeze_path.read_bytes()).hexdigest()

        units = []
        for model in range(1, 13):
            for resolution in (224, 320, 384):
                key = (f"M{model:02d}", resolution)
                recovery = {"status": "NOT_RECOVERED", "primary": None}
                if key in closure.EXPECTED_CANDIDATES:
                    recovery = {
                        "status": "PRESENT",
                        "primary": {
                            "source": "archive:LOCKED_TEST_PRIMARY_METRICS.json",
                            "path": "$",
                            "metrics": {
                                "n": 624,
                                "accuracy": 0.95,
                                "balanced_accuracy": 0.94,
                                "macro_precision": 0.93,
                                "macro_recall": 0.94,
                            },
                        },
                    }
                units.append(
                    {
                        "model_id": key[0],
                        "resolution": key[1],
                        "locked_test_recovery": recovery,
                    }
                )
        self.full = {
            "schema": "pneumonia.phase2.full_results.matrix.v1",
            "status": "PASS",
            "model_resolution_units": 36,
            "folds_represented": 180,
            "train_metric_folds": 180,
            "validation_metric_folds": 180,
            "extraction_only": True,
            "training_performed": False,
            "inference_performed": False,
            "threshold_tuning_performed": False,
            "locked_test_executed_by_this_run": False,
            "external_validation_executed_by_this_run": False,
            "units": units,
        }
        write_json(self.full_path, self.full)

        for model_id, resolution in closure.EXPECTED_CANDIDATES:
            report = {
                "schema": "pneumonia.phase2.finalist.external.rsna_pediatric.resolution.v1",
                "status": "SCIENTIFIC_RECEIPT_PASS",
                "model_id": model_id,
                "resolution": resolution,
                "selection_freeze_sha256": self.freeze_sha,
                "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",
                "locked_test_used_for_selection": False,
                "external_used_for_selection": False,
                "external_may_reselect": False,
                "training_performed": False,
                "hpo_performed": False,
                "external_threshold_tuning": False,
                "external_adaptation": False,
                "external_calibration_fitting": False,
                "primary_pediatric_lt10": {
                    "metrics": {
                        "accuracy": 0.75,
                        "macro_precision": 0.77,
                        "macro_recall": 0.76,
                    }
                },
                "expanded_pediatric_le18": {
                    "metrics": {
                        "accuracy": 0.80,
                        "macro_precision": 0.81,
                        "macro_recall": 0.80,
                    }
                },
            }
            zip_path = (
                self.external_root
                / f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_R{resolution}_V1_COMPLETE.zip"
            )
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_REPORT.json",
                    json.dumps(report, sort_keys=True),
                )

            receipt = {
                "schema": closure.EXTERNAL_SCHEMA,
                "status": "SCIENTIFIC_RECEIPT_PASS",
                "model_id": model_id,
                "resolution": resolution,
                "selection_freeze_sha256": self.freeze_sha,
                "training_performed": False,
                "hpo_performed": False,
                "external_threshold_tuning": False,
                "external_adaptation": False,
                "external_calibration_fitting": False,
                "locked_test_used_for_selection": False,
                "external_used_for_selection": False,
                "external_may_reselect": False,
                "artifact_sha256": {"report": "0" * 64},
            }
            receipt["receipt_sha256"] = hashlib.sha256(
                closure.canonical_json(receipt)
            ).hexdigest()
            receipt_path = (
                self.external_root
                / f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json"
            )
            write_json(receipt_path, receipt)

            handoff = {
                "schema": closure.HANDOFF_SCHEMA,
                "status": "SCIENTIFIC_RECEIPT_PASS",
                "model_id": model_id,
                "resolution": resolution,
                "selection_freeze_sha256": self.freeze_sha,
                "training_performed": False,
                "hpo_performed": False,
                "external_threshold_tuning": False,
                "external_adaptation": False,
                "external_calibration_fitting": False,
                "locked_test_used_for_selection": False,
                "external_used_for_selection": False,
                "external_may_reselect": False,
                "terminal_receipt_sha256": receipt["receipt_sha256"],
                "complete_zip_sha256": closure.sha256_file(zip_path),
                "kernel_ref": f"owner/{model_id.lower()}",
                "compute_mode": "GPU",
                "external_dataset": "nih-chest-xrays/data",
                "external_manifest_sha256": "a" * 64,
            }
            write_json(
                self.external_root / f"{model_id}_R{resolution}_RSNA_HANDOFF.json",
                handoff,
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_closure_inputs_pass(self):
        freeze, freeze_sha = closure.verify_selection_freeze(self.freeze_path)
        self.assertEqual(freeze_sha, self.freeze_sha)
        full, locked = closure.verify_full_results(self.full_path)
        external = closure.verify_external_root(self.external_root, freeze_sha)
        self.assertEqual(full["train_metric_folds"], 180)
        self.assertEqual(set(locked), closure.EXPECTED_CANDIDATES)
        self.assertEqual(set(external), closure.EXPECTED_CANDIDATES)
        self.assertEqual(
            {row["model_id"] for row in freeze["frozen_candidate_set"]},
            {"M09", "M10"},
        )

    def test_missing_finalist_locked_test_fails_closed(self):
        value = json.loads(self.full_path.read_text(encoding="utf-8"))
        for unit in value["units"]:
            if unit["model_id"] == "M09" and unit["resolution"] == 384:
                unit["locked_test_recovery"] = {
                    "status": "NOT_RECOVERED",
                    "primary": None,
                }
        write_json(self.full_path, value)
        with self.assertRaisesRegex(
            closure.ClosureError, "FINALIST_LOCKED_TEST_NOT_RECOVERED"
        ):
            closure.verify_full_results(self.full_path)


if __name__ == "__main__":
    unittest.main()
