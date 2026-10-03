from __future__ import annotations

import unittest

import phase2_full_results_reconcile_v1 as target


class ReconcileTests(unittest.TestCase):
    def test_pinned_output_coverage(self):
        self.assertEqual(len(target.PINNED_OUTPUTS), 11)
        for account_id, (kernel_ref, file_name) in target.PINNED_OUTPUTS.items():
            self.assertIn(target.SOURCE_RUN_ID, kernel_ref)
            suffix = (
                "MASTER"
                if account_id == "master"
                else account_id.replace("-", "_").upper()
            )
            self.assertEqual(file_name, f"PHASE2_FULL_RESULTS_{suffix}.json")
        self.assertEqual(target.M07_R224_LEGACY_DATASET_VERSION, 108)
        self.assertEqual(len(target.M07_R224_LEGACY_FILES), 5)

    def test_build_matrix_accepts_175_train_metrics_plus_5_legacy_histories(self):
        account_order = list(target.PINNED_OUTPUTS)
        account_units = {account_id: [] for account_id in account_order}
        cursor = 0
        finalists = {("M09", 384), ("M10", 320)}

        for model in range(1, 13):
            for resolution in (224, 320, 384):
                key = (f"M{model:02d}", resolution)
                account_id = account_order[cursor % len(account_order)]
                cursor += 1

                locked = {"status": "NOT_RECOVERED", "primary": None}
                if key in finalists:
                    locked = {
                        "status": "PRESENT",
                        "primary": {
                            "source": "archive:LOCKED_TEST/LOCKED_TEST_PRIMARY_METRICS.json",
                            "path": "$",
                            "metrics": {
                                "n": 624,
                                "accuracy": 0.95,
                                "balanced_accuracy": 0.94,
                                "macro_precision": 0.93,
                                "macro_recall": 0.94,
                            },
                        },
                        "candidate_count": 1,
                    }

                folds = []
                if key != ("M07", 224):
                    for fold in range(1, 6):
                        folds.append(
                            {
                                "receipt": {
                                    "fold_id": fold,
                                    "train_metrics": {"accuracy": 0.99},
                                    "validation_metrics": {"accuracy": 0.97},
                                }
                            }
                        )
                account_units[account_id].append(
                    {
                        "model_id": key[0],
                        "resolution": key[1],
                        "fold_archives": folds,
                        "direct_fold_receipts": [],
                        "_locked": locked,
                    }
                )

        legacy = [
            {
                "fold_id": fold,
                "train_metrics": None,
                "validation_metrics": {"n": 1000, "accuracy": 0.97},
                "training_evidence": {
                    "evidence_class": "LEGACY_TRAINING_HISTORY_AND_FROZEN_WEIGHTS",
                    "history_sha256": "a" * 64,
                    "weights_sha256": "b" * 64,
                },
            }
            for fold in range(1, 6)
        ]
        account_results = [
            {
                "schema": "pneumonia.phase2.full_results.account.v1",
                "status": "PASS",
                "account_id": account_id,
                "targets": account_units[account_id],
            }
            for account_id in account_order
        ]

        original = target.recovery._locked_test_summary
        try:
            target.recovery._locked_test_summary = lambda unit: unit["_locked"]
            matrix = target.build_matrix(account_results, "12345", legacy)
        finally:
            target.recovery._locked_test_summary = original

        self.assertEqual(matrix["accounts_complete"], 11)
        self.assertEqual(matrix["model_resolution_units"], 36)
        self.assertEqual(matrix["folds_represented"], 180)
        self.assertEqual(matrix["train_metric_folds"], 175)
        self.assertEqual(matrix["legacy_training_history_folds"], 5)
        self.assertEqual(matrix["training_evidence_folds"], 180)
        self.assertEqual(matrix["validation_metric_folds"], 180)
        self.assertFalse(
            matrix["legacy_training_metric_limitation"][
                "train_classification_metrics_persisted"
            ]
        )
        self.assertFalse(
            matrix["legacy_training_metric_limitation"][
                "posthoc_train_inference_performed"
            ]
        )

    def test_build_matrix_rejects_missing_legacy_history(self):
        with self.assertRaisesRegex(
            RuntimeError, "RECONCILE_ACCOUNT_COUNT_INVALID"
        ):
            target.build_matrix([], "12345", [])


if __name__ == "__main__":
    unittest.main()
