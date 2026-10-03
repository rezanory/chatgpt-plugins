from __future__ import annotations

import unittest

import phase2_full_results_reconcile_v1 as target


class ReconcileTests(unittest.TestCase):
    def test_pinned_output_coverage(self):
        self.assertEqual(len(target.PINNED_OUTPUTS), 11)
        self.assertEqual(
            set(target.PINNED_OUTPUTS),
            {
                "master",
                "kg-02",
                "kg-03",
                "kg-04",
                "kg-05",
                "kg-06",
                "kg-07",
                "kg-08",
                "kg-09",
                "kg-10",
                "kg-11",
            },
        )
        for account_id, (kernel_ref, file_name) in target.PINNED_OUTPUTS.items():
            self.assertIn(target.SOURCE_RUN_ID, kernel_ref)
            self.assertEqual(
                file_name,
                "PHASE2_FULL_RESULTS_"
                + ("MASTER" if account_id == "master" else account_id.replace("-", "_").upper())
                + ".json",
            )

    def test_build_matrix_requires_all_metrics_and_finalist_locked_test(self):
        account_units = {account_id: [] for account_id in target.PINNED_OUTPUTS}
        account_order = list(target.PINNED_OUTPUTS)
        cursor = 0
        for model in range(1, 13):
            for resolution in (224, 320, 384):
                account_id = account_order[cursor % len(account_order)]
                cursor += 1
                locked = {"status": "NOT_RECOVERED", "primary": None}
                if (f"M{model:02d}", resolution) in {("M09", 384), ("M10", 320)}:
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
                fold_archives = []
                for fold in range(1, 6):
                    fold_archives.append(
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
                        "model_id": f"M{model:02d}",
                        "resolution": resolution,
                        "fold_archives": fold_archives,
                        "direct_fold_receipts": [],
                        "_locked": locked,
                    }
                )

        original = target.recovery._locked_test_summary
        try:
            target.recovery._locked_test_summary = lambda unit: unit["_locked"]
            account_results = [
                {
                    "schema": "pneumonia.phase2.full_results.account.v1",
                    "status": "PASS",
                    "account_id": account_id,
                    "targets": account_units[account_id],
                }
                for account_id in account_order
            ]
            matrix = target.build_matrix(account_results, "12345")
        finally:
            target.recovery._locked_test_summary = original

        self.assertEqual(matrix["accounts_complete"], 11)
        self.assertEqual(matrix["model_resolution_units"], 36)
        self.assertEqual(matrix["folds_represented"], 180)
        self.assertEqual(matrix["train_metric_folds"], 180)
        self.assertEqual(matrix["validation_metric_folds"], 180)
        self.assertTrue(matrix["reconciled_from_existing_outputs"])
        self.assertFalse(matrix["training_performed"])
        self.assertFalse(matrix["inference_performed"])


if __name__ == "__main__":
    unittest.main()
