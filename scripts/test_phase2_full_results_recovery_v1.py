from __future__ import annotations

import unittest

import phase2_full_results_recovery_v1 as recovery


class FullResultsRecoveryTests(unittest.TestCase):
    def test_kernel_script_adds_full_result_sources_without_inference(self):
        target = {
            "model_id": "M01",
            "resolution": 224,
            "dataset_ref": "azadka/pneumonia-m01-r224-state-v1-7",
            "version": 1,
            "campaign_marker_name": "CAMPAIGN_STATE.json",
            "campaign_marker_sha256": "0" * 64,
            "campaign_receipt_sha256": None,
            "campaign_status": "COMPLETE",
            "campaign_split_fingerprint": "test-split",
            "campaign_completed_folds": [1, 2, 3, 4, 5],
        }
        source = recovery.kernel_script("master", [target])
        required = (
            "PHASE2_FULL_RESULTS_RECOVERY_PASS",
            "LOCKED_TEST_PRIMARY_METRICS",
            "GENERALIZATION_GAP",
            "direct_fold_receipts",
            "train_metrics",
            "validation_metrics",
            "training_evidence",
            "pneumonia.phase2.full_results.account.v1",
            'name="PHASE2_FULL_RESULTS_"+ACCOUNT_ID',
        )
        for marker in required:
            self.assertIn(marker, source)
        self.assertNotIn(
            'name="PHASE2_FINAL_EVIDENCE_"+ACCOUNT_ID',
            source,
        )
        self.assertNotIn("model.predict(", source)
        self.assertNotIn("model.fit(", source)

    def test_output_name_is_account_scoped(self):
        self.assertEqual(
            recovery._output_name("master"),
            "PHASE2_FULL_RESULTS_MASTER.json",
        )
        self.assertEqual(
            recovery._output_name("kg-05"),
            "PHASE2_FULL_RESULTS_KG_05.json",
        )

    def test_fold_receipts_prefers_direct_metrics_when_archive_is_minimal(self):
        unit = {
            "fold_archives": [
                {
                    "receipt": {
                        "fold_id": 1,
                        "receipt_sha256": "a",
                        "train_metrics": None,
                        "validation_metrics": None,
                    }
                }
            ],
            "direct_fold_receipts": [
                {
                    "fold_id": 1,
                    "receipt_sha256": "a",
                    "train_metrics": {"macro_precision": 0.99},
                    "validation_metrics": {"macro_recall": 0.98},
                }
            ],
        }
        rows = recovery._fold_receipts(unit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["train_metrics"]["macro_precision"], 0.99)

    def test_locked_test_summary_prioritizes_primary(self):
        unit = {
            "json_sources": [
                {
                    "source": "archive:LOCKED_TEST_MAJORITY_METRICS.json",
                    "metric_blocks": [
                        {
                            "path": "$",
                            "values": {
                                "n": 624,
                                "macro_precision": 0.90,
                                "macro_recall": 0.91,
                            },
                        }
                    ],
                },
                {
                    "source": "archive:LOCKED_TEST_PRIMARY_METRICS.json",
                    "metric_blocks": [
                        {
                            "path": "$",
                            "values": {
                                "n": 624,
                                "macro_precision": 0.92,
                                "macro_recall": 0.93,
                            },
                        }
                    ],
                },
            ]
        }
        summary = recovery._locked_test_summary(unit)
        self.assertEqual(summary["status"], "PRESENT")
        self.assertIn("PRIMARY", summary["primary"]["source"])
        self.assertEqual(summary["candidate_count"], 2)

    def test_locked_test_absence_is_explicit(self):
        summary = recovery._locked_test_summary({"json_sources": []})
        self.assertEqual(summary["status"], "NOT_RECOVERED")
        self.assertIsNone(summary["primary"])


if __name__ == "__main__":
    unittest.main()
