import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2_oof_leaderboard_v1",
    ROOT / "scripts" / "phase2_oof_leaderboard_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_unit(model_id: str, resolution: int, balacc: float) -> dict:
    metrics = {
        "accuracy": balacc,
        "balanced_accuracy": balacc,
        "macro_precision": balacc,
        "macro_recall": balacc,
        "macro_f1": balacc - 0.01,
        "mcc": balacc - 0.20,
        "auroc": min(0.999, balacc + 0.05),
        "precision_normal": balacc,
        "recall_normal": balacc - 0.02,
        "precision_pneumonia": balacc - 0.01,
        "recall_pneumonia": balacc + 0.02,
        "f1_normal": balacc - 0.01,
        "f1_pneumonia": balacc,
        "f2": balacc + 0.01,
    }
    return {
        "model_id": model_id,
        "resolution": resolution,
        "json_sources": [
            {
                "source": "archive:FINAL_REPORT.json",
                "metric_blocks": [{"path": "oof", "values": metrics}],
                "policy_paths": [
                    {
                        "path": "locked_test_used_for_selection",
                        "value": False,
                    },
                    {
                        "path": "external_used_for_selection",
                        "value": False,
                    },
                ],
            }
        ],
    }


class Phase2OofLeaderboardV1Tests(unittest.TestCase):
    def matrix(self) -> dict:
        units = []
        for model_index in range(1, 13):
            model_id = f"M{model_index:02d}"
            base = 0.70 + model_index / 100.0
            for offset, resolution in enumerate((224, 320, 384)):
                units.append(
                    synthetic_unit(
                        model_id,
                        resolution,
                        min(0.98, base + offset / 1000.0),
                    )
                )
        return {
            "schema": "pneumonia.phase2.final_evidence.matrix.v1",
            "status": "PASS",
            "units": units,
        }

    def test_exact_36_unit_coverage_and_descriptive_order(self):
        rows = MODULE.build_unit_rows(self.matrix())
        self.assertEqual(len(rows), 36)
        self.assertEqual(rows[0]["model_id"], "M12")
        self.assertEqual(rows[0]["resolution"], 384)
        self.assertEqual(rows[0]["descriptive_oof_rank"], 1)
        self.assertTrue(
            all(row["locked_test_used_for_selection"] is False for row in rows)
        )
        self.assertTrue(
            all(row["external_used_for_selection"] is False for row in rows)
        )

    def test_model_summary_uses_three_resolutions(self):
        unit_rows = MODULE.build_unit_rows(self.matrix())
        model_rows = MODULE.build_model_rows(unit_rows)
        self.assertEqual(len(model_rows), 12)
        self.assertEqual(model_rows[0]["model_id"], "M12")
        self.assertEqual(model_rows[0]["resolution_count"], 3)
        self.assertGreaterEqual(
            model_rows[0]["balanced_accuracy_sd_across_resolutions"],
            0.0,
        )

    def test_prefers_oof_metrics_json_over_final_report(self):
        unit = synthetic_unit("M01", 224, 0.80)
        unit["json_sources"].insert(
            0,
            {
                "source": "archive:OOF_METRICS.json",
                "metric_blocks": [
                    {
                        "path": "$",
                        "values": {
                            "balanced_accuracy": 0.81,
                            "macro_f1": 0.80,
                            "mcc": 0.60,
                            "auroc": 0.90,
                            "recall_normal": 0.79,
                            "recall_pneumonia": 0.83,
                        },
                    }
                ],
                "policy_paths": [],
            },
        )
        metrics, evidence = MODULE.extract_oof_metrics(unit)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.81)
        self.assertIn("OOF_METRICS.json", evidence["source"])

    def test_legacy_m07_r224_prefers_primary_fold_threshold_metrics(self):
        unit = synthetic_unit("M07", 224, 0.70)
        unit["json_sources"] = [
            {
                "source": "file:LIGHT_STATE/M07_OOF_GLOBAL_THRESHOLD_METRICS.json",
                "metric_blocks": [
                    {
                        "path": "$",
                        "values": {
                            "balanced_accuracy": 0.91,
                            "macro_f1": 0.90,
                            "mcc": 0.82,
                            "auroc": 0.97,
                            "recall_normal": 0.89,
                            "recall_pneumonia": 0.93,
                        },
                    }
                ],
                "policy_paths": [],
            },
            {
                "source": "file:LIGHT_STATE/M07_OOF_PRIMARY_METRICS.json",
                "metric_blocks": [
                    {
                        "path": "$",
                        "values": {
                            "balanced_accuracy": 0.98,
                            "macro_f1": 0.97,
                            "mcc": 0.95,
                            "auroc": 0.99,
                            "recall_normal": 0.97,
                            "recall_pneumonia": 0.99,
                        },
                    }
                ],
                "policy_paths": [],
            },
        ]
        metrics, evidence = MODULE.extract_oof_metrics(unit)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.98)
        self.assertIn("M07_OOF_PRIMARY_METRICS.json", evidence["source"])
        self.assertNotIn("GLOBAL_THRESHOLD", evidence["source"])

    def test_rejects_locked_test_selection_flag(self):
        unit = synthetic_unit("M01", 224, 0.80)
        unit["json_sources"][0]["policy_paths"][0]["value"] = True
        with self.assertRaisesRegex(
            MODULE.LeaderboardError,
            "LOCKED_TEST_SELECTION_POLICY_VIOLATION",
        ):
            MODULE.extract_oof_metrics(unit)

    def test_rejects_missing_matrix_unit(self):
        matrix = self.matrix()
        matrix["units"].pop()
        with self.assertRaisesRegex(
            MODULE.LeaderboardError,
            "MATRIX_UNIT_COUNT_INVALID",
        ):
            MODULE.build_unit_rows(matrix)

    def test_main_writes_receipt_and_csvs(self):
        matrix = self.matrix()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            out = root / "out"
            argv = [
                "phase2_oof_leaderboard_v1.py",
                "--matrix",
                str(matrix_path),
                "--output-dir",
                str(out),
            ]
            import sys

            original = sys.argv
            try:
                sys.argv = argv
                self.assertEqual(MODULE.main(), 0)
            finally:
                sys.argv = original
            receipt = json.loads(
                (out / "PHASE2_OOF_LEADERBOARD_RECEIPT_V1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["status"], "PASS_OOF_LEADERBOARD")
            self.assertFalse(receipt["selection_performed"])
            self.assertFalse(receipt["locked_test_used_for_selection"])
            self.assertFalse(receipt["external_used_for_selection"])
            self.assertEqual(
                receipt["next_action"],
                "RUN_PREDECLARED_PAIRED_PATIENT_BOOTSTRAP_HOLM",
            )


if __name__ == "__main__":
    unittest.main()
