import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2_master_oof_reconstruct_v1",
    ROOT / "scripts" / "phase2_master_oof_reconstruct_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fake_matrix():
    units = []
    for model_index, model in enumerate(MODULE.EXPECTED_MODELS, 1):
        for resolution in MODULE.EXPECTED_RESOLUTIONS:
            base = 0.70 + model_index * 0.005 + (resolution - 224) / 10000
            metrics = {
                "balanced_accuracy": base,
                "macro_f1": base - 0.01,
                "mcc": base - 0.20,
                "auroc": min(0.99, base + 0.10),
                "recall_normal": base - 0.03,
                "recall_pneumonia": base + 0.02,
                "precision_normal": base - 0.02,
                "precision_pneumonia": base + 0.01,
                "f1_normal": base - 0.025,
                "f1_pneumonia": base + 0.005,
                "f2": base + 0.015,
            }
            schema = (
                "m07.gate.multires.model_resolution.v1.6"
                if model == "M07"
                else "pneumonia.phase2.model_resolution.v1.6"
            )
            units.append(
                {
                    "model_id": model,
                    "resolution": resolution,
                    "dataset_ref": f"owner/{model.lower()}-r{resolution}",
                    "dataset_version_number": 1,
                    "json_sources": [
                        {
                            "source": "archive:FINAL_REPORT.json",
                            "schema": schema,
                            "status": "COMPLETE",
                            "metric_blocks": [
                                {"path": "oof", "values": metrics},
                            ],
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
            )
    return {
        "schema": "pneumonia.phase2.final_evidence.matrix.v1",
        "status": "PASS",
        "training_performed": False,
        "locked_test_executed": False,
        "units": units,
    }


class Phase2MasterOofReconstructV1Tests(unittest.TestCase):
    def test_exact_36_unit_matrix(self):
        rows = MODULE.build_master_matrix(fake_matrix())
        self.assertEqual(len(rows), 36)
        self.assertEqual(
            {(r["model_id"], r["resolution"]) for r in rows},
            {
                (model, resolution)
                for model in MODULE.EXPECTED_MODELS
                for resolution in MODULE.EXPECTED_RESOLUTIONS
            },
        )
        self.assertTrue(all(r["dataset"] == "OOF" for r in rows))
        self.assertTrue(all(r["locked_test_used_for_selection"] is False for r in rows))

    def test_aggregate_is_development_descriptive_only(self):
        rows = MODULE.build_master_matrix(fake_matrix())
        aggregate = MODULE.build_model_aggregate(rows)
        self.assertEqual(len(aggregate), 12)
        self.assertEqual(aggregate[0]["model_id"], "M12")
        self.assertEqual(aggregate[0]["development_descriptive_rank"], 1)
        self.assertTrue(
            all(
                row["selection_role"] == "DESCRIPTIVE_DEVELOPMENT_ORDER_ONLY"
                for row in aggregate
            )
        )

    def test_predeclared_contrast_shape(self):
        rows = MODULE.build_master_matrix(fake_matrix())
        contrasts = MODULE.build_predeclared_point_contrasts(rows)
        self.assertEqual(
            len(contrasts),
            3 * len(MODULE.PREDECLARED_MODEL_COMPARISONS) * 6,
        )
        self.assertTrue(
            all(
                row["inference_status"]
                == "POINT_ESTIMATE_ONLY_PENDING_PAIRED_PATIENT_BOOTSTRAP_HOLM"
                for row in contrasts
            )
        )

    def test_locked_test_selection_policy_fails_closed(self):
        payload = fake_matrix()
        payload["units"][0]["json_sources"][0]["policy_paths"][0]["value"] = True
        with self.assertRaisesRegex(
            MODULE.MasterAnalysisError, "LOCKED_TEST_SELECTION_POLICY_INVALID"
        ):
            MODULE.build_master_matrix(payload)

    def test_missing_unit_fails_closed(self):
        payload = fake_matrix()
        payload["units"].pop()
        with self.assertRaisesRegex(
            MODULE.MasterAnalysisError, "FINAL_EVIDENCE_MATRIX_COVERAGE_INVALID"
        ):
            MODULE.build_master_matrix(payload)


if __name__ == "__main__":
    unittest.main()
