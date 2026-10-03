import importlib.util
import inspect
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "phase2_oof_vector_extract_v1",
    ROOT / "scripts" / "phase2_oof_vector_extract_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2OofVectorExtractV1Tests(unittest.TestCase):
    def test_plan_covers_exact_36_units(self):
        observed = []
        for plan in MODULE.ACCOUNT_PLANS.values():
            observed.extend(
                (model, int(resolution))
                for model, resolution, _ in plan["targets"]
            )
        expected = {
            (f"M{i:02d}", resolution)
            for i in range(1, 13)
            for resolution in (224, 320, 384)
        }
        self.assertEqual(len(observed), 36)
        self.assertEqual(set(observed), expected)

    def test_generated_kernel_is_oof_only_and_compiles(self):
        target = {
            "model_id": "M07",
            "resolution": 320,
            "dataset_ref": "trickermark/m07-gate-r320-state-v1-7",
            "version": 2,
            "campaign_marker_name": "CAMPAIGN_STATE.json",
            "campaign_marker_sha256": "a" * 64,
            "campaign_receipt_sha256": "b" * 64,
            "campaign_status": "COMPLETE",
            "campaign_split_fingerprint": "c" * 64,
            "campaign_completed_folds": None,
        }
        source = MODULE.vector_kernel_script("kg-05", [target])
        compile(source, "<phase2-oof-vector-kernel>", "exec")
        self.assertIn("CANONICAL_PHASE2_OOF_FROM_SEALED_FOLDS", source)
        self.assertIn("validation_predictions.csv", source)
        self.assertIn("prediction_fold_threshold", source)
        self.assertIn("locked_test_used_for_selection", source)
        self.assertIn("threshold_tuning_performed", source)
        self.assertNotIn("LOCKED_TEST_PREDICTIONS.csv", source)
        self.assertNotIn("tensorflow", source.lower())
        self.assertNotIn("torch.", source.lower())
        self.assertNotIn(".fit(", source.lower())
        self.assertNotIn("optimizer", source.lower())

    def test_vector_names_are_deterministic(self):
        self.assertEqual(
            MODULE.vector_output_name("M07", 384),
            "PHASE2_OOF_VECTOR_M07_R384.json",
        )
        self.assertEqual(
            MODULE.account_summary_name("kg-05"),
            "PHASE2_OOF_VECTOR_ACCOUNT_KG_05.json",
        )

    def test_launch_policy_is_cpu_oof_only(self):
        source = inspect.getsource(MODULE.launch_account)
        self.assertIn('"enableGpu": False', source)
        self.assertIn('"enableInternet": False', source)
        self.assertIn("no training, inference, HPO, threshold tuning, Locked-Test", source)
        self.assertIn("reusable_account", source)

    def test_validate_vectors_requires_exact_36_and_one_identity(self):
        vectors = []
        for i in range(1, 13):
            for resolution in (224, 320, 384):
                vectors.append(
                    {
                        "model_id": f"M{i:02d}",
                        "resolution": resolution,
                        "identity_sequence_sha256": "a" * 64,
                        "row_count": 42,
                        "locked_test_used_for_selection": False,
                        "external_used_for_selection": False,
                    }
                )
        MODULE.validate_vectors(vectors)
        vectors[-1]["identity_sequence_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "OOF_VECTOR_GLOBAL_IDENTITY_MISMATCH"):
            MODULE.validate_vectors(vectors)


if __name__ == "__main__":
    unittest.main()
