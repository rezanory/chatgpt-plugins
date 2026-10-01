import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2_final_evidence_extract_v1",
    ROOT / "scripts" / "phase2_final_evidence_extract_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2FinalEvidenceExtractV1Tests(unittest.TestCase):
    def test_account_plan_covers_exact_36_units(self):
        observed = []
        for account_id, plan in MODULE.ACCOUNT_PLANS.items():
            self.assertTrue(account_id)
            self.assertTrue(plan["owner"])
            observed.extend((model, int(resolution)) for model, resolution, _ in plan["targets"])
        expected = {
            (f"M{index:02d}", resolution)
            for index in range(1, 13)
            for resolution in (224, 320, 384)
        }
        self.assertEqual(len(observed), 36)
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(set(observed)))

    def test_generated_kernel_is_cpu_evidence_only_python(self):
        target = {
            "model_id": "M04",
            "resolution": 384,
            "dataset_ref": "reyhanehazad/pneumonia-m04-r384-state-v1-7",
            "version": 6,
            "source": "reyhanehazad/pneumonia-m04-r384-state-v1-7/versions/6",
        }
        source = MODULE.kernel_script("kg-04", [target])
        compile(source, "<phase2-final-evidence-kernel>", "exec")
        self.assertIn("FINAL_EVIDENCE.cgpzip", source)
        self.assertIn("training_performed", source)
        self.assertNotIn("tensorflow", source.lower())
        self.assertNotIn("torch.", source.lower())
        self.assertNotIn(".fit(", source)
        self.assertNotIn("optimizer", source.lower())

    def test_recursive_status(self):
        self.assertEqual(
            MODULE.recursive_status({"result": {"kernelStatus": "running"}}),
            "RUNNING",
        )
        self.assertEqual(
            MODULE.recursive_status({"outer": [{"state": "complete"}]}),
            "COMPLETE",
        )
        self.assertEqual(MODULE.recursive_status({"ok": True}), "")


if __name__ == "__main__":
    unittest.main()
