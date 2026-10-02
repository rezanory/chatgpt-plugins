import hashlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2_collect_selection_bundles_v1",
    ROOT / "scripts" / "phase2_collect_selection_bundles_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_bundle(account_id="kg-02"):
    payloads = {
        "M01/R224/OOF_PREDICTIONS.csv": b"relative_path,patient_id,label,sha256,probability_pneumonia,prediction_fold_threshold\na,P1,0,"
        + b"a" * 64
        + b",0.1,0\n",
        "M01/R224/LOCKED_TEST_PREDICTIONS.csv": b"relative_path,patient_id,label,sha256,normalized_ensemble_score,prediction_primary\na,P1,0,"
        + b"a" * 64
        + b",-1.0,0\n",
        "M01/R224/FINAL_REPORT.json": b'{"status":"COMPLETE"}',
    }
    rows = [
        {
            "path": name,
            "source": "test",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in payloads.items()
    ]
    manifest = {
        "schema": "pneumonia.phase2.selection_prediction_bundle.v1",
        "status": "PASS",
        "account_id": account_id,
        "training_performed": False,
        "hpo_performed": False,
        "locked_test_executed": False,
        "external_validation_executed": False,
        "files": rows,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(manifest, sort_keys=True).encode(),
        )
    return buffer.getvalue()


class Phase2CollectSelectionBundlesV1Tests(unittest.TestCase):
    def test_validate_and_extract_verified_bundle(self):
        data = make_bundle()
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            extracted = MODULE.validate_and_extract_bundle_bytes(data, "kg-02", root)
            self.assertEqual(len(extracted), 3)
            self.assertTrue((root / "M01/R224/OOF_PREDICTIONS.csv").is_file())
            self.assertTrue((root / "M01/R224/LOCKED_TEST_PREDICTIONS.csv").is_file())

    def test_bundle_member_path_traversal_rejected(self):
        manifest = {
            "schema": "pneumonia.phase2.selection_prediction_bundle.v1",
            "status": "PASS",
            "account_id": "kg-02",
            "training_performed": False,
            "hpo_performed": False,
            "locked_test_executed": False,
            "external_validation_executed": False,
            "files": [],
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("../evil.csv", b"x")
            archive.writestr("M01/R224/OOF_PREDICTIONS.csv", b"x")
            archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest).encode())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                MODULE.SelectionBundleError, "BUNDLE_MEMBER_PATH_UNSAFE"
            ):
                MODULE.validate_and_extract_bundle_bytes(
                    buffer.getvalue(), "kg-02", pathlib.Path(temp)
                )

    def test_exact_prediction_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for model in MODULE.REQUIRED_MODELS:
                for resolution in MODULE.REQUIRED_RESOLUTIONS:
                    folder = root / model / f"R{resolution}"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "OOF_PREDICTIONS.csv").write_text("x\n", encoding="utf-8")
                    (folder / "LOCKED_TEST_PREDICTIONS.csv").write_text(
                        "x\n", encoding="utf-8"
                    )
            result = MODULE.validate_prediction_coverage(root)
            self.assertEqual(result["prediction_csv_count"], 72)
            self.assertEqual(result["expected_prediction_csv_count"], 72)


if __name__ == "__main__":
    unittest.main()
