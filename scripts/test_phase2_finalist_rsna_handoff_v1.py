from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest
import zipfile

import phase2_finalist_rsna_handoff_v1 as target


class FinalistHandoffTests(unittest.TestCase):
    def test_kernel_ref_validation_is_contract_bound(self):
        m09 = target.dispatch.CONTRACTS["PHASE2_FINALIST_RSNA_M09_R384_V1"]
        m10 = target.dispatch.CONTRACTS["PHASE2_FINALIST_RSNA_M10_R320_V1"]

        self.assertEqual(
            target.validate_kernel_ref(
                m09,
                "azadkk/phase2-finalist-rsna-m09-r384-v1-37142719732",
            ),
            37142719732,
        )
        self.assertEqual(
            target.validate_kernel_ref(
                m10,
                "mylovevpn1/phase2-finalist-rsna-m10-r320-v1-37142611856",
            ),
            37142611856,
        )
        with self.assertRaisesRegex(RuntimeError, "HANDOFF_KERNEL_OWNER_MISMATCH"):
            target.validate_kernel_ref(
                m09,
                "mylovevpn1/phase2-finalist-rsna-m09-r384-v1-37142719732",
            )
        with self.assertRaisesRegex(RuntimeError, "HANDOFF_KERNEL_SLUG_MISMATCH"):
            target.validate_kernel_ref(
                m09,
                "azadkk/phase2-finalist-rsna-m10-r384-v1-37142719732",
            )

    def test_main_selects_by_complete_output_evidence(self):
        import inspect
        source = inspect.getsource(target.main)
        self.assertIn("select_complete_output_candidate", source)
        self.assertNotIn("GetKernelSessionStatus", source)
        self.assertNotIn("resolved_kernel_status", source)

    def test_complete_output_candidate_prefers_highest_run(self):
        def item(name):
            return {"fileName": name, "url": "https://example.invalid/x"}

        receipt = "M10_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json"
        complete = "M10_RSNA_PEDIATRIC_EXTERNAL_R320_V1_COMPLETE.zip"
        ref, run_id, receipt_item, zip_item = (
            target.select_complete_output_candidate(
                [
                    (
                        "owner/phase2-finalist-rsna-m10-r320-v1-100",
                        100,
                        [item(receipt), item(complete)],
                    ),
                    (
                        "owner/phase2-finalist-rsna-m10-r320-v1-101",
                        101,
                        [item(receipt)],
                    ),
                    (
                        "owner/phase2-finalist-rsna-m10-r320-v1-99",
                        99,
                        [item(receipt), item(complete)],
                    ),
                ],
                receipt,
                complete,
            )
        )
        self.assertEqual(run_id, 100)
        self.assertTrue(ref.endswith("-100"))
        self.assertTrue(receipt_item["fileName"].endswith(receipt))
        self.assertTrue(zip_item["fileName"].endswith(complete))

    def test_choose_complete_kernel_selects_highest_complete_run(self):
        ref, run_id = target.choose_complete_kernel(
            [
                (
                    "owner/phase2-finalist-rsna-m10-r320-v1-100",
                    "COMPLETE",
                ),
                (
                    "owner/phase2-finalist-rsna-m10-r320-v1-101",
                    "RUNNING",
                ),
                (
                    "owner/phase2-finalist-rsna-m10-r320-v1-99",
                    "COMPLETE",
                ),
            ]
        )
        self.assertEqual(run_id, 100)
        self.assertTrue(ref.endswith("-100"))
        with self.assertRaisesRegex(
            RuntimeError, "HANDOFF_COMPLETE_KERNEL_NOT_FOUND"
        ):
            target.choose_complete_kernel(
                [
                    (
                        "owner/phase2-finalist-rsna-m10-r320-v1-101",
                        "RUNNING",
                    ),
                    (
                        "owner/phase2-finalist-rsna-m10-r320-v1-102",
                        "ERROR",
                    ),
                ]
            )

    def _fixture(
        self,
        root: pathlib.Path,
        *,
        corrupt_report_hash: bool = False,
    ) -> tuple[pathlib.Path, dict, str]:
        model_id = "M09"
        resolution = 384
        freeze_sha = "a" * 64
        report = {
            "schema": (
                "pneumonia.phase2.finalist.external."
                "rsna_pediatric.resolution.v1"
            ),
            "status": "SCIENTIFIC_RECEIPT_PASS",
            "model_id": model_id,
            "resolution": resolution,
            "selection_freeze_sha256": freeze_sha,
            "scientific_classification": (
                "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION"
            ),
            "training_performed": False,
            "hpo_performed": False,
            "external_threshold_tuning": False,
            "external_adaptation": False,
            "external_calibration_fitting": False,
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "external_may_reselect": False,
            "primary_pediatric_lt10": {
                "metrics": {
                    "accuracy": 0.70,
                    "macro_precision": 0.79,
                    "macro_recall": 0.72,
                }
            },
            "expanded_pediatric_le18": {
                "metrics": {
                    "accuracy": 0.76,
                    "macro_precision": 0.80,
                    "macro_recall": 0.77,
                }
            },
        }
        report_name = f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_REPORT.json"
        report_bytes = json.dumps(
            report, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        zip_path = root / "evidence.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(report_name, report_bytes)
            archive.writestr("A.txt", b"hello")

        report_sha = hashlib.sha256(report_bytes).hexdigest()
        if corrupt_report_hash:
            report_sha = "0" * 64
        receipt = {
            "artifact_sha256": {
                report_name: report_sha,
                "A.txt": hashlib.sha256(b"hello").hexdigest(),
            }
        }
        return zip_path, receipt, freeze_sha

    def test_zip_artifact_and_policy_verification_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path, receipt, freeze_sha = self._fixture(
                pathlib.Path(tmp)
            )
            report = target.verify_zip_artifacts(
                zip_path,
                receipt,
                "M09",
                384,
                freeze_sha,
            )
            self.assertEqual(
                report["primary_pediatric_lt10"]["metrics"]["macro_precision"],
                0.79,
            )

    def test_zip_artifact_hash_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path, receipt, freeze_sha = self._fixture(
                pathlib.Path(tmp),
                corrupt_report_hash=True,
            )
            with self.assertRaisesRegex(
                RuntimeError, "HANDOFF_ZIP_ARTIFACT_SHA_MISMATCH"
            ):
                target.verify_zip_artifacts(
                    zip_path,
                    receipt,
                    "M09",
                    384,
                    freeze_sha,
                )


if __name__ == "__main__":
    unittest.main()
