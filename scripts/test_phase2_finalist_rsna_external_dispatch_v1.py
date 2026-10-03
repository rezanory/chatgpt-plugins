from __future__ import annotations

import hashlib
import json
import pathlib
import unittest

import phase2_finalist_rsna_external_dispatch_v1 as target


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FinalistRsnaExternalTests(unittest.TestCase):
    def test_selection_freeze_is_exact_and_report_only(self):
        payload, digest = target.load_and_verify_selection_freeze(ROOT)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            {
                (row["model_id"], int(row["resolution"]))
                for row in payload["frozen_candidate_set"]
            },
            {("M09", 384), ("M10", 320)},
        )
        self.assertFalse(payload["locked_test_may_reselect"])
        self.assertFalse(payload["external_may_reselect"])
        self.assertEqual(
            payload["selection_basis"]["requested_metrics"],
            ["macro_precision", "macro_recall"],
        )

    def test_contracts_match_frozen_candidates(self):
        observed = {
            (row["model_id"], int(row["resolution"]))
            for row in target.CONTRACTS.values()
        }
        self.assertEqual(observed, target.EXPECTED_CANDIDATES)
        self.assertEqual(
            target.CONTRACTS["PHASE2_FINALIST_RSNA_M09_R384_V1"]["state_handle"],
            "azadkk/pneumonia-m09-r384-state-v1-7",
        )
        self.assertEqual(
            target.CONTRACTS["PHASE2_FINALIST_RSNA_M10_R320_V1"]["state_handle"],
            "mylovevpn1/pneumonia-m10-r320-state-v1-7",
        )

    def test_generated_kernel_is_inference_only_and_model_bound(self):
        _, freeze_sha = target.load_and_verify_selection_freeze(ROOT)
        source = (
            ROOT / "scripts" / "m07_rsna_pediatric_external_v1.py"
        ).read_text(encoding="utf-8")
        for token in (
            "PHASE2_FINALIST_RSNA_M09_R384_V1",
            "PHASE2_FINALIST_RSNA_M10_R320_V1",
        ):
            contract = target.CONTRACTS[token]
            code = target.build_kernel_script(
                source,
                contract,
                manifest_b64="eJyrVkrLz1eyUkpKLFKqBQAqWgQO",
                selection_freeze_sha256=freeze_sha,
            )
            compile(code, "<test-finalist-rsna>", "exec")
            self.assertIn(
                f"MODEL_ID={contract['model_id']!r}",
                code,
            )
            self.assertIn(
                f"SELECTION_FREEZE_SHA256={freeze_sha!r}",
                code,
            )
            self.assertIn(
                '"schema": "pneumonia.phase2.finalist.external.rsna_pediatric.terminal.v1"',
                code,
            )
            self.assertIn(
                'receipt.get("model_id") == MODEL_ID',
                code,
            )
            self.assertIn(
                'payload.get("model_id") == MODEL_ID',
                code,
            )
            self.assertIn(
                'state_root.rglob("OOF_PREDICTIONS.csv")',
                code,
            )
            self.assertIn(
                '"external_may_reselect": False',
                code,
            )
            self.assertNotIn("model.fit(", code)
            self.assertNotIn("optimizer.apply_gradients", code)

    def test_receipt_verifier_accepts_exact_policy(self):
        _, freeze_sha = target.load_and_verify_selection_freeze(ROOT)
        contract = target.CONTRACTS["PHASE2_FINALIST_RSNA_M09_R384_V1"]
        receipt = {
            "schema": "pneumonia.phase2.finalist.external.rsna_pediatric.terminal.v1",
            "status": "SCIENTIFIC_RECEIPT_PASS",
            "model_id": "M09",
            "resolution": 384,
            "primary_n": target.EXPECTED_COUNTS["primary_n"],
            "primary_normal": target.EXPECTED_COUNTS["primary_normal"],
            "primary_lung_opacity": target.EXPECTED_COUNTS["primary_lung_opacity"],
            "expanded_n": target.EXPECTED_COUNTS["expanded_n"],
            "expanded_normal": target.EXPECTED_COUNTS["expanded_normal"],
            "expanded_lung_opacity": target.EXPECTED_COUNTS["expanded_lung_opacity"],
            "training_performed": False,
            "hpo_performed": False,
            "external_threshold_tuning": False,
            "external_adaptation": False,
            "external_calibration_fitting": False,
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "external_may_reselect": False,
            "source_state": contract["state_handle"],
            "split_fingerprint": target.EXPECTED_SPLIT,
            "hpo_recipe_fingerprint": target.EXPECTED_RECIPE,
            "external_dataset": target.DATASET_REF,
            "external_manifest_sha256": target.EXPECTED_MANIFEST_SHA256,
            "selection_freeze_sha256": freeze_sha,
            "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",
            "artifact_sha256": {"x": "0" * 64},
        }
        body = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        receipt["receipt_sha256"] = hashlib.sha256(body).hexdigest()
        target.verify_receipt(
            receipt,
            contract,
            selection_freeze_sha256=freeze_sha,
        )

    def test_receipt_verifier_rejects_reselection(self):
        _, freeze_sha = target.load_and_verify_selection_freeze(ROOT)
        contract = target.CONTRACTS["PHASE2_FINALIST_RSNA_M10_R320_V1"]
        receipt = {
            "schema": "pneumonia.phase2.finalist.external.rsna_pediatric.terminal.v1",
            "status": "SCIENTIFIC_RECEIPT_PASS",
            "model_id": "M10",
            "resolution": 320,
            "primary_n": target.EXPECTED_COUNTS["primary_n"],
            "primary_normal": target.EXPECTED_COUNTS["primary_normal"],
            "primary_lung_opacity": target.EXPECTED_COUNTS["primary_lung_opacity"],
            "expanded_n": target.EXPECTED_COUNTS["expanded_n"],
            "expanded_normal": target.EXPECTED_COUNTS["expanded_normal"],
            "expanded_lung_opacity": target.EXPECTED_COUNTS["expanded_lung_opacity"],
            "training_performed": False,
            "hpo_performed": False,
            "external_threshold_tuning": False,
            "external_adaptation": False,
            "external_calibration_fitting": False,
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "external_may_reselect": True,
            "source_state": contract["state_handle"],
            "split_fingerprint": target.EXPECTED_SPLIT,
            "hpo_recipe_fingerprint": target.EXPECTED_RECIPE,
            "external_dataset": target.DATASET_REF,
            "external_manifest_sha256": target.EXPECTED_MANIFEST_SHA256,
            "selection_freeze_sha256": freeze_sha,
            "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",
            "artifact_sha256": {"x": "0" * 64},
        }
        body = json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        receipt["receipt_sha256"] = hashlib.sha256(body).hexdigest()
        with self.assertRaisesRegex(
            RuntimeError, "FINALIST_RSNA_TERMINAL_RECEIPT_MISMATCH"
        ):
            target.verify_receipt(
                receipt,
                contract,
                selection_freeze_sha256=freeze_sha,
            )


if __name__ == "__main__":
    unittest.main()
