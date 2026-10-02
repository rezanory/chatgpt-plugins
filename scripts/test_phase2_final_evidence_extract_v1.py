import importlib.util
import inspect
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
            observed.extend(
                (model, int(resolution))
                for model, resolution, _ in plan["targets"]
            )
        expected = {
            (f"M{index:02d}", resolution)
            for index in range(1, 13)
            for resolution in (224, 320, 384)
        }
        self.assertEqual(len(observed), 36)
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(set(observed)))

    def test_generated_kernel_is_cpu_attached_evidence_only_python(self):
        target = {
            "model_id": "M04",
            "resolution": 384,
            "dataset_ref": "reyhanehazad/pneumonia-m04-r384-state-v1-7",
            "version": 6,
            "campaign_marker_name": "CAMPAIGN_STATE.json",
            "campaign_marker_sha256": "a" * 64,
            "campaign_receipt_sha256": "b" * 64,
            "campaign_status": "COMPLETE",
            "campaign_split_fingerprint": None,
            "campaign_completed_folds": None,
        }
        source = MODULE.kernel_script("kg-04", [target])
        compile(source, "<phase2-final-evidence-kernel>", "exec")
        self.assertIn("FINAL_EVIDENCE.cgpzip", source)
        self.assertIn("FOLD_[1-5]_RECOVERY", source)
        self.assertIn("ATTACHED_DATASET", source)
        self.assertIn("CAMPAIGN_MARKER_SHA_MISMATCH", source)
        self.assertIn("CAMPAIGN_SPLIT_FINGERPRINT_MISMATCH", source)
        self.assertIn("CAMPAIGN_COMPLETED_FOLDS_MISMATCH", source)
        self.assertIn("ACCOUNT_EXTRACT_JSON_SIZE_BOUND_EXCEEDED", source)
        self.assertIn('"analysis_members"', source)
        self.assertIn('"fold_archives"', source)
        self.assertIn('"receipt_sha256"', source)
        self.assertIn("wanted_policy", source)
        self.assertIn("CANONICAL_PHASE2_OOF_FROM_SEALED_FOLDS", source)
        self.assertIn("M07_HIGHRES_VALIDATION_SHA_MISMATCH", source)
        self.assertIn("average_precision_score", source)
        self.assertIn("locked_test_used_for_selection", source)
        self.assertNotIn('"members"=', source)
        self.assertIn("ROOT.rglob(marker_name)", source)
        self.assertIn("DATASET_ATTACHMENT_MARKER_SHA_MATCH_INVALID", source)
        self.assertIn("sha256_file(marker)==expected_marker_sha", source)
        self.assertIn(r'[0-9a-f]{64}', source)
        self.assertNotIn(r'[0-9a-f]64', source)
        self.assertNotIn("kagglehub.dataset_download", source)
        self.assertIn("training_performed", source)
        self.assertNotIn("tensorflow", source.lower())
        self.assertNotIn("torch.", source.lower())
        self.assertNotIn(".fit(", source)
        self.assertNotIn("optimizer", source.lower())

    def test_resolve_standard_marker_uses_exact_dataset_version(self):
        calls = []
        original = MODULE.post_json
        marker_sha = "c" * 64
        try:
            def fake_post(endpoint, token, payload, timeout=180):
                calls.append(payload)
                if payload.get("action") == "raw_read":
                    return {
                        "ok": True,
                        "result": {
                            "datasets": [
                                {
                                    "ref": "reyhanehazad/pneumonia-m04-r384-state-v1-7",
                                    "currentVersionNumber": 6,
                                }
                            ]
                        },
                    }
                if payload.get("action") == "dataset_json_files":
                    return {
                        "ok": True,
                        "result": {
                            "files": [
                                {
                                    "sha256": marker_sha,
                                    "json": {
                                        "status": "COMPLETE",
                                        "receipt_sha256": "d" * 64,
                                    },
                                }
                            ]
                        },
                    }
                raise AssertionError(payload)

            MODULE.post_json = fake_post
            resolved = MODULE.resolve_version(
                "token",
                "kg-04",
                "reyhanehazad/pneumonia-m04-r384-state-v1-7",
            )
            self.assertEqual(resolved["version"], 6)
            self.assertEqual(resolved["campaign_marker_name"], "CAMPAIGN_STATE.json")
            self.assertEqual(resolved["campaign_marker_sha256"], marker_sha)
            self.assertEqual(resolved["campaign_receipt_sha256"], "d" * 64)
            self.assertEqual(resolved["campaign_status"], "COMPLETE")
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]["dataset_version_number"], 6)
            self.assertEqual(calls[1]["file_names"], ["CAMPAIGN_STATE.json"])
        finally:
            MODULE.post_json = original

    def test_resolve_legacy_m07_marker_derives_complete_from_five_folds(self):
        calls = []
        original = MODULE.post_json
        marker_sha = "e" * 64
        split = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
        ref = "rezanory/m07-final-5fold-fix2-d260914d"
        try:
            def fake_post(endpoint, token, payload, timeout=180):
                calls.append(payload)
                if payload.get("action") == "raw_read":
                    return {
                        "ok": True,
                        "result": {
                            "datasets": [
                                {"ref": ref, "currentVersionNumber": 108}
                            ]
                        },
                    }
                if payload.get("action") == "dataset_json_files":
                    return {
                        "ok": True,
                        "result": {
                            "files": [
                                {
                                    "sha256": marker_sha,
                                    "json": {
                                        "schema": "m07.persistence.state.run_safe.v1.6",
                                        "split_fingerprint": split,
                                        "completed_folds": [1, 2, 3, 4, 5],
                                    },
                                }
                            ]
                        },
                    }
                raise AssertionError(payload)

            MODULE.post_json = fake_post
            resolved = MODULE.resolve_version("token", "kg-03", ref)
            self.assertEqual(resolved["version"], 108)
            self.assertEqual(resolved["campaign_marker_name"], "M07_CAMPAIGN_STATE.json")
            self.assertEqual(resolved["campaign_marker_sha256"], marker_sha)
            self.assertIsNone(resolved["campaign_receipt_sha256"])
            self.assertEqual(resolved["campaign_status"], "COMPLETE")
            self.assertEqual(resolved["campaign_split_fingerprint"], split)
            self.assertEqual(resolved["campaign_completed_folds"], [1, 2, 3, 4, 5])
            self.assertEqual(calls[1]["file_names"], ["M07_CAMPAIGN_STATE.json"])
        finally:
            MODULE.post_json = original

    def test_launch_uses_unversioned_attached_states(self):
        source = inspect.getsource(MODULE.launch_account)
        self.assertIn(
            '"datasetDataSources": [item["dataset_ref"] for item in resolved]',
            source,
        )
        self.assertIn("EXTRACTOR_GENERATION", source)
        self.assertIn("phase2-final-evidence-{EXTRACTOR_GENERATION}", source)
        self.assertIn('"enableInternet": False', source)
        self.assertIn("campaign_marker_name", source)
        self.assertIn("campaign_marker_sha256", source)
        self.assertIn("invalidDatasetSources", source)
        self.assertIn('broker.get("provider_ref")', source)

    def test_action_envelope_accepts_direct_and_nested_broker_shapes(self):
        direct = {
            "ok": True,
            "provider": "kaggle",
            "operation_class": "compute",
            "provider_ref": "/code/example/kernel",
            "result": {"ref": "/code/example/kernel", "versionNumber": 1},
        }
        broker, provider = MODULE._action_envelope(direct)
        self.assertIs(broker, direct)
        self.assertEqual(provider["versionNumber"], 1)

        nested = {"ok": True, "result": dict(direct)}
        broker, provider = MODULE._action_envelope(nested)
        self.assertEqual(broker["provider"], "kaggle")
        self.assertEqual(provider["versionNumber"], 1)

    def test_read_payload_accepts_direct_and_nested_broker_shapes(self):
        payload = {"files": [{"file_name": "x.json"}]}
        direct = {"ok": True, "result": payload}
        nested = {
            "ok": True,
            "provider": "kaggle",
            "read_only": True,
            "result": {
                "ok": True,
                "provider": "kaggle",
                "read_only": True,
                "broker_run_id": "123",
                "result": payload,
            },
        }
        self.assertEqual(MODULE._read_payload(direct), payload)
        self.assertEqual(MODULE._read_payload(nested), payload)

    def test_reusable_extractor_selects_latest_complete_kernel(self):
        original = MODULE.post_json
        try:
            def fake_post(endpoint, token, payload, timeout=180):
                if payload.get("action") == "raw_read" and payload.get("method") == "ListKernels":
                    return {
                        "ok": True,
                        "result": {
                            "kernels": [
                                {
                                    "ref": "radlinaradlina/phase2-final-evidence-v2-kg-02-36923248897",
                                    "status": "ERROR",
                                },
                                {
                                    "ref": "radlinaradlina/phase2-final-evidence-kg-02-36927131810",
                                    "status": "COMPLETE",
                                },
                            ]
                        },
                    }
                if payload.get("action") == "raw_read" and payload.get("method") == "GetKernelSessionStatus":
                    return {"ok": True, "result": {"status": "COMPLETE"}}
                raise AssertionError(payload)

            MODULE.post_json = fake_post
            reused = MODULE.reusable_extractor("token", "kg-02", "radlinaradlina")
            self.assertIsNotNone(reused)
            self.assertEqual(
                reused["kernel_ref"],
                "radlinaradlina/phase2-final-evidence-kg-02-36927131810",
            )
            self.assertEqual(reused["status"], "COMPLETE")
            self.assertEqual(reused["source_run_id"], "36927131810")
        finally:
            MODULE.post_json = original

    def test_reusable_extractor_does_not_reuse_latest_failed_kernel(self):
        original = MODULE.post_json
        try:
            def fake_post(endpoint, token, payload, timeout=180):
                if payload.get("action") == "raw_read" and payload.get("method") == "ListKernels":
                    return {
                        "ok": True,
                        "result": {
                            "kernels": [
                                {
                                    "ref": "rezanory/phase2-final-evidence-v2-kg-03-36927131810",
                                    "status": "ERROR",
                                }
                            ]
                        },
                    }
                if payload.get("action") == "raw_read" and payload.get("method") == "GetKernelSessionStatus":
                    return {"ok": True, "result": {"status": "ERROR"}}
                raise AssertionError(payload)

            MODULE.post_json = fake_post
            self.assertIsNone(
                MODULE.reusable_extractor("token", "kg-03", "rezanory")
            )
        finally:
            MODULE.post_json = original

    def test_probe_status_falls_back_to_verified_output_receipt(self):
        original = MODULE.post_json
        try:
            def fake_post(endpoint, token, payload, timeout=180):
                if payload.get("action") == "raw_read" and payload.get("method") == "GetKernelSessionStatus":
                    return {"ok": False, "http_status": 403, "error": "Kaggle API HTTP 403"}
                if payload.get("action") == "output_json_files":
                    return {
                        "ok": True,
                        "provider": "kaggle",
                        "read_only": True,
                        "result": {
                            "ok": True,
                            "provider": "kaggle",
                            "read_only": True,
                            "broker_run_id": "nested",
                            "result": {
                                "files": [
                                    {
                                        "json": {"status": "PASS"},
                                    }
                                ]
                            },
                        },
                    }
                raise AssertionError(payload)

            MODULE.post_json = fake_post
            status = MODULE.probe_extractor_status(
                "token",
                "kg-09",
                "mylovevpn1/phase2-final-evidence-v2-kg-09-36927131810",
            )
            self.assertEqual(status, "COMPLETE")
        finally:
            MODULE.post_json = original

    def test_probe_status_falls_back_to_runtime_error_log(self):
        original = MODULE.post_json
        try:
            def fake_post(endpoint, token, payload, timeout=180):
                if payload.get("action") == "raw_read" and payload.get("method") == "GetKernelSessionStatus":
                    return {"ok": False, "http_status": 403, "error": "Kaggle API HTTP 403"}
                if payload.get("action") == "output_json_files":
                    return {"ok": False, "http_status": 403, "error": "not found"}
                if payload.get("action") == "raw_read" and payload.get("method") == "ListKernelSessionOutput":
                    return {
                        "ok": True,
                        "provider": "kaggle",
                        "read_only": True,
                        "result": {
                            "ok": True,
                            "provider": "kaggle",
                            "read_only": True,
                            "broker_run_id": "nested",
                            "result": {
                                "files": [],
                                "log": "Traceback (most recent call last):\\nRuntimeError: TEST",
                            },
                        },
                    }
                raise AssertionError(payload)

            MODULE.post_json = fake_post
            status = MODULE.probe_extractor_status(
                "token",
                "kg-09",
                "mylovevpn1/phase2-final-evidence-v2-kg-09-36927131810",
            )
            self.assertEqual(status, "ERROR")
        finally:
            MODULE.post_json = original

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
