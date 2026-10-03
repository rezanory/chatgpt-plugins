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
        self.assertIn("dataset_version_number", MODULE.vector_kernel_script("kg-05", [{
            "model_id":"M07",
            "resolution":320,
            "dataset_ref":"trickermark/m07-gate-r320-state-v1-7",
            "version":2,
            "campaign_marker_name":"CAMPAIGN_STATE.json",
            "campaign_marker_sha256":"a"*64,
            "campaign_receipt_sha256":"b"*64,
            "campaign_status":"COMPLETE",
            "campaign_split_fingerprint":"c"*64,
            "campaign_completed_folds":None,
        }]))

    def test_reuse_requires_exact_resolved_evidence_identity(self):
        targets=[{
            "model_id":"M01",
            "resolution":224,
            "dataset_ref":"azadka/pneumonia-m01-r224-state-v1-7",
            "version":7,
            "campaign_marker_sha256":"a"*64,
        }]
        original_post=MODULE.post_json
        original_summary=MODULE.fetch_account_summary
        try:
            def fake_post(endpoint,token,payload,timeout=180):
                if payload.get("method")=="ListKernels":
                    return {"ok":True,"result":{"kernels":[{
                        "ref":"azadka/phase2-oof-vectors-v1-master-123",
                        "status":"COMPLETE",
                    }]}}
                if payload.get("method")=="GetKernelSessionStatus":
                    return {"ok":True,"result":{"status":"COMPLETE"}}
                raise AssertionError(payload)

            MODULE.post_json=fake_post
            MODULE.fetch_account_summary=lambda *args: {
                "schema":"pneumonia.phase2.oof.vector.account.v1",
                "status":"PASS",
                "account_id":"master",
                "target_count":1,
                "targets":[{
                    "model_id":"M01",
                    "resolution":224,
                    "dataset_ref":"azadka/pneumonia-m01-r224-state-v1-7",
                    "dataset_version_number":6,
                    "campaign_marker_sha256":"a"*64,
                }],
            }
            self.assertIsNone(
                MODULE.reusable_account("token","master","azadka",targets)
            )

            MODULE.fetch_account_summary=lambda *args: {
                "schema":"pneumonia.phase2.oof.vector.account.v1",
                "status":"PASS",
                "account_id":"master",
                "target_count":1,
                "targets":[{
                    "model_id":"M01",
                    "resolution":224,
                    "dataset_ref":"azadka/pneumonia-m01-r224-state-v1-7",
                    "dataset_version_number":7,
                    "campaign_marker_sha256":"a"*64,
                }],
            }
            reused=MODULE.reusable_account("token","master","azadka",targets)
            self.assertIsNotNone(reused)
            self.assertEqual(reused["source_run_id"],"123")
        finally:
            MODULE.post_json=original_post
            MODULE.fetch_account_summary=original_summary

    def test_candidate_prefixes_accept_provider_normalized_ref(self):
        prefixes=MODULE._candidate_prefixes("kg-09")
        self.assertIn("phase2-oof-vectors-v1-kg-09-",prefixes)
        self.assertIn("phase2-oof-vectors-kg-09-",prefixes)

    def test_fetch_vectors_retries_output_propagation(self):
        launch={
            "account_id":"kg-09",
            "kernel_ref":"mylovevpn1/phase2-oof-vectors-kg-09-123",
            "targets":[{"model_id":"M10","resolution":224}],
        }
        vector={
            "schema":"pneumonia.phase2.oof.vector.v1",
            "status":"PASS",
            "model_id":"M10",
            "resolution":224,
        }
        original_post=MODULE.post_json
        original_sleep=MODULE.time.sleep
        calls=[]
        try:
            def fake_post(endpoint,token,payload,timeout=180):
                calls.append(payload)
                if len(calls)==1:
                    return {
                        "ok":False,
                        "http_status":403,
                        "error":(
                            '{"ok":false,"read_only":true,"error":'
                            '"requested Kaggle output JSON not found: '
                            'PHASE2_OOF_VECTOR_M10_R224.json"}'
                        ),
                    }
                if len(calls)==2:
                    return {"ok":True,"result":{"files":[]}}
                return {
                    "ok":True,
                    "provider":"kaggle",
                    "read_only":True,
                    "result":{
                        "ok":True,
                        "provider":"kaggle",
                        "read_only":True,
                        "result":{
                            "files":[{
                                "file_name":"PHASE2_OOF_VECTOR_M10_R224.json",
                                "json":vector,
                            }]
                        },
                    },
                }
            MODULE.post_json=fake_post
            MODULE.time.sleep=lambda _: None
            result=MODULE.fetch_vectors("token",launch)
            self.assertEqual(len(calls),3)
            self.assertEqual(result,[vector])
        finally:
            MODULE.post_json=original_post
            MODULE.time.sleep=original_sleep

    def test_fetch_vectors_fails_fast_on_permanent_read_error(self):
        launch={
            "account_id":"kg-09",
            "kernel_ref":"mylovevpn1/phase2-oof-vectors-kg-09-123",
            "targets":[{"model_id":"M10","resolution":224}],
        }
        original_post=MODULE.post_json
        original_sleep=MODULE.time.sleep
        calls=[]
        try:
            def fake_post(endpoint,token,payload,timeout=180):
                calls.append(payload)
                return {
                    "ok":False,
                    "http_status":403,
                    "error":"Kaggle API HTTP 403: permission denied",
                }
            MODULE.post_json=fake_post
            MODULE.time.sleep=lambda _: None
            with self.assertRaisesRegex(RuntimeError,"OOF_VECTOR_OUTPUT_FETCH_FAILED"):
                MODULE.fetch_vectors("token",launch)
            self.assertEqual(len(calls),1)
        finally:
            MODULE.post_json=original_post
            MODULE.time.sleep=original_sleep

    def test_reuse_accepts_exact_pass_summary_when_status_api_is_unavailable(self):
        targets=[{
            "model_id":"M10",
            "resolution":224,
            "dataset_ref":"mylovevpn1/pneumonia-m10-r224-state-v1-7",
            "version":5,
            "campaign_marker_sha256":"a"*64,
        }]
        original_post=MODULE.post_json
        original_summary=MODULE.fetch_account_summary
        try:
            def fake_post(endpoint,token,payload,timeout=180):
                if payload.get("method")=="ListKernels":
                    self.assertEqual(payload["body"]["search"],"phase2-oof-vectors")
                    return {"ok":True,"result":{"kernels":[{
                        "ref":"mylovevpn1/phase2-oof-vectors-kg-09-37115268419",
                    }]}}
                if payload.get("method")=="GetKernelSessionStatus":
                    return {"ok":False,"http_status":403,"error":"Kaggle API HTTP 403"}
                raise AssertionError(payload)
            MODULE.post_json=fake_post
            MODULE.fetch_account_summary=lambda *args: {
                "schema":"pneumonia.phase2.oof.vector.account.v1",
                "status":"PASS",
                "account_id":"kg-09",
                "target_count":1,
                "targets":[{
                    "model_id":"M10",
                    "resolution":224,
                    "dataset_ref":"mylovevpn1/pneumonia-m10-r224-state-v1-7",
                    "dataset_version_number":5,
                    "campaign_marker_sha256":"a"*64,
                }],
            }
            reused=MODULE.reusable_account("token","kg-09","mylovevpn1",targets)
            self.assertIsNotNone(reused)
            self.assertEqual(reused["kernel_ref"],"mylovevpn1/phase2-oof-vectors-kg-09-37115268419")
            self.assertEqual(reused["source_run_id"],"37115268419")
        finally:
            MODULE.post_json=original_post
            MODULE.fetch_account_summary=original_summary

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
