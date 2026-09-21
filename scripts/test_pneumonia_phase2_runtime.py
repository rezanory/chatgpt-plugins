import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pneumonia_phase2_runtime as runtime


class Phase2RuntimeTests(unittest.TestCase):
    def _state_marker(self, model_id="M02", resolution=224):
        return {
            "schema": "phase2.state.v2",
            "status": "COMPLETE",
            "model_id": model_id,
            "resolution": resolution,
            "run_contract": {
                "schema": "pneumonia.experiment.v1.7",
                "stage": "phase2_campaign",
                "model_id": model_id,
                "resolution": resolution,
                "split_fingerprint": runtime.SPLIT_FINGERPRINT,
                "extra": {"recipe_fingerprint": runtime.FROZEN_M07_RECIPE_FINGERPRINT},
            },
            "run_fingerprint": "a" * 64,
            "artifact_sha256": {"FOLD_1_RECOVERY.cgpzip": "b" * 64},
            "receipt_sha256": "c" * 64,
        }

    def test_state_marker_requires_exact_scientific_lineage(self):
        contract = runtime.unit_contract("PHASE2_UNIT_M02_R224_A09", "35599900009", "20260921")
        marker = self._state_marker()
        self.assertEqual(runtime._state_marker_folds(marker, contract), [1])
        drift = json.loads(json.dumps(marker))
        drift["run_contract"]["extra"]["recipe_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "state contract drift"):
            runtime._state_marker_folds(drift, contract)

    def test_state_readiness_requires_sealed_cgpzip_topology(self):
        snapshots = [
            {
                "completed_folds": [1],
                "topology": "LEGACY_KAGGLE_EXPANDED_ZIP",
            },
            {"completed_folds": [1], "topology": "SEALED_CGPZIP"},
        ]
        with (
            mock.patch.object(runtime, "_state_snapshot", side_effect=snapshots),
            mock.patch.object(runtime.time, "sleep"),
        ):
            result = runtime._wait_for_state_ready(
                mock.Mock(), {}, [1], max_polls=2, interval_seconds=0
            )
        self.assertEqual(result["readiness_poll_count"], 2)

    def test_state_snapshot_classifies_real_legacy_expanded_shape(self):
        token = "PHASE2_UNIT_M02_R224_A09"
        contract = runtime.unit_contract(token, "35599900009", "20260921")
        marker = self._state_marker()

        class Broker:
            def read(self, payload, timeout=90):
                if payload.get("method") == "ListDatasets":
                    return {
                        "datasets": [
                            {
                                "ref": contract["state_dataset_handle"],
                                "currentVersionNumber": 3,
                            }
                        ]
                    }
                if payload.get("action") == "dataset_json_files":
                    return {
                        "dataset_ref": contract["state_dataset_handle"],
                        "dataset_version_number": 3,
                        "signed_urls_returned": False,
                        "files": [{"sha256": "d" * 64, "json": marker}],
                    }
                if payload.get("method") == "ListDatasetFiles":
                    return {
                        "datasetFiles": [
                            {"name": "CAMPAIGN_STATE.json"},
                            {"name": "FOLD_1_RECOVERY/FOLDS/fold_1/COMPLETED.json"},
                        ]
                    }
                raise AssertionError(payload)

        snapshot = runtime._state_snapshot(Broker(), contract)
        self.assertEqual(snapshot["completed_folds"], [1])
        self.assertEqual(snapshot["topology"], "LEGACY_KAGGLE_EXPANDED_ZIP")
        self.assertEqual(snapshot["dataset_version_number"], 3)

    def test_preflight_rejects_existing_canonicalized_provider_candidate(self):
        token = "PHASE2_UNIT_M01_R224_A09"
        run_id = "35599900009"
        contract = runtime.unit_contract(token, run_id)
        provider_ref = runtime.expected_provider_kernel_ref(contract)

        class Broker:
            def read(self, payload, timeout=90):
                if payload.get("method") == "ListKernels":
                    return {"kernels": [{"ref": provider_ref, "status": "COMPLETE"}]}
                raise AssertionError(payload)

        with (
            mock.patch.object(runtime, "OidcReadBroker", return_value=Broker()),
            mock.patch.object(
                runtime,
                "_state_snapshot",
                return_value={
                    "completed_folds": [1],
                    "topology": "LEGACY_KAGGLE_EXPANDED_ZIP",
                },
            ),
            self.assertRaisesRegex(RuntimeError, "exact Phase2 kernel candidate already exists"),
        ):
            runtime.admission_preflight(token, run_id)

    def test_oidc_refresh_retries_transient_transport(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return ('{"value":"' + ("r" * 120) + '"}').encode()

        with (
            mock.patch.dict(
                os.environ,
                {
                    "CGP_READ_OIDC_TOKEN": "i" * 120,
                    "ACTIONS_ID_TOKEN_REQUEST_URL": "https://oidc.example/token",
                    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "request-token-long-enough",
                },
                clear=False,
            ),
            mock.patch.object(
                runtime.urllib.request,
                "urlopen",
                side_effect=[
                    runtime.urllib.error.URLError("one"),
                    runtime.urllib.error.URLError("two"),
                    Response(),
                ],
            ) as urlopen,
            mock.patch.object(runtime.time, "sleep"),
        ):
            broker = runtime.OidcReadBroker()
            self.assertEqual(broker.refresh(), "r" * 120)
            self.assertEqual(urlopen.call_count, 3)

    def test_exact_provider_ref_matches_kaggle_savekernel_canonicalization(self):
        contract = {
            "owner": "azadka",
            "model_id": "M01",
            "resolution": 224,
            "attempt": 1,
            "run_id": "35531382748",
        }
        self.assertEqual(
            runtime.expected_provider_kernel_ref(contract),
            "azadka/pneumonia-v1-7-phase2-m01-r224-a01-35531382748",
        )

    def test_session_state_normalizes_provider_payloads(self):
        cases = (
            ({"status": "RUNNING"}, "RUNNING"),
            ({"state": {"name": "COMPLETE"}}, "COMPLETE"),
            ({"message": "cancel_acknowledged"}, "CANCEL"),
            ({"failure": "ERROR"}, "ERROR"),
            ({"value": "not available"}, "UNKNOWN"),
        )
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(runtime.session_state(payload), expected)

    def test_cancel_wins_over_other_nested_text(self):
        payload = {"old": "RUNNING", "current": "CANCEL_ACKNOWLEDGED"}
        self.assertEqual(runtime.session_state(payload), "CANCEL")

    def test_exact_listing_identity_accepts_ref_or_owner_slug(self):
        self.assertTrue(
            runtime._listed_kernel_matches(
                {"ref": "azadka/p17-p2-m01-r224-a01"},
                "azadka",
                "p17-p2-m01-r224-a01",
            )
        )
        self.assertTrue(
            runtime._listed_kernel_matches(
                {"ownerRef": "azadka", "kernelSlug": "p17-p2-m01-r224-a01"},
                "azadka",
                "p17-p2-m01-r224-a01",
            )
        )
        self.assertFalse(
            runtime._listed_kernel_matches(
                {"ref": "azadka/p17-p2-m01-r224-a02"},
                "azadka",
                "p17-p2-m01-r224-a01",
            )
        )

    def test_listing_state_uses_explicit_fields_only(self):
        self.assertEqual(runtime._listed_kernel_state({"status": "RUNNING"}), "RUNNING")
        self.assertEqual(runtime._listed_kernel_state({"isRunning": True}), "RUNNING")
        self.assertEqual(
            runtime._listed_kernel_state({"title": "COMPLETE looking title"}),
            "UNKNOWN",
        )

    def test_status_falls_back_to_exact_listing_after_provider_403(self):
        class Broker:
            def __init__(self):
                self.calls = []

            def read(self, payload, timeout=90):
                self.calls.append((payload, timeout))
                method = payload.get("method")
                if method == "GetKernelSessionStatus":
                    raise RuntimeError(
                        'Phase2 read broker HTTP 403: {"error":"Kaggle API HTTP 403"}'
                    )
                if method == "ListKernels":
                    return {"kernels": [{"ref": "azadka/p17-p2-m01-r224-a01", "status": "RUNNING"}]}
                raise AssertionError(payload)

        broker = Broker()
        result = runtime._status(
            broker,
            {"account_id": "master", "owner": "azadka"},
            "azadka/p17-p2-m01-r224-a01",
        )
        self.assertEqual(result["state"], "RUNNING")
        self.assertEqual(result["status_transport"], "LIST_KERNELS_EXACT_FALLBACK")
        self.assertEqual(
            [call[0].get("method") for call in broker.calls],
            ["GetKernelSessionStatus", "ListKernels"],
        )

    def test_status_fallback_requires_one_exact_kernel(self):
        class Broker:
            def read(self, payload, timeout=90):
                return {"kernels": [{"ref": "azadka/some-other-kernel"}]}

        with self.assertRaisesRegex(RuntimeError, "expected one exact listed kernel"):
            runtime._status_from_exact_listing(
                Broker(),
                {"account_id": "master", "owner": "azadka"},
                "azadka/p17-p2-m01-r224-a01",
                "azadka",
                "p17-p2-m01-r224-a01",
            )

    def test_status_fallback_accepts_exact_terminal_receipt(self):
        class Broker:
            def read(self, payload, timeout=90):
                if payload.get("method") == "ListKernels":
                    return {"kernels": [{"ref": "azadka/p17-p2-m01-r224-a01"}]}
                if payload.get("action") == "output_json_files":
                    return {
                        "files": [
                            {
                                "source_file_name": "PHASE2_UNIT_TERMINAL_RECEIPT.json",
                                "json": {"status": "PARTIAL"},
                            }
                        ]
                    }
                raise AssertionError(payload)

        result = runtime._status_from_exact_listing(
            Broker(),
            {"account_id": "master", "owner": "azadka"},
            "azadka/p17-p2-m01-r224-a01",
            "azadka",
            "p17-p2-m01-r224-a01",
        )
        self.assertEqual(result["state"], "COMPLETE")
        self.assertEqual(result["status_transport"], "EXACT_TERMINAL_RECEIPT_FALLBACK")

    def test_terminal_verifier_survives_one_broker_transport_incident(self):
        class Broker:
            def read(self, payload, timeout=90):
                raise RuntimeError("live log unavailable")

        token = "PHASE2_UNIT_M01_R224_A03"
        run_id = "35539262689"
        kernel_ref = "azadka/p17-p2-m01-r224-a03-20260920-35539262689"
        provider_ref = "azadka/pneumonia-v1-7-phase2-m01-r224-a03-35539262689"
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(runtime, "OidcReadBroker", return_value=Broker()),
            mock.patch.object(
                runtime,
                "_status",
                side_effect=[
                    runtime.BrokerTransientError("temporary broker timeout"),
                    {"status": "ERROR"},
                ],
            ),
            mock.patch.object(runtime.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "PHASE2_UNIT_SCIENTIFIC_TERMINAL_ERROR"):
                runtime.verify_terminal(
                    token,
                    run_id,
                    kernel_ref,
                    provider_ref,
                    pathlib.Path(temp),
                    max_polls=2,
                    interval_seconds=0,
                )
            evidence = pathlib.Path(temp, "phase2-unit-verification.json")
            payload = __import__("json").loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(payload["terminal_state"], "ERROR")
            self.assertEqual(payload["poll_count"], 2)
            self.assertEqual(payload["history"][0]["state"], "UNKNOWN")
            self.assertIn("temporary broker timeout", payload["history"][0]["transport_error"])


if __name__ == "__main__":
    unittest.main()
