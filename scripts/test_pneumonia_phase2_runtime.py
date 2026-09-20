import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pneumonia_phase2_runtime as runtime


class Phase2RuntimeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
