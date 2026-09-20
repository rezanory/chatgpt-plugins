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


if __name__ == "__main__":
    unittest.main()
