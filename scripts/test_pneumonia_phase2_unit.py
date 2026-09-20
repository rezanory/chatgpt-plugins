import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pneumonia_phase2_unit as phase2

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "control-plane-payloads"
    / "pneumonia-v17-20260912"
    / "PNEUMONIA_MASTER_M01_M12_M07_GATE_RUN_SAFE_V1_7.ipynb"
)
MANIFEST = (
    ROOT
    / "control-plane-payloads"
    / "pneumonia-v17-20260912"
    / "PHASE2_DISTRIBUTED_MANIFEST_V1.json"
)


class Phase2UnitTests(unittest.TestCase):
    def test_manifest_has_exact_nonoverlapping_33_units(self):
        manifest = phase2.campaign_manifest()
        units = manifest["units"]
        identities = {(item["model_id"], item["resolution"]) for item in units}
        self.assertEqual(len(units), 33)
        self.assertEqual(len(identities), 33)
        self.assertEqual(set(manifest["models"]), set(phase2.MODEL_ASSIGNMENTS))
        self.assertNotIn("M07", manifest["models"])
        self.assertTrue(all(item["folds"] == [1, 2, 3, 4, 5] for item in units))

    def test_frozen_manifest_matches_generated_contract(self):
        frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(frozen, phase2.campaign_manifest())

    def test_assignment_covers_all_11_accounts_once(self):
        accounts = [item["account_id"] for item in phase2.MODEL_ASSIGNMENTS.values()]
        owners = [item["owner"] for item in phase2.MODEL_ASSIGNMENTS.values()]
        self.assertEqual(len(accounts), 11)
        self.assertEqual(len(set(accounts)), 11)
        self.assertEqual(len(set(owners)), 11)

    def test_token_contract_is_exact(self):
        contract = phase2.unit_contract(
            "PHASE2_UNIT_M08_R384_A07", "35599900001", "20260920"
        )
        self.assertEqual(contract["account_id"], "kg-07")
        self.assertEqual(contract["owner"], "nisabulutmark")
        self.assertEqual(contract["model_id"], "M08")
        self.assertEqual(contract["resolution"], 384)
        self.assertEqual(contract["max_new_folds"], 1)
        self.assertIn("m08-r384-a07-20260920-35599900001", contract["kernel_ref"])

    def test_invalid_or_m07_token_is_rejected(self):
        for token in (
            "PHASE2_UNIT_M07_R224_A01",
            "PHASE2_UNIT_M01_R256_A01",
            "PHASE2_UNIT_M01_R224_A00",
            "M07_PHASE2_UNLOCK_APPROVED_BY_USER",
        ):
            with self.subTest(token=token), self.assertRaises(phase2.Phase2ContractError):
                phase2.parse_token(token)

    def test_notebook_build_keeps_definitions_and_one_bounded_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            generated = pathlib.Path(temp) / "generated.ipynb"
            output = pathlib.Path(temp) / "unit.ipynb"
            source_notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
            source_notebook["cells"][12]["source"] = [
                "STAGEB_WINNER = {'params': {}}\n",
                "shared_params = STAGEB_WINNER['params']\n",
                "recipe = {'recipe_fingerprint_sha256': 'test'}\n",
                "print('M07_CONTINUATION_PRECHECK_PASS')\n",
            ]
            generated.write_text(json.dumps(source_notebook), encoding="utf-8")
            result = phase2.build_notebook(
                generated,
                output,
                "PHASE2_UNIT_M01_R224_A01",
                "35599900002",
                "20260920",
            )
            notebook = json.loads(output.read_text(encoding="utf-8"))
            joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
            self.assertEqual(result["output_cell_count"], 15)
            self.assertIn("PHASE2_UNIT_MODEL_ID = 'M01'", joined)
            self.assertIn("CGP_PHASE2_MAX_NEW_FOLDS = 1", joined)
            self.assertIn("PHASE2_UNIT_LEGACY_M07_PERSISTENCE_SKIPPED", joined)
            self.assertEqual(joined.count("PHASE2_UNIT_NEW_FOLD_BOUND_EXCEEDED"), 1)
            self.assertIn("run_phase2_model_resolution(\n    PHASE2_UNIT_MODEL_ID", joined)
            self.assertNotIn("RESTORE_SUMMARY = _try_restore_persisted_state_compat()", joined)
            self.assertNotIn("RESTORE_SUMMARY = _try_restore_persisted_state()", joined)
            self.assertNotIn("M07 run-safe persistence healthcheck before training", joined)
            self.assertNotIn("M07 RUNTIME PRE-FLIGHT PASS", joined)
            self.assertNotIn("M07_CONTINUATION_PRECHECK_PASS", joined)
            self.assertNotIn("MASTER / PHASE-2 ORCHESTRATOR", joined)
            self.assertNotIn("CONFIRMATION HPO — DETERMINISTIC", joined)

    def test_raw_canonical_notebook_is_rejected_before_hpo_can_run(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaises(phase2.Phase2ContractError),
        ):
            phase2.build_notebook(
                NOTEBOOK,
                pathlib.Path(temp) / "unit.ipynb",
                "PHASE2_UNIT_M01_R224_A01",
                "35599900004",
                "20260920",
            )

    def test_partial_and_complete_receipts_obey_fold_policy(self):
        contract = phase2.unit_contract(
            "PHASE2_UNIT_M12_R320_A05", "35599900003", "20260920"
        )
        partial = {
            "schema": "pneumonia.phase2.unit.terminal.v1",
            "status": "PARTIAL",
            "scientific_pass": False,
            "model_id": "M12",
            "resolution": 320,
            "completed_folds": [1, 2, 3],
            "new_folds_completed": 1,
            "five_fold_ready": False,
            "locked_test_started": False,
        }
        self.assertIs(phase2.validate_terminal_receipt(partial, contract), partial)
        complete = {
            **partial,
            "status": "COMPLETE",
            "scientific_pass": True,
            "completed_folds": [1, 2, 3, 4, 5],
            "five_fold_ready": True,
            "locked_test_started": True,
        }
        self.assertIs(phase2.validate_terminal_receipt(complete, contract), complete)
        bad = {**partial, "new_folds_completed": 2}
        with self.assertRaises(phase2.Phase2ContractError):
            phase2.validate_terminal_receipt(bad, contract)

    def test_successor_keeps_unit_until_complete_then_advances_resolution(self):
        self.assertEqual(
            phase2.next_token("PHASE2_UNIT_M03_R224_A01", "PARTIAL"),
            "PHASE2_UNIT_M03_R224_A02",
        )
        self.assertEqual(
            phase2.next_token("PHASE2_UNIT_M03_R224_A05", "COMPLETE"),
            "PHASE2_UNIT_M03_R320_A01",
        )
        self.assertEqual(
            phase2.next_token("PHASE2_UNIT_M03_R320_A05", "COMPLETE"),
            "PHASE2_UNIT_M03_R384_A01",
        )
        self.assertIsNone(
            phase2.next_token("PHASE2_UNIT_M03_R384_A05", "COMPLETE")
        )


if __name__ == "__main__":
    unittest.main()
