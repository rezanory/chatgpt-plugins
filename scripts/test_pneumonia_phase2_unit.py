import json
import pathlib
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

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


def _phase2_restore_helper_namespace():
    source = "\n".join(
        [
            "def phase2_restore(model_id, resolution):",
            "    try:",
            "        kagglehub.dataset_download(phase2_handle(model_id, resolution), "
            "output_dir=str(temp), force_download=True)",
            "    except Exception:",
            "        raise",
            "",
        ]
    )
    patched = phase2._patch_phase2_persistence_cell(source)
    helper = patched.split("def phase2_restore", 1)[0]
    namespace = {"Path": pathlib.Path, "json": json, "re": re}
    exec(helper, namespace)
    return namespace


class Phase2UnitTests(unittest.TestCase):
    def test_http_restore_downloads_manifest_and_hash_bound_archives_individually(self):
        namespace = _phase2_restore_helper_namespace()

        class Resolver:
            def __init__(self):
                self.paths = []

            def __call__(self, handle, path, *, output_dir, force_download):
                self.paths.append(path)
                target = pathlib.Path(output_dir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if path == "CAMPAIGN_STATE.json":
                    target.write_text(
                        json.dumps(
                            {"artifact_sha256": {"FOLD_1_RECOVERY.zip": "a" * 64}}
                        ),
                        encoding="utf-8",
                    )
                else:
                    target.write_bytes(b"sealed-archive")
                return str(target), 7

        resolver = Resolver()
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "kagglehub.http_resolver.DatasetHttpResolver", return_value=resolver
        ):
            result = namespace["_phase2_http_dataset_download"](
                "azadka/pneumonia-m01-r224-state-v1-7", temp
            )
            self.assertEqual(pathlib.Path(result), pathlib.Path(temp).resolve())
            self.assertEqual(
                resolver.paths, ["CAMPAIGN_STATE.json", "FOLD_1_RECOVERY.zip"]
            )
            self.assertEqual(
                (pathlib.Path(temp) / "FOLD_1_RECOVERY.zip").read_bytes(),
                b"sealed-archive",
            )

    def test_http_403_is_absence_only_after_exact_owner_inventory(self):
        namespace = _phase2_restore_helper_namespace()

        class Forbidden(RuntimeError):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=403)

        resolver = mock.Mock(side_effect=Forbidden())
        dataset_api = types.SimpleNamespace(
            list_datasets=lambda request: types.SimpleNamespace(
                datasets=[], next_page_token=""
            )
        )
        client = types.SimpleNamespace(
            username="azadka",
            datasets=types.SimpleNamespace(dataset_api_client=dataset_api),
        )
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "kagglehub.http_resolver.DatasetHttpResolver", return_value=resolver
        ), mock.patch("kagglehub.clients.build_kaggle_client", return_value=client):
            with self.assertRaises(
                namespace["_Phase2ConfirmedRemoteAbsence"]
            ) as captured:
                namespace["_phase2_http_dataset_download"](
                    "azadka/pneumonia-m01-r224-state-v1-7", temp
                )
            self.assertEqual(captured.exception.status_code, 404)
            self.assertIsInstance(captured.exception.__cause__, Forbidden)

    def test_http_403_token_only_client_uses_exact_my_inventory(self):
        namespace = _phase2_restore_helper_namespace()

        class Forbidden(RuntimeError):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=403)

        resolver = mock.Mock(side_effect=Forbidden())
        dataset_api = types.SimpleNamespace(
            list_datasets=lambda request: types.SimpleNamespace(
                datasets=[], next_page_token=""
            )
        )
        client = types.SimpleNamespace(
            username=None,
            datasets=types.SimpleNamespace(dataset_api_client=dataset_api),
        )
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "kagglehub.http_resolver.DatasetHttpResolver", return_value=resolver
        ), mock.patch("kagglehub.clients.build_kaggle_client", return_value=client):
            with self.assertRaises(
                namespace["_Phase2ConfirmedRemoteAbsence"]
            ) as captured:
                namespace["_phase2_http_dataset_download"](
                    "azadka/pneumonia-m01-r224-state-v1-7", temp
                )
            self.assertEqual(captured.exception.status_code, 404)

    def test_http_403_token_only_client_preserves_existing_dataset_failure(self):
        namespace = _phase2_restore_helper_namespace()

        class Forbidden(RuntimeError):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=403)

        handle = "azadka/pneumonia-m01-r224-state-v1-7"
        resolver = mock.Mock(side_effect=Forbidden())
        dataset_api = types.SimpleNamespace(
            list_datasets=lambda request: types.SimpleNamespace(
                datasets=[types.SimpleNamespace(ref=handle)], next_page_token=""
            )
        )
        client = types.SimpleNamespace(
            username=None,
            datasets=types.SimpleNamespace(dataset_api_client=dataset_api),
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch(
                "kagglehub.http_resolver.DatasetHttpResolver", return_value=resolver
            ),
            mock.patch("kagglehub.clients.build_kaggle_client", return_value=client),
            self.assertRaises(Forbidden),
        ):
            namespace["_phase2_http_dataset_download"](handle, temp)

    def test_http_403_rejects_positive_owner_identity_drift(self):
        namespace = _phase2_restore_helper_namespace()

        class Forbidden(RuntimeError):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=403)

        resolver = mock.Mock(side_effect=Forbidden())
        client = types.SimpleNamespace(username="wrong-owner")
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch(
                "kagglehub.http_resolver.DatasetHttpResolver", return_value=resolver
            ),
            mock.patch("kagglehub.clients.build_kaggle_client", return_value=client),
            self.assertRaisesRegex(
                RuntimeError, "PHASE2_KAGGLE_DATASET_OWNER_IDENTITY_DRIFT"
            ),
        ):
            namespace["_phase2_http_dataset_download"](
                "azadka/pneumonia-m01-r224-state-v1-7", temp
            )

    def test_manifest_has_exact_nonoverlapping_33_units(self):
        manifest = phase2.campaign_manifest()
        units = manifest["units"]
        identities = {(item["model_id"], item["resolution"]) for item in units}
        self.assertEqual(len(units), 33)
        self.assertEqual(len(identities), 33)
        self.assertEqual(set(manifest["models"]), set(phase2.MODEL_ASSIGNMENTS))
        self.assertNotIn("M07", manifest["models"])
        self.assertTrue(all(item["folds"] == [1, 2, 3, 4, 5] for item in units))
        self.assertEqual(
            manifest["shared_recipe"]["fingerprint"],
            phase2.FROZEN_M07_RECIPE_FINGERPRINT,
        )
        self.assertEqual(
            manifest["shared_recipe"]["receipt_sha256"],
            phase2.FROZEN_M07_RECIPE_RECEIPT_SHA256,
        )

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
            self.assertEqual(result["output_cell_count"], 16)
            self.assertIn("PHASE2_UNIT_MODEL_ID = 'M01'", joined)
            self.assertIn("CGP_PHASE2_MAX_NEW_FOLDS = 1", joined)
            self.assertIn("PHASE2_UNIT_LEGACY_M07_PERSISTENCE_SKIPPED", joined)
            self.assertIn("PHASE2_UNIT_FROZEN_M07_RECIPE_BOUND", joined)
            self.assertIn(phase2.FROZEN_M07_RECIPE_FINGERPRINT, joined)
            self.assertIn(phase2.FROZEN_M07_RECIPE_RECEIPT_SHA256, joined)
            self.assertIn("'batch_size': 12", joined)
            self.assertIn("'edge_filters': 16", joined)
            self.assertIn("import matplotlib.pyplot as plt", joined)
            self.assertIn("from kagglehub.http_resolver import DatasetHttpResolver", joined)
            self.assertIn("ApiListDatasetsRequest", joined)
            self.assertIn("DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY", joined)
            self.assertIn("PHASE2_KAGGLE_DATASET_OWNER_IDENTITY_DRIFT", joined)
            self.assertIn("PHASE2_REMOTE_STATE_CONFIRMED_ABSENT_BY_OWNER_INVENTORY", joined)
            self.assertIn('path="CAMPAIGN_STATE.json"', joined)
            self.assertIn("versioned = parsed.with_version(version)", joined)
            self.assertIn("path=artifact_name", joined)
            self.assertIn("_phase2_http_dataset_download", joined)
            self.assertNotIn(
                "kagglehub.dataset_download(phase2_handle(model_id, resolution)",
                joined,
            )
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    compile("".join(cell.get("source", [])), "<phase2-unit>", "exec")
            self.assertEqual(joined.count("PHASE2_UNIT_NEW_FOLD_BOUND_EXCEEDED"), 1)
            self.assertIn("run_phase2_model_resolution(\n    PHASE2_UNIT_MODEL_ID", joined)
            self.assertNotIn("RESTORE_SUMMARY = _try_restore_persisted_state_compat()", joined)
            self.assertNotIn("RESTORE_SUMMARY = _try_restore_persisted_state()", joined)
            self.assertNotIn("M07 run-safe persistence healthcheck before training", joined)
            self.assertNotIn("M07 RUNTIME PRE-FLIGHT PASS", joined)
            self.assertNotIn("M07_CONTINUATION_PRECHECK_PASS", joined)
            self.assertNotIn("STAGEB_WINNER", joined)
            self.assertNotIn("generate_confirmation_candidates()", joined)
            self.assertNotIn("run_or_restore_hpo_candidate_fold(", joined)
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
