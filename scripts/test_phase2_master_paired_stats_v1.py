import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2_master_paired_stats_v1",
    ROOT / "scripts" / "phase2_master_paired_stats_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Phase2MasterPairedStatsV1Tests(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_p_order(self):
        p = np.asarray([0.03, 0.001, 0.02, 0.9])
        adjusted = MODULE.holm_adjust(p)
        order = np.argsort(p, kind="mergesort")
        sorted_adjusted = adjusted[order]
        self.assertTrue(np.all(sorted_adjusted[1:] >= sorted_adjusted[:-1]))
        self.assertTrue(np.all((adjusted >= 0) & (adjusted <= 1)))

    def test_paired_bootstrap_reports_patient_cluster(self):
        rows = []
        for patient in range(20):
            label = patient % 2
            rows.append(
                {
                    "patient_id": f"P{patient:02d}",
                    "label": label,
                }
            )
        frame = pd.DataFrame(rows)
        y = frame["label"].to_numpy(dtype=int)
        pred_ref = y.copy()
        pred_ref[::5] = 1 - pred_ref[::5]
        pred_cand = y.copy()
        pred_cand[::10] = 1 - pred_cand[::10]
        score_ref = np.where(y == 1, 0.75, 0.25)
        score_cand = np.where(y == 1, 0.85, 0.15)
        result = MODULE.paired_patient_bootstrap_metrics(
            frame,
            pred_ref,
            score_ref,
            pred_cand,
            score_cand,
            metrics=("balanced_accuracy", "mcc"),
            n_boot=100,
            seed=7,
        )
        self.assertEqual({row["metric"] for row in result}, {"balanced_accuracy", "mcc"})
        self.assertTrue(all(row["bootstrap_unit"] == "PATIENT_CLUSTER_PAIRED" for row in result))
        self.assertTrue(all(row["n_bootstrap_valid"] >= 50 for row in result))

    def test_full_predeclared_matrix_shape_small_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            n = 24
            labels = np.asarray([i % 2 for i in range(n)], dtype=int)
            identities = pd.DataFrame(
                {
                    "relative_path": [f"image_{i:03d}.png" for i in range(n)],
                    "patient_id": [f"P{i:03d}" for i in range(n)],
                    "label": labels,
                    "sha256": [f"{i:064x}" for i in range(n)],
                }
            )
            for model_index in range(1, 13):
                model = f"M{model_index:02d}"
                for resolution in MODULE.EXPECTED_RESOLUTIONS:
                    folder = root / model / f"R{resolution}"
                    folder.mkdir(parents=True, exist_ok=True)
                    # Deterministic model-dependent perturbation while preserving shared identity.
                    prob = np.where(labels == 1, 0.70, 0.30).astype(float)
                    prob = np.clip(prob + (model_index - 6) * 0.004, 0.01, 0.99)
                    oof = identities.copy()
                    oof["probability_pneumonia"] = prob
                    oof["prediction_fold_threshold"] = (prob >= 0.5).astype(int)
                    oof.to_csv(folder / "OOF_PREDICTIONS.csv", index=False)

                    locked = identities.copy()
                    locked["normalized_ensemble_score"] = prob - 0.5
                    locked["prediction_primary"] = (prob >= 0.5).astype(int)
                    locked.to_csv(folder / "LOCKED_TEST_PREDICTIONS.csv", index=False)

            table = MODULE.build_statistics(root, n_boot=100)
            expected = (
                2
                * len(MODULE.EXPECTED_RESOLUTIONS)
                * len(MODULE.PREDECLARED_MODEL_COMPARISONS)
                * (1 + len(MODULE.PAIRED_SECONDARY_METRICS))
            )
            self.assertEqual(len(table), expected)
            self.assertEqual(set(table["dataset"]), {"OOF", "LOCKED_TEST"})
            self.assertTrue(
                table.loc[
                    table["metric"] == MODULE.PAIRED_PRIMARY_METRIC,
                    "p_holm_primary_global",
                ].notna().all()
            )


if __name__ == "__main__":
    unittest.main()
