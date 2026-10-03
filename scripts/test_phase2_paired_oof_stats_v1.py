import importlib.util
import pathlib
import tempfile
import unittest
import json, base64, zlib, hashlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2_paired_oof_stats_v1",
    ROOT / "scripts" / "phase2_paired_oof_stats_v1.py",
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)

def vector(model, resolution, delta=0.0, n=40):
    patients=[f"p{i//2:03d}" for i in range(n)]
    labels=[i%2 for i in range(n)]
    scores=[]
    preds=[]
    for i,y in enumerate(labels):
        base=0.75 if y else 0.25
        score=min(0.99,max(0.01,base+delta+((i%5)-2)*0.01))
        scores.append(score)
        preds.append(int(score>=0.5))
    inner={"patient_id":patients,"label":labels,"prediction":preds,"score":scores}
    raw=json.dumps(inner,sort_keys=True,separators=(",",":")).encode()
    return {
        "schema":"pneumonia.phase2.oof.vector.v1",
        "status":"PASS",
        "model_id":model,
        "resolution":resolution,
        "row_count":n,
        "identity_sequence_sha256":"a"*64,
        "payload_encoding":"zlib+base64+json",
        "payload_sha256":hashlib.sha256(raw).hexdigest(),
        "payload_b64":base64.b64encode(zlib.compress(raw,9)).decode(),
        "locked_test_used_for_selection":False,
        "external_used_for_selection":False,
    }

class Tests(unittest.TestCase):
    def test_holm_monotonic(self):
        out=M.holm_adjust([0.01,0.03,0.04])
        self.assertTrue(np.all((out>=0)&(out<=1)))
        self.assertAlmostEqual(float(out[0]),0.03)

    def test_bootstrap_shape(self):
        v=vector("M01",224)
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"PHASE2_OOF_VECTOR_M01_R224.json"
            p.write_text(json.dumps(v),encoding="utf-8")
            d=M.decode_vector(p)
            rows=M.paired_patient_bootstrap_metrics(
                d["patient_id"],d["label"],d["prediction"],d["score"],
                d["prediction"],d["score"],
                (M.PAIRED_PRIMARY_METRIC,)+M.PAIRED_SECONDARY_METRICS,
                n_boot=200,seed=123,
            )
            self.assertEqual(len(rows),6)
            for row in rows:
                self.assertAlmostEqual(row["delta_candidate_minus_reference"],0.0)
                self.assertEqual(row["bootstrap_unit"],"PATIENT_CLUSTER_PAIRED")

    def test_full_predeclared_matrix(self):
        vectors={}
        models={m for a,b,_ in M.PREDECLARED_MODEL_COMPARISONS for m in (a,b)}
        for m in models:
            for r in M.EXPECTED_RESOLUTIONS:
                v=vector(m,r)
                raw=zlib.decompress(base64.b64decode(v["payload_b64"]))
                inner=json.loads(raw)
                vectors[(m,r)]={
                    "model_id":m,"resolution":r,"row_count":len(inner["label"]),
                    "identity_sequence_sha256":"a"*64,
                    "patient_id":np.asarray(inner["patient_id"],dtype=object),
                    "label":np.asarray(inner["label"],dtype=int),
                    "prediction":np.asarray(inner["prediction"],dtype=int),
                    "score":np.asarray(inner["score"],dtype=float),
                }
        original=M.paired_patient_bootstrap_metrics
        def fake_bootstrap(patient_ids,y,pred_reference,score_reference,pred_candidate,score_candidate,metrics,n_boot=M.PAIRED_BOOTSTRAPS,seed=M.PAIRED_BOOTSTRAP_SEED):
            return [
                {
                    "metric":metric,
                    "reference":0.80,
                    "candidate":0.81,
                    "delta_candidate_minus_reference":0.01,
                    "ci95_low":0.001,
                    "ci95_high":0.02,
                    "p_raw_two_sided":0.01,
                    "bootstrap_unit":"PATIENT_CLUSTER_PAIRED",
                    "n_bootstrap_valid":5000,
                    "n_patients":20,
                }
                for metric in metrics
            ]
        M.paired_patient_bootstrap_metrics=fake_bootstrap
        try:
            rows=M.build_statistics(vectors)
        finally:
            M.paired_patient_bootstrap_metrics=original
        self.assertEqual(len(rows),12*3*6)
        self.assertEqual(sum(r["metric"]=="balanced_accuracy" for r in rows),36)
        primary=[r for r in rows if r["metric"]=="balanced_accuracy"]
        self.assertTrue(all(r.get("p_holm_primary_within_resolution") is not None for r in primary))
        self.assertTrue(all(r.get("p_holm_primary_global") is not None for r in primary))

if __name__=="__main__":
    unittest.main()
