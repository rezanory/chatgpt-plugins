"""Targeted executable acceptance checks; no Kaggle scores are fabricated here."""
import os,json,importlib.util,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
os.environ['SOILBIN_OUTPUT_ROOT']=str(ROOT/'preflight_output')
spec=importlib.util.spec_from_file_location('research_runner',ROOT/'runner.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
base=ROOT.parent
payload={'lane':'V10','campaign':'20260928-research-parallel-b1',
 'model_b64':(base/'soilbin-q1-v3/all_features_51.csv.gz.b64').read_text().strip(),
 'wave_b64':(base/'soilbin-q1-v4/waveform_used_columns.bin.xz.b64').read_text().strip()}
d,h=r.load_data(payload); c,sha=r.split_contract(h); checks=[]
assert len(d)==51 and len(h)==41 and len(c['folds'])==9; checks.append('frozen_input_51_41_9')
for f in c['folds']:
 tg=set(h.iloc[f['train']].Group); vg=set(h.iloc[f['test']].Group)
 assert not tg&vg
 for q in f['inner']:
  a=set(h.iloc[q['train']].Group);b=set(h.iloc[q['valid']].Group)
  assert not a&b and not a&vg and not b&vg and a|b==tg
checks.append('all_outer_inner_group_isolation')
f=c['folds'][0]; tr=np.array(f['train']);te=np.array(f['test']);eng=r.Engine(d,h)
for model in ['ABS_RF_WEIGHTED','ABS_ET_WEIGHTED','QRF_MEDIAN_WEIGHTED','PHYS_EXP','PHYS_POWER_GP']:
 p=eng.predict(model,tr,te);assert p.shape==(len(te),2) and np.isfinite(p).all();checks.append('predict_'+model)
try: eng.predict('RF_DEEP',tr,tr)
except ValueError as e: assert str(e)=='GROUP_LEAKAGE';checks.append('leakage_guard_rejects_shared_group')
else: raise AssertionError('leakage permitted')
for rank in (1,2):
 train=d[d.Group.isin(set(h.iloc[tr].Group))]
 model=r.KalmanSoil(rank=rank,logpass=True).fit(train)
 seq=d[d.Group==f['outer']].copy();ids=h.iloc[te].Run_ID.tolist()
 first=model.forecast(seq,ids)
 altered=seq.copy();tm=int(h.iloc[te[1]].Pass_T)
 altered.loc[altered.Pass_T>=tm,r.TARGETS]+=10000.
 second=model.forecast(altered,ids)
 prior=h.iloc[te].Pass_T.to_numpy()<=tm
 assert np.allclose(first[prior],second[prior],atol=1e-9)
 checks.append('state_rank'+str(rank)+'_no_current_or_future_target_leak')
q=r.QuantileForest().fit(np.ones((10,5)),np.array([0]*6+[100]*4),np.ones(10))
assert q.predict(np.ones((2,5))).tolist()==[0.,0.];checks.append('quantile_cdf_not_mean_40')
audit=r.audit_v10(payload,d,h,c,sha)
assert audit['trace_count']==102;checks.append('raw_waveform_102_traces_hash_verified')
# Build deliberately identical mock predictors to exercise ensemble/member removal
# without claiming that these numbers are outputs of a real trained experiment.
members=['V5','QRF_MEDIAN_WEIGHTED','TABPFN_DELTA','STATE_RANK1_LOG','PHYS_EXP_GP']
y=h[r.TARGETS].to_numpy(float); fake=y+float(r.FROZEN)
bank={'schema':'soilbin.parallel.bank.v1','lane':'TEST','split_sha256':sha,'source_sha256':r.EXPECTED_DATA,
'run_ids':h.Run_ID.tolist(),'groups':h.Group.tolist(),'models':{},'folds':c['folds']}
for name in members:
 bank['models'][name]={'outer_pred':fake.tolist(),'inner_oof':{f['outer']:{'indices':f['train'],'pred':fake[f['train']].tolist()} for f in c['folds']}}
wrapper={'complete':True,'payload_gzip_b64':r.pack(bank)}
ens=r.run_ensemble({'banks':{'TEST':wrapper}},d,h,c,sha)
assert len(ens['member_contributions'])==5 and ens['complete_planned_ensemble'];checks.append('ensemble_refitted_member_removal_contract')
result={'status':'PASS','checks':checks,'checks_passed':len(checks),'split_sha256':sha,'n_runs':51,'n_transitions':41,
'max_raw_peak_difference_N':audit['max_peak_reproduction_difference_N'],
'not_scientific_performance_run':True}
(ROOT/'PREFLIGHT_TEST_RECEIPT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
