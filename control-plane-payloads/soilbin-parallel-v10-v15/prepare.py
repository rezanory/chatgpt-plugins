from pathlib import Path
import json,ast,textwrap,hashlib
root=Path(__file__).resolve().parent
repo=root.parents[1]
p=root/'runner.py';s=p.read_text(encoding='utf-8')
s=s.replace("EXPECTED_DATA='dc96ffd8d48f6cc861c2ecbc95b95fa809c24c55688fbccba6cd8f61ce6059' # assigned below from audited literal\n",'')
s=s.replace("'V11':['V5','ABS_RF_UNWEIGHTED','ABS_RF_WEIGHTED','ABS_ET_WEIGHTED','QRF_MEDIAN_WEIGHTED']", "'V11':['V5','RF_SHALLOW','RF_SHALLOW_WEIGHTED','ET_SHALLOW_WEIGHTED','ABS_RF_UNWEIGHTED','ABS_RF_WEIGHTED','ABS_ET_WEIGHTED','QRF_MEDIAN_WEIGHTED']")
s=s.replace("                try: pred=m.predict(Xt,output_type='median')\n                except TypeError: pred=m.predict(Xt)","                pred=m.predict(Xt,output_type='median')")
old="    full=allpred['ANCHORED_BLEND']; effects=[]"
new="""    tests=[x for x in summary if 'paired_vs_v5' in x]
    order=sorted(range(len(tests)),key=lambda i:tests[i]['paired_vs_v5']['signflip_one_sided_p'])
    running=0.
    for rank,idx in enumerate(order):
        running=max(running,min(1.,(len(tests)-rank)*tests[idx]['paired_vs_v5']['signflip_one_sided_p']))
        tests[idx]['paired_vs_v5']['holm_adjusted_p']=running
        tests[idx]['paired_vs_v5']['comparison_family_size']=len(tests)
    full=allpred['ANCHORED_BLEND']; effects=[]"""
assert old in s;s=s.replace(old,new)
p.write_text(s,encoding='utf-8');ast.parse(s)

protocol={
 'campaign':'20260928-research-parallel-b1','version_count':6,
 'source_sha256':'dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059',
 'split_sha256':'ee49e9cbc3962eac2fc1a1fd8c933b6dc9712eea1d4c3cb3670c0b4f13c03a21',
 'target':'predict next-pass LC1/LC5 calibrated vertical force proxy (N)',
 'target_under_10_N':'research objective only, not acceptance threshold for selectively reporting models',
 'parallel_lanes':['V10','V11','V12','V13','V14'],'dependent_fan_in':'V15',
 'V10':{'purpose':'read-only signal/measurement audit','hardware_calibration_certificate':'not provided','new_target_definition':False},
 'V11':{'purpose':'controlled loss/weight/conditional-median ablations','model_count':8,'trees':250},
 'V12':{'purpose':'TabPFN v2 direct versus delta, predictive median','package':'tabpfn==2.1.3','estimators':2,'device':'cpu','no_external_data_API':True},
 'V13':{'purpose':'rank-one/rank-two probabilistic latent-state filtering','models':3,'inference':'forward only','optimizer_maxiter':90},
 'V14':{'purpose':'relaxation prior and Gaussian-process discrepancy','models':4,'validated_multifidelity_status':'BLOCKED_PENDING_PHYSICAL_METADATA','synthetic_rows_treated_as_measured':False},
 'V15':{'purpose':'fixed equal and anchored convex ensemble plus refitted member-removal ablation',
   'designated_members':['V5','QRF_MEDIAN_WEIGHTED','TABPFN_DELTA','STATE_RANK1_LOG','PHYS_EXP_GP','PERSISTENCE'],
   'V5_minimum_weight':0.5,'other_member_weight_grid':[0,0.125,0.25],'other_member_maximum_total':0.5,
   'sparsity_penalty_N':0.1,'member_selection_from_outer_scores':False,'inference':'exact grouped sign flip, bootstrap, Holm adjustment; exploratory'},
 'fairness':{'outer_groups':9,'history_pairs':41,'raw_runs':51,'random_split':False,'drop_hard_samples':False,'change_labels':False,
  'retrain_or_select_on_outer_test':False,'strict_meta_crossfit':True,'V5_reference_retained':True},
 'execution':{'private_kernels':True,'dedicated_distinct_account_per_version':True,'GPU_requested':False,'cancel_or_overwrite_existing_runs':False},
 'limitations':['Existing groups were used in earlier development; no new independent experiment.',
 'V14 analytic proxy tests are not a verified physical multifidelity simulator.',
 'V10 corrected traces cannot recover unknown original ADC/calibration parameters.',
 'Parallel starts do not imply identical completion times. V15 requires upstream banks.'],
 'primary_sources':[
 'https://www.nature.com/articles/s41586-024-08328-6',
 'https://jmlr.org/papers/v7/meinshausen06a.html',
 'https://proceedings.mlr.press/v5/alvarez09a.html',
 'https://arxiv.org/abs/2404.11965',
 'https://www.jmlr.org/papers/v11/cawley10a.html',
 'https://pypi.org/project/tabpfn/2.1.3/']}
(root/'EXPERIMENT_PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'README.md').write_text('''# SoilBin V10-V15 parallel research campaign

Five independent private CPU Kaggle lanes, followed by a sixth account for fan-in.
All measured targets, nine outer held-out groups and 41 transitions stay frozen.
A model below 10 N is a research objective, never a promise or a reason to hide negative results.

- V10: read-only audit of 102 corrected traces, timestamps, baseline-edge noise and calibration consistency.
- V11: controlled absolute-error, equal-group weighting, conditional-median forest ablation. Exact V5 reproduction required.
- V12: pinned TabPFN v2 direct/delta median prediction, entirely on Kaggle CPU; no data sent to a hosted prediction API.
- V13: actual probabilistic latent-state model, two sensors and forward filtering, trained by sequence likelihood.
- V14: explicit relaxation-prior and GP-discrepancy tests. A validated multifidelity simulator is blocked by missing physical metadata; never equate these proxy tests with that missing study.
- V15: predeclared cross-fitted ensemble and remove-one-member/refit-weight contribution assessment. No weight or model chosen from outer-test scores.

See EXPERIMENT_PROTOCOL.json for the frozen candidate catalog and caveats.
PREFLIGHT_TEST_RECEIPT.json is a technical smoke-test receipt, not scientific model-performance evidence.
''',encoding='utf-8')
(root/'.gitignore').write_text('__pycache__/\npreflight_output/\n*.pyc\n',encoding='utf-8')
workflow=repo/'.github/workflows/soilbin-q1-v6-23-extension.yml'
body=workflow.read_text(encoding='utf-8')
job='''
  research-parallel-v10-v15:
    if: ${{ github.event.issue.number == 134 && github.event.comment.user.id == 62356000 && github.event.comment.body == 'SOILBIN_PARALLEL_V10_V15_RESEARCH_B1' }}
    runs-on: ubuntu-latest
    timeout-minutes: 160
    steps:
      - name: Checkout frozen campaign source
        uses: actions/checkout@v4
        with:
          ref: ${{ github.sha }}
      - name: Validate campaign payload and protocol
        shell: bash
        run: |
          set -euo pipefail
          python -m py_compile control-plane-payloads/soilbin-parallel-v10-v15/runner.py control-plane-payloads/soilbin-parallel-v10-v15/controller.py
          python - <<'PY'
          import json,pathlib,hashlib,gzip,base64
          root=pathlib.Path('control-plane-payloads/soilbin-parallel-v10-v15')
          protocol=json.loads((root/'EXPERIMENT_PROTOCOL.json').read_text())
          raw=gzip.decompress(base64.b64decode((root.parent/'soilbin-q1-v3/all_features_51.csv.gz.b64').read_text().strip()))
          assert hashlib.sha256(raw).hexdigest()==protocol['source_sha256']
          assert protocol['fairness']['strict_meta_crossfit'] is True
          assert protocol['execution']['cancel_or_overwrite_existing_runs'] is False
          assert protocol['V14']['validated_multifidelity_status']=='BLOCKED_PENDING_PHYSICAL_METADATA'
          print('PROTOCOL_VALIDATED',hashlib.sha256((root/'EXPERIMENT_PROTOCOL.json').read_bytes()).hexdigest())
          PY
      - name: Capacity audit and five-account fan-out, sixth-account fan-in
        shell: bash
        env:
          GITHUB_TOKEN: ${{ github.token }}
        run: python -u control-plane-payloads/soilbin-parallel-v10-v15/controller.py
      - name: Upload every available campaign receipt
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: soilbin-parallel-v10-v15-evidence
          path: ${{ runner.temp }}/soilbin-parallel-v10-v15
          if-no-files-found: warn
          retention-days: 30
'''
if '  research-parallel-v10-v15:' not in body:
 workflow.write_text(body.rstrip()+'\n'+job,encoding='utf-8')
print('PREPARED',len(s),hashlib.sha256(s.encode()).hexdigest())
