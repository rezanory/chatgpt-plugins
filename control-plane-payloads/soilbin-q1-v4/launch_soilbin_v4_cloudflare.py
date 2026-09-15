from __future__ import annotations
import base64,gzip,lzma,hashlib,json,os,pathlib,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parent
RUNNER=ROOT/'soilbin_v4_raw_waveform_runner.py'
MODEL_B64=ROOT/'all_features_51.csv.gz.b64'
RAW_B64=ROOT/'waveform_used_columns.bin.xz.b64'
ENDPOINT='https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle'
KERNEL_REF='mylovevpn1/soilbin-q1-q2-v4-raw-waveform-20260914'
EXPECTED_MODEL_SHA='dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059'
EXPECTED_RAW_SHA='ea227f412827a962a1876ceb9aeb9946034b435052e9e74115e32bd7dca73cb0'
EXPECTED_WAVE_COMPACT_BLOB_SHA='786cd72be8a6e37aa0e34cee044f70b047c9a84f5a9230abc9028117db765677'
EXPECTED_WAVE_COMPACT_XZ_SHA='6ccca93e5e146e67908e37cf07575bdf1ed44b0f3b84fcf0ee3198cd2ffa02c9'
runner=RUNNER.read_text(encoding='utf-8'); compile(runner,str(RUNNER),'exec')
def verify(path,sha):
    b64=path.read_text(encoding='utf-8').strip(); raw=gzip.decompress(base64.b64decode(b64,validate=True)); got=hashlib.sha256(raw).hexdigest()
    if got!=sha: raise SystemExit(f'fingerprint mismatch {path.name}: {got}')
    return b64
mb64=verify(MODEL_B64,EXPECTED_MODEL_SHA)
rb64=RAW_B64.read_text(encoding='utf-8').strip()
rxz=base64.b64decode(rb64,validate=True)
if hashlib.sha256(rxz).hexdigest()!=EXPECTED_WAVE_COMPACT_XZ_SHA: raise SystemExit('wave compact xz fingerprint mismatch')
rblob=lzma.decompress(rxz)
if hashlib.sha256(rblob).hexdigest()!=EXPECTED_WAVE_COMPACT_BLOB_SHA: raise SystemExit('wave compact blob fingerprint mismatch')
for req in ['Only previous-pass raw LC1/LC5 waveforms','DCT','SPECTRAL','RAW_FLAT','COMPACT_CNN','GroupKFold','paired_representation_tests_v4.json','V3_D_DELTA_PREV_PEAK_FROZEN','v3_vs_raw_waveform_v4.json','V5_POLICY_SHA256']:
    if req not in runner: raise SystemExit('required V4 guard missing: '+req)
source='MODEL_CSV_GZ_B64 = '+repr(mb64)+'\nRAW_WAVE_MIN_XZ_B64 = '+repr(rb64)+'\n'+runner
notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# SoilBin Q1/Q2 V4 — previous-pass raw waveform representation learning\n','CPU-only; current-pass waveform forbidden; nested grouped validation; DCT/spectral/raw/CNN representations.\n']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'},'soilbin':{'schema':'soilbin.q1.v4','cpu_only':True,'external_labels_used':False,'current_waveform_predictor':False,'previous_raw_waveform':True}},'nbformat':4,'nbformat_minor':5}
text=json.dumps(notebook,ensure_ascii=False,indent=1); sha=hashlib.sha256(text.encode()).hexdigest(); print('SOILBIN_V4_NOTEBOOK_VALIDATED',sha,flush=True)
token=os.environ.get('CGP_ACTION_OIDC_TOKEN','')
if len(token)<100: raise SystemExit('CGP_ACTION_OIDC_TOKEN unavailable')
payload={'request_id':'soilbin-q1-v4-raw-waveform-kg09-20260914-r1','provider':'kaggle','operation_class':'compute','account_id':'kg-09','purpose':'PhD/Q1-Q2 SoilBin V4 CPU benchmark: previous-pass raw LC1/LC5 waveform representation learning with deterministic event resampling, DCT/spectral/raw flattened representations and compact CNN under nested leave-one-VxW-group-out validation. No current-pass waveform leakage; no external labels used.','service':'kernels.KernelsApiService','method':'SaveKernel','body':{'slug':KERNEL_REF,'newTitle':'SoilBin Q1 Q2 V4 Raw Waveform 20260914','text':text,'language':'PYTHON','kernelType':'NOTEBOOK','kernelExecutionType':'SAVE_AND_RUN_ALL','isPrivate':True,'enableGpu':False,'enableInternet':True}}
req=urllib.request.Request(ENDPOINT,data=json.dumps(payload,separators=(',',':')).encode(),method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','Accept':'application/json','User-Agent':'soilbin-q1-v4/1.0'})
try:
    with urllib.request.urlopen(req,timeout=240) as r: result=json.loads(r.read().decode('utf-8','replace') or '{}')
except urllib.error.HTTPError as e: raise SystemExit(f'Cloudflare action HTTP {e.code}: {e.read(4000).decode("utf-8","replace")}')
if not result.get('ok'): raise SystemExit('Cloudflare action returned ok=false: '+json.dumps(result)[:1800])
provider_ref=result.get('provider_ref') or (result.get('result') or {}).get('ref') or KERNEL_REF
receipt={'accepted':True,'account_id':'kg-09','owner':'mylovevpn1','kernel_ref':provider_ref,'notebook_sha256':sha,'runner_sha256':hashlib.sha256(runner.encode()).hexdigest(),'model_csv_sha256':EXPECTED_MODEL_SHA,'raw_timeseries_source_sha256':EXPECTED_RAW_SHA,'waveform_used_columns_compact_blob_sha256':EXPECTED_WAVE_COMPACT_BLOB_SHA,'cpu_only':True,'gpu':False,'external_labels_used':False,'broker_run_id':result.get('broker_run_id'),'broker_sha':result.get('broker_sha')}
print('SOILBIN_V4_LAUNCH_ACCEPTED',json.dumps(receipt,ensure_ascii=False),flush=True)
p=pathlib.Path(os.environ.get('RUNNER_TEMP','.'))/'soilbin_v4_launch_receipt.json'; p.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
with open(os.environ['GITHUB_ENV'],'a',encoding='utf-8') as h: h.write(f'SOILBIN_V4_RECEIPT_PATH={p}\nSOILBIN_V4_KERNEL_REF={provider_ref}\n')
