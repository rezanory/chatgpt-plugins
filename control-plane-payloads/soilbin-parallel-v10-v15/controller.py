"""Launch private, CPU-only Kaggle research lanes through the authorized broker.
No cancellation, deletion, credential export or modification of other kernels.
Five independent lanes fan out; V15 fans in from compatible prediction banks.
"""
from __future__ import annotations
import os,sys,json,time,base64,hashlib,gzip,threading,concurrent.futures,urllib.request,urllib.error,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAMPAIGN='20260928-research-parallel-b1'
READ='https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle'
WRITE='https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle'
ACCOUNT_POOL={'kg-02':'radlinaradlina','kg-03':'rezanory','kg-04':'reyhanehazad','kg-06':'msdenis','kg-07':'nisabulutmark','kg-08':'azadkk','kg-09':'mylovevpn1','kg-11':'jobreza1'}
# kg-05/M07 and kg-10/M11 are deliberately not used in this campaign.
NAMES={'V10':'quality-audit','V11':'mae-quantile','V12':'tabpfn-v2','V13':'latent-state','V14':'physics-prior','V15':'ensemble-ablation'}
ACTIVE={'RUNNING','QUEUED','PENDING','INITIALIZING','STARTING','UNKNOWN'}
TERMINAL={'COMPLETE','COMPLETED','ERROR','FAILED','CANCELLED','CANCELED'}
EVIDENCE=Path(os.environ.get('RUNNER_TEMP','.'))/'soilbin-parallel-v10-v15'
EVIDENCE.mkdir(parents=True,exist_ok=True)
TOKENS={}; LOCK=threading.Lock(); STATE_LOCK=threading.Lock()
STATE={'campaign':CAMPAIGN,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'lanes':{},'policy':{'cpu_only':True,'cancel_other_jobs':False,'notebooks_private':True,'overwrite_existing_kernel':False}}

def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def checkpoint():
    with STATE_LOCK: write_json(EVIDENCE/'CAMPAIGN_STATE.json',STATE)

def token(kind):
    with LOCK:
        cached=TOKENS.get(kind)
        if cached and time.time()-cached[0]<180: return cached[1]
        audience='cgp-control-plane-v3-action' if kind=='write' else 'cgp-control-plane-v3'
        url=os.environ['ACTIONS_ID_TOKEN_REQUEST_URL'];url+=('&' if '?' in url else '?')+'audience='+urllib.parse.quote(audience,safe='')
        req=urllib.request.Request(url,headers={'Authorization':'Bearer '+os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']})
        with urllib.request.urlopen(req,timeout=30) as r: value=json.load(r)['value']
        if len(value)<100: raise RuntimeError('OIDC_NOT_AVAILABLE')
        print('::add-mask::'+value,flush=True);TOKENS[kind]=(time.time(),value);return value

def call(body,kind='read',timeout=120):
    for attempt in range(3):
        req=urllib.request.Request(WRITE if kind=='write' else READ,data=json.dumps(body,separators=(',',':')).encode(),method='POST',
                headers={'Authorization':'Bearer '+token(kind),'Content-Type':'application/json','Accept':'application/json','User-Agent':'soilbin-parallel-b1/1.0'})
        try:
            with urllib.request.urlopen(req,timeout=timeout) as r: result=json.load(r)
            if result.get('ok') is not True: raise RuntimeError('BROKER_REJECTED:'+json.dumps(result)[:800])
            if kind=='read' and result.get('read_only') is not True: raise RuntimeError('READ_ONLY_GUARD')
            return result
        except urllib.error.HTTPError as e:
            detail=e.read(1800).decode('utf-8','replace')
            if e.code==403 and 'JWT expired' in detail and attempt<2:
                with LOCK: TOKENS.pop(kind,None)
                continue
            if e.code in (429,502,503,504) and attempt<2 and kind=='read': time.sleep(3*(attempt+1));continue
            raise RuntimeError(f'BROKER_HTTP_{e.code}:{detail}') from e
    raise RuntimeError('BROKER_RETRIES_EXHAUSTED')

def raw(account,method,body):
    return call({'action':'raw_read','account_id':account,'service':'kernels.KernelsApiService','method':method,'body':body})['result']

def normalize_ref(ref):
    ref=str(ref or '').strip()
    if ref.startswith('https://www.kaggle.com/'): ref=ref.split('kaggle.com/',1)[1]
    return ref.removeprefix('/code/').removeprefix('code/').lstrip('/')

def kernel_status(account,ref):
    owner,slug=ref.split('/',1)
    x=raw(account,'GetKernelSessionStatus',{'userName':owner,'kernelSlug':slug})
    return str(x.get('status') or 'PENDING').upper()

def inventory(account):
    owner=ACCOUNT_POOL[account]
    result=raw(account,'ListKernels',{'group':'PROFILE','sortBy':'DATE_RUN','pageSize':100})
    kernels=result.get('kernels',[])
    refs=[]
    for item in kernels:
        ref=normalize_ref(item.get('ref') or item.get('url') or '')
        if ref.startswith(owner+'/') and ref not in refs: refs.append(ref)
    # Probe every returned kernel, not merely titles or timestamps. Bounded at API page size.
    statuses=[]
    for ref in refs:
        try: st=kernel_status(account,ref)
        except Exception as exc: st='UNKNOWN'
        statuses.append({'ref':ref,'status':st})
    return {'account_id':account,'owner':owner,'kernels_scanned':len(refs),'listing_at_limit':len(kernels)>=100,
            'active_or_unknown':sum(x['status'] not in TERMINAL for x in statuses),'statuses':statuses,'read_verified':True}

def publish(title,body):
    gh=os.environ.get('GITHUB_TOKEN')
    if not gh: return
    url=f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/issues/134/comments"
    req=urllib.request.Request(url,data=json.dumps({'body':'## '+title+'\n\n'+body}).encode(),method='POST',
       headers={'Authorization':'Bearer '+gh,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'soilbin-parallel-b1'})
    with urllib.request.urlopen(req,timeout=30): pass

def output_files(account,ref,names):
    result=call({'action':'output_json_files','account_id':account,'kernel_ref':ref,'file_names':names,'max_bytes_per_file':262144},timeout=180)['result']
    answer={}
    for item in result.get('files',[]):
        name=Path(str(item.get('file_name','')).replace('\\','/')).name
        if isinstance(item.get('json'),dict): answer[name]=item['json']
    return answer

def make_notebook(lane,payload):
    code=(ROOT/'runner.py').read_text(encoding='utf-8');compile(code,'runner.py','exec')
    pp=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0)).decode()
    cc=base64.b64encode(code.encode()).decode()
    bootstrap=[]
    if lane=='V12':
        bootstrap=["import os, sys, subprocess\n","os.environ['TABPFN_MODEL_CACHE_DIR']='/tmp/soilbin-tabpfn-cache'\n",
                   "os.environ['DO_NOT_TRACK']='1'\n",
                   "subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','tabpfn==2.1.3'],check=True,timeout=900)\n"]
    bootstrap += ['import base64,gzip,json\n',f'PAYLOAD=json.loads(gzip.decompress(base64.b64decode({pp!r})))\n',
                  f'exec(compile(base64.b64decode({cc!r}),"soilbin_parallel_runner.py","exec"))\n']
    return json.dumps({'cells':[{'cell_type':'markdown','metadata':{},'source':[f'# SoilBin {lane}: {NAMES[lane]}\n',
        'Private post-lock research; shared grouped splits. No experimental targets are modified.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':bootstrap}],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},
          'soilbin':{'campaign':CAMPAIGN,'lane':lane,'private':True,'CPU_only':True,'post_lock_exploratory':True,'source_code_sha256':hashlib.sha256(code.encode()).hexdigest()}},
        'nbformat':4,'nbformat_minor':5},ensure_ascii=False)

def launch(lane,account,banks=None,unavailable=None):
    owner=ACCOUNT_POOL[account];slug=f'soilbin-{lane.lower()}-{NAMES[lane]}-20260928-b1';ref=owner+'/'+slug
    # Refuse overwriting. Listing is authoritative; a found run is recovered by reading only.
    existing=raw(account,'ListKernels',{'group':'PROFILE','sortBy':'DATE_RUN','pageSize':100,'search':slug}).get('kernels',[])
    if any(normalize_ref(x.get('ref'))==ref for x in existing):
        return {'account_id':account,'owner':owner,'kernel_ref':ref,'submitted':False,'recovered_existing':True,'state':kernel_status(account,ref)}
    payload={'lane':lane,'campaign':CAMPAIGN,'model_b64':(ROOT.parent/'soilbin-q1-v3/all_features_51.csv.gz.b64').read_text().strip()}
    if lane=='V10': payload['wave_b64']=(ROOT.parent/'soilbin-q1-v4/waveform_used_columns.bin.xz.b64').read_text().strip()
    if lane=='V15': payload.update({'banks':banks,'unavailable_lanes':unavailable or {}})
    text=make_notebook(lane,payload)
    response=call({'request_id':f'soilbin-{CAMPAIGN}-{lane.lower()}-{account}', 'provider':'kaggle','operation_class':'compute','account_id':account,
       'purpose':f'User-authorized parallel SoilBin {lane} research; private CPU kernel, no other jobs changed. Frozen V5 and targets retained.',
       'service':'kernels.KernelsApiService','method':'SaveKernel','body':{'slug':ref,'newTitle':slug,'text':text,'language':'PYTHON','kernelType':'NOTEBOOK',
          'kernelExecutionType':'SAVE_AND_RUN_ALL','isPrivate':True,'enableGpu':False,'enableInternet':lane=='V12'}},kind='write',timeout=240)
    actual=normalize_ref(response.get('provider_ref') or (response.get('result') or {}).get('ref') or ref)
    return {'account_id':account,'owner':owner,'kernel_ref':actual,'submitted':True,'notebook_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'code_sha256':hashlib.sha256((ROOT/'runner.py').read_bytes()).hexdigest(),'state':'SUBMITTED'}

def main():
    statefile=EVIDENCE/'CAMPAIGN_STATE.json'
    if statefile.exists(): raise RuntimeError('REFUSE_OVERWRITE_CAMPAIGN_RECEIPT')
    inv=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs={pool.submit(inventory,a):a for a in ACCOUNT_POOL}
        for fut in concurrent.futures.as_completed(fs):
            account=fs[fut]
            try: row=fut.result()
            except Exception as exc: row={'account_id':account,'read_verified':False,'error':str(exc)[:1500]}
            inv.append(row); print('ACCOUNT_PREFLIGHT',json.dumps(row),flush=True)
    write_json(EVIDENCE/'ACCOUNT_PREFLIGHT.json',inv)
    # Conservative headroom: at most one existing active/unknown session in the scanned page.
    # No rate/quota bypass or forced cancellation. Kaggle enforces the final capacity check.
    choices=sorted([v for v in inv if v.get('read_verified') and v['active_or_unknown']<2 and not v['listing_at_limit']],
                   key=lambda v:(v['active_or_unknown'],v['account_id']))
    if len(choices)<6:
        STATE['status']='BLOCKED_INSUFFICIENT_VERIFIED_ACCOUNT_HEADROOM'; STATE['inventory']=inv;checkpoint()
        publish('SoilBin parallel campaign: capacity blocked',f"Verified suitable accounts: {len(choices)}/6. No existing job was interrupted.")
        raise RuntimeError('NEED_SIX_SAFE_ACCOUNTS')
    assignment=dict(zip(['V10','V11','V12','V13','V14','V15'],[v['account_id'] for v in choices[:6]]))
    STATE['assignment']=assignment;STATE['status']='LAUNCHING';STATE['lanes']['V15']={'account_id':assignment['V15'],'owner':ACCOUNT_POOL[assignment['V15']],'state':'WAITING_FOR_PREDICTION_BANKS'};checkpoint()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        fs={pool.submit(launch,lane,assignment[lane]):lane for lane in ['V10','V11','V12','V13','V14']}
        for fut in concurrent.futures.as_completed(fs):
            lane=fs[fut]
            try: STATE['lanes'][lane]=fut.result()
            except Exception as exc: STATE['lanes'][lane]={'account_id':assignment[lane],'state':'LAUNCH_FAILED','error':str(exc)[:1500]}
            checkpoint(); print('LAUNCH_RECEIPT',lane,json.dumps(STATE['lanes'][lane]),flush=True)
    body='Campaign: '+CAMPAIGN+'\n\n'
    for lane in ['V10','V11','V12','V13','V14','V15']:
        z=STATE['lanes'][lane];body+=f"- {lane}: {z['account_id']} / {z.get('owner','')}; {z['state']}; {z.get('kernel_ref','fan-in pending')}\n"
    publish('SoilBin V10–V15 parallel launch receipts',body)
    banks={}; results={}; pending={l for l,z in STATE['lanes'].items() if l!='V15' and z['state']!='LAUNCH_FAILED'}
    deadline=time.time()+100*60
    while pending and time.time()<deadline:
        for lane in list(pending):
            z=STATE['lanes'][lane]
            try:
                st=kernel_status(z['account_id'],z['kernel_ref']);z['state']=st
                if st in TERMINAL:
                    try:
                        out=output_files(z['account_id'],z['kernel_ref'],['RESULTS.json'])
                        r=out.get('RESULTS.json',{})
                        if lane!='V10' and r.get('status','').startswith('COMPLETE'):
                            out.update(output_files(z['account_id'],z['kernel_ref'],['BANK.json']))
                    except Exception as exc:
                        z['terminal_collect_attempts']=z.get('terminal_collect_attempts',0)+1
                        if z['terminal_collect_attempts']<3: raise
                        out={};r={'status':'MISSING_TERMINAL_EVIDENCE','collection_error':str(exc)[:1200]}
                    lane_dir=EVIDENCE/lane;lane_dir.mkdir(exist_ok=True)
                    for name,val in out.items():write_json(lane_dir/name,val)
                    results[lane]=r;z['result_status']=r.get('status','MISSING_RESULT')
                    if out.get('BANK.json',{}).get('complete') and r.get('status','').startswith('COMPLETE'):
                        banks[lane]=out['BANK.json']
                    pending.remove(lane)
            except Exception as exc:
                z['last_read_error']=str(exc)[:1500]
            checkpoint()
        print('CAMPAIGN_PROGRESS',json.dumps({l:z.get('state') for l,z in STATE['lanes'].items()}),flush=True)
        if pending:time.sleep(30)
    if pending:
        STATE['status']='INCOMPLETE_UPSTREAM_TIMEOUT';checkpoint();raise RuntimeError('UPSTREAM_TIMEOUT')
    unavailable={l:STATE['lanes'][l].get('result_status',STATE['lanes'][l]['state']) for l in ['V11','V12','V13','V14'] if l not in banks}
    if 'V11' not in banks:
        STATE['lanes']['V15']['state']='BLOCKED_BASELINE_BANK';STATE['status']='INCOMPLETE';checkpoint()
        raise RuntimeError('BASELINE_BANK_MISSING')
    # V15 must consume the same group's INNER cross-fitted banks, not fit weights to outer test predictions.
    STATE['lanes']['V15']=launch('V15',assignment['V15'],banks=banks,unavailable=unavailable);checkpoint()
    publish('SoilBin V15 fan-in launched',json.dumps(STATE['lanes']['V15'],ensure_ascii=False)+'\nUnavailable source lanes: '+json.dumps(unavailable))
    z=STATE['lanes']['V15']; deadline=time.time()+30*60
    while time.time()<deadline:
        st=kernel_status(z['account_id'],z['kernel_ref']);z['state']=st;checkpoint()
        if st in TERMINAL:
            out=output_files(z['account_id'],z['kernel_ref'],['RESULTS.json']);r=out.get('RESULTS.json',{})
            results['V15']=r;z['result_status']=r.get('status','MISSING_RESULT');dest=EVIDENCE/'V15';dest.mkdir(exist_ok=True)
            write_json(dest/'RESULTS.json',r);break
        time.sleep(20)
    else:STATE['status']='INCOMPLETE_V15_TIMEOUT';checkpoint();raise RuntimeError('V15_TIMEOUT')
    STATE['status']='EXECUTION_TERMINAL';STATE['completed_with_known_physical_blocker']=True
    STATE['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());checkpoint()
    write_json(EVIDENCE/'RESULTS_ALL_LANES.json',results)
    publish('SoilBin V10–V15 execution receipts',json.dumps({l:{'account':z['account_id'],'state':z['state'],'result_status':z.get('result_status')} for l,z in STATE['lanes'].items()},ensure_ascii=False)+'\nV14 validated multifidelity remains blocked by missing physical metadata; proxy-model results are separate.')
    print('SOILBIN_PARALLEL_TERMINAL',json.dumps(STATE,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    finally:
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'],'a',encoding='utf-8') as h:h.write('SOILBIN_PARALLEL_EVIDENCE_DIR='+str(EVIDENCE)+'\n')
