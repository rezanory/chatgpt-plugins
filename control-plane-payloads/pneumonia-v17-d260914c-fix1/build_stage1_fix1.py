import argparse, ast, hashlib, json, pathlib

OLD_CAMPAIGN='d260914b'
NEW_CAMPAIGN='d260914c'
OLD_SHA={
 'W01':'6e0b6e463f769e8a8d919fc44b1a0bf50d81efa5fc31028cb3ce46e981a12620',
 'W02':'aa8ab4eaf8b6c864f8619b510e8e5fd1ce703948a29973464b2345ab79599ab0',
 'W03':'409f3e06b8305195a78d318f3ae474152f1ef1818f8725011eec70713368cf9f',
}
NEW_SHA={
 'W01':'ace1176fe2317f8c90596505283af6e624c5d7e58e655d8e513797857ce513ca',
 'W02':'f65f39f294c811bfa6bb95145d0e126c9191193b165ed9dd94d341fe6f5566f5',
 'W03':'3137c1242e5f301bbed5f3be82e8cc8d01c9c841ced4fe2782afb26c9519878e',
}
OLD_HELPER='''def _is_explicit_http_not_found(exc):\n    # Never infer absence from an error string, authentication, connection failure,\n    # or generic API exception. Only an explicit HTTP 404 admits a new campaign.\n    status = getattr(exc, "status_code", None)\n    response = getattr(exc, "response", None)\n    if response is not None:\n        status = getattr(response, "status_code", status)\n    return type(status) is int and status == 404\n'''
NEW_HELPER='''def _is_explicit_http_not_found(exc):\n    # Fail closed by default. Admit a fresh campaign only when absence is proven by\n    # either HTTP 404 or KaggleHub's canonical BackendError NOT_FOUND envelope.\n    status = getattr(exc, "status_code", None)\n    response = getattr(exc, "response", None)\n    if response is not None:\n        status = getattr(response, "status_code", status)\n    if type(status) is int and status == 404:\n        return True\n\n    if exc.__class__.__name__ != "BackendError":\n        return False\n    text = str(exc).strip()\n    marker = "POST failed with:"\n    if marker not in text:\n        return False\n    payload_text = text.split(marker, 1)[1].strip()\n    try:\n        payload = json.loads(payload_text)\n    except Exception:\n        return False\n    return (\n        isinstance(payload, dict)\n        and payload.get("wasSuccessful") is False\n        and isinstance(payload.get("error"), dict)\n        and payload["error"].get("code") == 5\n        and payload.get("errors") == ["Not found"]\n    )\n'''

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main(src,dst):
    src=pathlib.Path(src); dst=pathlib.Path(dst); dst.mkdir(parents=True,exist_ok=True)
    evidence=[]
    for slot in ('W01','W02','W03'):
        old=src/f'PNEUMONIA_V17_M07_STAGE1_SCREEN_{slot}_{OLD_CAMPAIGN}.ipynb'
        if sha(old)!=OLD_SHA[slot]: raise SystemExit(f'{slot}: predecessor SHA mismatch')
        raw=old.read_text(encoding='utf-8').replace(OLD_CAMPAIGN,NEW_CAMPAIGN)
        nb=json.loads(raw); hits=0
        for cell in nb['cells']:
            text=''.join(cell.get('source',[]))
            if OLD_HELPER in text:
                text=text.replace(OLD_HELPER,NEW_HELPER); hits+=1
                cell['source']=text.splitlines(keepends=True)
        if hits!=1: raise SystemExit(f'{slot}: expected one absence-classifier patch, got {hits}')
        out=dst/f'PNEUMONIA_V17_M07_STAGE1_SCREEN_{slot}_{NEW_CAMPAIGN}.ipynb'
        # Keep candidate identity independent of the runner OS. Path.write_text()
        # translates LF to CRLF on Windows and invalidates the frozen SHA values.
        out.write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
        code='\n'.join(''.join(c.get('source',[])) for c in nb['cells'] if c.get('cell_type')=='code')
        ast.parse(code)
        required=[NEW_CAMPAIGN,'DIST_ROLE = "M07_HPO_SCREEN"',f'DIST_SLOT = "{slot}"','UNLOCK_REMAINING_MODELS = False','paultimothymooney/chest-xray-pneumonia',f'm07-hpo-screen-{slot.lower()}-{NEW_CAMPAIGN}','payload.get("errors") == ["Not found"]']
        missing=[x for x in required if x not in code]
        if missing: raise SystemExit(f'{slot}: missing {missing}')
        got=sha(out)
        if got!=NEW_SHA[slot]: raise SystemExit(f'{slot}: superseding SHA mismatch {got}')
        evidence.append({'slot':slot,'file':out.name,'sha256':got,'json_ok':True,'python_ast_ok':True})
    class BackendError(Exception): pass
    class H404(Exception): status_code=404
    class H403(Exception): status_code=403
    ns={'json':json}; exec(NEW_HELPER,ns); pred=ns['_is_explicit_http_not_found']
    cases=[
      ('http404',H404('x'),True),
      ('canonical_not_found',BackendError('POST failed with: {"errors":["Not found"],"error":{"code":5},"wasSuccessful":false}'),True),
      ('http403',H403('x'),False),
      ('generic_string',Exception('Not found'),False),
      ('permission',BackendError('POST failed with: {"errors":["Permission denied"],"error":{"code":5},"wasSuccessful":false}'),False),
      ('wrong_code',BackendError('POST failed with: {"errors":["Not found"],"error":{"code":7},"wasSuccessful":false}'),False),
      ('malformed',BackendError('POST failed with: nope'),False),
    ]
    checks=[]
    for name,exc,want in cases:
      got=pred(exc); checks.append({'case':name,'expected':want,'actual':got,'pass':got==want})
    if not all(x['pass'] for x in checks): raise SystemExit('contract regression failed')
    ev={'campaign':NEW_CAMPAIGN,'supersedes':OLD_CAMPAIGN,'stage1':evidence,'contract_checks':checks,'pass':True}
    (dst/'RUNTIME_BUILD_EVIDENCE.json').write_bytes(json.dumps(ev,indent=2).encode('utf-8'))
    print(json.dumps(ev,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--src',required=True); ap.add_argument('--dst',required=True); a=ap.parse_args(); main(a.src,a.dst)
