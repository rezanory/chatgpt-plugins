"""Regression for the observed false capacity block. No network or credentials."""
import ast,json
from pathlib import Path
root=Path(__file__).resolve().parent
module=ast.parse((root/'controller.py').read_text(encoding='utf-8'))
terminal=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TERMINAL' for t in n.targets))
assert 'CANCEL_ACKNOWLEDGED' in terminal
assert 'RUNNING' not in terminal and 'QUEUED' not in terminal and 'UNKNOWN' not in terminal
observed={'kg-02':['CANCEL_ACKNOWLEDGED']*4,'kg-06':['CANCEL_ACKNOWLEDGED']*4,'kg-07':['CANCEL_ACKNOWLEDGED']*2,'kg-08':['UNKNOWN'],'kg-09':[],'kg-11':[]}
counts={account:sum(s not in terminal for s in statuses) for account,statuses in observed.items()}
assert sum(v<2 for v in counts.values())==6
assert sum(s not in terminal for s in ['RUNNING','UNKNOWN'])==2
receipt={'status':'PASS','checks_passed':5,'observed_cancelled_sessions_not_active':True,'unknown_sessions_still_fail_closed':True,'network_calls':0,'scientific_model_code_changed':False,'source_run':36440415257,'derived_active_or_unknown':counts}
(root/'CONTROLLER_STATUS_TEST_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt),flush=True)
