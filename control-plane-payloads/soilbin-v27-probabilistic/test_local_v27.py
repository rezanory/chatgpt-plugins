from pathlib import Path
import base64,gzip,hashlib,os
ROOT=Path(__file__).resolve().parent
runner=(ROOT/"runner.py").read_text(encoding="utf-8")
encoded=(ROOT/"all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
raw=gzip.decompress(base64.b64decode(encoded,validate=True))
assert hashlib.sha256(raw).hexdigest()=="dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
os.environ["SOILBIN_OUTPUT_ROOT"]=str(ROOT/"local_V27")
future="from __future__ import annotations"
payload={"lane":"V27","campaign":"local-v27-probabilistic","model_b64":encoded}
src=runner.replace(future,future+"\nPAYLOAD="+repr(payload),1)
exec(compile(src,str(ROOT/"runner.py"),"exec"),{"__name__":"__main__"})
