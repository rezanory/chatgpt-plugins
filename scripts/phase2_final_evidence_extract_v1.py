from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

ACTION_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"

# One lightweight CPU extractor per owning Kaggle account.  M07 is split across
# kg-03 (R224) and kg-05 (R320/R384), so those account jobs carry extra targets.
ACCOUNT_PLANS = {
    "master": {
        "owner": "azadka",
        "targets": [
            ("M01", 224, "azadka/pneumonia-m01-r224-state-v1-7"),
            ("M01", 320, "azadka/pneumonia-m01-r320-state-v1-7"),
            ("M01", 384, "azadka/pneumonia-m01-r384-state-v1-7"),
        ],
    },
    "kg-02": {
        "owner": "radlinaradlina",
        "targets": [
            ("M02", 224, "radlinaradlina/pneumonia-m02-r224-state-v1-7"),
            ("M02", 320, "radlinaradlina/pneumonia-m02-r320-state-v1-7"),
            ("M02", 384, "radlinaradlina/pneumonia-m02-r384-state-v1-7"),
        ],
    },
    "kg-03": {
        "owner": "rezanory",
        "targets": [
            ("M03", 224, "rezanory/pneumonia-m03-r224-state-v1-7"),
            ("M03", 320, "rezanory/pneumonia-m03-r320-state-v1-7"),
            ("M03", 384, "rezanory/pneumonia-m03-r384-state-v1-7"),
            ("M07", 224, "rezanory/m07-final-5fold-fix2-d260914d"),
        ],
    },
    "kg-04": {
        "owner": "reyhanehazad",
        "targets": [
            ("M04", 224, "reyhanehazad/pneumonia-m04-r224-state-v1-7"),
            ("M04", 320, "reyhanehazad/pneumonia-m04-r320-state-v1-7"),
            ("M04", 384, "reyhanehazad/pneumonia-m04-r384-state-v1-7"),
        ],
    },
    "kg-05": {
        "owner": "trickermark",
        "targets": [
            ("M05", 224, "trickermark/pneumonia-m05-r224-state-v1-7"),
            ("M05", 320, "trickermark/pneumonia-m05-r320-state-v1-7"),
            ("M05", 384, "trickermark/pneumonia-m05-r384-state-v1-7"),
            ("M07", 320, "trickermark/m07-gate-r320-state-v1-7"),
            ("M07", 384, "trickermark/m07-gate-r384-state-v1-7"),
        ],
    },
    "kg-06": {
        "owner": "msdenis",
        "targets": [
            ("M06", 224, "msdenis/pneumonia-m06-r224-state-v1-7"),
            ("M06", 320, "msdenis/pneumonia-m06-r320-state-v1-7"),
            ("M06", 384, "msdenis/pneumonia-m06-r384-state-v1-7"),
        ],
    },
    "kg-07": {
        "owner": "nisabulutmark",
        "targets": [
            ("M08", 224, "nisabulutmark/pneumonia-m08-r224-state-v1-7"),
            ("M08", 320, "nisabulutmark/pneumonia-m08-r320-state-v1-7"),
            ("M08", 384, "nisabulutmark/pneumonia-m08-r384-state-v1-7"),
        ],
    },
    "kg-08": {
        "owner": "azadkk",
        "targets": [
            ("M09", 224, "azadkk/pneumonia-m09-r224-state-v1-7"),
            ("M09", 320, "azadkk/pneumonia-m09-r320-state-v1-7"),
            ("M09", 384, "azadkk/pneumonia-m09-r384-state-v1-7"),
        ],
    },
    "kg-09": {
        "owner": "mylovevpn1",
        "targets": [
            ("M10", 224, "mylovevpn1/pneumonia-m10-r224-state-v1-7"),
            ("M10", 320, "mylovevpn1/pneumonia-m10-r320-state-v1-7"),
            ("M10", 384, "mylovevpn1/pneumonia-m10-r384-state-v1-7"),
        ],
    },
    "kg-10": {
        "owner": "computstu1",
        "targets": [
            ("M11", 224, "computstu1/pneumonia-m11-r224-state-v1-7"),
            ("M11", 320, "computstu1/pneumonia-m11-r320-state-v1-7"),
            ("M11", 384, "computstu1/pneumonia-m11-r384-state-v1-7"),
        ],
    },
    "kg-11": {
        "owner": "jobreza1",
        "targets": [
            ("M12", 224, "jobreza1/pneumonia-m12-r224-state-v1-7"),
            ("M12", 320, "jobreza1/pneumonia-m12-r320-state-v1-7"),
            ("M12", 384, "jobreza1/pneumonia-m12-r384-state-v1-7"),
        ],
    },
}

TERMINAL = {"COMPLETE", "ERROR", "CANCELLED"}
ACTIVE = {"QUEUED", "RUNNING"}

LEGACY_CAMPAIGN_MARKERS = {
    "rezanory/m07-final-5fold-fix2-d260914d": "M07_CAMPAIGN_STATE.json",
}


def post_json(endpoint: str, token: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "phase2-final-evidence-extract-v1/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "http_status": exc.code,
            "error": exc.read(12000).decode("utf-8", "replace"),
        }


def recursive_status(value) -> str:
    if isinstance(value, dict):
        for key in ("status", "kernelStatus", "state"):
            item = value.get(key)
            if isinstance(item, str) and item.upper() in TERMINAL | ACTIVE:
                return item.upper()
        for item in value.values():
            found = recursive_status(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = recursive_status(item)
            if found:
                return found
    return ""


def resolve_version(read_token: str, account_id: str, dataset_ref: str) -> dict:
    # Historical Phase-2 completion was established through account-scoped
    # ListDatasets inventory.  Keep that same proven read path here: some private
    # state datasets return 403 to GetDataset even for their owner account.
    owner, slug = dataset_ref.split("/", 1)
    listing = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "raw_read",
            "account_id": account_id,
            "service": "datasets.DatasetApiService",
            "method": "ListDatasets",
            "body": {
                "group": "MY",
                "search": slug,
                "page": 1,
                "pageSize": 100,
            },
        },
    )
    if not listing.get("ok"):
        raise RuntimeError(f"DATASET_LIST_FAILED:{dataset_ref}:{listing}")
    rows = (listing.get("result") or {}).get("datasets") or []
    exact = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("ref") or "").casefold() == dataset_ref.casefold()
    ]
    if len(exact) != 1:
        raise RuntimeError(
            f"DATASET_VERSION_EXACT_MATCH_INVALID:{dataset_ref}:{len(exact)}"
        )
    row = exact[0]
    version = row.get("currentVersionNumber", row.get("current_version_number"))
    if not isinstance(version, int) or version < 1:
        raise RuntimeError(f"DATASET_VERSION_UNRESOLVED:{dataset_ref}")
    marker_name = LEGACY_CAMPAIGN_MARKERS.get(dataset_ref, "CAMPAIGN_STATE.json")
    marker = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "dataset_json_files",
            "account_id": account_id,
            "dataset_ref": dataset_ref,
            "dataset_version_number": version,
            "file_names": [marker_name],
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    if not marker.get("ok"):
        raise RuntimeError(f"DATASET_MARKER_FETCH_FAILED:{dataset_ref}:{marker}")
    marker_result = marker.get("result") or {}
    marker_files = marker_result.get("files") or []
    if (
        len(marker_files) != 1
        or not isinstance(marker_files[0], dict)
        or not isinstance(marker_files[0].get("json"), dict)
    ):
        raise RuntimeError(f"DATASET_MARKER_SHAPE_INVALID:{dataset_ref}")
    marker_sha256 = str(marker_files[0].get("sha256") or "")
    if len(marker_sha256) != 64:
        raise RuntimeError(f"DATASET_MARKER_SHA_INVALID:{dataset_ref}")
    marker_json = marker_files[0]["json"]
    completed_folds = marker_json.get("completed_folds")
    derived_status = marker_json.get("status")
    if (
        derived_status is None
        and isinstance(completed_folds, list)
        and completed_folds == [1, 2, 3, 4, 5]
    ):
        derived_status = "COMPLETE"
    return {
        "version": version,
        "campaign_marker_name": marker_name,
        "campaign_marker_sha256": marker_sha256,
        "campaign_receipt_sha256": marker_json.get("receipt_sha256"),
        "campaign_status": derived_status,
        "campaign_split_fingerprint": marker_json.get("split_fingerprint"),
        "campaign_completed_folds": completed_folds,
    }

def kernel_script(account_id: str, targets: list[dict]) -> str:
    # The extractor never imports ML frameworks and never opens image data.
    # It only reads already-sealed evidence JSON/CSV metadata from mounted states.
    payload = json.dumps(targets, sort_keys=True, separators=(",", ":"))
    return f'''from pathlib import Path
import hashlib, json, re, zipfile

ACCOUNT_ID={account_id!r}
TARGETS=json.loads({payload!r})
ROOT=Path("/kaggle/input")
METRIC_KEYS=set("""accuracy balanced_accuracy precision recall sensitivity specificity f1 f2 mcc auroc auc auprc average_precision brier ece threshold tn fp fn tp precision_positive recall_positive precision_normal recall_normal macro_precision macro_recall macro_f1""".split())
POLICY_TOKENS=("locked","external","training","hpo","threshold","fingerprint","split","recipe","status","schema","model","resolution","calibrat")

def scalars(d):
    out={{}}
    for k,v in d.items():
        if isinstance(v,(str,int,float,bool)) or v is None:
            out[str(k)]=v
    return out

def summarize_json(value):
    metric_blocks=[]
    policy_paths=[]
    def walk(x,path):
        if isinstance(x,dict):
            keys=set(str(k).lower() for k in x)
            hit=sorted(keys & METRIC_KEYS)
            if len(hit)>=2 and len(metric_blocks)<80:
                block=scalars(x)
                if block:
                    metric_blocks.append({{"path":path or "$","values":block}})
            for k,v in x.items():
                kp=f"{{path}}.{{k}}" if path else str(k)
                lk=str(k).lower()
                if any(tok in lk for tok in POLICY_TOKENS) and isinstance(v,(str,int,float,bool,type(None))) and len(policy_paths)<60:
                    policy_paths.append({{"path":kp,"value":v}})
                walk(v,kp)
        elif isinstance(x,list):
            for i,v in enumerate(x[:100]):
                walk(v,f"{{path}}[{{i}}]")
    walk(value,"")
    return {{"metric_blocks":metric_blocks,"policy_paths":policy_paths}}

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def resolve_state_root(target):
    dataset_ref=str(target["dataset_ref"])
    version=int(target["version"])
    marker_name=str(target.get("campaign_marker_name") or "CAMPAIGN_STATE.json")
    expected_marker_sha=str(target.get("campaign_marker_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{{64}}",expected_marker_sha):
        raise RuntimeError("CAMPAIGN_MARKER_EXPECTED_SHA_INVALID:"+dataset_ref)

    slug=dataset_ref.split("/",1)[1]
    direct=ROOT/slug
    if direct.is_dir():
        direct_markers=[p for p in direct.rglob(marker_name) if p.is_file()]
        if len(direct_markers)==1 and sha256_file(direct_markers[0])==expected_marker_sha:
            return direct_markers[0].parent,"ATTACHED_DATASET_DIRECT_SHA"

    marker_candidates=[]
    scanned=0
    for marker in ROOT.rglob(marker_name):
        if not marker.is_file():
            continue
        scanned+=1
        if scanned>200:
            raise RuntimeError("DATASET_MARKER_SCAN_BOUND_EXCEEDED:"+dataset_ref)
        if sha256_file(marker)==expected_marker_sha:
            marker_candidates.append(marker)

    if len(marker_candidates)!=1:
        raise RuntimeError(
            "DATASET_ATTACHMENT_MARKER_SHA_MATCH_INVALID:"
            +dataset_ref+":"+str(version)+":"+str(len(marker_candidates))
            +":"+str([p.as_posix() for p in marker_candidates[:10]])
        )

    marker=marker_candidates[0]
    marker_path=marker.as_posix().casefold()
    slug_token=slug.casefold()
    if slug_token not in marker_path:
        # New Kaggle mount layouts can normalize path components, so SHA is the
        # authority. Keep this only as diagnostic metadata, never as a selector.
        access_mode="ATTACHED_DATASET_MARKER_SHA_PATH_NORMALIZED"
    else:
        access_mode="ATTACHED_DATASET_MARKER_SHA"
    return marker.parent,access_mode

def read_json_file(path):
    if path.stat().st_size>2_000_000:
        raise RuntimeError("JSON_TOO_LARGE:"+str(path))
    return json.loads(path.read_text(encoding="utf-8"))

def read_json_member(z,member):
    info=z.getinfo(member)
    if info.file_size>2_000_000:
        raise RuntimeError("ARCHIVE_JSON_TOO_LARGE:"+member)
    return json.loads(z.read(member).decode("utf-8"))

def candidate_name(name):
    base=Path(name).name.upper()
    if not base.endswith(".JSON"):
        return False
    return (
        base=="FINAL_REPORT.JSON"
        or ("OOF" in base and "METRIC" in base)
        or "CALIBRAT" in base
        or "LOCKED" in base
        or "TERMINAL" in base
        or "RECEIPT" in base
        or base=="COMPLETED.JSON"
    )

rows=[]
for target in TARGETS:
    model_id=str(target["model_id"])
    resolution=int(target["resolution"])
    dataset_ref=str(target["dataset_ref"])
    version=int(target["version"])
    state_root,access_mode=resolve_state_root(target)
    files=[p for p in state_root.rglob("*") if p.is_file()]
    if len(files)>5000:
        raise RuntimeError("DATASET_FILE_COUNT_UNBOUNDED:"+dataset_ref)
    row={{
        "model_id":model_id,
        "resolution":resolution,
        "dataset_ref":dataset_ref,
        "dataset_version_number":version,
        "state_root_name":state_root.name,
        "access_mode":access_mode,
        "archive":None,
        "json_sources":[],
    }}
    campaign_marker_name=str(target.get("campaign_marker_name") or "CAMPAIGN_STATE.json")
    campaign=[p for p in files if p.name==campaign_marker_name]
    if len(campaign)==1:
        observed_campaign_sha=sha256_file(campaign[0])
        expected_campaign_sha=str(target.get("campaign_marker_sha256") or "")
        if observed_campaign_sha != expected_campaign_sha:
            raise RuntimeError("CAMPAIGN_MARKER_SHA_MISMATCH:"+dataset_ref+":"+expected_campaign_sha+":"+observed_campaign_sha)
        cp=read_json_file(campaign[0])
        expected_receipt=target.get("campaign_receipt_sha256")
        if expected_receipt is not None and cp.get("receipt_sha256") != expected_receipt:
            raise RuntimeError("CAMPAIGN_RECEIPT_MISMATCH:"+dataset_ref)
        expected_status=target.get("campaign_status")
        observed_status=cp.get("status")
        if observed_status is None and cp.get("completed_folds")==[1,2,3,4,5]:
            observed_status="COMPLETE"
        if expected_status is not None and observed_status != expected_status:
            raise RuntimeError("CAMPAIGN_STATUS_MISMATCH:"+dataset_ref)
        expected_split=target.get("campaign_split_fingerprint")
        if expected_split is not None and cp.get("split_fingerprint") != expected_split:
            raise RuntimeError("CAMPAIGN_SPLIT_FINGERPRINT_MISMATCH:"+dataset_ref)
        expected_completed=target.get("campaign_completed_folds")
        if expected_completed is not None and cp.get("completed_folds") != expected_completed:
            raise RuntimeError("CAMPAIGN_COMPLETED_FOLDS_MISMATCH:"+dataset_ref)
        row["campaign"]={{
            "marker_name":campaign_marker_name,
            "schema":cp.get("schema"),
            "status":observed_status,
            "model_id":cp.get("model_id"),
            "resolution":cp.get("resolution"),
            "receipt_sha256":cp.get("receipt_sha256"),
            "split_fingerprint":cp.get("split_fingerprint"),
            "completed_folds":cp.get("completed_folds"),
            "marker_sha256":observed_campaign_sha,
            "artifact_sha256":cp.get("artifact_sha256"),
        }}
    else:
        raise RuntimeError("CAMPAIGN_MARKER_COUNT_INVALID:"+dataset_ref+":"+str(len(campaign)))
    archives=[p for p in files if p.name=="FINAL_EVIDENCE.cgpzip"]
    if len(archives)>1:
        raise RuntimeError("FINAL_EVIDENCE_ARCHIVE_AMBIGUOUS:"+dataset_ref)
    sources=[]
    if archives:
        archive=archives[0]
        row["archive"]={{"name":archive.name,"sha256":sha256_file(archive),"bytes":archive.stat().st_size}}
        with zipfile.ZipFile(archive) as z:
            members=[n for n in z.namelist() if not n.endswith("/")]
            row["archive"]["member_count"]=len(members)
            row["archive"]["members"]=members[:120]
            for name in members:
                if candidate_name(name):
                    sources.append(("archive:"+name,read_json_member(z,name)))

    fold_archives=sorted(
        [p for p in files if re.fullmatch(r"FOLD_[1-5]_RECOVERY\\.cgpzip",p.name)],
        key=lambda p:p.name,
    )
    row["fold_archives"]=[]
    for fold_archive in fold_archives:
        fold_record={{"name":fold_archive.name,"sha256":sha256_file(fold_archive),"bytes":fold_archive.stat().st_size}}
        with zipfile.ZipFile(fold_archive) as z:
            completed=[n for n in z.namelist() if Path(n).name=="COMPLETED.json"]
            if len(completed)!=1:
                raise RuntimeError("FOLD_COMPLETED_RECEIPT_COUNT_INVALID:"+dataset_ref+":"+fold_archive.name+":"+str(len(completed)))
            sources.append(("fold_archive:"+fold_archive.name+":"+completed[0],read_json_member(z,completed[0])))
            fold_record["completed_member"]=completed[0]
        row["fold_archives"].append(fold_record)

    for path in files:
        if candidate_name(path.name) and path.name!="CAMPAIGN_STATE.json":
            label="file:"+path.relative_to(state_root).as_posix()
            if not any(existing[0]==label for existing in sources):
                sources.append((label,read_json_file(path)))
    if not sources:
        # M07 R224 historically has direct evidence with nonstandard filenames.
        for path in files:
            if path.suffix.lower()==".json" and ("REPORT" in path.name.upper() or "OOF" in path.name.upper()):
                sources.append(("file:"+path.relative_to(state_root).as_posix(),read_json_file(path)))
    for source_name,value in sources[:18]:
        summary=summarize_json(value)
        row["json_sources"].append({{
            "source":source_name,
            "schema":value.get("schema") if isinstance(value,dict) else None,
            "status":value.get("status") if isinstance(value,dict) else None,
            "top_keys":list(value.keys())[:120] if isinstance(value,dict) else [],
            **summary,
        }})
    if not row["json_sources"]:
        raise RuntimeError("NO_FINAL_JSON_EVIDENCE:"+dataset_ref)
    rows.append(row)

out={{
    "schema":"pneumonia.phase2.final_evidence.extract.account.v1",
    "status":"PASS",
    "account_id":ACCOUNT_ID,
    "training_performed":False,
    "hpo_performed":False,
    "locked_test_executed":False,
    "external_validation_executed":False,
    "targets":rows,
}}
name="PHASE2_FINAL_EVIDENCE_"+ACCOUNT_ID.replace("-","_").upper()+".json"
Path("/kaggle/working",name).write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8")
print("PHASE2_FINAL_EVIDENCE_EXTRACT_PASS "+json.dumps({{"account_id":ACCOUNT_ID,"target_count":len(rows)}},sort_keys=True))
'''


def normalize_provider_ref(value: str, expected: str, owner: str) -> str:
    raw = str(value or expected).strip().strip("/")
    if raw.startswith("code/"):
        raw = raw[5:]
    if raw.count("/") != 1 or raw.split("/", 1)[0].casefold() != owner.casefold():
        raise RuntimeError(f"PROVIDER_REF_INVALID:{value}")
    return raw


def _listed_kernel_ref(item: dict, owner: str) -> str | None:
    raw = str(item.get("ref") or "").strip()
    if raw:
        raw = raw.strip("/")
        if raw.startswith("code/"):
            raw = raw[5:]
        if raw.count("/") == 1 and raw.split("/", 1)[0].casefold() == owner.casefold():
            return raw
    item_owner = str(
        item.get("ownerRef")
        or item.get("ownerUser")
        or item.get("owner")
        or item.get("userName")
        or ""
    ).strip()
    item_slug = str(item.get("kernelSlug") or item.get("slug") or "").strip()
    if item_owner.casefold() == owner.casefold() and item_slug:
        return f"{owner}/{item_slug}"
    return None


def _listing_status(item: dict) -> str:
    status = recursive_status(item)
    if status:
        return status
    if item.get("isRunning") is True or item.get("is_running") is True:
        return "RUNNING"
    return ""


def _extractor_output_name(account_id: str) -> str:
    return "PHASE2_FINAL_EVIDENCE_" + account_id.replace("-", "_").upper() + ".json"


def probe_extractor_status(
    read_token: str,
    account_id: str,
    kernel_ref: str,
    listing_item: dict | None = None,
) -> str:
    owner, slug = kernel_ref.split("/", 1)

    status_response = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "raw_read",
            "account_id": account_id,
            "service": "kernels.KernelsApiService",
            "method": "GetKernelSessionStatus",
            "body": {"userName": owner, "kernelSlug": slug},
        },
        timeout=120,
    )
    if status_response.get("ok"):
        status = recursive_status(status_response)
        if status in TERMINAL | ACTIVE:
            return status

    receipt_response = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "output_json_files",
            "account_id": account_id,
            "kernel_ref": kernel_ref,
            "file_names": [_extractor_output_name(account_id)],
            "max_bytes_per_file": 262144,
        },
        timeout=120,
    )
    if receipt_response.get("ok"):
        receipt_result = receipt_response.get("result") or {}
        receipt_files = receipt_result.get("files") or []
        if (
            len(receipt_files) == 1
            and isinstance(receipt_files[0], dict)
            and isinstance(receipt_files[0].get("json"), dict)
            and receipt_files[0]["json"].get("status") == "PASS"
        ):
            return "COMPLETE"

    output_response = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "raw_read",
            "account_id": account_id,
            "service": "kernels.KernelsApiService",
            "method": "ListKernelSessionOutput",
            "body": {
                "userName": owner,
                "kernelSlug": slug,
                "page": 1,
                "pageSize": 200,
            },
        },
        timeout=120,
    )
    if output_response.get("ok"):
        output_result = output_response.get("result") or {}
        names = {
            str(item.get("fileName") or "").split("/")[-1]
            for item in (output_result.get("files") or [])
            if isinstance(item, dict)
        }
        if _extractor_output_name(account_id) in names:
            return "COMPLETE"
        log = str(output_result.get("log") or "")
        if "PHASE2_FINAL_EVIDENCE_EXTRACT_PASS" in log:
            return "COMPLETE"
        if "Traceback (most recent call last)" in log or "RuntimeError:" in log:
            return "ERROR"

    if listing_item is not None:
        listed = _listing_status(listing_item)
        if listed in TERMINAL | ACTIVE:
            return listed
    return "UNKNOWN"


def reusable_extractor(
    read_token: str,
    account_id: str,
    owner: str,
) -> dict | None:
    search = f"phase2-final-evidence-{account_id}"
    listing = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "raw_read",
            "account_id": account_id,
            "service": "kernels.KernelsApiService",
            "method": "ListKernels",
            "body": {
                "group": "PROFILE",
                "user": owner,
                "search": "phase2-final-evidence",
                "sortBy": "DATE_RUN",
                "page": 1,
                "pageSize": 100,
            },
        },
        timeout=120,
    )
    if not listing.get("ok"):
        raise RuntimeError(f"EXTRACTOR_HISTORY_LIST_FAILED:{account_id}:{listing}")
    result = listing.get("result") or {}
    kernels = result.get("kernels") or []
    account_tokens = {account_id.casefold(), account_id.replace("-", "").casefold()}
    candidates = []
    for item in kernels:
        if not isinstance(item, dict):
            continue
        ref = _listed_kernel_ref(item, owner)
        if not ref:
            continue
        slug = ref.split("/", 1)[1]
        match = None
        for token in account_tokens:
            prefix = f"phase2-final-evidence-{token}-"
            if slug.casefold().startswith(prefix):
                suffix = slug[len(prefix):]
                if suffix.isdigit():
                    match = int(suffix)
                    break
        if match is None:
            continue
        candidates.append((match, ref, item))
    if not candidates:
        return None
    run_number, ref, item = max(candidates, key=lambda row: row[0])
    status = probe_extractor_status(
        read_token,
        account_id,
        ref,
        listing_item=item,
    )
    if status in ACTIVE | {"COMPLETE"}:
        return {
            "kernel_ref": ref,
            "status": status,
            "source_run_id": str(run_number),
        }
    if status in {"ERROR", "CANCELLED"}:
        return None
    raise RuntimeError(
        f"EXTRACTOR_LATEST_STATUS_UNCERTAIN:{account_id}:{ref}"
    )


def _action_envelope(result: dict) -> tuple[dict, dict]:
    if (
        result.get("provider") == "kaggle"
        and result.get("operation_class") is not None
    ):
        broker = result
    else:
        nested = result.get("result")
        broker = nested if isinstance(nested, dict) else {}
    provider = broker.get("result") if isinstance(broker.get("result"), dict) else {}
    return broker, provider


def launch_account(action_token: str, read_token: str, run_id: str, account_id: str, plan: dict) -> dict:
    owner = plan["owner"]
    resolved = []
    for model_id, resolution, dataset_ref in plan["targets"]:
        resolved_state = resolve_version(read_token, account_id, dataset_ref)
        version = int(resolved_state["version"])
        resolved.append(
            {
                "model_id": model_id,
                "resolution": resolution,
                "dataset_ref": dataset_ref,
                "version": version,
                "campaign_marker_name": resolved_state["campaign_marker_name"],
                "campaign_marker_sha256": resolved_state["campaign_marker_sha256"],
                "campaign_receipt_sha256": resolved_state["campaign_receipt_sha256"],
                "campaign_status": resolved_state["campaign_status"],
                "campaign_split_fingerprint": resolved_state["campaign_split_fingerprint"],
                "campaign_completed_folds": resolved_state["campaign_completed_folds"],
            }
        )
    reusable = reusable_extractor(read_token, account_id, owner)
    if reusable is not None:
        return {
            "account_id": account_id,
            "owner": owner,
            "kernel_ref": reusable["kernel_ref"],
            "status": reusable["status"],
            "reused": True,
            "source_run_id": reusable["source_run_id"],
            "targets": resolved,
        }

    slug_token = "master" if account_id == "master" else account_id
    slug = f"phase2-final-evidence-{slug_token}-{run_id}"
    expected_ref = f"{owner}/{slug}"

    payload = {
        "request_id": f"phase2-final-evidence-{account_id}-{run_id}",
        "provider": "kaggle",
        "operation_class": "compute",
        "account_id": account_id,
        "purpose": (
            "Phase-2 final scientific evidence extraction only. CPU-only, no model loading, no training, "
            "no HPO, no locked-test execution, no external validation."
        ),
        "service": "kernels.KernelsApiService",
        "method": "SaveKernel",
        "body": {
            "slug": expected_ref,
            "newTitle": f"Phase2 Final Evidence {account_id} {run_id}",
            "text": kernel_script(account_id, resolved),
            "language": "python",
            "kernelType": "script",
            "kernelExecutionType": "SAVE_AND_RUN_ALL",
            "isPrivate": True,
            "enableGpu": False,
            "enableTpu": False,
            "enableInternet": False,
            "kernelDataSources": [],
            "datasetDataSources": [item["dataset_ref"] for item in resolved],
            "competitionDataSources": [],
            "modelDataSources": [],
        },
    }
    result = post_json(ACTION_ENDPOINT, action_token, payload, timeout=240)
    broker, provider = _action_envelope(result)
    error = str(
        broker.get("error")
        or provider.get("error")
        or result.get("error")
        or ""
    )
    if not result.get("ok") or not broker.get("ok") or error:
        raise RuntimeError(
            f"EXTRACTOR_SAVEKERNEL_REJECTED:{account_id}:{error or result}"
        )
    invalid_datasets = (
        provider.get("invalidDatasetSources")
        or broker.get("invalidDatasetSources")
        or []
    )
    invalid_kernels = (
        provider.get("invalidKernelSources")
        or broker.get("invalidKernelSources")
        or []
    )
    if invalid_datasets or invalid_kernels:
        raise RuntimeError(
            f"EXTRACTOR_SAVEKERNEL_SOURCE_REJECTION:{account_id}:"
            f"datasets={invalid_datasets}:kernels={invalid_kernels}"
        )
    ref = normalize_provider_ref(
        broker.get("provider_ref")
        or result.get("provider_ref")
        or provider.get("ref")
        or expected_ref,
        expected_ref,
        owner,
    )
    return {
        "account_id": account_id,
        "owner": owner,
        "kernel_ref": ref,
        "status": "SUBMITTED",
        "reused": False,
        "source_run_id": str(run_id),
        "targets": resolved,
    }


def fetch_account_result(read_token: str, item: dict) -> dict:
    account_id = item["account_id"]
    name = _extractor_output_name(account_id)
    response = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "output_json_files",
            "account_id": account_id,
            "kernel_ref": item["kernel_ref"],
            "file_names": [name],
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    if not response.get("ok"):
        raise RuntimeError(f"EXTRACTOR_OUTPUT_FETCH_FAILED:{account_id}:{response}")
    result = response.get("result") or {}
    files = result.get("files") or []
    if len(files) != 1 or not isinstance(files[0].get("json"), dict):
        raise RuntimeError(f"EXTRACTOR_OUTPUT_SHAPE_INVALID:{account_id}")
    return files[0]["json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not str(args.run_id).isdigit():
        raise SystemExit("RUN_ID_INVALID")
    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("OIDC_TOKEN_INVALID")

    launches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
        futures = {
            pool.submit(launch_account, action_token, read_token, args.run_id, account_id, plan): account_id
            for account_id, plan in ACCOUNT_PLANS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            account_id = futures[future]
            row = future.result()
            launches.append(row)
            print("PHASE2_FINAL_EXTRACT_LAUNCH", json.dumps({"account_id": account_id, "kernel_ref": row["kernel_ref"], "status": row["status"]}, sort_keys=True), flush=True)
    launches.sort(key=lambda row: row["account_id"])

    terminal = {}
    deadline = time.monotonic() + 75 * 60
    while time.monotonic() < deadline and len(terminal) < len(launches):
        active_rows = [row for row in launches if row["account_id"] not in terminal]
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
            futures = {
                pool.submit(
                    probe_extractor_status,
                    read_token,
                    row["account_id"],
                    row["kernel_ref"],
                ): row
                for row in active_rows
            }
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                status = future.result()
                print("PHASE2_FINAL_EXTRACT_STATUS", json.dumps({"account_id": row["account_id"], "status": status or "UNKNOWN"}, sort_keys=True), flush=True)
                if status in TERMINAL:
                    terminal[row["account_id"]] = status
        if len(terminal) < len(launches):
            time.sleep(20)

    if len(terminal) != len(launches):
        raise SystemExit("EXTRACTOR_TERMINAL_TIMEOUT")
    failed = {account: status for account, status in terminal.items() if status != "COMPLETE"}
    if failed:
        raise SystemExit("EXTRACTOR_TERMINAL_FAILURE=" + json.dumps(failed, sort_keys=True))

    account_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
        futures = {pool.submit(fetch_account_result, read_token, row): row["account_id"] for row in launches}
        for future in concurrent.futures.as_completed(futures):
            account_results.append(future.result())
    account_results.sort(key=lambda row: row["account_id"])

    units = []
    for account_result in account_results:
        if account_result.get("status") != "PASS" or account_result.get("training_performed") is not False:
            raise RuntimeError("ACCOUNT_EXTRACT_RECEIPT_INVALID:" + str(account_result.get("account_id")))
        units.extend(account_result.get("targets") or [])
    expected = {(f"M{index:02d}", resolution) for index in range(1, 13) for resolution in (224, 320, 384)}
    observed = {(str(row.get("model_id")), int(row.get("resolution"))) for row in units}
    if observed != expected or len(units) != 36:
        raise RuntimeError(
            "FINAL_EVIDENCE_MATRIX_COVERAGE_INVALID="
            + json.dumps({"missing": sorted(expected - observed), "extra": sorted(observed - expected), "count": len(units)}, sort_keys=True)
        )

    output = {
        "schema": "pneumonia.phase2.final_evidence.matrix.v1",
        "status": "PASS",
        "github_run_id": str(args.run_id),
        "accounts_complete": len(account_results),
        "models": 12,
        "resolutions_per_model": 3,
        "model_resolution_units": len(units),
        "folds_represented": 180,
        "training_performed": False,
        "hpo_performed": False,
        "locked_test_executed": False,
        "external_validation_executed": False,
        "launches": launches,
        "units": sorted(units, key=lambda row: (row["model_id"], row["resolution"])),
    }
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print("PHASE2_FINAL_EVIDENCE_MATRIX_PASS", json.dumps({"units": len(units), "folds": 180, "accounts": len(account_results)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
