from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import time

from phase2_final_evidence_extract_v1 import (
    ACCOUNT_PLANS,
    ACTION_ENDPOINT,
    READ_ENDPOINT,
    TERMINAL,
    _action_envelope,
    _read_payload,
    normalize_provider_ref,
    post_json,
    recursive_status,
    resolve_version,
)

MAX_JSON_BYTES = 262_144
EXTRACTOR_GENERATION = "v1"


def vector_output_name(model_id: str, resolution: int) -> str:
    return f"PHASE2_OOF_VECTOR_{model_id}_R{int(resolution)}.json"


def account_summary_name(account_id: str) -> str:
    return "PHASE2_OOF_VECTOR_ACCOUNT_" + account_id.replace("-", "_").upper() + ".json"


def vector_kernel_script(account_id: str, targets: list[dict]) -> str:
    payload = json.dumps(targets, sort_keys=True, separators=(",", ":"))
    preamble = (
        "from pathlib import Path\n"
        "import base64,csv,hashlib,io,json,math,re,struct,zipfile,zlib\n\n"
        f"ACCOUNT_ID={account_id!r}\n"
        f"TARGETS=json.loads({payload!r})\n"
    )
    body = r'''
ROOT=Path("/kaggle/input")

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
    if not re.fullmatch(r"[0-9a-f]{64}",expected_marker_sha):
        raise RuntimeError("OOF_MARKER_EXPECTED_SHA_INVALID:"+dataset_ref)
    matches=[]
    scanned=0
    for marker in ROOT.rglob(marker_name):
        if not marker.is_file():
            continue
        scanned+=1
        if scanned>250:
            raise RuntimeError("OOF_MARKER_SCAN_BOUND_EXCEEDED:"+dataset_ref)
        if sha256_file(marker)==expected_marker_sha:
            matches.append(marker)
    if len(matches)!=1:
        raise RuntimeError("OOF_MARKER_SHA_MATCH_INVALID:"+dataset_ref+":"+str(version)+":"+str(len(matches)))
    return matches[0].parent

def read_json(path):
    if path.stat().st_size>2_000_000:
        raise RuntimeError("OOF_JSON_TOO_LARGE:"+path.as_posix())
    return json.loads(path.read_text(encoding="utf-8"))

def read_csv_path(path):
    with path.open("r",encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))

def read_csv_member(archive,member):
    with zipfile.ZipFile(archive) as z:
        info=z.getinfo(member)
        if info.file_size>25_000_000:
            raise RuntimeError("OOF_ARCHIVE_CSV_TOO_LARGE:"+member)
        with z.open(member,"r") as raw:
            with io.TextIOWrapper(raw,encoding="utf-8-sig",newline="") as handle:
                return list(csv.DictReader(handle))

def locate_standard_oof(files,archive):
    names={"OOF_PREDICTIONS.csv","M07_OOF_PREDICTIONS.csv"}
    direct=[p for p in files if p.name in names]
    if direct:
        hashes={sha256_file(p) for p in direct}
        if len(hashes)!=1:
            raise RuntimeError("OOF_DIRECT_CSV_CONFLICT:"+str([p.as_posix() for p in direct]))
        chosen=sorted(direct,key=lambda p:(len(p.as_posix()),p.as_posix()))[0]
        return read_csv_path(chosen),"file:"+chosen.as_posix(),sha256_file(chosen)
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            members=[name for name in z.namelist() if Path(name).name in names]
            if not members:
                return None,None,None
            ranked=sorted(members,key=lambda x:(0 if Path(x).name=="OOF_PREDICTIONS.csv" else 1,len(x),x))
            best_name=Path(ranked[0]).name
            tied=[m for m in ranked if Path(m).name==best_name]
            if len(tied)>1:
                payloads={hashlib.sha256(z.read(m)).hexdigest() for m in tied}
                if len(payloads)!=1:
                    raise RuntimeError("OOF_ARCHIVE_CSV_CONFLICT:"+str(tied))
            chosen=tied[0]
            data=z.read(chosen)
            return read_csv_member(archive,chosen),"archive:"+chosen,hashlib.sha256(data).hexdigest()
    return None,None,None

def reconstruct_m07_highres(files,state_root,target):
    dataset_ref=str(target["dataset_ref"])
    resolution=int(target["resolution"])
    expected_split=target.get("campaign_split_fingerprint")
    rows=[]
    fold_evidence=[]
    for fold in range(1,6):
        fold_token=("FOLD_"+str(fold)+"_RECOVERY").upper()
        fold_dir_token=("/fold_"+str(fold)+"/").lower()
        completed=[
            p for p in files
            if p.name=="COMPLETED.json"
            and fold_token in p.as_posix().upper()
            and fold_dir_token in p.as_posix().lower()
        ]
        validation=[
            p for p in files
            if p.name=="validation_predictions.csv"
            and fold_token in p.as_posix().upper()
            and fold_dir_token in p.as_posix().lower()
        ]
        if len(completed)!=1 or len(validation)!=1:
            raise RuntimeError(
                "OOF_M07_FOLD_EVIDENCE_COUNT_INVALID:"+dataset_ref+":"+str(fold)
                +":"+str(len(completed))+":"+str(len(validation))
            )
        receipt=read_json(completed[0])
        contract=receipt.get("run_contract") if isinstance(receipt.get("run_contract"),dict) else {}
        extra=contract.get("extra") if isinstance(contract.get("extra"),dict) else {}
        if (
            receipt.get("schema")!="pneumonia.phase2.fold.v1.7"
            or receipt.get("status")!="COMPLETED"
            or receipt.get("model_id")!="M07"
            or int(receipt.get("resolution") or 0)!=resolution
            or int(receipt.get("fold_id") or 0)!=fold
            or receipt.get("locked_test_used_for_training") is not False
            or receipt.get("external_used_for_training") is not False
            or contract.get("stage")!="phase2_fold"
            or contract.get("model_id")!="M07"
            or int(contract.get("resolution") or 0)!=resolution
            or int(contract.get("fold_id") or 0)!=fold
            or extra.get("campaign_role")!="M07_GATE_MULTIRES"
        ):
            raise RuntimeError("OOF_M07_FOLD_RECEIPT_POLICY_INVALID:"+dataset_ref+":"+str(fold))
        observed_split=receipt.get("split_fingerprint") or contract.get("split_fingerprint")
        if expected_split is not None and observed_split!=expected_split:
            raise RuntimeError("OOF_M07_FOLD_SPLIT_MISMATCH:"+dataset_ref+":"+str(fold))
        receipt_sha=str(receipt.get("receipt_sha256") or "")
        run_fingerprint=str(receipt.get("run_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{64}",receipt_sha) or not re.fullmatch(r"[0-9a-f]{64}",run_fingerprint):
            raise RuntimeError("OOF_M07_FOLD_RECEIPT_ID_INVALID:"+dataset_ref+":"+str(fold))
        expected_csv_sha=str((receipt.get("artifact_sha256") or {}).get("validation_predictions.csv") or "")
        observed_csv_sha=sha256_file(validation[0])
        if observed_csv_sha!=expected_csv_sha:
            raise RuntimeError("OOF_M07_VALIDATION_SHA_MISMATCH:"+dataset_ref+":"+str(fold))
        threshold=float((receipt.get("validation_metrics") or {}).get("threshold"))
        if not math.isfinite(threshold) or not 0.0<=threshold<=1.0:
            raise RuntimeError("OOF_M07_THRESHOLD_INVALID:"+dataset_ref+":"+str(fold))
        fold_rows=read_csv_path(validation[0])
        if not fold_rows:
            raise RuntimeError("OOF_M07_VALIDATION_EMPTY:"+dataset_ref+":"+str(fold))
        for item in fold_rows:
            item=dict(item)
            probability=float(item["probability_pneumonia"])
            if not math.isfinite(probability) or not 0.0<=probability<=1.0:
                raise RuntimeError("OOF_M07_PROBABILITY_INVALID:"+dataset_ref+":"+str(fold))
            item["prediction_fold_threshold"]=str(int(probability>=threshold))
            rows.append(item)
        fold_evidence.append({
            "fold_id":fold,
            "rows":len(fold_rows),
            "threshold":threshold,
            "receipt_sha256":receipt_sha,
            "run_fingerprint":run_fingerprint,
            "validation_predictions_sha256":observed_csv_sha,
            "validation_predictions_path":validation[0].relative_to(state_root).as_posix(),
        })
    return rows,fold_evidence

def encode_payload(inner):
    raw=json.dumps(inner,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(),base64.b64encode(zlib.compress(raw,9)).decode("ascii")

def vectorize(rows,model_id,resolution,dataset_ref,version,source,source_sha,fold_evidence):
    if not rows:
        raise RuntimeError("OOF_VECTOR_EMPTY:"+dataset_ref)
    required={"relative_path","patient_id","label","sha256","probability_pneumonia","prediction_fold_threshold"}
    if not required.issubset(set(rows[0])):
        raise RuntimeError("OOF_VECTOR_COLUMNS_MISSING:"+dataset_ref+":"+str(sorted(required-set(rows[0]))))
    normalized=[]
    for row in rows:
        relative_path=str(row["relative_path"]).strip()
        patient_id=str(row["patient_id"]).strip()
        label=int(row["label"])
        sha=str(row["sha256"]).strip()
        score=float(row["probability_pneumonia"])
        pred=int(float(row["prediction_fold_threshold"]))
        if (
            not relative_path or not patient_id or not re.fullmatch(r"[0-9a-f]{64}",sha)
            or label not in (0,1) or pred not in (0,1)
            or not math.isfinite(score) or not 0.0<=score<=1.0
        ):
            raise RuntimeError("OOF_VECTOR_ROW_INVALID:"+dataset_ref)
        normalized.append(((relative_path,patient_id,label,sha),pred,score))
    normalized.sort(key=lambda item:item[0])
    identities=[item[0] for item in normalized]
    if len(identities)!=len(set(identities)):
        raise RuntimeError("OOF_VECTOR_IDENTITY_DUPLICATE:"+dataset_ref)
    identity_raw=json.dumps(identities,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    inner={
        "patient_id":[item[0][1] for item in normalized],
        "label":[item[0][2] for item in normalized],
        "prediction":[item[1] for item in normalized],
        "score":[item[2] for item in normalized],
    }
    payload_sha,payload_b64=encode_payload(inner)
    return {
        "schema":"pneumonia.phase2.oof.vector.v1",
        "status":"PASS",
        "model_id":model_id,
        "resolution":resolution,
        "row_count":len(normalized),
        "identity_sequence_sha256":hashlib.sha256(identity_raw).hexdigest(),
        "payload_encoding":"zlib+base64+json",
        "payload_sha256":payload_sha,
        "payload_b64":payload_b64,
        "source_dataset_ref":dataset_ref,
        "source_dataset_version":int(version),
        "source_oof":source,
        "source_oof_sha256":source_sha,
        "source_fold_evidence":fold_evidence,
        "locked_test_used_for_selection":False,
        "external_used_for_selection":False,
        "training_performed":False,
        "inference_performed":False,
        "threshold_tuning_performed":False,
    }

receipts=[]
for target in TARGETS:
    model_id=str(target["model_id"])
    resolution=int(target["resolution"])
    dataset_ref=str(target["dataset_ref"])
    state_root=resolve_state_root(target)
    files=[p for p in state_root.rglob("*") if p.is_file()]
    if len(files)>5000:
        raise RuntimeError("OOF_DATASET_FILE_COUNT_UNBOUNDED:"+dataset_ref)
    archives=[p for p in files if p.name=="FINAL_EVIDENCE.cgpzip"]
    if len(archives)>1:
        raise RuntimeError("OOF_FINAL_EVIDENCE_AMBIGUOUS:"+dataset_ref)
    archive=archives[0] if archives else None
    rows,source,source_sha=locate_standard_oof(files,archive)
    fold_evidence=[]
    if rows is None and model_id=="M07" and resolution in (320,384):
        rows,fold_evidence=reconstruct_m07_highres(files,state_root,target)
        source="derived:CANONICAL_PHASE2_OOF_FROM_SEALED_FOLDS"
        source_sha=hashlib.sha256(
            json.dumps(fold_evidence,sort_keys=True,separators=(",",":")).encode("utf-8")
        ).hexdigest()
    if rows is None:
        raise RuntimeError("OOF_PREDICTIONS_NOT_FOUND:"+dataset_ref)
    value=vectorize(
        rows,model_id,resolution,dataset_ref,target["version"],source,source_sha,fold_evidence
    )
    name=f"PHASE2_OOF_VECTOR_{model_id}_R{resolution}.json"
    encoded=json.dumps(value,sort_keys=True,separators=(",",":")).encode("utf-8")
    if len(encoded)>250000:
        raise RuntimeError("OOF_VECTOR_JSON_TOO_LARGE:"+name+":"+str(len(encoded)))
    Path("/kaggle/working",name).write_bytes(encoded)
    receipts.append({
        "model_id":model_id,
        "resolution":resolution,
        "dataset_ref":dataset_ref,
        "dataset_version_number":int(target["version"]),
        "campaign_marker_sha256":str(target["campaign_marker_sha256"]),
        "file_name":name,
        "bytes":len(encoded),
    })

summary={
    "schema":"pneumonia.phase2.oof.vector.account.v1",
    "status":"PASS",
    "account_id":ACCOUNT_ID,
    "target_count":len(receipts),
    "targets":receipts,
    "training_performed":False,
    "inference_performed":False,
    "locked_test_used_for_selection":False,
    "external_used_for_selection":False,
}
summary_name="PHASE2_OOF_VECTOR_ACCOUNT_"+ACCOUNT_ID.replace("-","_").upper()+".json"
Path("/kaggle/working",summary_name).write_text(
    json.dumps(summary,sort_keys=True,separators=(",",":")),encoding="utf-8"
)
print("PHASE2_OOF_VECTOR_EXTRACT_PASS "+json.dumps({"account_id":ACCOUNT_ID,"target_count":len(receipts)},sort_keys=True))
'''
    return preamble + body


def _vector_slug(account_id: str, run_id: str) -> str:
    token = "master" if account_id == "master" else account_id
    return f"phase2-oof-vectors-{EXTRACTOR_GENERATION}-{token}-{run_id}"


def _candidate_prefixes(account_id: str) -> tuple[str, ...]:
    token = "master" if account_id == "master" else account_id
    return (
        f"phase2-oof-vectors-{EXTRACTOR_GENERATION}-{token}-",
        f"phase2-oof-vectors-{token}-",
    )


def fetch_account_summary(read_token: str, account_id: str, kernel_ref: str) -> dict | None:
    response = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "output_json_files",
            "account_id": account_id,
            "kernel_ref": kernel_ref,
            "file_names": [account_summary_name(account_id)],
            "max_bytes_per_file": MAX_JSON_BYTES,
        },
        timeout=180,
    )
    if not response.get("ok"):
        return None
    files = _read_payload(response).get("files") or []
    if len(files) != 1 or not isinstance(files[0], dict):
        return None
    value = files[0].get("json")
    if (
        not isinstance(value, dict)
        or value.get("schema") != "pneumonia.phase2.oof.vector.account.v1"
        or value.get("status") != "PASS"
        or value.get("account_id") != account_id
        or value.get("training_performed") is not False
        or value.get("inference_performed") is not False
        or value.get("locked_test_used_for_selection") is not False
        or value.get("external_used_for_selection") is not False
    ):
        return None
    return value


def reusable_account(
    read_token: str,
    account_id: str,
    owner: str,
    targets: list[dict],
) -> dict | None:
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
                "search": "phase2-oof-vectors",
                "sortBy": "DATE_RUN",
                "page": 1,
                "pageSize": 100,
            },
        },
        timeout=120,
    )
    if not listing.get("ok"):
        return None
    kernels = _read_payload(listing).get("kernels") or []
    candidates: list[tuple[int, str]] = []
    prefixes = _candidate_prefixes(account_id)
    for row in kernels:
        if not isinstance(row, dict):
            continue
        ref = str(row.get("ref") or "")
        if ref.count("/") != 1:
            continue
        slug = ref.split("/", 1)[1]
        for prefix in prefixes:
            if slug.casefold().startswith(prefix.casefold()):
                suffix = slug[len(prefix):]
                if suffix.isdigit():
                    candidates.append((int(suffix), ref))
                break
    for source_run_id, ref in sorted(candidates, reverse=True):
        status = recursive_status(
            post_json(
                READ_ENDPOINT,
                read_token,
                {
                    "action": "raw_read",
                    "account_id": account_id,
                    "service": "kernels.KernelsApiService",
                    "method": "GetKernelSessionStatus",
                    "body": {
                        "userName": ref.split("/", 1)[0],
                        "kernelSlug": ref.split("/", 1)[1],
                    },
                },
                timeout=120,
            )
        )
        if status in {"ERROR", "CANCELLED"}:
            continue
        summary = fetch_account_summary(read_token, account_id, ref)
        if summary is None or int(summary.get("target_count") or 0) != len(targets):
            continue
        expected_targets = sorted(
            (
                str(target["model_id"]),
                int(target["resolution"]),
                str(target["dataset_ref"]),
                int(target["version"]),
                str(target["campaign_marker_sha256"]),
            )
            for target in targets
        )
        observed_targets = []
        for target in summary.get("targets") or []:
            if not isinstance(target, dict):
                observed_targets = []
                break
            try:
                observed_targets.append(
                    (
                        str(target["model_id"]),
                        int(target["resolution"]),
                        str(target["dataset_ref"]),
                        int(target["dataset_version_number"]),
                        str(target["campaign_marker_sha256"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                observed_targets = []
                break
        if sorted(observed_targets) != expected_targets:
            continue
        return {
            "account_id": account_id,
            "owner": owner,
            "kernel_ref": ref,
            "targets": [],
            "status": "COMPLETE",
            "reused": True,
            "source_run_id": str(source_run_id),
        }
    return None


def launch_account(
    action_token: str,
    read_token: str,
    run_id: str,
    account_id: str,
    plan: dict,
) -> dict:
    owner = str(plan["owner"])
    targets = []
    for model_id, resolution, dataset_ref in plan["targets"]:
        resolved = resolve_version(read_token, account_id, dataset_ref)
        targets.append(
            {
                "model_id": model_id,
                "resolution": int(resolution),
                "dataset_ref": dataset_ref,
                **resolved,
            }
        )

    reused = reusable_account(read_token, account_id, owner, targets)
    if reused is not None:
        reused["targets"] = targets
        return reused
    slug = _vector_slug(account_id, run_id)
    kernel_ref = f"{owner}/{slug}"
    payload = {
        "request_id": f"phase2-oof-vectors-{EXTRACTOR_GENERATION}-{account_id}-{run_id}",
        "provider": "kaggle",
        "operation_class": "compute",
        "account_id": account_id,
        "purpose": (
            "Read sealed Phase-2 development OOF predictions and emit compact paired-statistics "
            "vectors only. CPU-only; no training, inference, HPO, threshold tuning, Locked-Test "
            "access, external validation, or model mutation."
        ),
        "service": "kernels.KernelsApiService",
        "method": "SaveKernel",
        "body": {
            "slug": kernel_ref,
            "newTitle": f"Phase2 OOF Vectors {account_id} {run_id}",
            "text": vector_kernel_script(account_id, targets),
            "language": "python",
            "kernelType": "script",
            "kernelExecutionType": "SAVE_AND_RUN_ALL",
            "isPrivate": True,
            "enableGpu": False,
            "enableTpu": False,
            "enableInternet": False,
            "kernelDataSources": [],
            "datasetDataSources": [target["dataset_ref"] for target in targets],
            "competitionDataSources": [],
            "modelDataSources": [],
        },
    }
    response = post_json(ACTION_ENDPOINT, action_token, payload, timeout=240)
    broker, provider = _action_envelope(response)
    error = str(broker.get("error") or provider.get("error") or response.get("error") or "")
    if not response.get("ok") or not broker.get("ok") or error:
        raise RuntimeError(f"OOF_VECTOR_SAVEKERNEL_REJECTED:{account_id}:{error or response}")
    invalid_datasets = provider.get("invalidDatasetSources") or []
    invalid_kernels = provider.get("invalidKernelSources") or []
    if invalid_datasets or invalid_kernels:
        raise RuntimeError(
            f"OOF_VECTOR_SOURCE_REJECTION:{account_id}:datasets={invalid_datasets}:kernels={invalid_kernels}"
        )
    ref = normalize_provider_ref(
        broker.get("provider_ref")
        or response.get("provider_ref")
        or provider.get("ref")
        or kernel_ref,
        kernel_ref,
        owner,
    )
    return {
        "account_id": account_id,
        "owner": owner,
        "kernel_ref": ref,
        "targets": targets,
        "status": "SUBMITTED",
        "reused": False,
        "source_run_id": str(run_id),
    }


def probe_status(read_token: str, row: dict) -> str:
    if row.get("reused"):
        return "COMPLETE"
    owner, slug = row["kernel_ref"].split("/", 1)
    response = post_json(
        READ_ENDPOINT,
        read_token,
        {
            "action": "raw_read",
            "account_id": row["account_id"],
            "service": "kernels.KernelsApiService",
            "method": "GetKernelSessionStatus",
            "body": {"userName": owner, "kernelSlug": slug},
        },
        timeout=120,
    )
    status = recursive_status(response)
    if status:
        return status
    summary = fetch_account_summary(read_token, row["account_id"], row["kernel_ref"])
    if summary is not None:
        return "COMPLETE"
    return "UNKNOWN"


def _retryable_output_propagation(response: dict) -> bool:
    error = str(response.get("error") or "")
    return (
        "requested Kaggle output JSON not found:" in error
        or "output JSON HTTP 404:" in error
    )


def _fetch_one_vector(read_token: str, launch: dict, name: str) -> dict:
    last_files = []
    last_error = None
    for attempt in range(31):
        response = post_json(
            READ_ENDPOINT,
            read_token,
            {
                "action": "output_json_files",
                "account_id": launch["account_id"],
                "kernel_ref": launch["kernel_ref"],
                "file_names": [name],
                "max_bytes_per_file": MAX_JSON_BYTES,
            },
            timeout=240,
        )
        if not response.get("ok"):
            last_error = response.get("error")
            if _retryable_output_propagation(response) and attempt < 30:
                time.sleep(10)
                continue
            if _retryable_output_propagation(response):
                raise RuntimeError(
                    f"OOF_VECTOR_OUTPUT_PROPAGATION_TIMEOUT:{launch['account_id']}:"
                    f"{name}:error={last_error}"
                )
            raise RuntimeError(
                f"OOF_VECTOR_OUTPUT_FETCH_FAILED:{launch['account_id']}:{name}:{response}"
            )

        payload = _read_payload(response)
        if payload.get("truncated") is True:
            raise RuntimeError(
                f"OOF_VECTOR_SINGLE_OUTPUT_RESPONSE_TRUNCATED:{launch['account_id']}:"
                f"{name}:{payload.get('original_json_chars')}"
            )
        files = payload.get("files") or []
        last_files = files
        if len(files) == 0:
            if attempt < 30:
                time.sleep(10)
                continue
            raise RuntimeError(
                f"OOF_VECTOR_OUTPUT_PROPAGATION_TIMEOUT:{launch['account_id']}:"
                f"{name}:files=0"
            )
        if len(files) != 1 or not isinstance(files[0], dict):
            raise RuntimeError(
                f"OOF_VECTOR_SINGLE_OUTPUT_COUNT_INVALID:{launch['account_id']}:"
                f"{name}:{len(files)}"
            )
        item = files[0]
        observed_name = str(item.get("file_name") or item.get("source_file_name") or "")
        value = item.get("json")
        if (
            observed_name != name
            or not isinstance(value, dict)
            or value.get("schema") != "pneumonia.phase2.oof.vector.v1"
            or value.get("status") != "PASS"
        ):
            raise RuntimeError(
                f"OOF_VECTOR_OUTPUT_INVALID:{launch['account_id']}:{name}:{observed_name}"
            )
        return value

    raise RuntimeError(
        f"OOF_VECTOR_OUTPUT_PROPAGATION_TIMEOUT:{launch['account_id']}:"
        f"{name}:files={len(last_files)}:error={last_error}"
    )


def fetch_vectors(read_token: str, launch: dict) -> list[dict]:
    names = [
        vector_output_name(target["model_id"], target["resolution"])
        for target in launch["targets"]
    ]
    # Read each vector separately. The read broker caps the serialized response at
    # 200k characters, so batching 3-5 otherwise-valid compressed vectors can turn
    # a complete result into a bounded {truncated, preview} envelope.
    return [_fetch_one_vector(read_token, launch, name) for name in names]


def validate_vectors(vectors: list[dict]) -> None:
    expected = {
        (f"M{i:02d}", resolution)
        for i in range(1, 13)
        for resolution in (224, 320, 384)
    }
    observed = {(str(v.get("model_id")), int(v.get("resolution"))) for v in vectors}
    if len(vectors) != 36 or observed != expected:
        raise RuntimeError(
            "OOF_VECTOR_COVERAGE_INVALID="
            + json.dumps(
                {
                    "count": len(vectors),
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                },
                sort_keys=True,
            )
        )
    identities = {str(v.get("identity_sequence_sha256") or "") for v in vectors}
    row_counts = {int(v.get("row_count") or 0) for v in vectors}
    if len(identities) != 1 or "" in identities or len(row_counts) != 1 or 0 in row_counts:
        raise RuntimeError(
            "OOF_VECTOR_GLOBAL_IDENTITY_MISMATCH:"
            + json.dumps({"identities": sorted(identities), "row_counts": sorted(row_counts)}, sort_keys=True)
        )
    for vector in vectors:
        if vector.get("locked_test_used_for_selection") not in (None, False):
            raise RuntimeError("OOF_VECTOR_LOCKED_SELECTION_VIOLATION")
        if vector.get("external_used_for_selection") not in (None, False):
            raise RuntimeError("OOF_VECTOR_EXTERNAL_SELECTION_VIOLATION")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not str(args.run_id).isdigit():
        raise SystemExit("OOF_VECTOR_RUN_ID_INVALID")
    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("OOF_VECTOR_OIDC_INVALID")

    launches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
        futures = {
            pool.submit(
                launch_account, action_token, read_token, str(args.run_id), account_id, plan
            ): account_id
            for account_id, plan in ACCOUNT_PLANS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            launches.append(row)
            print(
                "PHASE2_OOF_VECTOR_LAUNCH "
                + json.dumps(
                    {
                        "account_id": row["account_id"],
                        "kernel_ref": row["kernel_ref"],
                        "status": row["status"],
                        "reused": bool(row.get("reused")),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    launches.sort(key=lambda row: row["account_id"])

    terminal: dict[str, str] = {}
    for row in launches:
        if row.get("reused"):
            terminal[row["account_id"]] = "COMPLETE"

    deadline = time.monotonic() + 60 * 60
    while time.monotonic() < deadline and len(terminal) < len(launches):
        active = [row for row in launches if row["account_id"] not in terminal]
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
            futures = {pool.submit(probe_status, read_token, row): row for row in active}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                status = future.result()
                print(
                    "PHASE2_OOF_VECTOR_STATUS "
                    + json.dumps({"account_id": row["account_id"], "status": status}, sort_keys=True),
                    flush=True,
                )
                if status in TERMINAL:
                    terminal[row["account_id"]] = status
        if len(terminal) < len(launches):
            time.sleep(20)

    if len(terminal) != len(launches):
        raise SystemExit("OOF_VECTOR_TERMINAL_TIMEOUT")
    failures = {account: status for account, status in terminal.items() if status != "COMPLETE"}
    if failures:
        raise SystemExit("OOF_VECTOR_TERMINAL_FAILURE=" + json.dumps(failures, sort_keys=True))

    vectors: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
        futures = {pool.submit(fetch_vectors, read_token, launch): launch["account_id"] for launch in launches}
        for future in concurrent.futures.as_completed(futures):
            vectors.extend(future.result())
    vectors.sort(key=lambda v: (v["model_id"], int(v["resolution"])))
    validate_vectors(vectors)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for vector in vectors:
        path = out_dir / vector_output_name(vector["model_id"], vector["resolution"])
        encoded = json.dumps(vector, sort_keys=True, separators=(",", ":"))
        path.write_text(encoded, encoding="utf-8")
        files.append(path.name)

    manifest = {
        "schema": "pneumonia.phase2.oof.vector.collection.v1",
        "status": "PASS",
        "github_run_id": str(args.run_id),
        "accounts": len(launches),
        "model_resolution_units": len(vectors),
        "models": 12,
        "resolutions": [224, 320, 384],
        "training_performed": False,
        "inference_performed": False,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "files": files,
        "launches": [
            {
                "account_id": row["account_id"],
                "kernel_ref": row["kernel_ref"],
                "reused": bool(row.get("reused")),
                "source_run_id": row.get("source_run_id"),
            }
            for row in launches
        ],
    }
    (out_dir / "PHASE2_OOF_VECTOR_COLLECTION_V1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "PHASE2_OOF_VECTOR_COLLECTION_PASS "
        + json.dumps({"units": len(vectors), "accounts": len(launches)}, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
