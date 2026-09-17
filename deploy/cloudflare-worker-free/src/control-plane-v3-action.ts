import type { AccountId, WorkerEnv } from "./kaggle";
import type { KaggleOperationClass } from "./control-plane-v3-kaggle";
import { verifyGitHubActionBrokerOidc } from "./control-plane-v3-oidc";

type Rec = Record<string, unknown>;

type ActionEnv = WorkerEnv;

const API_ROOT = "https://api.kaggle.com/v1";
const IDENTIFIER = /^[A-Za-z][A-Za-z0-9_.]{0,127}$/;
const METHOD = /^[A-Za-z][A-Za-z0-9_]{0,127}$/;
const ACTION_CLASSES = new Set<KaggleOperationClass>([
  "write",
  "compute",
  "destructive",
  "privileged",
]);
const M07_REPAIR_REF = "refs/heads/fix/m07-r320-producer-source-v1";
const M07_REPAIR_WORKFLOW_REF =
  "rezanory/chatgpt-plugins/.github/workflows/pneumonia-v17-m07-continuation-20260914.yml@refs/heads/fix/m07-r320-producer-source-v1";
const ACCOUNTS: Record<
  AccountId,
  { owner: string; username: string; envKey: keyof WorkerEnv }
> = {
  "kg-02": { owner: "radlinaradlina", username: "radlinaradlina", envKey: "CGP_KAGGLE_KG02_TOKEN" },
  "kg-03": { owner: "rezanory", username: "rezanory", envKey: "CGP_KAGGLE_KG03_TOKEN" },
  "kg-04": { owner: "reyhanehazad", username: "reyhanehazad", envKey: "CGP_KAGGLE_KG04_TOKEN" },
  "kg-05": { owner: "trickermark", username: "trickermark", envKey: "CGP_KAGGLE_KG05_TOKEN" },
  "kg-06": { owner: "msdenis", username: "msdenis", envKey: "CGP_KAGGLE_KG06_TOKEN" },
  "kg-07": { owner: "nisabulutmark", username: "nisabulutmark", envKey: "CGP_KAGGLE_KG07_TOKEN" },
  "kg-08": { owner: "azadkk", username: "azadkk", envKey: "CGP_KAGGLE_KG08_TOKEN" },
  "kg-09": { owner: "mylovevpn1", username: "mylovevpn1", envKey: "CGP_KAGGLE_KG09_TOKEN" },
  "kg-10": { owner: "computstu1", username: "computstu1", envKey: "CGP_KAGGLE_KG10_TOKEN" },
  "kg-11": { owner: "jobreza1", username: "jobreza1", envKey: "CGP_KAGGLE_KG11_TOKEN" },
  master: { owner: "azadka", username: "azadka", envKey: "CGP_KAGGLE_MASTER_TOKEN" },
};

export function kaggleActionAccountIds(): string[] {
  return Object.keys(ACCOUNTS).sort();
}

function object(value: unknown): Rec {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("action payload/body must be an object");
  }
  return value as Rec;
}

function accountId(value: unknown): AccountId {
  const id = String(value ?? "") as AccountId;
  if (!(id in ACCOUNTS)) throw new Error("unknown Kaggle account_id");
  return id;
}

function operationClass(value: unknown): KaggleOperationClass {
  const candidate = String(value ?? "") as KaggleOperationClass;
  if (!ACTION_CLASSES.has(candidate)) {
    throw new Error("action operation_class must be write/compute/destructive/privileged");
  }
  return candidate;
}

function tokenFor(env: WorkerEnv, id: AccountId): string {
  const raw = env[ACCOUNTS[id].envKey];
  const token = typeof raw === "string" ? raw.trim() : "";
  if (!token) throw new Error(`Kaggle credential is not configured for ${id}`);
  return token;
}

function authorization(id: AccountId, token: string): string {
  if (token.startsWith("KGAT_")) return `Bearer ${token}`;
  return `Basic ${btoa(`${ACCOUNTS[id].username}:${token}`)}`;
}

async function kaggleAction(
  env: ActionEnv,
  id: AccountId,
  service: string,
  method: string,
  body: Rec,
): Promise<Rec> {
  if (!IDENTIFIER.test(service)) throw new Error("invalid Kaggle service identifier");
  if (!METHOD.test(method)) throw new Error("invalid Kaggle method identifier");
  const token = tokenFor(env, id);
  const response = await fetch(`${API_ROOT}/${service}/${method}`, {
    method: "POST",
    headers: {
      Authorization: authorization(id, token),
      "Content-Type": "application/json",
      "User-Agent": "chatgpt-control-plane-v3-action/1.0",
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let parsed: unknown = {};
  try {
    parsed = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Kaggle ${method} returned non-JSON HTTP ${response.status}`);
  }
  const result = object(parsed);
  const code = typeof result.code === "number" ? result.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) {
    const message = typeof result.message === "string" ? result.message : `${method} HTTP ${response.status}`;
    throw new Error(message.slice(0, 1200));
  }
  return result;
}

const M07_R320_PRODUCER_REF = "rezanory/m07-phase2-unlock-20260915";
const M07_R320_CAMPAIGN_RECEIPT = "e3884dd10e323d2f0925660599a420964781111d850664e8c88a830324460c77";
const M07_R320_FOLD4_RECEIPT = "93e6a75a26b9e9c2d911d0703d9e4683c218b0e51e85ada0d3146cb4255abb38";
const M07_R320_RECIPE_FP = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b";
const M07_R320_SPLIT_FP = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9";
const M07_R320_ARCHIVE_HASHES: Record<string, string> = {
  "FOLD_1_RECOVERY.zip": "cd18468bf9305bb801f809098f8917462b16fb69fe9313b8bd17f2a95a7d1531",
  "FOLD_2_RECOVERY.zip": "a75967888cec5cea6133a2ed457bb998edb9ada01442be88b2cc35e970e6f72d",
  "FOLD_3_RECOVERY.zip": "4b04f657ddfcab61dc720f17ba4a8a5a8f234cd2bb7ed10b64e7357e049a9ad6",
  "FOLD_4_RECOVERY.zip": "74a7c69c572a66a139beb99dfa375185ce2f25e96689642eac2a1e56928deb93",
};
const M07_R320_OUTPUT_PREFIX = "M07_GATE_MULTIRES_V17/STATE/R320/";

async function createM07R320ProducerBridge(env: ActionEnv, body: Rec): Promise<Rec> {
  const bridgeRef = String(body.slug ?? "").trim();
  if (!/^trickermark\/m07-r320-producer-bridge-[0-9]+$/.test(bridgeRef)) {
    throw new Error("M07 producer bridge slug is invalid");
  }

  const required = new Set<string>([
    `${M07_R320_OUTPUT_PREFIX}CAMPAIGN_STATE.json`,
    ...Object.keys(M07_R320_ARCHIVE_HASHES).map((name) => `${M07_R320_OUTPUT_PREFIX}${name}`),
  ]);
  const found = new Map<string, { url: string; bytes: number | null }>();
  let pageToken = "";
  for (let page = 0; page < 20 && found.size < required.size; page += 1) {
    const requestBody: Rec = {
      userName: "rezanory",
      kernelSlug: "m07-phase2-unlock-20260915",
      pageSize: 100,
    };
    if (pageToken) requestBody.pageToken = pageToken;
    const listing = await kaggleAction(
      env,
      "kg-03",
      "kernels.KernelsApiService",
      "ListKernelSessionOutput",
      requestBody,
    );
    const rows = Array.isArray(listing.files) ? listing.files : [];
    for (const item of rows) {
      if (!item || typeof item !== "object" || Array.isArray(item)) continue;
      const record = item as Rec;
      const name = String(record.fileName ?? "").replaceAll("\\", "/");
      if (!required.has(name)) continue;
      const rawUrl = String(record.url ?? "");
      let url: URL;
      try {
        url = new URL(rawUrl);
      } catch {
        throw new Error(`M07 producer output URL is invalid for ${name}`);
      }
      if (url.protocol !== "https:" || url.hostname !== "www.kaggleusercontent.com") {
        throw new Error(`M07 producer output URL host is invalid for ${name}`);
      }
      const rawBytes = Number(record.totalBytes ?? record.bytes ?? record.fileSize ?? record.size ?? NaN);
      found.set(name, {
        url: rawUrl,
        bytes: Number.isFinite(rawBytes) && rawBytes >= 0 ? rawBytes : null,
      });
    }
    const next = String(listing.nextPageToken ?? listing.next_page_token ?? "").trim();
    if (!next || next === pageToken) break;
    pageToken = next;
  }
  const missing = [...required].filter((name) => !found.has(name));
  if (missing.length) throw new Error(`M07 producer bridge source files missing: ${missing.join(",")}`);

  const files: Rec = {};
  for (const name of required) {
    const item = found.get(name)!;
    files[name] = {
      url: item.url,
      sha256: M07_R320_ARCHIVE_HASHES[name.slice(M07_R320_OUTPUT_PREFIX.length)] ?? null,
      bytes: item.bytes,
    };
  }
  const config = {
    producer_ref: M07_R320_PRODUCER_REF,
    campaign_receipt_sha256: M07_R320_CAMPAIGN_RECEIPT,
    fold4_receipt_sha256: M07_R320_FOLD4_RECEIPT,
    recipe_fingerprint: M07_R320_RECIPE_FP,
    split_fingerprint: M07_R320_SPLIT_FP,
    archive_sha256: M07_R320_ARCHIVE_HASHES,
    files,
  };
  const configLiteral = JSON.stringify(JSON.stringify(config));
  const script = [
    "from pathlib import Path",
    "import hashlib, json, urllib.request, urllib.parse, zipfile, shutil",
    `CONFIG=json.loads(${configLiteral})`,
    "work=Path('/kaggle/working')",
    "state=work/'M07_GATE_MULTIRES_V17'/'STATE'/'R320'",
    "state.mkdir(parents=True,exist_ok=True)",
    "observed={}",
    "for rel,spec in CONFIG['files'].items():",
    "    url=spec['url']",
    "    parsed=urllib.parse.urlparse(url)",
    "    assert parsed.scheme=='https' and parsed.hostname=='www.kaggleusercontent.com', 'M07_BRIDGE_URL_HOST_INVALID'",
    "    target=state/Path(rel).name",
    "    req=urllib.request.Request(url,headers={'User-Agent':'m07-r320-producer-bridge/1.0'})",
    "    h=hashlib.sha256()",
    "    with urllib.request.urlopen(req,timeout=300) as response, target.open('wb') as out:",
    "        while True:",
    "            chunk=response.read(8*1024*1024)",
    "            if not chunk: break",
    "            out.write(chunk); h.update(chunk)",
    "    digest=h.hexdigest()",
    "    observed[target.name]=digest",
    "    expected=spec.get('sha256')",
    "    if expected is not None: assert digest==expected, 'M07_BRIDGE_ARCHIVE_SHA_MISMATCH:'+target.name",
    "marker=state/'CAMPAIGN_STATE.json'",
    "campaign=json.loads(marker.read_text(encoding='utf-8'))",
    "assert campaign.get('schema')=='phase2.state.v2' and campaign.get('status')=='COMPLETE', 'M07_BRIDGE_CAMPAIGN_SCHEMA_STATUS_INVALID'",
    "assert campaign.get('model_id')=='M07' and int(campaign.get('resolution'))==320, 'M07_BRIDGE_CAMPAIGN_IDENTITY_INVALID'",
    "assert campaign.get('receipt_sha256')==CONFIG['campaign_receipt_sha256'], 'M07_BRIDGE_CAMPAIGN_RECEIPT_MISMATCH'",
    "assert campaign.get('run_contract',{}).get('split_fingerprint')==CONFIG['split_fingerprint'], 'M07_BRIDGE_SPLIT_FP_MISMATCH'",
    "assert campaign.get('run_contract',{}).get('extra',{}).get('recipe_fingerprint')==CONFIG['recipe_fingerprint'], 'M07_BRIDGE_RECIPE_FP_MISMATCH'",
    "assert campaign.get('artifact_sha256')==CONFIG['archive_sha256'], 'M07_BRIDGE_ARCHIVE_INVENTORY_MISMATCH'",
    "with zipfile.ZipFile(state/'FOLD_4_RECOVERY.zip','r') as z:",
    "    matches=[n for n in z.namelist() if n.endswith('FOLDS/fold_4/COMPLETED.json') or n.endswith('fold_4/COMPLETED.json')]",
    "    assert len(matches)==1, 'M07_BRIDGE_FOLD4_RECEIPT_PATH_INVALID'",
    "    fold4=json.loads(z.read(matches[0]).decode('utf-8'))",
    "assert fold4.get('status')=='COMPLETED' and int(fold4.get('fold_id'))==4 and int(fold4.get('resolution'))==320, 'M07_BRIDGE_FOLD4_IDENTITY_INVALID'",
    "assert fold4.get('receipt_sha256')==CONFIG['fold4_receipt_sha256'], 'M07_BRIDGE_FOLD4_RECEIPT_MISMATCH'",
    "assert fold4.get('confirmed_m07_recipe_fingerprint')==CONFIG['recipe_fingerprint'], 'M07_BRIDGE_FOLD4_RECIPE_MISMATCH'",
    "assert fold4.get('split_fingerprint')==CONFIG['split_fingerprint'], 'M07_BRIDGE_FOLD4_SPLIT_MISMATCH'",
    "assert fold4.get('locked_test_used_for_training') is False and fold4.get('external_used_for_training') is False, 'M07_BRIDGE_FOLD4_POLICY_MISMATCH'",
    "receipt={'schema':'m07.r320.producer.bridge.v1','status':'COMPLETE','producer_ref':CONFIG['producer_ref'],'campaign_receipt_sha256':CONFIG['campaign_receipt_sha256'],'fold4_receipt_sha256':CONFIG['fold4_receipt_sha256'],'recipe_fingerprint':CONFIG['recipe_fingerprint'],'split_fingerprint':CONFIG['split_fingerprint'],'archive_sha256':CONFIG['archive_sha256'],'folds_sealed':[1,2,3,4],'fold5_sealed':False}",
    "(work/'M07_R320_PRODUCER_BRIDGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True),encoding='utf-8')",
    "print('CGP_PHASE:M07_R320_PRODUCER_BRIDGE_COMPLETE')",
    "print('M07_R320_PRODUCER_BRIDGE '+json.dumps({k:v for k,v in receipt.items() if k!='archive_sha256'},sort_keys=True,separators=(',',':')))",
  ].join("\n");

  const save = await kaggleAction(
    env,
    "kg-05",
    "kernels.KernelsApiService",
    "SaveKernel",
    {
      slug: bridgeRef,
      newTitle: `M07 R320 Producer Bridge ${bridgeRef.split("-").at(-1) ?? ""}`,
      text: script,
      language: "PYTHON",
      kernelType: "SCRIPT",
      kernelExecutionType: "SAVE_AND_RUN_ALL",
      isPrivate: true,
      enableGpu: false,
      enableInternet: true,
    },
  );
  return {
    ref: typeof save.ref === "string" ? save.ref : bridgeRef,
    versionNumber: save.versionNumber ?? null,
    bridge_ref: bridgeRef,
    producer_ref: M07_R320_PRODUCER_REF,
    source_file_count: required.size,
    expected_archive_sha256: M07_R320_ARCHIVE_HASHES,
    signed_urls_returned: false,
    gpu: false,
    training: false,
  };
}

export async function executeKaggleOidcAction(request: Request, env: ActionEnv): Promise<Rec> {
  const identity = await verifyGitHubActionBrokerOidc(request);
  const payload = object(await request.json());
  const id = accountId(payload.account_id);
  const opClass = operationClass(payload.operation_class);
  const service = String(payload.service ?? "");
  const method = String(payload.method ?? "");
  const body = object(payload.body ?? {});
  let repairBridge = false;
  if (identity.ref === M07_REPAIR_REF) {
    const slug = typeof body.slug === "string" ? body.slug : "";
    const repairSaveKernel =
      service === "kernels.KernelsApiService" &&
      method === "SaveKernel" &&
      slug.startsWith("trickermark/m07-") &&
      body.isPrivate === true;
    repairBridge =
      service === "m07.RecoveryBridge" &&
      method === "CreateProducerMirror" &&
      /^trickermark\/m07-r320-producer-bridge-[0-9]+$/.test(slug);
    const repairAuthorized =
      identity.workflow_ref === M07_REPAIR_WORKFLOW_REF &&
      id === "kg-05" &&
      opClass === "compute" &&
      (repairSaveKernel || repairBridge);
    if (!repairAuthorized) {
      throw new Error("M07 repair branch is restricted to private kg-05 M07 SaveKernel/producer-bridge compute");
    }
  }
  const result = repairBridge
    ? await createM07R320ProducerBridge(env, body)
    : await kaggleAction(env, id, service, method, body);

  const requestedRef = typeof body.slug === "string" ? body.slug : null;
  const providerRef = typeof result.ref === "string" ? result.ref : null;
  return {
    ok: true,
    provider: "kaggle",
    operation_class: opClass,
    broker_run_id: identity.run_id,
    broker_sha: identity.sha,
    account_id: id,
    service,
    method,
    requested_ref: requestedRef,
    provider_ref: providerRef,
    provider_ref_authoritative: providerRef !== null,
    result,
  };
}
