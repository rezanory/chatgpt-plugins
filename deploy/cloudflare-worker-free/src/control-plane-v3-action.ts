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

export async function executeKaggleOidcAction(request: Request, env: ActionEnv): Promise<Rec> {
  const identity = await verifyGitHubActionBrokerOidc(request);
  const payload = object(await request.json());
  const id = accountId(payload.account_id);
  const opClass = operationClass(payload.operation_class);
  const service = String(payload.service ?? "");
  const method = String(payload.method ?? "");
  const body = object(payload.body ?? {});
  if (identity.ref === M07_REPAIR_REF) {
    const slug = typeof body.slug === "string" ? body.slug : "";
    const repairAuthorized =
      identity.workflow_ref === M07_REPAIR_WORKFLOW_REF &&
      id === "kg-05" &&
      opClass === "compute" &&
      service === "kernels.KernelsApiService" &&
      method === "SaveKernel" &&
      slug.startsWith("trickermark/m07-") &&
      body.isPrivate === true;
    if (!repairAuthorized) {
      throw new Error("M07 repair branch is restricted to private kg-05 M07 SaveKernel compute");
    }
  }
  const result = await kaggleAction(env, id, service, method, body);

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
