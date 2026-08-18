import { kernelStatus, type AccountId, type WorkerEnv } from "./kaggle";

type Resolution = 224 | 320 | 384;

interface CanonicalTask {
  workerId: string;
  accountId: AccountId;
  ownerSlug: string;
  modelId: string;
  resolution: Resolution;
  kernelRef: string;
}

interface AccountAuth {
  username: string;
  token: string;
}

const KAGGLE_API_ROOT = "https://api.kaggle.com/v1";
const KAGGLE_SERVICE = "kernels.KernelsApiService";
const RESOLUTIONS: readonly Resolution[] = [224, 320, 384];
const OUTPUT_PAGE_SIZE = 100;
const MAX_OUTPUT_PAGES = 4;
const MAX_JSON_BYTES = 2 * 1024 * 1024;

const WORKER_ACCOUNTS = [
  { accountId: "kg-02" as const, ownerSlug: "radlinaradlina" },
  { accountId: "kg-03" as const, ownerSlug: "rezanory" },
  { accountId: "kg-04" as const, ownerSlug: "reyhanehazad" },
  { accountId: "kg-05" as const, ownerSlug: "trickermark" },
  { accountId: "kg-06" as const, ownerSlug: "msdenis" },
  { accountId: "kg-07" as const, ownerSlug: "nisabulutmark" },
] as const;

function tokenFor(env: WorkerEnv, accountId: AccountId): string {
  const value =
    accountId === "master" ? env.CGP_KAGGLE_MASTER_TOKEN :
    accountId === "kg-02" ? env.CGP_KAGGLE_KG02_TOKEN :
    accountId === "kg-03" ? env.CGP_KAGGLE_KG03_TOKEN :
    accountId === "kg-04" ? env.CGP_KAGGLE_KG04_TOKEN :
    accountId === "kg-05" ? env.CGP_KAGGLE_KG05_TOKEN :
    accountId === "kg-06" ? env.CGP_KAGGLE_KG06_TOKEN :
    env.CGP_KAGGLE_KG07_TOKEN;
  const token = value?.trim();
  if (!token) throw new Error(`Kaggle token missing for ${accountId}`);
  return token;
}

function authFor(env: WorkerEnv, task: CanonicalTask): AccountAuth {
  return { username: task.ownerSlug, token: tokenFor(env, task.accountId) };
}

function authorization(auth: AccountAuth): string {
  return auth.token.startsWith("KGAT_")
    ? `Bearer ${auth.token}`
    : `Basic ${btoa(`${auth.username}:${auth.token}`)}`;
}

function normalizeStatus(payload: Record<string, unknown>): string {
  const value = payload.status;
  return typeof value === "string" && value.trim() ? value.trim().toUpperCase() : "UNKNOWN";
}

function taskForWave(wave: number, slot: number): CanonicalTask {
  if (!Number.isInteger(wave) || wave < 1 || wave > 6) throw new Error("wave must be 1..6");
  if (!Number.isInteger(slot) || slot < 0 || slot > 5) throw new Error("slot must be 0..5");

  if (wave === 1) {
    const baseline = [
      { workerId: "W01", accountId: "master" as const, ownerSlug: "azadka", modelId: "M01", resolution: 224 as const },
      { workerId: "W02", accountId: "kg-02" as const, ownerSlug: "radlinaradlina", modelId: "M01", resolution: 320 as const },
      { workerId: "W03", accountId: "kg-04" as const, ownerSlug: "reyhanehazad", modelId: "M01", resolution: 384 as const },
      { workerId: "W04", accountId: "kg-05" as const, ownerSlug: "trickermark", modelId: "M02", resolution: 224 as const },
      { workerId: "W05", accountId: "kg-06" as const, ownerSlug: "msdenis", modelId: "M02", resolution: 320 as const },
      { workerId: "W06", accountId: "kg-07" as const, ownerSlug: "nisabulutmark", modelId: "M02", resolution: 384 as const },
    ] as const;
    const row = baseline[slot];
    const slug = `pneumonia-v6-2-2-train-${row.workerId.toLowerCase()}-${row.modelId.toLowerCase()}-r${row.resolution}`;
    return { ...row, kernelRef: `${row.ownerSlug}/${slug}` };
  }

  const firstModelNumber = 1 + (wave - 1) * 2;
  const modelNumber = firstModelNumber + (slot >= 3 ? 1 : 0);
  const modelId = `M${String(modelNumber).padStart(2, "0")}`;
  const resolution = RESOLUTIONS[slot % 3];
  const workerNumber = 1 + (wave - 1) * 6 + slot;
  const workerId = `W${String(workerNumber).padStart(2, "0")}`;
  const account = WORKER_ACCOUNTS[slot];
  const slug = `pneumonia-v6-2-2-train-${workerId.toLowerCase()}-${modelId.toLowerCase()}-r${resolution}`;
  return {
    workerId,
    accountId: account.accountId,
    ownerSlug: account.ownerSlug,
    modelId,
    resolution,
    kernelRef: `${account.ownerSlug}/${slug}`,
  };
}

function allowedOutputUrl(rawUrl: string): URL {
  const url = new URL(rawUrl);
  if (url.protocol !== "https:") throw new Error("Kaggle output URL is not HTTPS");
  const host = url.hostname.toLocaleLowerCase("en-US");
  const allowed =
    host === "api.kaggle.com" ||
    host === "www.kaggle.com" ||
    host === "storage.googleapis.com" ||
    host.endsWith(".kaggleusercontent.com") ||
    host.endsWith(".googleusercontent.com");
  if (!allowed) throw new Error("Kaggle output URL host is not allowlisted");
  return url;
}

function nextPageToken(value: Record<string, unknown>): string {
  if (typeof value.nextPageToken === "string") return value.nextPageToken;
  if (typeof value.next_page_token === "string") return value.next_page_token;
  return "";
}

async function findValidationMetricsUrl(env: WorkerEnv, task: CanonicalTask): Promise<string> {
  const auth = authFor(env, task);
  const kernelSlug = task.kernelRef.split("/")[1];
  let pageToken = "";
  for (let page = 0; page < MAX_OUTPUT_PAGES; page += 1) {
    const body: Record<string, unknown> = {
      userName: task.ownerSlug,
      kernelSlug,
      pageSize: OUTPUT_PAGE_SIZE,
    };
    if (pageToken) body.pageToken = pageToken;
    const response = await fetch(`${KAGGLE_API_ROOT}/${KAGGLE_SERVICE}/ListKernelSessionOutput`, {
      method: "POST",
      headers: {
        Authorization: authorization(auth),
        "Content-Type": "application/json",
        "User-Agent": "chatgpt-kaggle-v622-metrics/0.1",
      },
      body: JSON.stringify(body),
    });
    const text = await response.text();
    let value: Record<string, unknown> = {};
    try {
      const parsed: unknown = text ? JSON.parse(text) : {};
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) value = parsed as Record<string, unknown>;
    } catch {
      throw new Error(`ListKernelSessionOutput returned non-JSON HTTP ${response.status}`);
    }
    if (!response.ok) throw new Error(`ListKernelSessionOutput HTTP ${response.status}`);
    const files = Array.isArray(value.files) ? value.files : [];
    for (const file of files) {
      if (!file || typeof file !== "object" || Array.isArray(file)) continue;
      const row = file as Record<string, unknown>;
      const fileName = typeof row.fileName === "string" ? row.fileName : "";
      const url = typeof row.url === "string" ? row.url : "";
      if (fileName.endsWith("/validation_metrics.json") && url) return url;
    }
    const next = nextPageToken(value);
    if (!next || next === pageToken) break;
    pageToken = next;
  }
  throw new Error("validation_metrics.json not found");
}

async function readValidationMetrics(env: WorkerEnv, task: CanonicalTask): Promise<Record<string, unknown>> {
  const rawUrl = await findValidationMetricsUrl(env, task);
  const url = allowedOutputUrl(rawUrl);
  const response = await fetch(url.toString(), { redirect: "follow" });
  if (!response.ok) throw new Error(`validation_metrics.json HTTP ${response.status}`);
  const declaredLength = Number(response.headers.get("content-length") ?? "0");
  if (declaredLength > MAX_JSON_BYTES) throw new Error("validation_metrics.json is unexpectedly large");
  const text = await response.text();
  if (text.length > MAX_JSON_BYTES) throw new Error("validation_metrics.json exceeds safety limit");
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("validation_metrics.json is not an object");
  return value as Record<string, unknown>;
}

export async function v622ValidationWaveResults(env: WorkerEnv, wave: number): Promise<Record<string, unknown>> {
  if (!Number.isInteger(wave) || wave < 1 || wave > 6) throw new Error("metrics wave must be 1..6");
  const tasks = Array.from({ length: 6 }, (_, slot) => taskForWave(wave, slot));
  const results: Array<Record<string, unknown>> = [];
  for (const task of tasks) {
    try {
      const statusPayload = await kernelStatus(env, task.accountId, task.kernelRef);
      const status = normalizeStatus(statusPayload);
      if (status !== "COMPLETE") {
        results.push({
          worker_id: task.workerId,
          account_id: task.accountId,
          model_id: task.modelId,
          resolution: task.resolution,
          kernel_ref: task.kernelRef,
          status,
          metrics: null,
        });
        continue;
      }
      const metrics = await readValidationMetrics(env, task);
      results.push({
        worker_id: task.workerId,
        account_id: task.accountId,
        model_id: task.modelId,
        resolution: task.resolution,
        kernel_ref: task.kernelRef,
        status,
        metrics,
      });
    } catch (error) {
      results.push({
        worker_id: task.workerId,
        account_id: task.accountId,
        model_id: task.modelId,
        resolution: task.resolution,
        kernel_ref: task.kernelRef,
        status: "READ_ERROR",
        metrics: null,
        error: error instanceof Error ? error.message.slice(0, 500) : "unknown error",
      });
    }
  }
  return {
    project: "PNEUMONIA V6.2.2",
    report: "validation_metrics",
    wave,
    generated_at: new Date().toISOString(),
    results,
  };
}
