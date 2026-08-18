interface SelectionEnv {
  CGP_PROJECT_CONTROL_TOKEN?: string;
  CGP_KAGGLE_KG02_TOKEN?: string;
  CGP_KAGGLE_KG03_TOKEN?: string;
  CGP_KAGGLE_KG04_TOKEN?: string;
  CGP_KAGGLE_KG05_TOKEN?: string;
  CGP_KAGGLE_KG06_TOKEN?: string;
  CGP_KAGGLE_KG07_TOKEN?: string;
  CGP_KAGGLE_MASTER_TOKEN?: string;
}

type AccountId = "master" | "kg-02" | "kg-03" | "kg-04" | "kg-05" | "kg-06" | "kg-07";

type CanonicalTask = {
  workerId: string;
  accountId: AccountId;
  ownerSlug: string;
  kernelRef: string;
};

const API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService";
const MAX_BYTES = 8 * 1024 * 1024;
const MAX_PAGES = 20;
const PAGE_SIZE = 100;
const VALIDATION_ARTIFACTS = new Set([
  "final_report.json",
  "validation_metrics.json",
  "val_predictions.csv",
]);
const GOVERNANCE_SOURCE_ARTIFACTS = new Set([
  "select_champion.py",
  "auto_ensemble_selection.py",
  "backbone_ensemble_selection.py",
  "balanced_metrics.py",
  "backbone_comparison_contracts.py",
  "source_package_integrity.py",
  "v6_fingerprint_contracts.py",
  "test_auto_ensemble_selection.py",
  "test_backbone_ensemble_selection.py",
]);

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function safeError(error: unknown): Response {
  return json({
    ok: false,
    error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
  }, 502);
}

function authorized(request: Request, env: SelectionEnv): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  if (!expected || expected.length < 32) return false;
  return request.headers.get("authorization") === `Bearer ${expected}`;
}

function taskForWorker(workerIdRaw: string): CanonicalTask {
  const match = workerIdRaw.trim().toUpperCase().match(/^W(\d{2})$/);
  if (!match) throw new Error("worker_id must use W01..W36 form");
  const workerNumber = Number(match[1]);
  if (workerNumber < 1 || workerNumber > 36) throw new Error("worker_id must be W01..W36");
  const workerId = `W${String(workerNumber).padStart(2, "0")}`;
  const wave = Math.floor((workerNumber - 1) / 6) + 1;
  const slot = (workerNumber - 1) % 6;

  let accountId: AccountId;
  let ownerSlug: string;
  let modelNumber: number;
  let resolution: number;

  if (wave === 1) {
    const baseline = [
      { accountId: "master" as const, ownerSlug: "azadka", modelNumber: 1, resolution: 224 },
      { accountId: "kg-02" as const, ownerSlug: "radlinaradlina", modelNumber: 1, resolution: 320 },
      { accountId: "kg-04" as const, ownerSlug: "reyhanehazad", modelNumber: 1, resolution: 384 },
      { accountId: "kg-05" as const, ownerSlug: "trickermark", modelNumber: 2, resolution: 224 },
      { accountId: "kg-06" as const, ownerSlug: "msdenis", modelNumber: 2, resolution: 320 },
      { accountId: "kg-07" as const, ownerSlug: "nisabulutmark", modelNumber: 2, resolution: 384 },
    ];
    const row = baseline[slot];
    accountId = row.accountId;
    ownerSlug = row.ownerSlug;
    modelNumber = row.modelNumber;
    resolution = row.resolution;
  } else {
    const accounts = [
      { accountId: "kg-02" as const, ownerSlug: "radlinaradlina" },
      { accountId: "kg-03" as const, ownerSlug: "rezanory" },
      { accountId: "kg-04" as const, ownerSlug: "reyhanehazad" },
      { accountId: "kg-05" as const, ownerSlug: "trickermark" },
      { accountId: "kg-06" as const, ownerSlug: "msdenis" },
      { accountId: "kg-07" as const, ownerSlug: "nisabulutmark" },
    ];
    const row = accounts[slot];
    accountId = row.accountId;
    ownerSlug = row.ownerSlug;
    modelNumber = 1 + (wave - 1) * 2 + (slot >= 3 ? 1 : 0);
    resolution = [224, 320, 384][slot % 3];
  }

  const modelId = `M${String(modelNumber).padStart(2, "0")}`;
  const slug = `pneumonia-v6-2-2-train-${workerId.toLowerCase()}-${modelId.toLowerCase()}-r${resolution}`;
  return { workerId, accountId, ownerSlug, kernelRef: `${ownerSlug}/${slug}` };
}

function tokenFor(env: SelectionEnv, accountId: AccountId): string {
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

function authHeader(token: string, username: string): string {
  return token.startsWith("KGAT_") ? `Bearer ${token}` : `Basic ${btoa(`${username}:${token}`)}`;
}

function nextPageToken(value: Record<string, unknown>): string {
  if (typeof value.nextPageToken === "string") return value.nextPageToken;
  if (typeof value.next_page_token === "string") return value.next_page_token;
  return "";
}

function outputRows(value: unknown): Array<{ fileName: string; url: string }> {
  if (!Array.isArray(value)) return [];
  const rows: Array<{ fileName: string; url: string }> = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const row = item as Record<string, unknown>;
    if (typeof row.fileName === "string" && typeof row.url === "string") {
      rows.push({ fileName: row.fileName.replaceAll("\\", "/"), url: row.url });
    }
  }
  return rows;
}

function allowedUrl(raw: string): URL {
  const url = new URL(raw);
  if (url.protocol !== "https:") throw new Error("artifact URL must be HTTPS");
  const host = url.hostname.toLowerCase();
  const allowed = host === "api.kaggle.com" || host === "www.kaggle.com" || host === "storage.googleapis.com" || host.endsWith(".kaggleusercontent.com") || host.endsWith(".googleusercontent.com");
  if (!allowed) throw new Error("artifact URL host is not allowlisted");
  return url;
}

async function locateArtifact(env: SelectionEnv, task: CanonicalTask, artifactName: string): Promise<{ fileName: string; url: string }> {
  const token = tokenFor(env, task.accountId);
  const kernelSlug = task.kernelRef.split("/")[1];
  let pageToken = "";
  const matches: Array<{ fileName: string; url: string }> = [];
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const body: Record<string, unknown> = { userName: task.ownerSlug, kernelSlug, pageSize: PAGE_SIZE };
    if (pageToken) body.pageToken = pageToken;
    const response = await fetch(`${API_ROOT}/ListKernelSessionOutput`, {
      method: "POST",
      headers: {
        Authorization: authHeader(token, task.ownerSlug),
        "Content-Type": "application/json",
        "User-Agent": "chatgpt-v622-selection-artifact-bridge/0.1",
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
    for (const row of outputRows(value.files)) {
      if (row.fileName === artifactName || row.fileName.endsWith(`/${artifactName}`)) matches.push(row);
    }
    const next = nextPageToken(value);
    if (!next || next === pageToken) break;
    pageToken = next;
  }
  if (matches.length < 1) throw new Error(`${artifactName} not found for ${task.workerId}`);
  matches.sort((a, b) => a.fileName.length - b.fileName.length || a.fileName.localeCompare(b.fileName));
  return matches[0];
}

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), byte => byte.toString(16).padStart(2, "0")).join("");
}

async function readArtifact(env: SelectionEnv, workerId: string, artifactNameRaw: string): Promise<Record<string, unknown>> {
  const artifactName = artifactNameRaw.trim();
  const task = taskForWorker(workerId);
  const isValidation = VALIDATION_ARTIFACTS.has(artifactName);
  const isGovernance = GOVERNANCE_SOURCE_ARTIFACTS.has(artifactName);
  if (!isValidation && !isGovernance) throw new Error("artifact_name is not in the V6.2.2 selection allowlist");
  if (isGovernance && task.workerId !== "W01") throw new Error("governance source artifacts are recovered only from canonical W01");
  const lower = artifactName.toLowerCase();
  if (lower.includes("test") && !lower.startsWith("test_auto_") && !lower.startsWith("test_backbone_")) {
    throw new Error("locked-test artifacts are forbidden");
  }
  if (lower.includes("external") || lower.includes("lockbox")) throw new Error("external/lockbox artifacts are forbidden");

  const located = await locateArtifact(env, task, artifactName);
  const url = allowedUrl(located.url);
  const response = await fetch(url.toString(), { redirect: "follow" });
  if (!response.ok) throw new Error(`${artifactName} download HTTP ${response.status}`);
  const declared = Number(response.headers.get("content-length") ?? "0");
  if (declared > MAX_BYTES) throw new Error(`${artifactName} exceeds selection artifact size limit`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > MAX_BYTES) throw new Error(`${artifactName} exceeds selection artifact size limit`);
  const sha256 = hex(await crypto.subtle.digest("SHA-256", bytes));
  const content = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  return {
    project: "PNEUMONIA V6.2.2",
    purpose: "validation_only_selection_artifact",
    worker_id: task.workerId,
    account_id: task.accountId,
    kernel_ref: task.kernelRef,
    artifact_name: artifactName,
    file_name: located.fileName,
    bytes: bytes.byteLength,
    sha256,
    content,
  };
}

export default {
  async fetch(request: Request, env: SelectionEnv): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") {
      return json({ service: "v622-selection-artifact-bridge", status: "ready", protected: true });
    }
    if (url.pathname === "/control/v6-2-2/selection-artifact" && request.method === "POST") {
      if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
      try {
        const raw: unknown = await request.json();
        const body = raw && typeof raw === "object" && !Array.isArray(raw) ? raw as Record<string, unknown> : {};
        return json(await readArtifact(env, String(body.worker_id ?? ""), String(body.artifact_name ?? "")));
      } catch (error) {
        return safeError(error);
      }
    }
    return new Response("Not found", { status: 404 });
  },
} satisfies ExportedHandler<SelectionEnv>;
