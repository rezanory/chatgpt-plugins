export type AccountId = "kg-02" | "kg-03" | "kg-04" | "kg-05" | "kg-06" | "kg-07";

export interface WorkerEnv {
  CGP_CONTROL_REPOSITORY: string;
  CGP_GITHUB_ALLOWED_ACTORS: string;
  CGP_WRITE_ENABLED: string;
  CGP_MCP_PATH_TOKEN?: string;
  CGP_GITHUB_WEBHOOK_SECRET?: string;
  CGP_GITHUB_TOKEN?: string;
  CGP_KAGGLE_KG02_TOKEN?: string;
  CGP_KAGGLE_KG03_TOKEN?: string;
  CGP_KAGGLE_KG04_TOKEN?: string;
  CGP_KAGGLE_KG05_TOKEN?: string;
  CGP_KAGGLE_KG06_TOKEN?: string;
  CGP_KAGGLE_KG07_TOKEN?: string;
}

interface AccountDescriptor {
  accountId: AccountId;
  ownerSlug: string;
  username: string;
}

interface AccountCredentials extends AccountDescriptor {
  token: string;
}

export const ACCOUNTS: readonly AccountDescriptor[] = [
  { accountId: "kg-02", ownerSlug: "radlinaradlina", username: "radlinaradlina" },
  { accountId: "kg-03", ownerSlug: "rezanory", username: "rezanory" },
  { accountId: "kg-04", ownerSlug: "reyhanehazad", username: "reyhanehazad" },
  { accountId: "kg-05", ownerSlug: "trickermark", username: "trickermark" },
  { accountId: "kg-06", ownerSlug: "msdenis", username: "msdenis" },
  { accountId: "kg-07", ownerSlug: "nisabulutmark", username: "nisabulutmark" },
] as const;

const KAGGLE_SERVICE = "kernels.KernelsApiService";
const KAGGLE_API_ROOT = "https://api.kaggle.com/v1";
const MAX_LOG_CHARS = 200_000;
const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024;
const MAX_ARTIFACT_TOTAL_BYTES = 16 * 1024 * 1024;
const MAX_ARTIFACT_FILES = 8;

export class KaggleGatewayError extends Error {
  readonly status: number;
  readonly code?: number;

  constructor(message: string, status: number, code?: number) {
    super(message);
    this.name = "KaggleGatewayError";
    this.status = status;
    this.code = code;
  }
}

function tokenFor(env: WorkerEnv, accountId: AccountId): string | undefined {
  switch (accountId) {
    case "kg-02":
      return env.CGP_KAGGLE_KG02_TOKEN;
    case "kg-03":
      return env.CGP_KAGGLE_KG03_TOKEN;
    case "kg-04":
      return env.CGP_KAGGLE_KG04_TOKEN;
    case "kg-05":
      return env.CGP_KAGGLE_KG05_TOKEN;
    case "kg-06":
      return env.CGP_KAGGLE_KG06_TOKEN;
    case "kg-07":
      return env.CGP_KAGGLE_KG07_TOKEN;
  }
}

function accountDescriptor(accountId: string): AccountDescriptor {
  const account = ACCOUNTS.find((candidate) => candidate.accountId === accountId);
  if (!account) {
    throw new Error(`unknown or disabled Kaggle account: ${accountId}`);
  }
  return account;
}

function accountCredentials(env: WorkerEnv, accountId: string): AccountCredentials {
  const account = accountDescriptor(accountId);
  const token = tokenFor(env, account.accountId)?.trim();
  if (!token) {
    throw new Error(`Kaggle credential is not configured for ${account.accountId}`);
  }
  return { ...account, token };
}

function splitKernelRef(account: AccountDescriptor, kernelRef: string): [string, string] {
  const parts = kernelRef.split("/");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    throw new Error("kernel_ref must use owner/slug form");
  }
  const [owner, slug] = parts;
  if (owner.toLocaleLowerCase("en-US") !== account.ownerSlug.toLocaleLowerCase("en-US")) {
    throw new Error(`kernel owner does not match ${account.accountId}`);
  }
  if (!/^[A-Za-z0-9._-]+$/.test(slug)) {
    throw new Error("kernel slug contains unsupported characters");
  }
  return [owner, slug];
}

async function parseResponse(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();
  if (!text) return {};
  try {
    const value: unknown = JSON.parse(text);
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
  } catch {
    if (!response.ok) {
      throw new KaggleGatewayError(`Kaggle HTTP ${response.status}`, response.status);
    }
  }
  throw new KaggleGatewayError("Kaggle returned a non-object response", response.status);
}

async function kaggleCall(
  env: WorkerEnv,
  accountId: string,
  requestName: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const account = accountCredentials(env, accountId);
  const authorization = account.token.startsWith("KGAT_")
    ? `Bearer ${account.token}`
    : `Basic ${btoa(`${account.username}:${account.token}`)}`;
  const response = await fetch(`${KAGGLE_API_ROOT}/${KAGGLE_SERVICE}/${requestName}`, {
    method: "POST",
    headers: {
      Authorization: authorization,
      "Content-Type": "application/json",
      "User-Agent": "chatgpt-kaggle-gateway-worker-free/0.1",
    },
    body: JSON.stringify(body),
  });
  const payload = await parseResponse(response);
  const code = typeof payload.code === "number" ? payload.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) {
    const rawMessage = typeof payload.message === "string" ? payload.message : "Kaggle API request failed";
    const message = rawMessage.slice(0, 1000);
    throw new KaggleGatewayError(message, response.status, code);
  }
  return payload;
}

export function publicAccounts(): Array<Record<string, string | boolean>> {
  return ACCOUNTS.map((account) => ({
    account_id: account.accountId,
    owner_slug: account.ownerSlug,
    enabled: true,
  }));
}

export async function listKernels(
  env: WorkerEnv,
  accountId: string,
  search = "",
  pageSize = 20,
): Promise<unknown[]> {
  if (!Number.isInteger(pageSize) || pageSize < 1 || pageSize > 100) {
    throw new Error("page_size must be between 1 and 100");
  }
  accountCredentials(env, accountId);
  const request: Record<string, unknown> = {
    group: "PROFILE",
    sortBy: "DATE_RUN",
    pageSize,
  };
  if (search.trim()) request.search = search.trim().slice(0, 200);
  const response = await kaggleCall(env, accountId, "ListKernels", request);
  return Array.isArray(response.kernels) ? response.kernels : [];
}

export async function authCheck(env: WorkerEnv, accountId: string): Promise<Record<string, unknown>> {
  const account = accountCredentials(env, accountId);
  const kernels = await listKernels(env, accountId, "", 1);
  return {
    account_id: account.accountId,
    owner_slug: account.ownerSlug,
    auth_ok: true,
    probe_count: kernels.length,
  };
}

export async function authCheckAll(env: WorkerEnv): Promise<Array<Record<string, unknown>>> {
  return Promise.all(
    ACCOUNTS.map(async (account) => {
      try {
        return await authCheck(env, account.accountId);
      } catch (error) {
        return {
          account_id: account.accountId,
          owner_slug: account.ownerSlug,
          auth_ok: false,
          error_type: error instanceof Error ? error.name : "Error",
          error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
        };
      }
    }),
  );
}

export async function inventoryAll(
  env: WorkerEnv,
  search: string,
  pageSize = 20,
): Promise<Array<Record<string, unknown>>> {
  if (!search.trim()) throw new Error("search is required");
  return Promise.all(
    ACCOUNTS.map(async (account) => {
      try {
        return {
          account_id: account.accountId,
          owner_slug: account.ownerSlug,
          ok: true,
          kernels: await listKernels(env, account.accountId, search, pageSize),
        };
      } catch (error) {
        return {
          account_id: account.accountId,
          owner_slug: account.ownerSlug,
          ok: false,
          error_type: error instanceof Error ? error.name : "Error",
          error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
        };
      }
    }),
  );
}

export async function kernelStatus(
  env: WorkerEnv,
  accountId: string,
  kernelRef: string,
): Promise<Record<string, unknown>> {
  const account = accountCredentials(env, accountId);
  const [, kernelSlug] = splitKernelRef(account, kernelRef);
  return kaggleCall(env, accountId, "GetKernelSessionStatus", {
    userName: account.ownerSlug,
    kernelSlug,
  });
}

export async function kernelLogs(
  env: WorkerEnv,
  accountId: string,
  kernelRef: string,
): Promise<string> {
  const account = accountCredentials(env, accountId);
  const [, kernelSlug] = splitKernelRef(account, kernelRef);
  const response = await kaggleCall(env, accountId, "ListKernelSessionOutput", {
    userName: account.ownerSlug,
    kernelSlug,
  });
  const log = String(response.log ?? "");
  return log.length > MAX_LOG_CHARS ? log.slice(-MAX_LOG_CHARS) : log;
}

interface OutputFile {
  fileName: string;
  url: string;
}

function outputFiles(value: unknown): OutputFile[] {
  if (!Array.isArray(value)) return [];
  const files: OutputFile[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const record = item as Record<string, unknown>;
    if (typeof record.fileName !== "string" || typeof record.url !== "string") continue;
    files.push({ fileName: record.fileName, url: record.url });
  }
  return files;
}

function safeArtifactName(value: string): string {
  const name = value.trim();
  if (!name || name.length > 160 || name.startsWith("/") || name.includes("..")) {
    throw new Error("artifact name is invalid");
  }
  if (!/^[A-Za-z0-9._/ -]+$/.test(name)) {
    throw new Error("artifact name contains unsupported characters");
  }
  return name;
}

function isSelectedFile(fileName: string, artifactNames: string[]): boolean {
  return artifactNames.some((name) => fileName === name || fileName.endsWith(`/${name}`));
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

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function kernelOutputManifest(
  env: WorkerEnv,
  accountId: string,
  kernelRef: string,
  requestedArtifactNames: string[],
  expectedFingerprint = "",
): Promise<Record<string, unknown>> {
  if (requestedArtifactNames.length < 1 || requestedArtifactNames.length > MAX_ARTIFACT_FILES) {
    throw new Error(`artifact_names must contain between 1 and ${MAX_ARTIFACT_FILES} entries`);
  }
  const artifactNames = requestedArtifactNames.map(safeArtifactName);
  const account = accountCredentials(env, accountId);
  const [, kernelSlug] = splitKernelRef(account, kernelRef);
  const output = await kaggleCall(env, accountId, "ListKernelSessionOutput", {
    userName: account.ownerSlug,
    kernelSlug,
    pageSize: 100,
  });
  const availableFiles = outputFiles(output.files);
  const selected = availableFiles.filter((file) => isSelectedFile(file.fileName, artifactNames));
  let totalBytes = 0;
  const files: Array<Record<string, unknown>> = [];
  const fingerprintHits: string[] = [];

  for (const file of selected.slice(0, MAX_ARTIFACT_FILES)) {
    const url = allowedOutputUrl(file.url);
    const response = await fetch(url.toString(), { redirect: "follow" });
    if (!response.ok) {
      files.push({ file_name: file.fileName, ok: false, status: response.status });
      continue;
    }
    const declaredLength = Number(response.headers.get("content-length") ?? "0");
    if (declaredLength > MAX_ARTIFACT_BYTES) {
      files.push({ file_name: file.fileName, ok: false, skipped: "file_too_large" });
      continue;
    }
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > MAX_ARTIFACT_BYTES || totalBytes + bytes.byteLength > MAX_ARTIFACT_TOTAL_BYTES) {
      files.push({ file_name: file.fileName, ok: false, skipped: "artifact_budget_exceeded" });
      continue;
    }
    totalBytes += bytes.byteLength;
    const digest = hex(await crypto.subtle.digest("SHA-256", bytes));
    let containsFingerprint = false;
    if (expectedFingerprint) {
      const text = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
      containsFingerprint = text.includes(expectedFingerprint);
      if (containsFingerprint) fingerprintHits.push(file.fileName);
    }
    files.push({
      file_name: file.fileName,
      ok: true,
      bytes: bytes.byteLength,
      sha256: digest,
      expected_fingerprint_present: containsFingerprint,
    });
  }

  return {
    account_id: account.accountId,
    kernel_ref: kernelRef,
    requested_artifact_names: artifactNames,
    available_file_count: availableFiles.length,
    available_file_names: availableFiles.slice(0, 100).map((file) => file.fileName),
    files,
    fingerprint_hits: fingerprintHits,
    expected_fingerprint: expectedFingerprint || null,
  };
}

export async function getKernel(
  env: WorkerEnv,
  accountId: string,
  kernelRef: string,
): Promise<Record<string, unknown>> {
  const account = accountCredentials(env, accountId);
  const [, kernelSlug] = splitKernelRef(account, kernelRef);
  return kaggleCall(env, accountId, "GetKernel", {
    userName: account.ownerSlug,
    kernelSlug,
  });
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
}

function copyIfPresent(target: Record<string, unknown>, source: Record<string, unknown>, key: string): void {
  const value = source[key];
  if (value !== undefined && value !== null && value !== "") target[key] = value;
}

export async function rerunExisting(
  env: WorkerEnv,
  accountId: string,
  kernelRef: string,
): Promise<Record<string, unknown>> {
  const account = accountCredentials(env, accountId);
  splitKernelRef(account, kernelRef);
  const current = await getKernel(env, accountId, kernelRef);
  const metadata = asRecord(current.metadata);
  const blob = asRecord(current.blob);
  const source = typeof blob.source === "string" ? blob.source : "";
  if (!source) throw new Error("GetKernel returned no source; refusing rerun");

  const request: Record<string, unknown> = {
    slug: kernelRef,
    text: source,
    kernelExecutionType: "SAVE_AND_RUN_ALL",
  };
  if (typeof metadata.id === "number" && metadata.id > 0) request.id = metadata.id;
  copyIfPresent(request, metadata, "title");
  if (request.title !== undefined) {
    request.newTitle = request.title;
    delete request.title;
  }
  copyIfPresent(request, blob, "language");
  if (request.language === undefined) copyIfPresent(request, metadata, "language");
  copyIfPresent(request, blob, "kernelType");
  if (request.kernelType === undefined) copyIfPresent(request, metadata, "kernelType");
  for (const key of [
    "datasetDataSources",
    "kernelDataSources",
    "competitionDataSources",
    "categoryIds",
    "isPrivate",
    "enableGpu",
    "enableInternet",
    "dockerImagePinningType",
    "modelDataSources",
    "enableTpu",
    "sessionTimeoutSeconds",
    "priority",
    "dockerImage",
    "machineShape",
  ]) {
    copyIfPresent(request, metadata, key);
  }

  return kaggleCall(env, accountId, "SaveKernel", request);
}
