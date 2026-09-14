import type { AccountId, WorkerEnv } from "./kaggle";

export type ControlPlaneV3Env = WorkerEnv & {
  CGP_PROJECT_CONTROL_TOKEN?: string;
  CGP_CONTROL_SCOPES?: string;
  CGP_CONTROL_EXPIRES_AT?: string;
  CGP_CONTROL_ALLOW_GLOBAL_SCOPE?: string;
  CGP_CONTROL_V3_MUTATION_ENABLED?: string;
};

export type KaggleOperationClass =
  | "read"
  | "write"
  | "compute"
  | "destructive"
  | "privileged";

export interface KaggleCallSpec {
  accountId: AccountId;
  service: string;
  method: string;
  body: Record<string, unknown>;
  operationClass: KaggleOperationClass;
  requiredScope?: string;
}

const API_ROOT = "https://api.kaggle.com/v1";
const IDENTIFIER = /^[A-Za-z][A-Za-z0-9_.]{0,127}$/;
const METHOD = /^[A-Za-z][A-Za-z0-9_]{0,127}$/;
const KERNEL_SLUG = /^[A-Za-z0-9._-]{1,200}$/;
const READ_METHOD = /^(Get|List|Search|Query|Read|Fetch|Download|Check|Describe|Validate)/;
const PHASE_MARKER = /(?:^|\n)CGP_PHASE:([A-Za-z0-9_.:-]{1,120})/g;
const LOG_SECRET = /(KGAT_[A-Za-z0-9_-]+|(?:token|secret|password|authorization)\s*[:=]\s*\S+)/gi;

const ACCOUNTS: Record<
  AccountId,
  { owner: string; username: string; envKey: keyof WorkerEnv }
> = {
  "kg-02": {
    owner: "radlinaradlina",
    username: "radlinaradlina",
    envKey: "CGP_KAGGLE_KG02_TOKEN",
  },
  "kg-03": {
    owner: "rezanory",
    username: "rezanory",
    envKey: "CGP_KAGGLE_KG03_TOKEN",
  },
  "kg-04": {
    owner: "reyhanehazad",
    username: "reyhanehazad",
    envKey: "CGP_KAGGLE_KG04_TOKEN",
  },
  "kg-05": {
    owner: "trickermark",
    username: "trickermark",
    envKey: "CGP_KAGGLE_KG05_TOKEN",
  },
  "kg-06": {
    owner: "msdenis",
    username: "msdenis",
    envKey: "CGP_KAGGLE_KG06_TOKEN",
  },
  "kg-07": {
    owner: "nisabulutmark",
    username: "nisabulutmark",
    envKey: "CGP_KAGGLE_KG07_TOKEN",
  },
  "kg-08": {
    owner: "azadkk",
    username: "azadkk",
    envKey: "CGP_KAGGLE_KG08_TOKEN",
  },
  "kg-09": {
    owner: "mylovevpn1",
    username: "mylovevpn1",
    envKey: "CGP_KAGGLE_KG09_TOKEN",
  },
  "kg-10": {
    owner: "computstu1",
    username: "computstu1",
    envKey: "CGP_KAGGLE_KG10_TOKEN",
  },
  "kg-11": {
    owner: "jobreza1",
    username: "jobreza1",
    envKey: "CGP_KAGGLE_KG11_TOKEN",
  },
  master: {
    owner: "azadka",
    username: "azadka",
    envKey: "CGP_KAGGLE_MASTER_TOKEN",
  },
};

function tokenFor(env: WorkerEnv, accountId: AccountId): string {
  const descriptor = ACCOUNTS[accountId];
  const value = env[descriptor.envKey];
  const token = typeof value === "string" ? value.trim() : "";
  if (!token) throw new Error(`Kaggle credential is not configured for ${accountId}`);
  return token;
}

function authHeader(accountId: AccountId, token: string): string {
  if (token.startsWith("KGAT_")) return `Bearer ${token}`;
  const username = ACCOUNTS[accountId].username;
  return `Basic ${btoa(`${username}:${token}`)}`;
}

function scopes(env: ControlPlaneV3Env): Set<string> {
  return new Set(
    String(env.CGP_CONTROL_SCOPES ?? "")
      .split(/[\s,]+/)
      .map((value) => value.trim())
      .filter(Boolean),
  );
}

function notExpired(env: ControlPlaneV3Env): boolean {
  const raw = env.CGP_CONTROL_EXPIRES_AT?.trim();
  if (!raw) return false;
  const numeric = Number(raw);
  const expiresAt = Number.isFinite(numeric)
    ? numeric < 10_000_000_000
      ? numeric * 1000
      : numeric
    : Date.parse(raw);
  if (!Number.isFinite(expiresAt)) return false;
  return Date.now() < expiresAt;
}

function scopeMatches(env: ControlPlaneV3Env, requiredScope: string): boolean {
  const allowed = scopes(env);
  if (allowed.has(requiredScope)) return true;
  for (const candidate of allowed) {
    if (candidate === "*" && env.CGP_CONTROL_ALLOW_GLOBAL_SCOPE === "1") return true;
    if (!candidate.endsWith(":*")) continue;
    const prefix = candidate.slice(0, -1);
    if (requiredScope.startsWith(prefix)) return true;
  }
  return false;
}

export function controlPlaneCapabilityAuthorized(
  request: Request,
  env: ControlPlaneV3Env,
  requiredScope: string,
): boolean {
  if (env.CGP_CONTROL_V3_MUTATION_ENABLED !== "1") return false;
  const token = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  if (!token || token.length < 32) return false;
  if (request.headers.get("authorization") !== `Bearer ${token}`) return false;
  if (!notExpired(env)) return false;
  return scopeMatches(env, requiredScope);
}

function defaultScope(spec: KaggleCallSpec): string {
  return `kaggle:${spec.operationClass}:${spec.service}:${spec.method}`;
}

function validateSpec(spec: KaggleCallSpec): void {
  if (!IDENTIFIER.test(spec.service)) throw new Error("invalid Kaggle service identifier");
  if (!METHOD.test(spec.method)) throw new Error("invalid Kaggle method identifier");
  if (!(spec.accountId in ACCOUNTS)) throw new Error("unknown Kaggle account");
  if (!spec.body || typeof spec.body !== "object" || Array.isArray(spec.body)) {
    throw new Error("Kaggle body must be an object");
  }
  if (spec.operationClass === "read" && !READ_METHOD.test(spec.method)) {
    throw new Error(
      `Kaggle method ${spec.method} is not read-like; use a scoped non-read operation class`,
    );
  }
}

function kernelSlug(accountId: AccountId, kernelRef: string): string {
  const [owner, slug, ...rest] = kernelRef.split("/");
  if (rest.length || !owner || !slug) throw new Error("kernel_ref must use owner/slug form");
  if (owner.toLowerCase() !== ACCOUNTS[accountId].owner.toLowerCase()) {
    throw new Error(`kernel owner does not match ${accountId}`);
  }
  if (!KERNEL_SLUG.test(slug)) throw new Error("kernel slug contains unsupported characters");
  return slug;
}

function sanitizeLog(raw: string, maxChars: number): string {
  const bounded = raw.length > maxChars ? raw.slice(-maxChars) : raw;
  return bounded.replace(LOG_SECRET, "<redacted>");
}

async function parseResponse(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();
  let value: unknown = {};
  if (text) {
    try {
      value = JSON.parse(text);
    } catch {
      throw new Error(`Kaggle returned non-JSON HTTP ${response.status}`);
    }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Kaggle returned a non-object response");
  }
  const record = value as Record<string, unknown>;
  const code = typeof record.code === "number" ? record.code : undefined;
  if (!response.ok || (code !== undefined && code >= 400)) {
    const message = typeof record.message === "string"
      ? record.message
      : `Kaggle API HTTP ${response.status}`;
    throw new Error(message.slice(0, 1200));
  }
  return record;
}

async function execute(
  env: ControlPlaneV3Env,
  spec: KaggleCallSpec,
): Promise<Record<string, unknown>> {
  validateSpec(spec);
  const token = tokenFor(env, spec.accountId);
  const url = `${API_ROOT}/${spec.service}/${spec.method}`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: authHeader(spec.accountId, token),
      "Content-Type": "application/json",
      "User-Agent": "chatgpt-control-plane-v3/1.0",
    },
    body: JSON.stringify(spec.body),
  });
  return parseResponse(response);
}

export async function kaggleReadCall(
  env: ControlPlaneV3Env,
  spec: Omit<KaggleCallSpec, "operationClass">,
): Promise<Record<string, unknown>> {
  return execute(env, { ...spec, operationClass: "read" });
}

export async function kaggleScopedCall(
  request: Request,
  env: ControlPlaneV3Env,
  spec: KaggleCallSpec,
): Promise<Record<string, unknown>> {
  if (spec.operationClass === "read") return execute(env, spec);
  const requiredScope = spec.requiredScope?.trim() || defaultScope(spec);
  if (!controlPlaneCapabilityAuthorized(request, env, requiredScope)) {
    throw new Error(`Control-plane capability missing/disabled/expired scope: ${requiredScope}`);
  }
  return execute(env, spec);
}

export async function kaggleLiveLog(
  env: ControlPlaneV3Env,
  accountId: AccountId,
  kernelRef: string,
  maxChars = 120_000,
): Promise<Record<string, unknown>> {
  if (!Number.isInteger(maxChars) || maxChars < 1000 || maxChars > 200_000) {
    throw new Error("maxChars must be an integer between 1000 and 200000");
  }
  const slug = kernelSlug(accountId, kernelRef);
  const value = await kaggleReadCall(env, {
    accountId,
    service: "kernels.KernelsApiService",
    method: "ListKernelSessionOutput",
    body: { userName: ACCOUNTS[accountId].owner, kernelSlug: slug, pageSize: 1 },
  });
  const log = sanitizeLog(String(value.log ?? ""), maxChars);
  const matches = [...log.matchAll(PHASE_MARKER)];
  const phase = matches.length ? matches[matches.length - 1][1] : null;
  return {
    account_id: accountId,
    kernel_ref: kernelRef,
    phase,
    phase_marker_found: phase !== null,
    log_tail: log,
    bounded: true,
  };
}

export async function kaggleOutputJsonFiles(
  env: ControlPlaneV3Env,
  accountId: AccountId,
  kernelRef: string,
  fileNames: string[],
  maxBytesPerFile = 65_536,
): Promise<Record<string, unknown>> {
  if (!Array.isArray(fileNames) || fileNames.length < 1 || fileNames.length > 10) {
    throw new Error("file_names must contain between 1 and 10 paths");
  }
  if (!Number.isInteger(maxBytesPerFile) || maxBytesPerFile < 1024 || maxBytesPerFile > 262_144) {
    throw new Error("max_bytes_per_file must be between 1024 and 262144");
  }
  const slug = kernelSlug(accountId, kernelRef);
  const exactNames = new Set<string>();
  for (const rawName of fileNames) {
    const name = String(rawName ?? "").trim();
    if (!/^[A-Za-z0-9._/-]{1,500}$/.test(name) || !name.endsWith(".json") ||
        name.startsWith("/") || name.includes("..") || name.includes("\\")) {
      throw new Error("file_names contains an unsafe or non-JSON path");
    }
    if (exactNames.has(name)) throw new Error("file_names must be unique");
    exactNames.add(name);
  }
  const listing = await kaggleReadCall(env, {
    accountId,
    service: "kernels.KernelsApiService",
    method: "ListKernelSessionOutput",
    body: { userName: ACCOUNTS[accountId].owner, kernelSlug: slug, pageSize: 100 },
  });
  const files = Array.isArray(listing.files) ? listing.files : [];
  const byName = new Map<string, Record<string, unknown>>();
  for (const item of files) {
    if (item && typeof item === "object" && !Array.isArray(item)) {
      const rec = item as Record<string, unknown>;
      const name = String(rec.fileName ?? "");
      if (name) byName.set(name, rec);
    }
  }
  const outputs: Array<Record<string, unknown>> = [];
  for (const name of exactNames) {
    const item = byName.get(name);
    if (!item) throw new Error(`requested Kaggle output JSON not found: ${name}`);
    const rawUrl = String(item.url ?? "");
    let url: URL;
    try { url = new URL(rawUrl); }
    catch { throw new Error(`invalid output URL for: ${name}`); }
    if (url.protocol !== "https:" || url.hostname !== "www.kaggleusercontent.com") {
      throw new Error(`output URL host is not allowed for: ${name}`);
    }
    const response = await fetch(url.toString(), {
      headers: { "User-Agent": "chatgpt-control-plane-v3/output-json/1.0" },
    });
    if (!response.ok) throw new Error(`output JSON HTTP ${response.status}: ${name}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > maxBytesPerFile) {
      throw new Error(`output JSON exceeds bounded size: ${name}`);
    }
    let value: unknown;
    try { value = JSON.parse(new TextDecoder().decode(bytes)); }
    catch { throw new Error(`output file is not valid JSON: ${name}`); }
    if (!value || typeof value !== "object") {
      throw new Error(`output JSON must be object/array: ${name}`);
    }
    outputs.push({ file_name: name, json: value });
  }
  return {
    account_id: accountId,
    kernel_ref: kernelRef,
    file_count: outputs.length,
    files: outputs,
    signed_urls_returned: false,
  };
}

export async function kagglePhaseProbe(
  env: ControlPlaneV3Env,
  accountId: AccountId,
  kernelRef: string,
): Promise<Record<string, unknown>> {
  const result = await kaggleLiveLog(env, accountId, kernelRef, 40_000);
  return {
    account_id: result.account_id,
    kernel_ref: result.kernel_ref,
    phase: result.phase,
    phase_marker_found: result.phase_marker_found,
    live_log_read_used: true,
    labels_or_metrics_required: false,
  };
}

export function kaggleAccountOwner(accountId: AccountId): string {
  return ACCOUNTS[accountId].owner;
}
