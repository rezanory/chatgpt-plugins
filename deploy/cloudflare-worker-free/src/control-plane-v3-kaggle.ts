import type { AccountId, WorkerEnv } from "./kaggle";

export type ControlPlaneV3Env = WorkerEnv & {
  CGP_PROJECT_CONTROL_TOKEN?: string;
  CGP_CONTROL_SCOPES?: string;
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

export function controlPlaneCapabilityAuthorized(
  request: Request,
  env: ControlPlaneV3Env,
  requiredScope: string,
): boolean {
  const token = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  if (!token || token.length < 32) return false;
  if (request.headers.get("authorization") !== `Bearer ${token}`) return false;
  const allowed = scopes(env);
  return allowed.has("*") || allowed.has(requiredScope);
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
    throw new Error(`Control-plane capability missing required scope: ${requiredScope}`);
  }
  return execute(env, spec);
}

export function kaggleAccountOwner(accountId: AccountId): string {
  return ACCOUNTS[accountId].owner;
}
