import canonicalWorker from "./index";
import {
  kaggleLiveLog,
  kagglePhaseProbe,
  kaggleReadCall,
  type ControlPlaneV3Env,
} from "./control-plane-v3-kaggle";
import { verifyGitHubReadBrokerOidc } from "./control-plane-v3-oidc";
import type { AccountId } from "./kaggle";

type Rec = Record<string, unknown>;

const ACCOUNTS = new Set<AccountId>([
  "kg-02",
  "kg-03",
  "kg-04",
  "kg-05",
  "kg-06",
  "kg-07",
  "master",
]);
const SECRET_KEY = /(token|secret|password|authorization|credential|api[_-]?key|cookie|signed[_-]?url)/i;
const SECRET_VALUE = /(KGAT_[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._~-]+|Basic\s+[A-Za-z0-9+/=]+|X-Goog-Signature=|X-Amz-Signature=)/i;
const MAX_RESPONSE_CHARS = 200_000;

function object(value: unknown): Rec {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Rec;
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function accountId(value: unknown): AccountId {
  const candidate = String(value ?? "") as AccountId;
  if (!ACCOUNTS.has(candidate)) throw new Error("unknown Kaggle account_id");
  return candidate;
}

function sanitize(value: unknown, depth = 0): unknown {
  if (depth > 12) return "<depth-limit>";
  if (typeof value === "string") {
    if (SECRET_VALUE.test(value)) return "<redacted>";
    return value.length > 120_000 ? `${value.slice(0, 120_000)}<truncated>` : value;
  }
  if (typeof value === "number" || typeof value === "boolean" || value === null) return value;
  if (Array.isArray(value)) return value.slice(0, 2000).map((item) => sanitize(item, depth + 1));
  if (!value || typeof value !== "object") return String(value ?? "");
  const result: Rec = {};
  for (const [key, child] of Object.entries(value as Rec).slice(0, 2000)) {
    result[key] = SECRET_KEY.test(key) ? "<redacted>" : sanitize(child, depth + 1);
  }
  return result;
}

function bounded(value: unknown): unknown {
  const sanitized = sanitize(value);
  const encoded = JSON.stringify(sanitized);
  if (encoded.length <= MAX_RESPONSE_CHARS) return sanitized;
  return {
    truncated: true,
    original_json_chars: encoded.length,
    preview: encoded.slice(0, MAX_RESPONSE_CHARS),
  };
}

async function handleKaggleRead(request: Request, env: ControlPlaneV3Env): Promise<Response> {
  const identity = await verifyGitHubReadBrokerOidc(request);
  const body = object(await request.json());
  const action = String(body.action ?? "raw_read");
  const account = accountId(body.account_id);
  let result: unknown;

  if (action === "raw_read") {
    const service = String(body.service ?? "");
    const method = String(body.method ?? "");
    result = await kaggleReadCall(env, {
      accountId: account,
      service,
      method,
      body: object(body.body),
    });
  } else if (action === "phase_probe") {
    result = await kagglePhaseProbe(env, account, String(body.kernel_ref ?? ""));
  } else if (action === "live_log") {
    const maxChars = Number(body.max_chars ?? 40_000);
    result = await kaggleLiveLog(env, account, String(body.kernel_ref ?? ""), maxChars);
  } else {
    throw new Error("unsupported Kaggle read action");
  }

  return json({
    ok: true,
    provider: "kaggle",
    read_only: true,
    broker_run_id: identity.run_id,
    broker_sha: identity.sha,
    result: bounded(result),
  });
}

function errorResponse(error: unknown, status = 403): Response {
  return json(
    {
      ok: false,
      read_only: true,
      error: error instanceof Error ? error.message.slice(0, 1200) : "unknown error",
    },
    status,
  );
}

export default {
  async fetch(request: Request, env: ControlPlaneV3Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/control-plane/v3/healthz" && request.method === "GET") {
      return json({
        service: "chatgpt-control-plane-v3",
        status: "ready",
        github_oidc_read_broker: true,
        trusted_repository_id: "1337215097",
        mutation_default_enabled: env.CGP_CONTROL_V3_MUTATION_ENABLED === "1",
      });
    }

    if (url.pathname === "/control-plane/v3/read/kaggle" && request.method === "POST") {
      try {
        return await handleKaggleRead(request, env);
      } catch (error) {
        return errorResponse(error);
      }
    }

    return canonicalWorker.fetch(request, env, ctx);
  },
} satisfies ExportedHandler<ControlPlaneV3Env>;
