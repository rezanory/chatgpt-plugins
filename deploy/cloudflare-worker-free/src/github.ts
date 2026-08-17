import { rerunExisting, type WorkerEnv } from "./kaggle";

const CONTROL_MARKER = "<!-- chatgpt-plugins-kaggle-control:v1 -->";
const CLAIM_MARKER = "<!-- chatgpt-plugins-kaggle-control-claim:v1";
const RECEIPT_MARKER = "<!-- chatgpt-plugins-kaggle-control-receipt:v1";
const FAILURE_MARKER = "<!-- chatgpt-plugins-kaggle-control-failure:v1";
const CONTROL_SCHEMA = "chatgpt.kaggle.control/v1";

interface ControlCommand {
  schema: string;
  action: "rerun_existing";
  jobId: string;
  accountId: string;
  kernelRef: string;
}

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function parseCommand(title: string, body: string): ControlCommand {
  if (!title.startsWith("[KAGGLE-RUN]")) {
    throw new Error("control issue title must start with [KAGGLE-RUN]");
  }
  if (body.length > 50_000 || !body.includes(CONTROL_MARKER)) {
    throw new Error("control issue marker is missing or body is too large");
  }
  let payloadText = body.split(CONTROL_MARKER, 2)[1].trim();
  if (payloadText.startsWith("```json")) payloadText = payloadText.slice(7).trim();
  if (payloadText.endsWith("```")) payloadText = payloadText.slice(0, -3).trim();
  const payload: unknown = JSON.parse(payloadText);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("control payload must be a JSON object");
  }
  const record = payload as Record<string, unknown>;
  const allowed = new Set(["schema", "action", "job_id", "account_id", "kernel_ref"]);
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) throw new Error(`unsupported control field: ${key}`);
  }
  const schema = String(record.schema ?? "");
  const action = String(record.action ?? "");
  const jobId = String(record.job_id ?? "");
  const accountId = String(record.account_id ?? "");
  const kernelRef = String(record.kernel_ref ?? "");
  if (schema !== CONTROL_SCHEMA) throw new Error("unsupported control schema");
  if (action !== "rerun_existing") throw new Error("V0.1 only supports rerun_existing");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(jobId)) throw new Error("invalid job_id");
  if (!/^kg-[0-9]{2}$/.test(accountId)) throw new Error("invalid account_id");
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(kernelRef)) {
    throw new Error("kernel_ref must use owner/slug form");
  }
  return { schema, action: "rerun_existing", jobId, accountId, kernelRef };
}

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(left: string, right: string): boolean {
  const length = Math.max(left.length, right.length);
  let diff = left.length ^ right.length;
  for (let index = 0; index < length; index += 1) {
    diff |= (left.charCodeAt(index) || 0) ^ (right.charCodeAt(index) || 0);
  }
  return diff === 0;
}

async function validGitHubSignature(secret: string, rawBody: ArrayBuffer, signature: string): Promise<boolean> {
  if (!secret || !signature.startsWith("sha256=")) return false;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const digest = hex(await crypto.subtle.sign("HMAC", key, rawBody));
  return constantTimeEqual(`sha256=${digest}`, signature);
}

async function githubApi(
  env: WorkerEnv,
  path: string,
  init: RequestInit = {},
): Promise<unknown> {
  const token = env.CGP_GITHUB_TOKEN?.trim();
  if (!token) throw new Error("CGP_GITHUB_TOKEN is not configured");
  const response = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "chatgpt-kaggle-gateway-worker-free/0.1",
      ...(init.headers ?? {}),
    },
  });
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text.slice(0, 500);
    }
  }
  if (!response.ok) throw new Error(`GitHub API ${response.status}`);
  return payload;
}

async function comments(env: WorkerEnv, issueNumber: number): Promise<Array<Record<string, unknown>>> {
  const repository = env.CGP_CONTROL_REPOSITORY || "rezanory/chatgpt-plugins";
  const value = await githubApi(env, `/repos/${repository}/issues/${issueNumber}/comments?per_page=100`);
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    : [];
}

async function hasJobMarker(env: WorkerEnv, issueNumber: number, jobId: string): Promise<boolean> {
  const needles = [CLAIM_MARKER, RECEIPT_MARKER, FAILURE_MARKER].map(
    (marker) => `${marker} job_id=${jobId} -->`,
  );
  for (const comment of await comments(env, issueNumber)) {
    const body = String(comment.body ?? "");
    if (needles.some((needle) => body.includes(needle))) return true;
  }
  return false;
}

async function comment(env: WorkerEnv, issueNumber: number, body: string): Promise<void> {
  const repository = env.CGP_CONTROL_REPOSITORY || "rezanory/chatgpt-plugins";
  await githubApi(env, `/repos/${repository}/issues/${issueNumber}/comments`, {
    method: "POST",
    body: JSON.stringify({ body: body.slice(0, 60_000) }),
  });
}

async function executeClaimed(
  env: WorkerEnv,
  issueNumber: number,
  command: ControlCommand,
): Promise<void> {
  try {
    const result = await rerunExisting(env, command.accountId, command.kernelRef);
    const safeResult: Record<string, unknown> = {};
    for (const key of ["ref", "url", "versionNumber", "version_number", "error", "status", "kernelId"]) {
      if (result[key] !== undefined) safeResult[key] = result[key];
    }
    const receipt = {
      schema: "chatgpt.kaggle.control.receipt/v1",
      job_id: command.jobId,
      status: "submitted",
      account_id: command.accountId,
      kernel_ref: command.kernelRef,
      provider_result: safeResult,
    };
    await comment(
      env,
      issueNumber,
      `${RECEIPT_MARKER} job_id=${command.jobId} -->\nDirect Kaggle HTTP API submission accepted.\n\n` +
        `\`\`\`json\n${JSON.stringify(receipt)}\n\`\`\``,
    );
  } catch (error) {
    const failure = {
      schema: "chatgpt.kaggle.control.failure/v1",
      job_id: command.jobId,
      status: "failed",
      account_id: command.accountId,
      kernel_ref: command.kernelRef,
      error_type: error instanceof Error ? error.name : "Error",
      error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
    };
    try {
      await comment(
        env,
        issueNumber,
        `${FAILURE_MARKER} job_id=${command.jobId} -->\n` +
          `\`\`\`json\n${JSON.stringify(failure)}\n\`\`\``,
      );
    } catch {
      // The durable claim comment still prevents a duplicate execution if receipt publication fails.
    }
  }
}

export async function handleGitHubWebhook(
  request: Request,
  env: WorkerEnv,
  ctx: ExecutionContext,
): Promise<Response> {
  const secret = env.CGP_GITHUB_WEBHOOK_SECRET?.trim() ?? "";
  if (!secret) return jsonResponse({ accepted: false, reason: "webhook_not_configured" }, 503);
  const rawBody = await request.arrayBuffer();
  const signature = request.headers.get("x-hub-signature-256") ?? "";
  if (!(await validGitHubSignature(secret, rawBody, signature))) {
    return jsonResponse({ accepted: false, reason: "invalid_signature" }, 401);
  }
  if ((request.headers.get("x-github-event") ?? "") !== "issues") {
    return jsonResponse({ accepted: false, reason: "ignored_event" }, 202);
  }

  let payload: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(new TextDecoder().decode(rawBody));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("invalid_json");
    payload = parsed as Record<string, unknown>;
  } catch {
    return jsonResponse({ accepted: false, reason: "invalid_json" }, 400);
  }
  if (payload.action !== "opened") return jsonResponse({ accepted: false, reason: "ignored_action" }, 202);

  const repository = (payload.repository && typeof payload.repository === "object")
    ? String((payload.repository as Record<string, unknown>).full_name ?? "")
    : "";
  const expectedRepository = env.CGP_CONTROL_REPOSITORY || "rezanory/chatgpt-plugins";
  if (repository !== expectedRepository) {
    return jsonResponse({ accepted: false, reason: "wrong_repository" }, 403);
  }

  const issue = payload.issue && typeof payload.issue === "object"
    ? (payload.issue as Record<string, unknown>)
    : {};
  const issueUser = issue.user && typeof issue.user === "object"
    ? (issue.user as Record<string, unknown>)
    : {};
  const actor = String(issueUser.login ?? "").toLocaleLowerCase("en-US");
  const allowedActors = (env.CGP_GITHUB_ALLOWED_ACTORS || "rezanory")
    .split(",")
    .map((value) => value.trim().toLocaleLowerCase("en-US"))
    .filter(Boolean);
  if (!allowedActors.includes(actor)) {
    return jsonResponse({ accepted: false, reason: "actor_not_allowed" }, 403);
  }
  if (env.CGP_WRITE_ENABLED !== "1") {
    return jsonResponse({ accepted: false, reason: "write_disabled_until_recovery" }, 202);
  }
  if (!env.CGP_GITHUB_TOKEN?.trim()) {
    return jsonResponse({ accepted: false, reason: "github_journal_not_configured" }, 503);
  }

  let issueNumber: number;
  let command: ControlCommand;
  try {
    issueNumber = Number(issue.number);
    if (!Number.isInteger(issueNumber) || issueNumber < 1) throw new Error("invalid issue number");
    command = parseCommand(String(issue.title ?? ""), String(issue.body ?? ""));
  } catch (error) {
    return jsonResponse(
      {
        accepted: false,
        reason: error instanceof Error ? error.name : "Error",
        detail: error instanceof Error ? error.message.slice(0, 500) : "invalid control issue",
      },
      400,
    );
  }

  if (await hasJobMarker(env, issueNumber, command.jobId)) {
    return jsonResponse({ accepted: false, duplicate: true, job_id: command.jobId }, 202);
  }
  await comment(
    env,
    issueNumber,
    `${CLAIM_MARKER} job_id=${command.jobId} -->\n` +
      `Accepted \`${command.action}\` for \`${command.accountId}\` / \`${command.kernelRef}\`. ` +
      "Execution uses direct Kaggle HTTPS API from Cloudflare Workers Free.",
  );
  ctx.waitUntil(executeClaimed(env, issueNumber, command));
  return jsonResponse({ accepted: true, duplicate: false, job_id: command.jobId }, 202);
}
