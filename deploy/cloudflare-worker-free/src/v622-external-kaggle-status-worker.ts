import canonicalWorker from "./index";
import { kernelLogs, kernelStatus, type WorkerEnv } from "./kaggle";

type Rec = Record<string, unknown>;
const KERNEL_REF = "azadka/pneumonia-v6-2-2-external-validation";

function rec(v: unknown): Rec {
  return v && typeof v === "object" && !Array.isArray(v) ? v as Rec : {};
}

function normalizedStatus(raw: Rec): string {
  const session = rec(raw.session);
  for (const value of [raw.status, raw.statusName, raw.state, raw.sessionStatus, session.status, session.statusName, session.state]) {
    if (typeof value === "string" && value.trim()) return value.trim().toUpperCase();
  }
  return "UNKNOWN";
}

function errorExcerpt(log: string): string {
  if (!log) return "";
  const lines = log.split(/\r?\n/);
  let start = -1;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (/Traceback \(most recent call last\)|(?:Error|Exception):|RuntimeError|ValueError|FileNotFoundError|PermissionError|KeyError|TypeError/i.test(lines[i])) {
      start = Math.max(0, i - 18);
      break;
    }
  }
  const selected = start >= 0 ? lines.slice(start, Math.min(lines.length, start + 80)) : lines.slice(-80);
  return selected.join("\n").slice(-12000);
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

export default {
  async fetch(request: Request, env: WorkerEnv, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/recovery/v6-2-2/external-kernel-status" && request.method === "GET") {
      try {
        const raw = await kernelStatus(env, "master", KERNEL_REF);
        const status = normalizedStatus(raw);
        const terminalError = ["ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status);
        const log = terminalError ? await kernelLogs(env, "master", KERNEL_REF) : "";
        return json({
          project: "PNEUMONIA V6.2.2",
          stage: "EXTERNAL_VALIDATION",
          kernel_ref: KERNEL_REF,
          status,
          terminal: ["COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status),
          complete: status === "COMPLETE",
          needs_incident_review: terminalError,
          probe_only: true,
          kaggle_compute_launched_by_probe: false,
          error_excerpt: terminalError ? errorExcerpt(log) : "",
          log_tail: terminalError ? log.slice(-20000) : "",
          raw_status: raw,
        });
      } catch (error) {
        return json({
          project: "PNEUMONIA V6.2.2",
          stage: "EXTERNAL_VALIDATION",
          kernel_ref: KERNEL_REF,
          status: "PROBE_ERROR",
          probe_only: true,
          kaggle_compute_launched_by_probe: false,
          error: error instanceof Error ? error.message.slice(0, 1200) : "unknown error",
        }, 502);
      }
    }
    return canonicalWorker.fetch(request, env, ctx);
  },
} satisfies ExportedHandler<WorkerEnv>;
