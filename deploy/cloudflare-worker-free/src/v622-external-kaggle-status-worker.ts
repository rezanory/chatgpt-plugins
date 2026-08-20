import canonicalWorker from "./index";
import { kernelStatus, type WorkerEnv } from "./kaggle";

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
        return json({
          project: "PNEUMONIA V6.2.2",
          stage: "EXTERNAL_VALIDATION",
          kernel_ref: KERNEL_REF,
          status,
          terminal: ["COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status),
          complete: status === "COMPLETE",
          needs_incident_review: ["ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status),
          probe_only: true,
          kaggle_compute_launched_by_probe: false,
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
