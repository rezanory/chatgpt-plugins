import canonicalWorker from "./index";
import { kernelStatus, type WorkerEnv } from "./kaggle";

const ACCOUNT_ID = "kg-05";
const KERNEL_REF = "trickermark/pneumonia-v6-2-2-backbone-m06-r224";

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function normalizedStatus(raw: Record<string, unknown>): string {
  const session = record(raw.session);
  const candidates = [
    raw.status,
    raw.statusName,
    raw.status_name,
    raw.state,
    raw.sessionStatus,
    raw.kernelSessionStatus,
    session.status,
    session.statusName,
    session.state,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim().toUpperCase();
    }
  }
  return "UNKNOWN";
}

export default {
  async fetch(request: Request, env: WorkerEnv, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/recovery/v6-2-2/backbone-status" && request.method === "GET") {
      try {
        const raw = await kernelStatus(env, ACCOUNT_ID, KERNEL_REF);
        const status = normalizedStatus(raw);
        return json({
          project: "PNEUMONIA V6.2.2",
          stage: "post_ablation_backbone_comparison",
          account_id: ACCOUNT_ID,
          kernel_ref: KERNEL_REF,
          status,
          complete: status === "COMPLETE",
          terminal: ["COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status),
          needs_repair: ["ERROR", "FAILED", "CANCELLED", "CANCELED", "DEAD"].includes(status),
          probe_error: false,
          raw_status: raw,
          no_kaggle_compute_launched_by_probe: true,
          locked_test_used: false,
          external_data_used: false,
        });
      } catch (error) {
        return json({
          project: "PNEUMONIA V6.2.2",
          stage: "post_ablation_backbone_comparison",
          account_id: ACCOUNT_ID,
          kernel_ref: KERNEL_REF,
          status: "PROBE_ERROR",
          complete: false,
          terminal: false,
          needs_repair: false,
          probe_error: true,
          error_type: error instanceof Error ? error.name : "Error",
          error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
          no_kaggle_compute_launched_by_probe: true,
          locked_test_used: false,
          external_data_used: false,
        }, 502);
      }
    }
    return canonicalWorker.fetch(request, env, ctx);
  },
} satisfies ExportedHandler<WorkerEnv>;
