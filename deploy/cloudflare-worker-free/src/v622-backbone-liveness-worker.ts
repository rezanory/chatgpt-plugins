import monitorWorker from "./v622-backbone-monitor-index";
import { kernelLogs, kernelStatus, type WorkerEnv } from "./kaggle";

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

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export default {
  async fetch(request: Request, env: WorkerEnv, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/recovery/v6-2-2/backbone-liveness" && request.method === "GET") {
      try {
        const rawStatus = await kernelStatus(env, ACCOUNT_ID, KERNEL_REF);
        const status = normalizedStatus(rawStatus);
        const log = await kernelLogs(env, ACCOUNT_ID, KERNEL_REF);
        const lines = log ? log.split(/\r?\n/).length : 0;
        return json({
          project: "PNEUMONIA V6.2.2",
          stage: "post_ablation_backbone_comparison_liveness",
          account_id: ACCOUNT_ID,
          kernel_ref: KERNEL_REF,
          status,
          log_chars: log.length,
          log_lines: lines,
          log_sha256: await sha256(log),
          probe_generated_at: new Date().toISOString(),
          no_kaggle_compute_launched_by_probe: true,
          locked_test_used: false,
          external_data_used: false,
          raw_log_exposed: false,
        });
      } catch (error) {
        return json({
          project: "PNEUMONIA V6.2.2",
          stage: "post_ablation_backbone_comparison_liveness",
          account_id: ACCOUNT_ID,
          kernel_ref: KERNEL_REF,
          status: "PROBE_ERROR",
          probe_error: true,
          error_type: error instanceof Error ? error.name : "Error",
          error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error",
          probe_generated_at: new Date().toISOString(),
          no_kaggle_compute_launched_by_probe: true,
          locked_test_used: false,
          external_data_used: false,
          raw_log_exposed: false,
        }, 502);
      }
    }
    return monitorWorker.fetch(request, env, ctx);
  },
} satisfies ExportedHandler<WorkerEnv>;
