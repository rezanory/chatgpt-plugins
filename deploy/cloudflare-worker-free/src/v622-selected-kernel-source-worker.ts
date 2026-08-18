import { getKernel, type WorkerEnv } from "./kaggle";

type Env = WorkerEnv & { CGP_PROJECT_CONTROL_TOKEN?: string };

const ACCOUNT_ID = "kg-05";
const KERNEL_REF = "trickermark/pneumonia-v6-2-2-train-w16-m06-r224";
const EXPECTED_SOURCE_FINGERPRINT = "fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838";

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function authorized(request: Request, env: Env): boolean {
  const expected = env.CGP_PROJECT_CONTROL_TOKEN?.trim();
  return Boolean(expected && expected.length >= 32 && request.headers.get("authorization") === `Bearer ${expected}`);
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/healthz" && request.method === "GET") {
      return json({ service: "v622-selected-kernel-source", status: "ready", protected: true });
    }
    if (url.pathname !== "/control/v6-2-2/selected-kernel-source" || request.method !== "POST") {
      return new Response("Not found", { status: 404 });
    }
    if (!authorized(request, env)) return new Response("Forbidden", { status: 403 });
    try {
      const value = await getKernel(env, ACCOUNT_ID, KERNEL_REF);
      const blob = record(value.blob);
      const metadata = record(value.metadata);
      const source = typeof blob.source === "string" ? blob.source : "";
      if (!source) throw new Error("GetKernel returned no notebook source");
      if (!source.includes(EXPECTED_SOURCE_FINGERPRINT)) throw new Error("W16 source fingerprint mismatch");
      return json({
        project: "PNEUMONIA V6.2.2",
        purpose: "selected_kernel_source_recovery",
        account_id: ACCOUNT_ID,
        kernel_ref: KERNEL_REF,
        expected_source_fingerprint: EXPECTED_SOURCE_FINGERPRINT,
        metadata,
        blob: {
          source,
          language: blob.language ?? metadata.language ?? null,
          kernelType: blob.kernelType ?? metadata.kernelType ?? null,
        },
      });
    } catch (error) {
      return json({ ok: false, error: error instanceof Error ? error.message.slice(0, 1000) : "unknown error" }, 502);
    }
  },
} satisfies ExportedHandler<Env>;
